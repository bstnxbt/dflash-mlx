# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import logging
import os
import shutil
import struct
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Optional

import mlx.core as mx

import dflash_mlx
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot

_LOG = logging.getLogger(__name__)

L2_SCHEMA_VERSION = 4
L2_FILE_SUFFIX = ".safetensors"
L2_TMP_SUFFIX = ".tmp.safetensors"
L2_LOCK_NAME = ".dflash_l2.lock"
L2_LAYOUT_ROOT = "v2"
CONTEXT_REPRESENTATION = "draft_projected"

_VALID_KINDS = ("prefill", "generation")
_HEX = set("0123456789abcdef")

def _is_json_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)

def _json_int_field(d: dict[str, Any], name: str) -> int:
    value = d[name]
    if not _is_json_int(value):
        raise ValueError(f"expected integer key field {name!r}")
    return value

def _sha256_field(d: dict[str, Any], name: str) -> str:
    value = d[name]
    if not isinstance(value, str) or len(value) != 64 or any(c not in _HEX for c in value):
        raise ValueError(f"expected sha256 key field {name!r}")
    return value

def _key_to_dict(key: DFlashPrefixKey) -> dict[str, Any]:
    return {
        "target_model_id": key.target_model_id,
        "draft_model_id": key.draft_model_id,
        "capture_layer_ids": list(key.capture_layer_ids),
        "draft_sink_size": int(key.draft_sink_size),
        "draft_window_size": int(key.draft_window_size),
        "template_hash": key.template_hash,
        "prompt_policy_hash": key.prompt_policy_hash,
        "target_fa_window": int(key.target_fa_window),
        "format_version": int(key.format_version),
    }

def _key_from_dict(d: dict[str, Any]) -> DFlashPrefixKey:
    if not isinstance(d, dict):
        raise ValueError("expected key metadata object")
    target_model_id = d["target_model_id"]
    draft_model_id = d["draft_model_id"]
    capture_layer_ids = d["capture_layer_ids"]
    template_hash = _sha256_field(d, "template_hash")
    prompt_policy_hash = _sha256_field(d, "prompt_policy_hash")
    if not isinstance(target_model_id, str):
        raise ValueError("expected string target_model_id")
    if not isinstance(draft_model_id, str):
        raise ValueError("expected string draft_model_id")
    if (
        not isinstance(capture_layer_ids, list)
        or not all(_is_json_int(x) for x in capture_layer_ids)
    ):
        raise ValueError("expected integer capture_layer_ids")
    return DFlashPrefixKey(
        target_model_id=target_model_id,
        draft_model_id=draft_model_id,
        capture_layer_ids=tuple(capture_layer_ids),
        draft_sink_size=_json_int_field(d, "draft_sink_size"),
        draft_window_size=_json_int_field(d, "draft_window_size"),
        template_hash=template_hash,
        prompt_policy_hash=prompt_policy_hash,
        target_fa_window=_json_int_field(d, "target_fa_window"),
        format_version=_json_int_field(d, "format_version"),
    )

def _canon_key_blob(key: DFlashPrefixKey) -> bytes:
    return json.dumps(
        _key_to_dict(key), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

def _canon_tokens_blob(tokens: tuple[int, ...] | list[int]) -> bytes:
    n = len(tokens)
    if n == 0:
        return b""
    return struct.pack(f"<{n}q", *(int(t) for t in tokens))

def _key_hash(key: DFlashPrefixKey) -> str:
    return hashlib.sha256(_canon_key_blob(key)).hexdigest()[:32]

def _token_hash(tokens: tuple[int, ...] | list[int]) -> str:
    return hashlib.sha256(_canon_tokens_blob(tokens)).hexdigest()[:16]

def _token_hashes_for_lengths(
    tokens: tuple[int, ...],
    lengths: set[int],
) -> dict[int, str]:
    wanted = sorted(int(n) for n in lengths if int(n) >= 0)
    if not wanted:
        return {}
    out: dict[int, str] = {}
    if wanted[0] == 0:
        out[0] = hashlib.sha256(b"").hexdigest()[:16]
    max_len = wanted[-1]
    h = hashlib.sha256()
    next_idx = 0
    while next_idx < len(wanted) and wanted[next_idx] == 0:
        next_idx += 1
    for pos, token in enumerate(tokens[:max_len], start=1):
        h.update(struct.pack("<q", int(token)))
        while next_idx < len(wanted) and wanted[next_idx] == pos:
            out[pos] = h.copy().hexdigest()[:16]
            next_idx += 1
    return out

def _runtime_layout_hash() -> str:
    blob = f"schema={L2_SCHEMA_VERSION}|dflash={dflash_mlx.__version__}".encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]

def _fingerprint(snapshot: DFlashPrefixSnapshot) -> str:
    payload = {
        "key": _key_to_dict(snapshot.key),
        "kind": snapshot.kind,
        "tokens": list(snapshot.token_ids),
        "schema": L2_SCHEMA_VERSION,
        "runtime": dflash_mlx.__version__,
        "context_representation": CONTEXT_REPRESENTATION,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

@dataclass(frozen=True)
class _Parts:
    token_len: int
    token_hash: str
    kind: str
    fp_short: str

def _format_filename(
    *, token_len: int, token_hash: str, kind: str, fp_short: str
) -> str:
    return f"{token_len:010d}-{token_hash}-{kind}-{fp_short}{L2_FILE_SUFFIX}"

def _parse_filename(name: str) -> Optional[_Parts]:
    if not name.endswith(L2_FILE_SUFFIX) or name.endswith(L2_TMP_SUFFIX):
        return None
    base = name[: -len(L2_FILE_SUFFIX)]
    parts = base.split("-")
    if len(parts) != 4:
        return None
    len_s, t_hash, kind, fp_short = parts
    if not len_s.isdigit():
        return None
    if len(t_hash) != 16 or not all(c in _HEX for c in t_hash):
        return None
    if kind not in _VALID_KINDS:
        return None
    if len(fp_short) != 16 or not all(c in _HEX for c in fp_short):
        return None
    try:
        token_len = int(len_s)
    except ValueError:
        return None
    if token_len < 0:
        return None
    return _Parts(token_len, t_hash, kind, fp_short)

def _eval_arrays(arrays_mlx: dict[str, mx.array]) -> None:
    items = list(arrays_mlx.values())
    if items:
        mx.eval(*items)

def _serialize(snapshot: DFlashPrefixSnapshot) -> tuple[dict[str, mx.array], dict[str, str]]:
    arrays: dict[str, mx.array] = {}
    fa_present: list[bool] = []
    fa_offsets: dict[str, int] = {}
    fa_indices: dict[str, int] = {}
    for i, fa in enumerate(snapshot.fa_states):
        if fa is None:
            fa_present.append(False)
        else:
            k, v, offset = fa[:3]
            arrays[f"fa_{i}_k"] = k
            arrays[f"fa_{i}_v"] = v
            fa_present.append(True)
            fa_offsets[str(i)] = int(offset)
            if len(fa) > 3:
                fa_indices[str(i)] = int(fa[3])

    gdn_present: list[bool] = []
    gdn_arity: list[int] = []
    gdn_array_present: list[list[bool]] = []
    for i, gdn in enumerate(snapshot.gdn_states):
        if gdn is None:
            gdn_present.append(False)
            gdn_arity.append(0)
            gdn_array_present.append([])
        else:
            gdn_present.append(True)
            gdn_arity.append(len(gdn))
            mask: list[bool] = []
            for j, a in enumerate(gdn):
                if a is None:
                    mask.append(False)
                else:
                    arrays[f"gdn_{i}_{j}"] = a
                    mask.append(True)
            gdn_array_present.append(mask)

    chunk_spans: list[list[int]] = []
    for c, chunk in enumerate(snapshot.target_hidden_chunks):
        arrays[f"target_hidden_{c}"] = chunk
    for span in snapshot.target_hidden_chunk_spans:
        chunk_spans.append([int(span[0]), int(span[1])])

    has_last_logits = snapshot.last_logits is not None
    if has_last_logits:
        arrays["last_logits"] = snapshot.last_logits

    meta = {
        "schema_version": L2_SCHEMA_VERSION,
        "runtime_version": dflash_mlx.__version__,
        "context_representation": CONTEXT_REPRESENTATION,
        "key": _key_to_dict(snapshot.key),
        "kind": snapshot.kind,
        "token_ids": list(snapshot.token_ids),
        "created_at": float(snapshot.created_at),
        "fa_present": fa_present,
        "fa_offsets": fa_offsets,
        "fa_indices": fa_indices,
        "gdn_present": gdn_present,
        "gdn_arity": gdn_arity,
        "gdn_array_present": gdn_array_present,
        "target_hidden_chunk_spans": chunk_spans,
        "target_hidden_total_len": int(snapshot.target_hidden_total_len),
        "has_last_logits": has_last_logits,
    }
    metadata_dict = {"dflash_meta": json.dumps(meta, separators=(",", ":"))}
    return arrays, metadata_dict

def _deserialize(arrays: dict[str, mx.array], meta: dict[str, Any]) -> DFlashPrefixSnapshot:
    key = _key_from_dict(meta["key"])

    fa_present = meta["fa_present"]
    fa_offsets = meta.get("fa_offsets", {}) or {}
    fa_indices = meta.get("fa_indices", {}) or {}
    fa_states = []
    for i, present in enumerate(fa_present):
        if not present:
            fa_states.append(None)
        else:
            k_name = f"fa_{i}_k"
            v_name = f"fa_{i}_v"
            offset = int(fa_offsets.get(str(i), 0))
            k = arrays[k_name]
            v = arrays[v_name]
            if str(i) in fa_indices:
                fa_states.append((k, v, offset, int(fa_indices[str(i)])))
            else:
                fa_states.append((k, v, offset))

    gdn_present = meta["gdn_present"]
    gdn_arity = meta["gdn_arity"]
    gdn_array_present = meta["gdn_array_present"]
    gdn_states: list[Optional[tuple[Optional[mx.array], ...]]] = []
    for i, present in enumerate(gdn_present):
        if not present:
            gdn_states.append(None)
        else:
            arity = int(gdn_arity[i])
            mask = gdn_array_present[i]
            sub: list[Optional[mx.array]] = []
            for j in range(arity):
                if j < len(mask) and mask[j]:
                    name = f"gdn_{i}_{j}"
                    sub.append(arrays[name])
                else:
                    sub.append(None)
            gdn_states.append(tuple(sub))

    chunk_spans_raw = meta["target_hidden_chunk_spans"]
    target_hidden_chunks: list[mx.array] = []
    chunk_count = len(chunk_spans_raw)
    for c in range(chunk_count):
        name = f"target_hidden_{c}"
        target_hidden_chunks.append(arrays[name])

    last_logits = None
    if meta.get("has_last_logits"):
        last_logits = arrays["last_logits"]

    _eval_arrays(arrays)

    return DFlashPrefixSnapshot(
        token_ids=tuple(int(t) for t in meta["token_ids"]),
        fa_states=tuple(fa_states),
        gdn_states=tuple(gdn_states),
        target_hidden_chunks=tuple(target_hidden_chunks),
        target_hidden_chunk_spans=tuple(
            (int(s), int(e)) for s, e in chunk_spans_raw
        ),
        target_hidden_total_len=int(meta["target_hidden_total_len"]),
        last_logits=last_logits,
        key=key,
        kind=str(meta["kind"]),
        created_at=float(meta.get("created_at", time.time())),
    )

@dataclass
class _WritePayload:
    tmp_path: Optional[Path]
    final_path: Path
    nbytes: int
    epoch: int

@dataclass(frozen=True)
class _WriteJob:
    """Holds pre-eval'd arrays ready for disk write. Created on the request thread."""
    arrays_mlx: dict
    meta: dict
    final_path: Path
    nbytes: int
    epoch: int

class DFlashPrefixL2Cache:

    def __init__(
        self,
        *,
        cache_dir: str | Path,
        max_bytes: int,
        max_in_flight: int = 1,
    ):
        self._dir = Path(cache_dir).expanduser().resolve()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_bytes = int(max_bytes)
        self._max_in_flight = max(1, int(max_in_flight))
        self._lock = threading.Lock()

        self._writer_slots = threading.Semaphore(self._max_in_flight)
        self._write_queue: Queue[Optional[_WriteJob]] = Queue()
        self._epoch = 0

        self._tracked_disk_bytes = 0
        self._stats: dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "writes": 0,
            "write_drops_queue_full": 0,
            "write_drops_epoch_invalidated": 0,
            "write_errors": 0,
            "materialize_errors": 0,
            "load_errors": 0,
            "schema_rejects": 0,
            "evictions": 0,
            "bytes_in": 0,
            "bytes_out": 0,
            "load_total_us": 0,
            "lookup_loads": 0,
            "lookup_hash_filtered": 0,
        }
        self._stop = threading.Event()
        self._lock_fp = self._try_acquire_dir_lock()

        self._tracked_disk_bytes = self._snapshot_disk_bytes()

        if self.writable:
            self._writer_thread: Optional[threading.Thread] = threading.Thread(
                target=self._writer_loop,
                name="dflash-l2-writer",
                daemon=True,
            )
            self._writer_thread.start()
        else:
            self._writer_thread = None

    @property
    def writable(self) -> bool:
        return self._lock_fp is not None

    @property
    def cache_dir(self) -> Path:
        return self._dir

    def _try_acquire_dir_lock(self):
        path = self._dir / L2_LOCK_NAME
        try:
            fp = open(path, "w")
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fp
        except OSError:
            _LOG.warning(
                "L2 cache dir %s held by another process; falling back to read-only",
                path,
            )
            return None

    def _bucket_for(self, key: DFlashPrefixKey) -> Path:
        kh = _key_hash(key)
        return self._dir / L2_LAYOUT_ROOT / _runtime_layout_hash() / kh[:2] / kh

    def _layout_root(self) -> Path:
        return self._dir / L2_LAYOUT_ROOT

    def _final_path_for(self, snapshot: DFlashPrefixSnapshot) -> Path:
        bucket = self._bucket_for(snapshot.key)
        name = _format_filename(
            token_len=len(snapshot.token_ids),
            token_hash=_token_hash(snapshot.token_ids),
            kind=snapshot.kind,
            fp_short=_fingerprint(snapshot)[:16],
        )
        return bucket / name

    def lookup(
        self,
        req_tokens: tuple[int, ...],
        key: DFlashPrefixKey,
        *,
        min_token_len: int = 0,
    ) -> Optional[DFlashPrefixSnapshot]:
        t0 = time.perf_counter_ns()
        req_tokens = tuple(int(t) for t in req_tokens)
        min_token_len = max(0, int(min_token_len))
        bucket = self._bucket_for(key)
        if not bucket.is_dir():
            with self._lock:
                self._stats["misses"] += 1
            return None

        candidates: list[tuple[_Parts, Path]] = []
        try:
            for path in bucket.iterdir():
                parts = _parse_filename(path.name)
                if parts is None:
                    continue
                if parts.token_len <= min_token_len or parts.token_len > len(req_tokens):
                    continue
                candidates.append((parts, path))
        except OSError as e:
            _LOG.debug("L2 bucket scan failed: %s", e)
            with self._lock:
                self._stats["misses"] += 1
            return None

        candidates.sort(key=lambda c: (-c[0].token_len, 0 if c[0].kind == "prefill" else 1))

        candidate_lengths = {parts.token_len for parts, _path in candidates}
        hashes = _token_hashes_for_lengths(req_tokens, candidate_lengths)

        for parts, path in candidates:
            if hashes[parts.token_len] != parts.token_hash:
                with self._lock:
                    self._stats["lookup_hash_filtered"] += 1
                continue
            if parts.token_len == len(req_tokens) and parts.kind != "prefill":
                continue
            with self._lock:
                self._stats["lookup_loads"] += 1
            snap = self._load_and_validate(
                path, key=key, req_tokens=req_tokens, parts=parts
            )
            if snap is None:
                continue

            if parts.token_len == len(req_tokens) and snap.last_logits is None:
                continue
            elapsed_us = (time.perf_counter_ns() - t0) // 1000
            with self._lock:
                self._stats["hits"] += 1
                self._stats["bytes_out"] += snap.nbytes
                self._stats["load_total_us"] += int(elapsed_us)
            return snap

        with self._lock:
            self._stats["misses"] += 1
        return None

    def insert_async(self, snapshot: DFlashPrefixSnapshot) -> bool:
        if not self.writable:
            return False
        final_path = self._final_path_for(snapshot)
        if final_path.exists():
            with self._lock:
                self._stats["writes"] += 1
            return True
        if not self._writer_slots.acquire(blocking=False):
            with self._lock:
                self._stats["write_drops_queue_full"] += 1
            return False
        try:
            arrays_mlx, meta = _serialize(snapshot)
            _eval_arrays(arrays_mlx)
        except Exception:
            self._writer_slots.release()
            raise
        with self._lock:
            epoch = self._epoch
        self._write_queue.put(_WriteJob(
            arrays_mlx=arrays_mlx,
            meta=meta,
            final_path=final_path,
            nbytes=int(snapshot.nbytes),
            epoch=epoch,
        ))
        return True

    def stats(self) -> dict[str, Any]:
        with self._lock:
            out = dict(self._stats)
            current_bytes = self._tracked_disk_bytes
        out["max_bytes"] = self._max_bytes
        out["writable"] = self.writable
        out["current_bytes"] = current_bytes
        return out

    def shutdown(self, wait: bool = True) -> None:
        if self._writer_thread is not None:

            self._write_queue.put(None)
            if wait:
                self._writer_thread.join(timeout=10.0)
        self._stop.set()
        if self._lock_fp is not None:
            try:
                fcntl.flock(self._lock_fp.fileno(), fcntl.LOCK_UN)
                self._lock_fp.close()
            except OSError:
                pass
            self._lock_fp = None

    def __del__(self):
        try:
            self.shutdown(wait=False)
        except Exception as exc:
            _LOG.debug("L2 finalizer shutdown failed: %s", exc)

    def _writer_loop(self) -> None:
        while True:
            try:
                job = self._write_queue.get(timeout=0.5)
            except Empty:
                if self._stop.is_set():
                    break
                continue
            if job is None:
                break
            try:
                with self._lock:
                    if job.epoch != self._epoch:
                        self._stats["write_drops_epoch_invalidated"] += 1
                        continue
                try:
                    payload = self._write_job_to_disk(job)
                except OSError as e:
                    if e.errno in (errno.ENOSPC, errno.EDQUOT):
                        _LOG.warning("L2 write skipped (disk full): %s", e)
                        with self._lock:
                            self._stats["write_errors"] += 1
                    else:
                        _LOG.warning("L2 materialize failed: %s", e)
                        with self._lock:
                            self._stats["materialize_errors"] += 1
                    continue
                try:
                    self._write_payload(payload)
                except Exception as e:
                    _LOG.warning("L2 write failed: %s", e)
                    with self._lock:
                        self._stats["write_errors"] += 1
            finally:
                job = None
                self._writer_slots.release()

    def _write_job_to_disk(self, job: _WriteJob) -> _WritePayload:
        """Write pre-eval'd arrays to a temp file. Called from the writer thread."""
        if job.final_path.exists():
            return _WritePayload(
                tmp_path=None,
                final_path=job.final_path,
                nbytes=job.nbytes,
                epoch=job.epoch,
            )
        job.final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f".{job.final_path.stem}.",
            suffix=L2_TMP_SUFFIX,
            dir=str(job.final_path.parent),
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        try:
            mx.save_safetensors(str(tmp_path), job.arrays_mlx, metadata=job.meta)
        except Exception:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise
        return _WritePayload(
            tmp_path=tmp_path,
            final_path=job.final_path,
            nbytes=job.nbytes,
            epoch=job.epoch,
        )

    def _write_payload(self, payload: _WritePayload) -> None:
        if not self.writable:
            if payload.tmp_path is not None:
                try:
                    payload.tmp_path.unlink()
                except OSError:
                    pass
            return

        with self._lock:
            if payload.epoch != self._epoch:
                if payload.tmp_path is not None:
                    try:
                        payload.tmp_path.unlink()
                    except OSError:
                        pass
                self._stats["write_drops_epoch_invalidated"] += 1
                return
        if payload.final_path.exists():
            if payload.tmp_path is not None:
                try:
                    payload.tmp_path.unlink()
                except OSError:
                    pass
            with self._lock:
                self._stats["writes"] += 1
            return
        if payload.tmp_path is None:
            with self._lock:
                self._stats["write_errors"] += 1
            return
        payload.final_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            if payload.epoch != self._epoch:
                try:
                    payload.tmp_path.unlink()
                except OSError:
                    pass
                self._stats["write_drops_epoch_invalidated"] += 1
                return
            try:

                payload.final_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(str(payload.tmp_path), str(payload.final_path))
            except OSError as e:
                try:
                    payload.tmp_path.unlink()
                except OSError:
                    pass
                if e.errno in (errno.ENOSPC, errno.EDQUOT):
                    _LOG.warning("L2 write skipped (disk full): %s", e)
                    self._stats["write_errors"] += 1
                    return
                raise
            try:
                actual_size = payload.final_path.stat().st_size
            except OSError:
                actual_size = int(payload.nbytes)
            self._stats["writes"] += 1
            self._stats["bytes_in"] += payload.nbytes
            self._tracked_disk_bytes += actual_size
        self._evict_to_budget()

    def _load_and_validate(
        self,
        path: Path,
        *,
        key: DFlashPrefixKey,
        req_tokens: tuple[int, ...],
        parts: _Parts,
    ) -> Optional[DFlashPrefixSnapshot]:
        try:
            arrays, metadata = mx.load(
                str(path), format="safetensors", return_metadata=True
            )
        except Exception as e:
            _LOG.warning("L2 load failed for %s: %s", path.name, e)
            with self._lock:
                self._stats["load_errors"] += 1
            self._unlink_if_writable(path)
            return None
        meta_raw = metadata.get("dflash_meta") if isinstance(metadata, dict) else None
        if not meta_raw:
            with self._lock:
                self._stats["schema_rejects"] += 1
            self._unlink_if_writable(path)
            return None
        try:
            meta = json.loads(meta_raw)
        except (TypeError, json.JSONDecodeError):
            with self._lock:
                self._stats["schema_rejects"] += 1
            self._unlink_if_writable(path)
            return None
        if not isinstance(meta, dict):
            with self._lock:
                self._stats["schema_rejects"] += 1
            self._unlink_if_writable(path)
            return None
        schema_version = meta.get("schema_version")
        if not _is_json_int(schema_version):
            with self._lock:
                self._stats["schema_rejects"] += 1
            self._unlink_if_writable(path)
            return None
        if schema_version != L2_SCHEMA_VERSION:
            with self._lock:
                self._stats["schema_rejects"] += 1
            self._unlink_if_writable(path)
            return None
        if meta.get("context_representation") != CONTEXT_REPRESENTATION:
            with self._lock:
                self._stats["schema_rejects"] += 1
            self._unlink_if_writable(path)
            return None
        if str(meta.get("runtime_version", "")) != dflash_mlx.__version__:
            with self._lock:
                self._stats["schema_rejects"] += 1
            self._unlink_if_writable(path)
            return None
        try:
            file_key = _key_from_dict(meta["key"])
        except (KeyError, TypeError, ValueError):
            with self._lock:
                self._stats["schema_rejects"] += 1
            self._unlink_if_writable(path)
            return None
        if file_key != key:

            with self._lock:
                self._stats["schema_rejects"] += 1
            self._unlink_if_writable(path)
            return None
        file_tokens_raw = meta.get("token_ids")
        if not isinstance(file_tokens_raw, list):
            with self._lock:
                self._stats["schema_rejects"] += 1
            self._unlink_if_writable(path)
            return None
        if not all(_is_json_int(t) for t in file_tokens_raw):
            with self._lock:
                self._stats["schema_rejects"] += 1
            self._unlink_if_writable(path)
            return None
        file_tokens = tuple(file_tokens_raw)
        n = len(file_tokens)
        if n != parts.token_len:
            with self._lock:
                self._stats["schema_rejects"] += 1
            self._unlink_if_writable(path)
            return None
        if n == 0 or n > len(req_tokens):
            return None
        if tuple(req_tokens[:n]) != file_tokens:

            return None
        if str(meta.get("kind", "")) != parts.kind:
            with self._lock:
                self._stats["schema_rejects"] += 1
            self._unlink_if_writable(path)
            return None
        try:
            return _deserialize(arrays, meta)
        except Exception as e:
            _LOG.warning("L2 deserialize failed for %s: %s", path.name, e)
            with self._lock:
                self._stats["load_errors"] += 1
            self._unlink_if_writable(path)
            return None

    def _unlink_if_writable(self, path: Path) -> None:
        if not self.writable:
            return
        try:
            path.unlink()
        except OSError:
            return
        with self._lock:
            self._tracked_disk_bytes = self._snapshot_disk_bytes()

    def _walk_snapshots(self):
        root = self._layout_root()
        if not root.is_dir():
            return
        try:
            runtime_dirs = list(root.iterdir())
        except OSError:
            return
        for runtime_dir in runtime_dirs:
            if not runtime_dir.is_dir():
                continue
            try:
                shards = list(runtime_dir.iterdir())
            except OSError:
                continue
            for shard in shards:
                if not shard.is_dir():
                    continue
                try:
                    buckets = list(shard.iterdir())
                except OSError:
                    continue
                for bucket in buckets:
                    if not bucket.is_dir():
                        continue
                    try:
                        files = list(bucket.iterdir())
                    except OSError:
                        continue
                    for path in files:
                        if _parse_filename(path.name) is None:
                            continue
                        yield path

    def _snapshot_disk_bytes(self) -> int:
        total = 0
        for path in self._walk_snapshots():
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total

    def _evict_to_budget(self) -> None:
        if not self.writable or self._max_bytes <= 0:
            return
        with self._lock:
            if self._tracked_disk_bytes <= self._max_bytes:
                return

        files: list[tuple[int, int, Path]] = []
        observed_bytes = 0
        for p in self._walk_snapshots():
            try:
                st = p.stat()
            except OSError:
                continue
            observed_bytes += int(st.st_size)
            files.append((st.st_mtime_ns, st.st_size, p))
        with self._lock:
            self._tracked_disk_bytes = int(observed_bytes)
        files.sort()
        for _, sz, p in files:
            with self._lock:
                if self._tracked_disk_bytes <= self._max_bytes:
                    return
            try:
                p.unlink()
            except OSError:
                continue
            with self._lock:
                self._stats["evictions"] += 1
                self._tracked_disk_bytes = max(0, self._tracked_disk_bytes - int(sz))

    def clear(self) -> None:
        if not self.writable:
            return
        with self._lock:
            self._epoch += 1
            root = self._layout_root()
            if root.is_dir():
                try:
                    shutil.rmtree(root)
                except OSError:
                    self._tracked_disk_bytes = self._snapshot_disk_bytes()
                    raise
            self._tracked_disk_bytes = 0
