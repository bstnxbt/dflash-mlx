# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
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
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Optional

import mlx.core as mx

import dflash_mlx
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot

_LOG = logging.getLogger(__name__)

L2_SCHEMA_VERSION = 1
L2_FILE_SUFFIX = ".safetensors"
L2_TMP_SUFFIX = ".tmp.safetensors"
L2_LOCK_NAME = ".dflash_l2.lock"
L2_LAYOUT_ROOT = "v1"

_VALID_KINDS = ("prefill", "generation")
_HEX = set("0123456789abcdef")

def _key_to_dict(key: DFlashPrefixKey) -> dict[str, Any]:
    return {
        "target_model_id": key.target_model_id,
        "draft_model_id": key.draft_model_id,
        "capture_layer_ids": list(key.capture_layer_ids),
        "draft_sink_size": int(key.draft_sink_size),
        "draft_window_size": int(key.draft_window_size),
        "target_fa_window": int(key.target_fa_window),
        "format_version": int(key.format_version),
    }

def _key_from_dict(d: dict[str, Any]) -> DFlashPrefixKey:
    return DFlashPrefixKey(
        target_model_id=str(d["target_model_id"]),
        draft_model_id=str(d["draft_model_id"]),
        capture_layer_ids=tuple(int(x) for x in d["capture_layer_ids"]),
        draft_sink_size=int(d["draft_sink_size"]),
        draft_window_size=int(d["draft_window_size"]),
        target_fa_window=int(d.get("target_fa_window", 0)),
        format_version=int(d.get("format_version", 1)),
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
    for i, fa in enumerate(snapshot.fa_states):
        if fa is None:
            fa_present.append(False)
        else:
            k, v, offset = fa
            arrays[f"fa_{i}_k"] = k
            arrays[f"fa_{i}_v"] = v
            fa_present.append(True)
            fa_offsets[str(i)] = int(offset)

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
        "key": _key_to_dict(snapshot.key),
        "kind": snapshot.kind,
        "token_ids": list(snapshot.token_ids),
        "created_at": float(snapshot.created_at),
        "fa_present": fa_present,
        "fa_offsets": fa_offsets,
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
    def _clone(a: mx.array, name: str) -> mx.array:

        cloned = mx.array(a)
        mx.eval(cloned)
        return cloned

    key = _key_from_dict(meta["key"])

    fa_present = meta["fa_present"]
    fa_offsets = meta.get("fa_offsets", {}) or {}
    fa_states: list[Optional[tuple[mx.array, mx.array, int]]] = []
    for i, present in enumerate(fa_present):
        if not present:
            fa_states.append(None)
        else:
            k_name = f"fa_{i}_k"
            v_name = f"fa_{i}_v"
            offset = int(fa_offsets.get(str(i), 0))
            fa_states.append(
                (_clone(arrays[k_name], k_name), _clone(arrays[v_name], v_name), offset)
            )

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
                    sub.append(_clone(arrays[name], name))
                else:
                    sub.append(None)
            gdn_states.append(tuple(sub))

    chunk_spans_raw = meta["target_hidden_chunk_spans"]
    target_hidden_chunks: list[mx.array] = []
    chunk_count = len(chunk_spans_raw)
    for c in range(chunk_count):
        name = f"target_hidden_{c}"
        target_hidden_chunks.append(_clone(arrays[name], name))

    last_logits = None
    if meta.get("has_last_logits"):
        last_logits = _clone(arrays["last_logits"], "last_logits")

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
    arrays: dict[str, mx.array]
    meta: dict[str, str]
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
        self._write_queue: Queue[Optional[_WritePayload]] = Queue()
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

        try:
            self._tracked_disk_bytes = sum(
                p.stat().st_size for p in self._walk_snapshots()
            )
        except OSError:
            self._tracked_disk_bytes = 0

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
    ) -> Optional[DFlashPrefixSnapshot]:
        t0 = time.perf_counter_ns()
        req_tokens = tuple(int(t) for t in req_tokens)
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
                if parts.token_len > len(req_tokens):
                    continue
                candidates.append((parts, path))
        except OSError as e:
            _LOG.debug("L2 bucket scan failed: %s", e)
            with self._lock:
                self._stats["misses"] += 1
            return None

        candidates.sort(key=lambda c: (-c[0].token_len, 0 if c[0].kind == "prefill" else 1))

        hashes: dict[int, str] = {}

        for parts, path in candidates:
            if parts.token_len not in hashes:
                hashes[parts.token_len] = _token_hash(req_tokens[: parts.token_len])
            if hashes[parts.token_len] != parts.token_hash:
                with self._lock:
                    self._stats["lookup_hash_filtered"] += 1
                continue
            with self._lock:
                self._stats["lookup_loads"] += 1
            snap = self._load_and_validate(
                path, key=key, req_tokens=req_tokens, parts=parts
            )
            if snap is None:
                continue

            if parts.token_len == len(req_tokens):
                if snap.kind != "prefill" or snap.last_logits is None:
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

    def insert_async(self, snapshot: DFlashPrefixSnapshot) -> None:
        if not self.writable:
            return
        if not self._writer_slots.acquire(blocking=False):
            with self._lock:
                self._stats["write_drops_queue_full"] += 1
            return
        try:
            arrays_mlx, meta = _serialize(snapshot)
            _eval_arrays(arrays_mlx)
            final_path = self._final_path_for(snapshot)
            with self._lock:
                epoch_at_submit = self._epoch
            payload = _WritePayload(
                arrays=arrays_mlx,
                meta=meta,
                final_path=final_path,
                nbytes=int(snapshot.nbytes),
                epoch=int(epoch_at_submit),
            )
        except Exception as e:
            _LOG.warning("L2 materialize failed: %s", e)
            with self._lock:
                self._stats["materialize_errors"] += 1
            self._writer_slots.release()
            return

        self._write_queue.put(payload)

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
        except Exception:
            pass

    def _writer_loop(self) -> None:
        while True:
            try:
                payload = self._write_queue.get(timeout=0.5)
            except Empty:
                if self._stop.is_set():
                    break
                continue
            if payload is None:
                break
            try:
                self._write_payload(payload)
            except Exception as e:
                _LOG.warning("L2 write failed: %s", e)
                with self._lock:
                    self._stats["write_errors"] += 1
            finally:

                payload = None
                self._writer_slots.release()

    def _write_one(self, snapshot: DFlashPrefixSnapshot) -> None:
        if not self.writable:
            return
        try:
            arrays_mlx, meta = _serialize(snapshot)
            _eval_arrays(arrays_mlx)
            final_path = self._final_path_for(snapshot)
            with self._lock:
                epoch = self._epoch
            payload = _WritePayload(
                arrays=arrays_mlx,
                meta=meta,
                final_path=final_path,
                nbytes=int(snapshot.nbytes),
                epoch=int(epoch),
            )
        except Exception as e:
            _LOG.warning("L2 sync write materialize failed: %s", e)
            with self._lock:
                self._stats["materialize_errors"] += 1
            return
        self._write_payload(payload)

    def _write_payload(self, payload: _WritePayload) -> None:
        if not self.writable:
            return

        with self._lock:
            if payload.epoch != self._epoch:
                self._stats["write_drops_epoch_invalidated"] += 1
                return
        if payload.final_path.exists():
            with self._lock:
                self._stats["writes"] += 1
            return
        payload.final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f".{payload.final_path.stem}.",
            suffix=L2_TMP_SUFFIX,
            dir=str(payload.final_path.parent),
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        try:
            mx.save_safetensors(str(tmp_path), payload.arrays, metadata=payload.meta)
        except OSError as e:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            if e.errno in (errno.ENOSPC, errno.EDQUOT):
                _LOG.warning("L2 write skipped (disk full): %s", e)
                with self._lock:
                    self._stats["write_errors"] += 1
                return
            raise

        with self._lock:
            if payload.epoch != self._epoch:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
                self._stats["write_drops_epoch_invalidated"] += 1
                return
            try:

                payload.final_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(str(tmp_path), str(payload.final_path))
            except OSError as e:
                try:
                    tmp_path.unlink()
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
        except Exception:
            with self._lock:
                self._stats["schema_rejects"] += 1
            self._unlink_if_writable(path)
            return None
        if int(meta.get("schema_version", 0)) != L2_SCHEMA_VERSION:
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
        except Exception:
            with self._lock:
                self._stats["schema_rejects"] += 1
            self._unlink_if_writable(path)
            return None
        if file_key != key:

            with self._lock:
                self._stats["schema_rejects"] += 1
            self._unlink_if_writable(path)
            return None
        file_tokens = meta.get("token_ids", [])
        n = len(file_tokens)
        if n != parts.token_len:
            with self._lock:
                self._stats["schema_rejects"] += 1
            self._unlink_if_writable(path)
            return None
        if n == 0 or n > len(req_tokens):
            return None
        if tuple(req_tokens[:n]) != tuple(int(t) for t in file_tokens):

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
            sz = path.stat().st_size
        except OSError:
            sz = 0
        try:
            path.unlink()
        except OSError:
            return
        if sz > 0:
            with self._lock:
                self._tracked_disk_bytes = max(0, self._tracked_disk_bytes - sz)

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

    def _evict_to_budget(self) -> None:
        if not self.writable or self._max_bytes <= 0:
            return
        with self._lock:
            if self._tracked_disk_bytes <= self._max_bytes:
                return

        files: list[tuple[int, int, Path]] = []
        for p in self._walk_snapshots():
            try:
                st = p.stat()
            except OSError:
                continue
            files.append((st.st_mtime_ns, st.st_size, p))
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
                self._tracked_disk_bytes = max(0, self._tracked_disk_bytes - sz)

    def clear(self) -> None:
        if not self.writable:
            return
        with self._lock:
            self._epoch += 1
            root = self._layout_root()
            if root.is_dir():
                shutil.rmtree(root, ignore_errors=True)
            self._tracked_disk_bytes = 0
