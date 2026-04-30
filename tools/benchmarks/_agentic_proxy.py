# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)


from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import sys
import threading
import time
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dflash_mlx.artifacts import create_run_dir

_REQ_LOCK = threading.Lock()
_REQ_INDEX = 0

def _next_req_idx() -> int:
    global _REQ_INDEX
    with _REQ_LOCK:
        _REQ_INDEX += 1
        return _REQ_INDEX

def _json_compact(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

def _parse_sse_event(block: bytes) -> dict[str, Any]:
    text = block.decode("utf-8", errors="replace")
    fields: dict[str, str] = {}
    data_lines: list[str] = []
    for line in text.split("\n"):
        if not line:
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        elif ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip()] = v.lstrip()
    out: dict[str, Any] = {"raw": text}
    if data_lines:
        joined = "\n".join(data_lines)
        out["data_raw"] = joined
        if joined != "[DONE]":
            try:
                out["data"] = json.loads(joined)
            except Exception:
                pass
    if fields:
        out["fields"] = fields
    return out

class TraceHandler(BaseHTTPRequestHandler):
    server_version = "agentic-proxy/1.0"
    out_dir: Path
    upstream_host: str
    upstream_port: int
    upstream_scheme: str
    proxy_log_path: Path

    def log_message(self, *_, **__):
        pass

    def _proxy_log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.proxy_log_path.open("a") as f:
            f.write(f"{ts} {msg}\n")

    def _open_upstream(self) -> http.client.HTTPConnection:
        if self.upstream_scheme == "https":
            return http.client.HTTPSConnection(self.upstream_host, self.upstream_port, timeout=600)
        return http.client.HTTPConnection(self.upstream_host, self.upstream_port, timeout=600)

    def do_GET(self):

        self._forward_simple("GET")

    def do_POST(self):
        if self.path.startswith("/v1/chat/completions"):
            self._handle_chat_completions()
        else:
            self._forward_simple("POST")

    def _forward_simple(self, method: str):
        try:
            conn = self._open_upstream()
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length else None
            headers = {k: v for k, v in self.headers.items() if k.lower() != "host"}
            conn.request(method, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() in ("transfer-encoding", "content-encoding"):
                    continue
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            conn.close()
        except Exception as e:
            self._proxy_log(f"forward_simple error {method} {self.path}: {e!r}")
            try:
                self.send_response(502)
                self.end_headers()
            except Exception:
                pass

    def _handle_chat_completions(self):
        idx = _next_req_idx()
        t0 = time.perf_counter()
        wall_t0 = time.time()
        length = int(self.headers.get("Content-Length", "0") or "0")
        body_bytes = self.rfile.read(length) if length else b""

        try:
            body_obj = json.loads(body_bytes) if body_bytes else {}
        except Exception:
            body_obj = {"_decode_error": True, "_raw_len": len(body_bytes)}

        is_stream = bool(body_obj.get("stream"))
        req_path = self.out_dir / "requests" / f"{idx:03d}.json"
        sse_path = self.out_dir / "sse" / f"{idx:03d}.jsonl"
        req_path.parent.mkdir(parents=True, exist_ok=True)
        sse_path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "idx": idx,
            "method": "POST",
            "path": self.path,
            "wall_ts": wall_t0,
            "stream": is_stream,
            "headers": {k: v for k, v in self.headers.items() if k.lower() != "authorization"},
            "body": body_obj,
            "body_bytes": len(body_bytes),
        }
        req_path.write_bytes(json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))

        self._proxy_log(
            f"req#{idx} {self.path} stream={is_stream} body_bytes={len(body_bytes)} "
            f"max_tokens={body_obj.get('max_tokens')} model={body_obj.get('model')}"
        )

        try:
            conn = self._open_upstream()
            up_headers = {k: v for k, v in self.headers.items() if k.lower() != "host"}
            up_headers["Host"] = f"{self.upstream_host}:{self.upstream_port}"
            up_headers["Content-Length"] = str(len(body_bytes))
            t_upstream_start = time.perf_counter() - t0
            conn.request("POST", self.path, body=body_bytes, headers=up_headers)
            resp = conn.getresponse()
            t_status = time.perf_counter() - t0
        except Exception as e:
            self._proxy_log(f"req#{idx} upstream connect failed: {e!r}")
            self.send_response(502)
            self.end_headers()
            return

        self.send_response(resp.status)
        forwarded_hdrs = []
        for k, v in resp.getheaders():
            if k.lower() in ("transfer-encoding", "content-encoding", "content-length", "connection", "keep-alive"):
                continue
            self.send_header(k, v)
            forwarded_hdrs.append((k, v))
        if is_stream:
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
        self.end_headers()

        sse_f = sse_path.open("a", buffering=1)
        sse_f.write(json.dumps({
            "type": "meta",
            "t_ms": 0.0,
            "t_upstream_start_ms": t_upstream_start * 1000.0,
            "t_status_ms": t_status * 1000.0,
            "status": resp.status,
            "headers": forwarded_hdrs,
        }) + "\n")

        try:
            if is_stream:
                self._stream_sse(idx, t0, resp, sse_f)
            else:
                data = resp.read()
                if data:
                    self.wfile.write(data)
                self._log_chunk(sse_f, t0, "non_stream_body", {"bytes": len(data), "preview": data[:512].decode("utf-8", "replace")})
        except (BrokenPipeError, ConnectionResetError) as e:
            self._proxy_log(f"req#{idx} client disconnect: {e!r}")
        except Exception as e:
            self._proxy_log(f"req#{idx} stream error: {e!r}")
        finally:
            t_total = (time.perf_counter() - t0) * 1000.0
            sse_f.write(json.dumps({"type": "end", "t_ms": t_total}) + "\n")
            sse_f.close()
            try:
                conn.close()
            except Exception:
                pass
            self._proxy_log(f"req#{idx} done t_total_ms={t_total:.1f}")

    def _write_chunk(self, data: bytes) -> None:
        if not data:
            return
        self.wfile.write(f"{len(data):x}\r\n".encode("ascii"))
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _write_chunk_end(self) -> None:
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except Exception:
            pass

    def _stream_sse(self, idx: int, t0: float, resp, sse_f):

        buf: list[bytes] = []
        first_byte_logged = False
        client_alive = True
        try:
            while True:
                try:
                    line = resp.fp.readline()
                except Exception as e:
                    self._proxy_log(f"req#{idx} readline err: {e!r}")
                    break
                if not line:
                    break
                if not first_byte_logged:
                    t = (time.perf_counter() - t0) * 1000.0
                    sse_f.write(json.dumps({"type": "first_byte", "t_ms": t}) + "\n")
                    first_byte_logged = True

                if client_alive:
                    try:
                        self._write_chunk(line)
                    except (BrokenPipeError, ConnectionResetError):
                        self._proxy_log(f"req#{idx} client gone mid-stream")
                        client_alive = False
                if line in (b"\n", b"\r\n"):
                    if buf:
                        block = b"".join(buf)
                        parsed = _parse_sse_event(block)
                        self._log_chunk(sse_f, t0, "event", parsed)
                    buf = []
                else:
                    buf.append(line)
            if buf:
                parsed = _parse_sse_event(b"".join(buf))
                self._log_chunk(sse_f, t0, "event_tail", parsed)
        finally:
            if client_alive:
                self._write_chunk_end()

    def _log_chunk(self, sse_f, t0: float, kind: str, payload: Any):
        t_ms = (time.perf_counter() - t0) * 1000.0
        sse_f.write(json.dumps({"type": kind, "t_ms": t_ms, "payload": payload}, ensure_ascii=False) + "\n")

def _make_handler(out_dir: Path, upstream_url: str):
    parsed = urlparse(upstream_url)
    if not parsed.hostname or not parsed.port:
        raise SystemExit(f"upstream-url must include host:port, got {upstream_url!r}")

    class Bound(TraceHandler):
        pass

    Bound.out_dir = out_dir
    Bound.upstream_host = parsed.hostname
    Bound.upstream_port = parsed.port
    Bound.upstream_scheme = parsed.scheme or "http"
    Bound.proxy_log_path = out_dir / "proxy.log"
    return Bound

def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--listen-host", default="127.0.0.1")
    p.add_argument("--listen-port", type=int, required=True)
    p.add_argument("--upstream-url", required=True, help="e.g. http://127.0.0.1:8000")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(list(argv) if argv is not None else None)

    out_dir = Path(args.out_dir) if args.out_dir else create_run_dir("trace", "proxy")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "requests").mkdir(exist_ok=True)
    (out_dir / "sse").mkdir(exist_ok=True)
    handler = _make_handler(out_dir, args.upstream_url)

    srv = ThreadingHTTPServer((args.listen_host, args.listen_port), handler)
    sys.stderr.write(
        f"[proxy] listen={args.listen_host}:{args.listen_port} upstream={args.upstream_url} out={out_dir}\n"
    )
    sys.stderr.write(f"Output: {out_dir}\n")
    sys.stderr.flush()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
