# Observability

`dflash-serve` ships with two layers of observability. The first is a stream of
human-readable lines on stderr that the server always emits — startup banner,
per-request lifecycle, periodic decode rate, prefix-cache events, peak memory.
The second is a machine-readable JSONL stream gated by the
`DFLASH_BENCH_LOG_DIR` environment variable, intended for post-hoc analysis
with `pandas` or `jq`. Both layers are documented below, with the source file
and line number where each format string is defined so you can map a line you
see back to the code that produced it.

## 1. Stderr log lines

All stderr lines are prefixed with a wall-clock timestamp (`%Y-%m-%d %H:%M:%S`)
and the literal tag `[dflash]`. Examples below show the post-prefix payload
only.

### Startup banner

Printed once, after the model load returns and just before the HTTP listener
binds. Source: `dflash_mlx/serve.py:491` (`_print_startup_banner`). The box
border is drawn with `+` and `-`; the body lines are formatted as
`| <field>: <value> |` and right-padded to a uniform width.

```
+--------------------------------------------------------+
| DFlash v0.1.4.1 - speculative decoding engine          |
| Target:       mlx-community/Qwen3.6-27B-4bit           |
| Draft:        z-lab/Qwen3.6-27B-DFlash (auto-detected) |
| Mode:         DFlash (speculative decoding active)     |
| Prefix cache: enabled                                  |
| Server:       mlx_lm.server on port 8765               |
+--------------------------------------------------------+
```

Field meanings:

- **DFlash v…** — package version reported by `importlib.metadata`. Falls back
  to `unknown` if the package is not installed.
- **Target** — full HuggingFace ID of the verifier model that was loaded.
- **Draft** — full HuggingFace ID of the draft model. Suffix is
  `(auto-detected)` when resolved from the target's family map, or
  `(explicit)` when supplied via `--draft-model`.
- **Mode** — always `DFlash (speculative decoding active)` when the server
  starts; the line exists so log scrapers can detect a successful boot.
- **Prefix cache** — `enabled` when the L1 prefix snapshot cache is active, or
  `disabled (--no-prefix-cache)` when the user opted out.
- **Server** — the underlying HTTP runner module name and bound port.

### Per-request lifecycle lines

These fire once each per HTTP request, in the order shown.

#### Fast-path AR

```
2026-04-27 10:14:02 [dflash] fast-path AR | max_tokens=128
```

Source: `dflash_mlx/serve.py:107`. Fires when the incoming request has
`max_tokens <= 256`. The request bypasses speculative decoding and is delegated
to the stock `mlx_lm.server` autoregressive path with the draft model
temporarily detached.

#### Lazy model load (stock path only)

```
2026-04-27 10:14:03 [dflash] loading model on generation worker thread...
```

Source: `dflash_mlx/server/model_provider.py:95`. The DFlash banner is printed
*after* the model is resident, so this line only appears on the fast-path AR
branch when the underlying `mlx_lm.server` worker hits a model swap on first
use.

#### Prefix cache enabled (one-shot)

```
2026-04-27 10:14:00 [dflash] prefix cache enabled (max_entries=4, max_bytes=8589934592)
```

Source: `dflash_mlx/server/prefix_cache_manager.py:20`. Printed once when the
prefix-cache singleton is constructed, on the first request that needs it.

#### Prefix cache hit

```
2026-04-27 10:14:05 [dflash] prefix cache hit 873/1024 tokens (stable prefix 873)
```

Source: `dflash_mlx/serve.py:209`. The first number is the matched prefix
length (tokens reused), the second is the full prompt length, and
`stable prefix N` is the conservative truncation point at which the chat
template's last `<|im_start|>assistant` block begins, so cross-turn matches
align on a safe boundary.

#### Prefill progress

```
2026-04-27 10:14:06 [dflash] prefill: 4096/12000 tokens | 1.3s
```

Source: `dflash_mlx/serve.py:252` (incremental) and `dflash_mlx/serve.py:262`
(final). Long prompts are prefilled in chunks; the line repeats once per
chunk and a final summary line is printed when prefill completes. The trailing
`Z.Zs` is wall time elapsed since the request started.

#### End-of-request snapshot

```
2026-04-27 10:14:42 [dflash] end-of-request snapshot saved (1247 tokens)
```

Source: `dflash_mlx/serve.py:320`. Fires after the assistant turn finishes when
the runtime has emitted a `generation_snapshot_ready` event and the snapshot
was successfully inserted into the L1 cache. The token count is
`prompt_tokens + generated_tokens`, i.e. the full prefix that subsequent
requests can reuse.

#### Peak memory

```
2026-04-27 10:14:42 [dflash] req#37 peak_memory=18.42 GB
```

Source: `dflash_mlx/serve.py:458`. Printed at the end of every request from
`mlx.core.get_peak_memory()`. Note that the value is cumulative across the
process lifetime — it never goes down, so a flat reading across requests means
the high-water mark has not been raised since the previous request.

#### Periodic prefix-cache stats

```
2026-04-27 10:14:05 [dflash] prefix-cache-stats [lookup] entries=3/4 bytes=4811234560/8589934592 hits=12+5 misses=2 insertions=8 evictions=4 prefill_tokens_saved=18432
```

Source: `dflash_mlx/server/prefix_cache_manager.py:28` (`format_stats_line`).
Emitted once per request after each cache lookup. The optional `[label]`
identifies the call site (`lookup` is the only one wired up today). Fields:

- `entries=A/B` — current snapshot count vs. cap.
- `bytes=A/B` — total bytes held vs. cap.
- `hits=E+P` — cumulative exact hits plus prefix hits.
- `misses` — cumulative misses since process start.
- `insertions` — cumulative inserts.
- `evictions` — cumulative LRU evictions.
- `prefill_tokens_saved` — cumulative tokens whose prefill cost was avoided
  thanks to a hit.

### Live decode stats line

```
2026-04-27 10:14:35 [dflash] 142.7 tok/s | 88.4% accepted | 2048 tokens | 14.6s | prompt: 873 tokens
```

Source: `dflash_mlx/serve.py:359`. Printed during generation every 2048 emitted
tokens. Fields:

- `tok/s` — tokens generated divided by elapsed seconds since the end of
  prefill (decode-only throughput).
- `% accepted` — running acceptance ratio of drafted tokens, as reported by
  the speculative engine.
- `N tokens` — generated tokens so far in this request.
- `Ts` — wall time since the request started, including prefill.
- `prompt: M tokens` — input prompt length.

A final summary line in the same shape is printed once when the request ends,
sourced from `dflash_mlx/server/metrics.py:27` (`write_summary_line`).

## 2. JSONL events

When `DFLASH_BENCH_LOG_DIR` points at a writable directory, the server appends
to three line-delimited JSON files. The directory is created on demand. Files
are opened with line buffering and writes are mutex-guarded, so it is safe to
tail them while the server runs.

### Enabling

```bash
DFLASH_BENCH_LOG_DIR=/tmp/dflash-logs dflash-serve --model mlx-community/Qwen3.6-27B-4bit
```

After the first request, the directory will contain:

```
/tmp/dflash-logs/
  post_events.jsonl
  cycle_events.jsonl
  cache_events.jsonl
```

Every event row carries an automatically-injected `ts` field (Unix epoch
seconds, float) added by `bench_logger._BenchLogger._write`.

### post_events.jsonl

One row per HTTP request, written when the request closes. Source:
`dflash_mlx/server/metrics.py:35` (`log_bench_post`) → `bench_logger.log_post`.
On the fast-path AR branch, a smaller row is emitted directly from
`dflash_mlx/serve.py:120`.

| Field | Type | Meaning |
| --- | --- | --- |
| `ts` | float | Unix time at which the row was written. |
| `request_id` | int | Monotonic counter, starts at 1 per process. |
| `mode_used` | string | `dflash`, `dflash_fallback` (engine fell back to AR mid-request), or `dflash_ar_fastpath` (max\_tokens ≤ 256). |
| `prompt_tokens` | int | Tokenized prompt length seen by the engine. |
| `generated_tokens` | int | Number of tokens emitted to the client. |
| `wall_ms` | float | Total request wall time, in milliseconds. |
| `ttft_ms` | float \| null | Time to first token, in ms. Null if no token was emitted. |
| `prefill_ms` | float \| null | Time from request start to end of prefill, in ms. |
| `decode_ms` | float \| null | Time from end of prefill to request close, in ms. |
| `cache_lookup_ms` | float | Time spent in `DFlashPrefixCache.lookup`. |
| `cache_hit_tokens` | int | Tokens reused from a prefix-cache hit (0 on miss). |
| `cache_insert_ms` | float | Cumulative time spent inserting snapshots for this request. |
| `acceptance_ratio` | float | Final speculative-decoding acceptance ratio (0.0 – 1.0). |
| `cycles_completed` | int | Number of speculative cycles run for this request. |
| `adaptive_fallback_enabled` | bool | Whether `DFLASH_ADAPTIVE_FALLBACK` was enabled for this request. |
| `adaptive_fallback_triggered` | bool | Whether adaptive fallback entered target-AR mode at least once. |
| `adaptive_fallback_count` | int | Number of adaptive fallback transitions to target-AR mode. |
| `adaptive_reprobe_count` | int | Number of post-cooldown reprobe transitions. Defaults to `0` with the conservative cooldown unless generation continues past the cooldown. |
| `adaptive_fallback_tokens` | int | Tokens emitted while adaptive fallback was in target-AR cooldown. |
| `adaptive_final_block_tokens` | int | Active speculative block size at request end. |
| `adaptive_last_probe_tokens_per_cycle` | float | Average committed tokens per cycle in the most recent completed adaptive probe window. |
| `adaptive_bad_probe_windows` | int | Configured number of consecutive bad probe windows required before fallback. |
| `adaptive_pending_bad_probe_windows` | int | Consecutive bad probe windows observed but not yet acted on at request end. |
| `adaptive_last_probe_ms_per_token` | float \| null | Milliseconds per committed token in the most recent completed adaptive probe window, when cycle timing was available. |
| `adaptive_ar_ms_per_token` | float \| null | Milliseconds per token observed during the most recent target-AR cooldown, when available. |
| `adaptive_latency_reject_count` | int | Number of cooldown or reprobe paths rejected by the latency guard. |
| `adaptive_latency_locked` | bool | Whether adaptive fallback locked back to the reference block after a latency rejection. |
| `adaptive_fallback_reason` | string \| null | Reason string for the latest adaptive fallback transition. |
| `finish_reason` | string \| null | `stop` or `length`. |
| `max_tokens` | int | Upper bound from the request body. |

Example row:

```json
{"ts":1745740445.123,"request_id":37,"mode_used":"dflash","prompt_tokens":873,"generated_tokens":1247,"wall_ms":18421.4,"ttft_ms":215.7,"prefill_ms":210.2,"decode_ms":18211.1,"cache_lookup_ms":0.31,"cache_hit_tokens":873,"cache_insert_ms":4.18,"acceptance_ratio":0.884,"cycles_completed":167,"finish_reason":"stop","max_tokens":2048}
```

### cycle_events.jsonl

One row per speculative cycle, written when `DFLASH_PROFILE=1` is set (the
runtime emits `cycle_complete` events only under profiling). Source:
`dflash_mlx/engine/spec_epoch.py:558` → `dflash_mlx/serve.py:235` →
`bench_logger.log_cycle`.

| Field | Type | Meaning |
| --- | --- | --- |
| `ts` | float | Unix time at which the row was written. |
| `request_id` | int | Matches `post_events.request_id`. |
| `cycle` | int | 1-based cycle index within the request. |
| `block_len` | int | Drafted block length for this cycle. |
| `commit_count` | int | Tokens committed to the output (1 + `acceptance_len`). |
| `acceptance_len` | int | Drafted tokens accepted by the verifier. |
| `draft_us` | float | Microseconds spent drafting (or launching the draft). |
| `verify_us` | float | Microseconds in the target verify pass. |
| `acceptance_us` | float | Microseconds in the acceptance comparison. |
| `hidden_extraction_us` | float | Microseconds spent slicing committed hidden states. |
| `rollback_us` | float | Microseconds spent rolling the target KV cache back to the commit boundary. |
| `other_us` | float | Residual cycle time (`cycle_total_us` − sum of named phases). |
| `cycle_total_us` | float | Wall time of the full cycle, in microseconds. |

Example row:

```json
{"ts":1745740445.512,"request_id":37,"cycle":42,"block_len":5,"commit_count":6,"acceptance_len":5,"draft_us":210.4,"verify_us":3815.7,"acceptance_us":42.1,"hidden_extraction_us":18.6,"rollback_us":94.0,"other_us":12.3,"cycle_total_us":4193.1}
```

### cache_events.jsonl

One row per prefix-cache `lookup` or `insert`. Source:
`dflash_mlx/cache/prefix_l1.py:67` (lookup miss), `:88` (lookup hit), and
`:111` (insert).

Common fields (always present):

| Field | Type | Meaning |
| --- | --- | --- |
| `ts` | float | Unix time at which the row was written. |
| `op` | string | `lookup` or `insert`. |
| `elapsed_us` | float | Time spent under the cache lock for this op. |

Lookup-only fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `result` | string | `exact_hit`, `prefix_hit`, or `miss`. |
| `req_tokens` | int | Length of the prompt being looked up. |
| `matched_len` | int | Tokens reused (0 on miss). |
| `entries` | int | Snapshots in cache at lookup time. |
| `fingerprint_reject` | bool | Present on miss; true if at least one stored snapshot was rejected because its model fingerprint did not match. |

Insert-only fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `kind` | string | `prefill` or `generation`. |
| `prefix_len` | int | Tokens covered by the snapshot. |
| `nbytes` | int | Snapshot size in bytes. |
| `entries_before` | int | Cache occupancy before insert. |
| `entries_after` | int | Cache occupancy after pruning + eviction. |
| `pruned` | int | Snapshots removed because the new one strictly dominates them. |
| `evicted` | int | Snapshots removed by LRU to fit the new one. |

Example rows:

```json
{"ts":1745740443.071,"op":"lookup","result":"prefix_hit","req_tokens":1024,"matched_len":873,"entries":3,"elapsed_us":312.4}
{"ts":1745740445.480,"op":"insert","kind":"generation","prefix_len":2120,"nbytes":402653184,"entries_before":3,"entries_after":3,"pruned":1,"evicted":0,"elapsed_us":4180.6}
```

## 3. DFLASH_PROFILE

`cycle_events.jsonl` is only populated when `DFLASH_PROFILE=1` (or `true`,
`yes`, `on`) is set in the environment. Without profiling, the engine takes a
faster path that skips cycle-level instrumentation; with profiling, each cycle
synchronizes more aggressively (`mx.eval` instead of `mx.async_eval`) so the
per-phase microsecond fields above are meaningful. See `RUNTIME_FLAGS.md` for
the full env-var reference.

## 4. Reading the JSONL with pandas

```python
import pandas as pd
import pathlib

LOG_DIR = pathlib.Path("/tmp/dflash-logs")

posts = pd.read_json(LOG_DIR / "post_events.jsonl", lines=True)
cycles = pd.read_json(LOG_DIR / "cycle_events.jsonl", lines=True)

# Per-request shape.
print(posts[["request_id", "wall_ms", "prompt_tokens", "generated_tokens",
             "acceptance_ratio", "cache_hit_tokens"]])

# Aggregate decode throughput from cycle data.
total_committed = cycles["commit_count"].sum()
total_wall_s = cycles["cycle_total_us"].sum() / 1_000_000.0
print(f"avg cycle tok/s: {total_committed / total_wall_s:.1f}")

# Time-in-phase breakdown.
phase_cols = ["draft_us", "verify_us", "acceptance_us",
              "hidden_extraction_us", "rollback_us", "other_us"]
print((cycles[phase_cols].sum() / cycles["cycle_total_us"].sum()).rename("share"))
```

For the cache file:

```python
cache = pd.read_json(LOG_DIR / "cache_events.jsonl", lines=True)
hits = cache[(cache["op"] == "lookup") & (cache["result"] != "miss")]
print(f"hit rate: {len(hits) / (cache['op'] == 'lookup').sum():.1%}")
```

## 5. Quick troubleshooting

- **The banner says `Prefix cache: disabled (--no-prefix-cache)` and
  `prefix cache hit` lines never appear.** Pass `--prefix-cache` (or omit the
  flag entirely; the default is enabled). The L1 cache is a process-local
  singleton, so the flag is read once at startup.
- **The HTTP server accepts connections but the first request hangs for tens
  of seconds.** With `dflash-serve`, the model load is synchronous and the
  banner only prints when the worker is ready, so a hang at this stage means
  the model is still loading. If you are running stock `mlx_lm.server`
  instead, the load happens lazily on the first request and the
  `[dflash] loading model on generation worker thread...` line will print
  while you wait.
- **The JSONL files are empty (or do not exist).** `DFLASH_BENCH_LOG_DIR`
  must be set in the same shell that launches `dflash-serve`, before the
  process starts — `bench_logger` reads the env var once on first use and
  caches the result. For cycle-level rows, also set `DFLASH_PROFILE=1`.
