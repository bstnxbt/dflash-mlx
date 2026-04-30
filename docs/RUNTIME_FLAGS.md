# Runtime flags

This file is the reference for every flag and environment variable that affects
dflash-mlx at runtime. CLI flags take precedence over environment variables;
environment variables remain useful when launching from systemd or similar
non-CLI contexts.

## dflash-serve CLI flags

`dflash-serve` exposes an OpenAI-compatible HTTP server. Flags are grouped
below; the default column reflects the actual values defined in
`dflash_mlx/server/config.py`.

### Server transport

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--host` | str | `127.0.0.1` | Interface the HTTP server binds to. Use `0.0.0.0` to expose on all interfaces. |
| `--port` | int | `8000` | TCP port the server listens on. |

### Model selection

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--model` | str | (required) | Hugging Face repo ID or local path of the target model, e.g. `mlx-community/Qwen3.6-27B-4bit`. |
| `--draft`, `--draft-model` | str | auto-resolved | Draft model reference. When omitted, dflash-mlx looks up a known draft for the target. Supply a HF repo ID or local path to override. |
| `--dflash-max-ctx` | int | unlimited | Hard cap on the maximum context length the runtime is allowed to use. When unset, the runtime uses the model's own limit. Useful for memory budgeting. |
| `--target-fa-window` | int | `0` | Experimental target-verifier full-attention KV window. `0` keeps the current full KV cache; `N > 0` uses a rotating KV cache with the last `N` tokens for target full-attention layers only. Draft, GDN, and fallback AR are not windowed. Prefix-cache is disabled in this mode. |

### Draft quantization

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--draft-quant SPEC` | str | unset (use draft as published) | Re-quantize the draft model in memory at load time. The format is `w{2,4,8}[a{16,32}][:gs{32,64,128}]`, for example `w4`, `w8a16`, `w4a32:gs128`. Weight bits select the integer width; the optional `aN` controls activation precision (16 = bfloat16, 32 = float32); the optional `gsN` selects group size. |

Example: `--draft-quant w4:gs64` requantizes the draft to 4-bit weights with
group size 64 while keeping bfloat16 activations.

### Generation defaults

These values seed the server's defaults; per-request fields in the JSON body
override them.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--temp` | float | `0.0` | Sampling temperature. `0.0` selects greedy decoding. |
| `--top-p` | float | `1.0` | Nucleus sampling cutoff. `1.0` disables nucleus filtering. |
| `--top-k` | int | `0` | Top-k sampling. `0` disables top-k filtering. |
| `--min-p` | float | `0.0` | Minimum probability filter. `0.0` disables it. |
| `--max-tokens` | int | `512` | Default ceiling on generated tokens per request. |
| `--chat-template` | str | `""` | Inline Jinja chat template that overrides the tokenizer's bundled template. Empty string keeps the bundled one. |
| `--use-default-chat-template` | flag | off | When set, force the tokenizer's `default_chat_template` even if a custom template ships with the model. |
| `--chat-template-args` | JSON dict | `{}` | Extra keyword arguments forwarded to the chat-template renderer, e.g. `'{"enable_thinking": true}'`. |

### Prompt cache

The mlx_lm-style LRU prompt cache. Stores prompt KV state per session so that
repeated requests with the same prefix skip prefill.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--prompt-cache-size` | int | `10` | Maximum number of prompt cache slots retained in the LRU. |
| `--prompt-cache-bytes` | int | unset (no byte cap) | Hard byte ceiling for the prompt cache. When unset, only the entry count is bounded. |

### Prefix cache

The dflash prefix cache stores cross-turn KV snapshots so that multi-turn
agentic workloads avoid re-prefilling shared prefixes.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--prefix-cache` / `--no-prefix-cache` | flag | enabled | Toggles the dflash prefix cache. Big win on multi-turn agentic workloads, roughly neutral on single-turn. |
| `--prefix-cache-max-entries` | int | `4` | Maximum number of cached prefix snapshots. Must be `> 0`. |
| `--prefix-cache-max-bytes` | int | `8589934592` (8 GiB) | Maximum total bytes the prefix cache may retain. Must be `>= 0`; `0` effectively disables retention. |

### Logging

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--log-level` | str | `INFO` | One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Sets the root logger level for the server process. |

## dflash CLI flags

`dflash` is a one-shot generation entry point defined in
`dflash_mlx/generate.py`. It loads the target and draft, runs a single prompt,
streams tokens to stdout, and prints a one-line summary to stderr.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--model` | str | (required) | Target model reference (HF repo ID or local path). |
| `--prompt` | str | (required) | Prompt string passed to the model. |
| `--max-tokens` | int | `2048` | Maximum number of tokens to generate. |
| `--no-chat-template` | flag | off | When set, the prompt is fed verbatim instead of being wrapped by the tokenizer's chat template. |
| `--draft` | str | auto-resolved | Optional draft model override. When omitted, dflash-mlx resolves a published draft for the target. |
| `--target-fa-window` | int | `0` | Experimental target-verifier full-attention KV window. Same semantics as `dflash-serve --target-fa-window`. |

## Environment variables

Every variable below is read directly from `os.environ`. CLI flags on
`dflash-serve` write the corresponding env var during startup, so downstream
readers see a single source of truth.

### Runtime tuning

| Variable | Default | Effect |
| --- | --- | --- |
| `DFLASH_MAX_CTX` | `0` (unlimited) | Hard cap on the runtime context length, mirroring `--dflash-max-ctx`. Set to a positive integer to clamp; values `<= 0` disable the cap. Example: `DFLASH_MAX_CTX=32768`. |
| `DFLASH_TARGET_FA_WINDOW` | `0` | Experimental target-verifier full-attention KV window, mirroring `--target-fa-window` on `dflash`, `dflash-serve`, and `dflash-benchmark`. `0` preserves full KV; `N > 0` uses `RotatingKVCache(max_size=N)` for target FA layers only. Prefix-cache is disabled while this is non-zero. |
| `DFLASH_PREFILL_STEP_SIZE` | `8192` | Number of prompt tokens processed per prefill chunk. Larger values reduce launch overhead at the cost of peak memory. Example: `DFLASH_PREFILL_STEP_SIZE=4096`. |

### Draft

| Variable | Default | Effect |
| --- | --- | --- |
| `DFLASH_DRAFT_QUANT` | unset | Same SPEC syntax as `--draft-quant`. Applied at draft load when the CLI flag is not provided. Example: `DFLASH_DRAFT_QUANT=w4:gs64`. |
| `DFLASH_DRAFT_SINK` | `64` | Number of leading tokens kept as a permanent attention sink in the draft KV window. Set to `0` to disable. Example: `DFLASH_DRAFT_SINK=128`. |
| `DFLASH_DRAFT_WINDOW` | `1024` | Sliding window length the draft uses for non-sink tokens. Setting any non-empty value also flips an internal "user override" bit, so explicit assignment is honoured even if it matches the default. Example: `DFLASH_DRAFT_WINDOW=2048`. |

### Adaptive fallback

Adaptive fallback is disabled by default. When enabled, the engine watches
speculative decode progress over probe windows; if sustained low committed
tokens per cycle indicates harmful speculation, it falls back to target
autoregressive decoding for a cooldown window. The default cooldown is
conservative, so most single requests remain in target-AR after fallback instead
of automatically reprobeing with a smaller verifier block. Smaller-block
reprobe is available for experiments, but changing verifier chunk shape can
alter greedy output in near-tie cases.

| Variable | Default | Effect |
| --- | --- | --- |
| `DFLASH_ADAPTIVE_FALLBACK` | unset (off) | Master switch. Truthy values enable adaptive fallback; `0`, `false`, `no`, `off`, or empty disable it. Example: `DFLASH_ADAPTIVE_FALLBACK=1`. |
| `DFLASH_ADAPTIVE_MIN_TOKENS_PER_CYCLE` | `2.0` | Minimum average committed tokens per probe cycle. Probe windows below this threshold count as bad speculation. Example: `DFLASH_ADAPTIVE_MIN_TOKENS_PER_CYCLE=2.5`. |
| `DFLASH_ADAPTIVE_PROBE_CYCLES` | `8` | Number of speculative cycles in each probe window. Clamped to `>= 1`. Example: `DFLASH_ADAPTIVE_PROBE_CYCLES=12`. |
| `DFLASH_ADAPTIVE_BAD_PROBE_WINDOWS` | `2` | Consecutive bad probe windows required before entering fallback. Clamped to `>= 1`. Example: `DFLASH_ADAPTIVE_BAD_PROBE_WINDOWS=3`. |
| `DFLASH_ADAPTIVE_TARGET_AR_COOLDOWN` | `4096` | Number of generated tokens to hold target-AR mode after fallback before considering reprobe. Clamped to `>= 1`. Example: `DFLASH_ADAPTIVE_TARGET_AR_COOLDOWN=512`. |
| `DFLASH_ADAPTIVE_LATENCY_MARGIN` | `1.0` | Latency guard for cooldown/reprobe paths. A new path must be no slower than the reference path multiplied by this margin. Example: `DFLASH_ADAPTIVE_LATENCY_MARGIN=1.1`. |
| `DFLASH_ADAPTIVE_REPROBE_BLOCK_TOKENS` | unset | Explicit block size for the post-cooldown reprobe. When unset and the cooldown completes, the runtime probes a smaller block derived from the current block. Example: `DFLASH_ADAPTIVE_REPROBE_BLOCK_TOKENS=2`. |

### Prefix cache

These variables are overridden by the corresponding CLI flags on
`dflash-serve` when both are present.

| Variable | Default | Effect |
| --- | --- | --- |
| `DFLASH_PREFIX_CACHE` | `0` | Master switch for the prefix cache. Truthy values (`1`, `true`, `yes`) enable it; `0`, `false`, `no`, or empty disable it. The `dflash-serve` CLI sets this to `1` unless `--no-prefix-cache` is passed. Example: `DFLASH_PREFIX_CACHE=1`. |
| `DFLASH_PREFIX_CACHE_MAX_ENTRIES` | `4` | Maximum number of cached prefix snapshots. Clamped to `>= 1`. Example: `DFLASH_PREFIX_CACHE_MAX_ENTRIES=8`. |
| `DFLASH_PREFIX_CACHE_MAX_BYTES` | `8589934592` (8 GiB) | Maximum bytes retained across all prefix snapshots. Clamped to `>= 0`. Example: `DFLASH_PREFIX_CACHE_MAX_BYTES=4294967296`. |

### Verify path

The verify path runs custom Metal kernels for selected `QuantizedLinear`
layers. These knobs are advanced and are typically left at their defaults.

| Variable | Default | Effect |
| --- | --- | --- |
| `DFLASH_VERIFY_LINEAR` | unset (auto) | Force-enables (`1`) or force-disables (`0`) verify-linear installation. When unset, the runtime auto-selects based on the target architecture. Example: `DFLASH_VERIFY_LINEAR=1`. |
| `DFLASH_VERIFY_QMM` | `""` (off) | Set to `1` to route eligible quantized matmuls through the verify-QMM kernel path. Auto-set to `1` when verify-linear installs successfully. Example: `DFLASH_VERIFY_QMM=1`. |
| `DFLASH_VERIFY_VARIANT` | `auto` | Selects the verify-QMM kernel variant. `auto` picks between `mma2big` and `mma2big_pipe` based on `K`/`N`. Other accepted values are `mma2big` and `mma2big_pipe`. Example: `DFLASH_VERIFY_VARIANT=mma2big_pipe`. |
| `DFLASH_VERIFY_LEN` | unset (use full block) | Caps the verify length per cycle to at most this many tokens. Values `<= 0` are ignored. Example: `DFLASH_VERIFY_LEN=8`. |
| `DFLASH_VERIFY_MAX_N` | `100000` | Excludes layers whose output dimension `N` is at or above this value from the verify path. Example: `DFLASH_VERIFY_MAX_N=65536`. |
| `DFLASH_VERIFY_QMM_KPARTS` | `4` | Number of K-axis partitions for the pipelined verify-QMM kernel. Auto-overridden when the variant selector returns its own `K_PARTS` recommendation. Example: `DFLASH_VERIFY_QMM_KPARTS=8`. |
| `DFLASH_VERIFY_INCLUDE` | `all` | Comma-separated allow-list of projection tags eligible for the verify path. Group aliases `mlp`, `attn`, and `gdn` expand to their member tags (e.g. `mlp` becomes `mlp_gate,mlp_up,mlp_down`). Individual tags include `attn_q`, `attn_k`, `attn_v`, `attn_o`, `gdn_qkv`, `gdn_z`, `gdn_o`. Example: `DFLASH_VERIFY_INCLUDE=mlp,attn_o`. |

### Telemetry

| Variable | Default | Effect |
| --- | --- | --- |
| `DFLASH_PROFILE` | unset (off) | Enables per-cycle profiling instrumentation. Truthy values (anything other than empty, `0`, `false`, `no`) turn it on. Setting `DFLASH_BENCH_LOG_DIR` also implicitly enables profiling. Example: `DFLASH_PROFILE=1`. |
| `DFLASH_BENCH_LOG_DIR` | unset | Directory where the bench logger appends `post_events.jsonl`, `cycle_events.jsonl`, and `cache_events.jsonl`. The directory is created on demand. Setting this also enables `DFLASH_PROFILE`-style per-cycle telemetry. Example: `DFLASH_BENCH_LOG_DIR=/tmp/dflash-logs`. |

## Env variables without a public CLI flag

These remain env-only on purpose or because the surface is still experimental.
Do not treat them as product-facing knobs until they have a real CLI flag and a
benchmark protocol.

| Variable | Current status |
| --- | --- |
| `DFLASH_DRAFT_SINK` | Env-only draft cache tuning. No CLI yet. |
| `DFLASH_DRAFT_WINDOW` | Env-only draft cache tuning. No CLI yet. |
| `DFLASH_PREFILL_STEP_SIZE` | Env-only in the runtime path. `dflash-serve` has an old hidden parser argument, but it is not a public wired flag. |
| `DFLASH_ADAPTIVE_FALLBACK` | Env-only adaptive fallback switch. No CLI yet. |
| `DFLASH_ADAPTIVE_MIN_TOKENS_PER_CYCLE` | Env-only adaptive fallback threshold. No CLI yet. |
| `DFLASH_ADAPTIVE_PROBE_CYCLES` | Env-only adaptive fallback probe window size. No CLI yet. |
| `DFLASH_ADAPTIVE_BAD_PROBE_WINDOWS` | Env-only adaptive fallback confirmation count. No CLI yet. |
| `DFLASH_ADAPTIVE_TARGET_AR_COOLDOWN` | Env-only adaptive fallback target-AR hold length. No CLI yet. |
| `DFLASH_ADAPTIVE_LATENCY_MARGIN` | Env-only adaptive fallback latency guard. No CLI yet. |
| `DFLASH_ADAPTIVE_REPROBE_BLOCK_TOKENS` | Env-only experimental adaptive fallback reprobe block override. No CLI yet. |
| `DFLASH_VERIFY_LINEAR` | Env-only verify kernel override. No CLI yet. |
| `DFLASH_VERIFY_QMM` | Env-only verify-QMM switch, also auto-set by runtime in some load paths. No CLI yet. |
| `DFLASH_VERIFY_VARIANT` | Env-only verify-QMM kernel variant selector. No CLI yet. |
| `DFLASH_VERIFY_LEN` | Env-only per-cycle verify length cap. No CLI yet. |
| `DFLASH_VERIFY_MAX_N` | Env-only verify layer exclusion threshold. No CLI yet. |
| `DFLASH_VERIFY_QMM_KPARTS` | Env-only verify-QMM K partition override. No CLI yet. |
| `DFLASH_VERIFY_INCLUDE` | Env-only projection allow-list for verify kernels. No CLI yet. |
| `DFLASH_PROFILE` | Env-only profiling switch. Agentic trace enables profiling indirectly through `DFLASH_BENCH_LOG_DIR`, but there is no general public CLI flag. |
| `DFLASH_BENCH_LOG_DIR` | Env-only structured logging destination. `bench_agentic_trace` sets it internally for managed server runs, but `dflash-serve` has no public `--bench-log-dir` flag. |

## Precedence rules

When both a CLI flag and the corresponding environment variable are set, the
CLI flag wins. `dflash-serve` writes its parsed values back into the
environment via `normalize_cli_args`, so every downstream component reads from a
single source of truth: the environment after CLI parsing has run. This means
launching from a wrapper script that sets env vars and then invokes
`dflash-serve` with explicit flags will reflect the flag values, not the
pre-existing env values.

## Quick examples

```
dflash-serve --model mlx-community/Qwen3.6-27B-4bit
```

Launches the server with all defaults: localhost binding on port 8000, prefix
cache enabled, draft auto-resolved from the target name.

```
dflash-serve --model mlx-community/Qwen3.6-35B-A3B-4bit --no-prefix-cache --max-tokens 8192
```

Runs the 35B-A3B target with the prefix cache disabled (single-turn or
benchmark scenarios) and lifts the per-request token ceiling to 8192.

```
dflash-serve --model mlx-community/Qwen3.6-27B-4bit --target-fa-window 4096
```

Runs the experimental target full-attention rotating KV probe. This is not a
product default; it disables prefix-cache and only affects target verifier
full-attention layers.

```
DFLASH_BENCH_LOG_DIR=/tmp/dflash-logs dflash-serve --model mlx-community/Qwen3.6-27B-4bit --log-level DEBUG
```

Enables structured per-cycle telemetry under `/tmp/dflash-logs` and turns on
debug-level logging, suitable for diagnosing acceptance or scheduling issues
during a bench run.
