# Runtime Flags

This document describes the live product config surface. Historical experiment
flags belong in `old_status/`, not here.

## Config Model

For `dflash serve` and `dflash doctor`, runtime config is resolved once at
startup:

1. CLI flag;
2. startup env var;
3. profile value;
4. product default.

After that, engine code consumes typed `RuntimeContext`. Product behavior should
not depend on rereading `os.environ` during a request.

`dflash generate` and `dflash benchmark` expose a smaller explicit flag surface.
They do not support the full server profile system.

## Profiles

Use:

```bash
dflash profiles
```

Current profiles:

| Profile | Prefill | Draft window | Prefix cache | L1 entries / byte budget | L2 | Intent |
| --- | ---: | --- | --- | --- | --- | --- |
| `balanced` | 4096 | `64+1024` | on | `4 / 8GiB` | off | default normal coding |
| `fast` | 8192 | `64+1024` | on | `4 / 16GiB` | off | throughput first |
| `low-memory` | 1024 | `64+1024` | on | `2 / 2GiB` | off | lower pressure, slower prefill |
| `long-session` | 4096 | `64+1024` | on | `8 / 8GiB` | on / `50GiB` | revisit-oriented long sessions |

`clear_cache_boundaries` is currently off in all profiles.

## `dflash serve`

Core:

| Flag | Meaning |
| --- | --- |
| `--profile {balanced,fast,low-memory,long-session}` | preset defaults |
| `--list-profiles` | print profiles and exit |
| `--model REF_OR_PATH` | target model |
| `--draft-model REF_OR_PATH`, `--draft REF_OR_PATH` | draft override |
| `--host HOST` | server host |
| `--port PORT` | server port |
| `--log-level LEVEL` | Python log level |

Generation defaults:

| Flag | Meaning |
| --- | --- |
| `--temp FLOAT` | default request temperature |
| `--top-p FLOAT` | default nucleus sampling cutoff |
| `--top-k INT` | default top-k, `0` disables |
| `--min-p FLOAT` | default min-p filter |
| `--max-tokens INT` | default request max tokens |
| `--chat-template TEMPLATE` | inline chat template override |
| `--use-default-chat-template` | force tokenizer default template |
| `--chat-template-args JSON` | JSON args for template rendering |

DFlash runtime:

| Flag | Meaning |
| --- | --- |
| `--prefill-step-size INT` | target prefill chunk size |
| `--draft-sink-size INT` | draft cache sink tokens |
| `--draft-window-size INT` | draft cache rolling window tokens |
| `--verify-len-cap INT` | max tokens per verify forward, `0` means block size |
| `--verify-mode {auto,off}` | verifier path mode; `off` is debug/parity only |
| `--dflash-max-ctx INT` | DFlash runtime context cap; `0` means no cap |
| `--target-fa-window INT` | experimental target FA rotating window; `0` means full KV |
| `--draft-quant SPEC` | optional in-memory draft quantization, e.g. `w4:gs64` |

Prefix cache:

| Flag | Meaning |
| --- | --- |
| `--prefix-cache`, `--no-prefix-cache` | enable/disable DFlash prefix snapshots |
| `--prefix-cache-max-entries INT` | L1 snapshot entry budget |
| `--prefix-cache-max-bytes BYTES` | L1 snapshot byte budget |
| `--max-snapshot-tokens INT` | snapshot insert token cap; `0` disables the cap |
| `--prefix-cache-l2`, `--no-prefix-cache-l2` | enable/disable SSD L2 for evicted snapshots |
| `--prefix-cache-l2-dir PATH` | L2 root directory |
| `--prefix-cache-l2-max-bytes BYTES` | L2 disk budget |

Notes:

- `target_fa_window > 0` disables prefix cache reuse because snapshot cache shape
  differs from full-KV verification.
- `max_snapshot_tokens` skips oversized inserts when L2 is absent. With L2
  enabled, oversized snapshots may still be accepted and later spilled by budget.
- L2 helps revisits after L1 eviction. It does not reduce the active request KV
  bucket.

Upstream prompt cache:

| Flag | Meaning |
| --- | --- |
| `--prompt-cache-size INT` | upstream `mlx_lm` prompt-cache entry count |
| `--prompt-cache-bytes BYTES` | parsed byte cap; verify current upstream wiring before relying on it |

Diagnostics:

| Flag | Meaning |
| --- | --- |
| `--diagnostics {off,basic,full}` | structured diagnostics mode |
| `--diagnostics-dir PATH` | output directory |
| `--memory-waterfall`, `--no-memory-waterfall` | advanced direct toggle for memory bucket events |
| `--bench-log-dir PATH` | advanced direct event log directory |

Metal limits:

| Flag | Meaning |
| --- | --- |
| `--wired-limit auto|none|BYTES` | MLX wired memory limit policy |
| `--cache-limit auto|none|BYTES` | MLX cache limit policy |

## `dflash generate`

```bash
dflash generate --model Qwen/Qwen3.5-4B --prompt "Explain DFlash."
```

Flags:

| Flag | Meaning |
| --- | --- |
| `--model REF_OR_PATH` | target model, required |
| `--prompt TEXT` | prompt, required |
| `--max-tokens INT` | generated token cap |
| `--no-chat-template` | use raw prompt text |
| `--draft REF_OR_PATH` | draft override |
| `--verify-mode {auto,off}` | verifier path mode |
| `--target-fa-window INT` | target FA rotating window |
| `--draft-sink-size INT` | draft cache sink tokens |
| `--draft-window-size INT` | draft cache rolling window tokens |
| `--verify-len-cap INT` | max tokens per verify forward |

Generate disables cross-request prefix caching. Use it for smoke tests, not for
benchmark claims.

## `dflash benchmark`

```bash
dflash benchmark --model Qwen/Qwen3.5-4B --max-tokens 64
```

Flags:

| Flag | Meaning |
| --- | --- |
| `--prompt TEXT` | prompt text |
| `--ctx INT` | build an approximate synthetic long-context prompt |
| `--max-tokens INT` | generated token count |
| `--block-tokens INT` | DFlash verify block size |
| `--repeat INT` | measured runs |
| `--cooldown SECONDS` | sleep between runs |
| `--model REF_OR_PATH` | target model |
| `--draft REF_OR_PATH` | draft override |
| `--no-chat-template` | raw prompt mode |
| `--quantize-draft` | quantize draft after load |
| `--no-eos` | suppress EOS so generation reaches token cap |
| `--split-sdpa`, `--no-split-sdpa` | target split-SDPA verifier path; enabled by default |
| `--target-fa-window INT` | target FA rotating window |
| `--draft-sink-size INT` | draft cache sink tokens |
| `--draft-window-size INT` | draft cache rolling window tokens |
| `--verify-len-cap INT` | max tokens per verify forward |
| `--no-memory` | omit memory medians from summary |
| `--out PATH` | artifact directory |

New benchmark outputs default to `.artifacts/dflash/benchmarks/...`.

## `dflash doctor`

Doctor accepts the same runtime config flags as the server for validation:

```bash
dflash doctor --profile low-memory
dflash doctor --profile long-session --prefix-cache-l2 --json
dflash doctor --model Qwen/Qwen3.5-4B --load-model
```

Common flags:

| Flag | Meaning |
| --- | --- |
| `--json` | machine-readable report |
| `--strict` | return non-zero on warnings |
| `--load-model` | actually load model/draft |
| `--model REF_OR_PATH` | target model for registry/load checks |
| `--draft REF_OR_PATH` | draft override |
| runtime flags | profile, prefill, draft window, prefix cache, L2, FA window, max ctx |

## Startup Env Vars

These are accepted as startup inputs for server/doctor config. CLI flags override
them.

| Env var | Matching config |
| --- | --- |
| `DFLASH_RUNTIME_PROFILE` | `--profile` |
| `DFLASH_PREFILL_STEP_SIZE` | `--prefill-step-size` |
| `DFLASH_DRAFT_SINK_SIZE` | `--draft-sink-size` |
| `DFLASH_DRAFT_WINDOW_SIZE` | `--draft-window-size` |
| `DFLASH_VERIFY_LEN_CAP` | `--verify-len-cap` |
| `DFLASH_PREFIX_CACHE` | `--prefix-cache` / `--no-prefix-cache` |
| `DFLASH_PREFIX_CACHE_MAX_ENTRIES` | `--prefix-cache-max-entries` |
| `DFLASH_PREFIX_CACHE_MAX_BYTES` | `--prefix-cache-max-bytes` |
| `DFLASH_CLEAR_CACHE_BOUNDARIES` | `--clear-cache-boundaries` |
| `DFLASH_MAX_SNAPSHOT_TOKENS` | `--max-snapshot-tokens` |
| `DFLASH_PREFIX_CACHE_L2_ENABLED` | `--prefix-cache-l2` |
| `DFLASH_PREFIX_CACHE_L2_DIR` | `--prefix-cache-l2-dir` |
| `DFLASH_PREFIX_CACHE_L2_MAX_BYTES` | `--prefix-cache-l2-max-bytes` |
| `DFLASH_TARGET_FA_WINDOW` | `--target-fa-window` |
| `DFLASH_MAX_CTX` | `--dflash-max-ctx` |
| `DFLASH_VERIFY_MODE` | `--verify-mode` |

Prefer CLI flags for reproducible local runs. Env vars are mainly for Docker,
CI, and wrappers.

## Internal Debug Env Vars

These are not product flags. They are kept for kernel and verifier experiments:

- `DFLASH_VERIFY_LINEAR`
- `DFLASH_VERIFY_QMM`
- `DFLASH_VERIFY_VARIANT`
- `DFLASH_VERIFY_MAX_N`
- `DFLASH_VERIFY_QMM_KPARTS`
- `DFLASH_VERIFY_INCLUDE`

Do not use them for public benchmark claims.

## Validation Rules

The runtime rejects invalid config before serving:

- `prefill_step_size > 0`
- `draft_sink_size >= 0`
- `draft_window_size > 0`
- `verify_len_cap >= 0`
- `prefix_cache_max_entries > 0`
- `prefix_cache_max_bytes >= 0`
- `max_snapshot_tokens >= 0`
- `prefix_cache_l2_max_bytes >= 0`
- `target_fa_window >= 0`
- `dflash_max_ctx >= 0`
- `verify_mode in {auto, off}`

Use `dflash doctor --json` to see the resolved effective config.
