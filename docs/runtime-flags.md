# Runtime Flags

This document describes the live product config surface. Historical experiment
flags belong in `old_status/`, not here.

## Config Model

For `dflash serve` and `dflash doctor`, runtime config is resolved once at
startup:

1. CLI flag;
2. startup env var;
3. product default.

After that, engine code consumes typed `RuntimeContext`. Product behavior should
not depend on rereading `os.environ` during a request.

`dflash generate` and `dflash benchmark` expose a smaller explicit flag surface.

`dflash serve` defaults are the product session policy: prefill `2048`, draft
sink/window `64+1024`, prefix cache on, L1 `8 / 8GiB`, boundary cache clears on,
L2 on with `50GiB`, and MLX cache limit `4GB`. Explicit flags override those
values. `--cache-limit auto` restores the generic wired-limit/4 policy, and
`--cache-limit none` skips `mx.set_cache_limit`.

Gemma4 long-context memory-pressure note: set `--prefill-step-size 1024`
explicitly. The measured 31B serve probes also showed lower peak memory and TTFT
with `--prefill-step-size 512` under tighter memory pressure, but validate that
on your prompt; it is not a public benchmark throughput claim.

## `dflash serve`

Core:

| Flag | Meaning |
| --- | --- |
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
| `--fastpath-max-tokens INT` | short-output target-only AR fast path threshold; default `0` disables; set a positive value to opt in; does not change DFlash max context |
| `--chat-template TEMPLATE` | inline chat template override |
| `--use-default-chat-template` | force tokenizer default template |
| `--enable-thinking` | set chat-template arg `enable_thinking=true`; default follows tokenizer/model template |
| `--chat-template-args JSON` | JSON args for template rendering |

DFlash runtime:

| Flag | Meaning |
| --- | --- |
| `--prefill-step-size INT` | target prefill chunk size |
| `--draft-sink-size INT` | draft cache sink tokens |
| `--draft-window-size INT` | draft cache rolling window tokens |
| `--verify-len-cap INT` | max tokens per verify forward, `0` means block size |
| `--verify-mode {dflash,adaptive,ddtree,off}` | verifier path mode; default `adaptive` probes shorter low-acceptance blocks, `ddtree` verifies a small branch batch, `off` is debug/parity only |
| `--dflash-max-ctx INT` | DFlash runtime context cap; `0` means no cap |
| `--target-fa-window INT` | experimental target FA rotating window; `0` means full KV |
| `--clear-cache-boundaries`, `--no-clear-cache-boundaries` | clear the MLX cache at safe request boundaries |

Draft loading:

| Flag | Meaning |
| --- | --- |
| `--draft-quant SPEC` | draft quantization override, e.g. `w4:gs64`; use `none` to disable model defaults |

Validated large DFlash drafts default to `w4` in memory because current Qwen3.5,
Qwen3.6, and Gemma4 README-prompt probes showed the best practical
memory/throughput tradeoff. Use `--draft-quant none` for bf16/non-quant draft
debugging or A/B comparisons.

On M1/M2 GPUs, quantized DFlash drafts are loaded with FP16 floating tensors to
avoid the BF16 emulation path. Run metadata records this as
`draft_load_dtype: "float16"`. `--draft-quant none` preserves the checkpoint
dtype for explicit A/B comparisons.

Prefix cache:

| Flag | Meaning |
| --- | --- |
| `--prefix-cache`, `--no-prefix-cache` | enable/disable DFlash prefix snapshots |
| `--prefix-cache-max-entries INT` | L1 snapshot entry budget |
| `--prefix-cache-max-bytes BYTES` | L1 snapshot byte budget; raw integer bytes or suffixes like `2GB` |
| `--max-snapshot-tokens INT` | snapshot insert token cap; `0` disables the cap |
| `--prefix-cache-l2`, `--no-prefix-cache-l2` | enable/disable SSD L2 for persisted/spilled snapshots |
| `--prefix-cache-l2-dir PATH` | L2 root directory |
| `--prefix-cache-l2-max-bytes BYTES` | L2 disk budget; raw integer bytes or suffixes like `50GB` |

Notes:

- `target_fa_window > 0` disables prefix cache reuse because snapshot cache shape
  differs from full-KV verification.
- `max_snapshot_tokens` bounds L1 snapshot inserts. With L2 enabled, oversized
  prefill snapshots may still be persisted to disk for later restores.
- L2 prefill frontiers are stored at a coarser internal stride, currently at
  least `8192` tokens, to avoid writing every prefill chunk as a large snapshot.
- For chat-template requests with a stable assistant/model boundary, DFlash
  stores reusable prefill snapshots and may also publish end-of-generation
  snapshots so later agentic turns can restore deeper tool/result prefixes.
- L2 persists admitted prefill snapshots and stores snapshots spilled from L1.
  It helps revisits after process restart or L1 eviction. It does not reduce the
  active request KV bucket.

Upstream prompt cache:

| Flag | Meaning |
| --- | --- |
| `--prompt-cache-size INT` | upstream `mlx_lm` prompt-cache entry count |

Diagnostics:

| Flag | Meaning |
| --- | --- |
| `--diagnostics {off,basic,full}` | structured diagnostics mode |
| `--diagnostics-dir PATH` | output directory |

Metal limits:

| Flag | Meaning |
| --- | --- |
| `--wired-limit auto\|none\|BYTES` | MLX wired memory limit policy |
| `--cache-limit auto\|none\|BYTES` | MLX cache limit policy; omitted uses the serve default `4GB` |

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
| `--draft-quant SPEC` | draft quantization override, e.g. `w4:gs64`; use `none` to disable model defaults |
Runtime override flags:

| Flag | Meaning |
| --- | --- |
| `--verify-mode {dflash,adaptive,ddtree,off}` | verifier path mode; default `adaptive` probes shorter low-acceptance blocks, `ddtree` verifies a small branch batch, `off` is debug/parity only |
| `--prefill-step-size INT` | target prefill chunk size |
| `--target-fa-window INT` | experimental target FA rotating window; `0` means full KV |
| `--draft-sink-size INT` | draft cache sink tokens |
| `--draft-window-size INT` | draft cache rolling window tokens |
| `--verify-len-cap INT` | max tokens per verify forward, `0` means block size |

Generate disables cross-request prefix caching. Use it for local sanity checks, not for
benchmark claims.

## `dflash benchmark`

```bash
PROMPT='The function $f$ satisfies the functional equation \[ f(x) + f(y) = f(x + y) - xy - 1 \] for all real numbers $x$ and $y$. If $f(1) = 1$, then find all integers $n$ such that $f(n) = n$. Enter all such integers, separated by commas. Please reason step by step, and put your final answer within \boxed{}.'

dflash benchmark \
  --model Qwen/Qwen3.5-9B \
  --prompt "$PROMPT" \
  --max-tokens 1024 \
  --repeat 3 \
  --cooldown 60 \
  --no-eos
```

`dflash benchmark` measures runtime behavior. `humaneval`, `gsm8k`, `math500`,
and `aime25` load real Hugging Face datasets via the optional `datasets`
package. `aime25` also records exact integer score fields for baseline and
DFlash and defaults to `65536` generated tokens; the other dataset suites are runtime prompt suites only. `smoke` is a
local CLI sanity check only; `longctx` is local/offline synthetic context
stress. Use `--prompt-file PATH` for offline/custom prompt JSONL.

Flags:

| Flag | Meaning |
| --- | --- |
| `--suite {smoke,humaneval,gsm8k,math500,aime25,longctx}` | named benchmark prompt suite |
| `--limit N` | deterministic prompt count limit |
| `--ctx-tokens N` | synthetic context target for `longctx` |
| `--prompt-file PATH` | JSONL prompt override with `id`, `suite`, `prompt` rows |
| `--shuffle` | shuffle HF dataset rows before applying `--limit` |
| `--seed INT` | shuffle seed used only with `--shuffle` |
| `--prompt TEXT` | prompt text |
| `--max-tokens INT` | generated token count; default is `65536` for `aime25`, `64` otherwise |
| `--block-tokens INT` | DFlash verify block size |
| `--repeat INT` | measured runs |
| `--cooldown SECONDS` | sleep between baseline/DFlash legs and repeated runs |
| `--model REF_OR_PATH` | target model |
| `--draft REF_OR_PATH` | draft override |
| `--no-chat-template` | raw prompt mode |
| `--draft-quant SPEC` | draft quantization override, e.g. `w4:gs64`; use `none` to disable model defaults |
| `--no-eos` | suppress EOS so generation reaches token cap |
| `--only-dflash` | skip baseline MLX and run DFlash only |
Runtime override flags:

| Flag | Meaning |
| --- | --- |
| `--verify-mode {dflash,adaptive,ddtree,off}` | verifier path mode; default `adaptive` probes shorter low-acceptance blocks, `ddtree` verifies a small branch batch, `off` is debug/parity only |
| `--prefill-step-size INT` | target prefill chunk size |
| `--target-fa-window INT` | experimental target FA rotating window; `0` means full KV |
| `--draft-sink-size INT` | draft cache sink tokens |
| `--draft-window-size INT` | draft cache rolling window tokens |
| `--verify-len-cap INT` | max tokens per verify forward, `0` means block size |

Benchmark output flags:

| Flag | Meaning |
| --- | --- |
| `--no-memory` | omit memory medians from summary |
| `--out PATH` | artifact directory |

New benchmark outputs default to `.artifacts/dflash/benchmarks/...`.

## `dflash doctor`

Doctor accepts the same runtime config flags as the server for validation:

```bash
dflash doctor --prefill-step-size 1024
dflash doctor --no-prefix-cache-l2 --json
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
| runtime flags | prefill, draft window, prefix cache, L2, FA window, max ctx |

## Startup Env Vars

These are accepted as startup inputs for server/doctor config. CLI flags override
them.

| Env var | Matching config |
| --- | --- |
| `DFLASH_PREFILL_STEP_SIZE` | `--prefill-step-size INT` |
| `DFLASH_DRAFT_SINK_SIZE` | `--draft-sink-size INT` |
| `DFLASH_DRAFT_WINDOW_SIZE` | `--draft-window-size INT` |
| `DFLASH_VERIFY_LEN_CAP` | `--verify-len-cap INT` |
| `DFLASH_CLEAR_CACHE_BOUNDARIES` | `--clear-cache-boundaries`, `--no-clear-cache-boundaries` |
| `DFLASH_VERIFY_MODE` | `--verify-mode {dflash,adaptive,ddtree,off}` |
| `DFLASH_MAX_SNAPSHOT_TOKENS` | `--max-snapshot-tokens INT` |
| `DFLASH_PREFIX_CACHE_L2_ENABLED` | `--prefix-cache-l2`, `--no-prefix-cache-l2` |
| `DFLASH_PREFIX_CACHE_L2_DIR` | `--prefix-cache-l2-dir PATH` |
| `DFLASH_PREFIX_CACHE_L2_MAX_BYTES` | `--prefix-cache-l2-max-bytes BYTES` |
| `DFLASH_PREFIX_CACHE` | `--prefix-cache`, `--no-prefix-cache` |
| `DFLASH_PREFIX_CACHE_MAX_ENTRIES` | `--prefix-cache-max-entries INT` |
| `DFLASH_PREFIX_CACHE_MAX_BYTES` | `--prefix-cache-max-bytes BYTES` |
| `DFLASH_TARGET_FA_WINDOW` | `--target-fa-window INT` |
| `DFLASH_MAX_CTX` | `--dflash-max-ctx INT` |

Byte-budget env vars accept raw integer bytes or suffixes such as `2GB` and
`50GB`, matching the CLI flags.

Prefer CLI flags for reproducible local runs. Env vars are mainly for Docker,
CI, and wrappers.

## Internal Debug Env Vars

These are not product flags. They are kept for kernel and verifier experiments:

- `DFLASH_VERIFY_LINEAR`
- `DFLASH_VERIFY_QMM`
- `DFLASH_VERIFY_VARIANT` (`auto` selects the M16 NAX verify kernel on
  `applegpu_g17*` when the shape and macOS runtime support it; forced values
  are for experiments only)
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
- `prefix_cache_l2_dir` must be non-empty when L2 is enabled
- `prefix_cache_l2_max_bytes >= 0`
- `target_fa_window >= 0`
- `dflash_max_ctx >= 0`
- `verify_mode in {dflash, adaptive, ddtree, off}`

Use `dflash doctor --json` to see the resolved effective config.
