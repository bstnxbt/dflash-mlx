# Benchmarking

Use `dflash benchmark` for public local performance claims. Keep lab harnesses
separate from product claims.

## Public Runtime Benchmark

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

Default protocol:

1. load target and matching DFlash draft;
2. resolve a deterministic prompt suite;
3. render/tokenize each prompt once;
4. run baseline MLX first;
5. run DFlash second using the same prompt token ids;
6. repeat according to `--repeat`;
7. write artifacts under `.artifacts/dflash/benchmarks/...`.

Pass `--only-dflash` for long or expensive runs when you only want the DFlash
leg. Default remains a baseline-vs-DFlash comparison.

This command is for local runtime numbers and regression checks. It is not an
agentic benchmark and not a general accuracy leaderboard. The `aime25` suite
does record exact-answer score fields so speed and score can be inspected in the
same artifact.

## Public Runtime Suites

| Suite | Purpose | Default limit |
| --- | --- | ---: |
| `smoke` | cheap CLI sanity check only, not a performance claim | 1 |
| `humaneval` | `openai_humaneval`, split `test`, field `prompt` | 10 |
| `gsm8k` | `gsm8k`, config `main`, split `test`, formatted as question/answer | 10 |
| `math500` | `HuggingFaceH4/MATH-500`, split `test`, formatted as problem/solution | 10 |
| `aime25` | `math-ai/aime25`, split `test`, exact integer answer scoring | 30 |
| `longctx` | synthetic long-context prompt for prefill/memory/cache stress | 1 |

`humaneval`, `gsm8k`, `math500`, and `aime25` use real Hugging Face datasets
through the optional `datasets` package (`pip install 'dflash-mlx[bench]'`).
First use may download or populate the local HF cache. They are still runtime
prompt suites: they measure speed, memory, acceptance, and cache behavior on
familiar prompt shapes. `aime25` additionally records exact integer score fields
for baseline and DFlash. The other HF suites do not report official HumanEval
pass@1, GSM8K exact match, Math500 accuracy, or any other accuracy score.

`smoke` and `longctx` remain local/offline. `smoke` exists to catch broken CLI,
model loading, tokenizer, and artifact plumbing. Do not use it for published
performance numbers.

## Important Defaults

| Setting | Default |
| --- | --- |
| suite | `smoke` |
| prompt limit | `1` for `smoke`/`longctx`, `30` for `aime25`, `10` otherwise |
| target model | required via `--model` |
| chat template | enabled |
| generated tokens | `64`, or `65536` for `aime25` |
| block tokens | `16` |
| repeat | `1` |
| cooldown | `10` seconds |
| memory summary | enabled |
| MLX wired/cache limits | wired `auto`; cache `4GB` for `longctx`, `auto` otherwise |
| output dir | `.artifacts/dflash/benchmarks/<timestamp>-<suite>-<model>` |

The default `dflash benchmark` invocation uses `smoke`; that is intentionally a
sanity check. For comparable numbers, pass an explicit `--prompt` or a real
runtime suite, plus `--repeat`, `--cooldown`, and `--no-eos` when you need a
fixed generation length.

`--suite longctx --ctx-tokens INT` builds an approximate synthetic long-context
prompt. It is useful for cheap stress testing, but it is not the same as a real
multi-turn coding agent session.

HF suite selection is deterministic by default: no shuffle, first `--limit` rows
from the split. Use `--shuffle --seed INT` to shuffle explicitly before the
limit is applied. The manifest records shuffle, seed, dataset id/config/split,
selected row indices, and selected prompt ids.

## Benchmark Flags

| Flag | Meaning |
| --- | --- |
| `--suite {smoke,humaneval,gsm8k,math500,aime25,longctx}` | named benchmark prompt suite |
| `--limit N` | deterministic prompt count limit |
| `--ctx-tokens N` | synthetic context target for `longctx` |
| `--prompt-file PATH` | JSONL prompt override, rows contain `id`, `suite`, `prompt` |
| `--shuffle` | shuffle HF dataset rows before applying `--limit` |
| `--seed INT` | shuffle seed used only with `--shuffle` |
| `--prompt TEXT` | prompt text |
| `--max-tokens INT` | generation length; default is `65536` for `aime25`, `64` otherwise |
| `--block-tokens INT` | DFlash verify block size |
| `--repeat INT` | measured runs |
| `--cooldown SECONDS` | sleep between baseline/DFlash legs and repeated runs |
| `--sustained-minutes MINUTES` | continuous-load mode: one model load, back-to-back generations for N minutes; reports fresh vs plateau decode tok/s, throttle factor, and cliff onset |
| `--wired-limit auto\|none\|BYTES` | MLX wired memory limit for reproducible memory runs |
| `--cache-limit auto\|none\|BYTES` | MLX allocator cache limit; default is `4GB` for `longctx`, `auto` otherwise |
| `--model REF_OR_PATH` | target model; required |
| `--draft REF_OR_PATH` | draft override |
| `--no-chat-template` | raw prompt text |
| `--draft-quant SPEC` | draft quantization override, e.g. `w4:gs64`; use `none` to disable model defaults |
| `--no-eos` | suppress EOS for fixed-length runs |
| `--only-dflash` | skip the baseline MLX leg and run DFlash only |
| `--prefill-step-size INT` | target prefill chunk size |
| `--target-fa-window INT` | experimental target FA rotating window |
| `--draft-sink-size INT` | draft cache sink tokens |
| `--draft-window-size INT` | draft cache rolling window tokens |
| `--verify-len-cap INT` | max tokens per verify forward |
| `--verify-mode {dflash,adaptive,ddtree,off}` | verifier path mode |
| `--no-memory` | omit memory medians |
| `--out PATH` | artifact directory |

## Artifacts

Each public benchmark run writes:

- `manifest.json` - repo/runtime metadata;
- `invocation.json` - command, model refs, prompt token mode, protocol;
- `results.json` - raw per-prompt metrics plus aggregate metrics, including
  end-of-leg `phys_footprint`, `mlx_active`, and `mlx_cache` memory breakdown
  when available; `aime25` rows include `score` objects with expected answer,
  prediction, and correctness;
- `runs.jsonl` - per-run measurements;
- `summary.json` - aggregate numbers;
- `summary.md` - human-readable report.

Validated large DFlash drafts default to `w4` in memory. When comparing draft
quantization, use separate artifact directories and pass `--draft-quant none`
for the bf16/non-quant draft leg. Compact reports include the effective
`draft_quant` value in `summary.md` and `runs.jsonl` so quantized and
non-quantized runs do not collapse into the same row during review.

The artifact directory is local by default. New raw benchmark outputs should not
be committed.

## Legacy Results

`benchmark/results/*.json` contains pinned historical JSON reports. They are
kept as legacy evidence and are not the default destination for new runs.

When quoting an old result, quote the file path and its recorded git hash. When
quoting a new result, quote the `.artifacts/...` directory.

## Lab Harnesses

`tools/benchmarks/` contains private/lab harnesses:

- agentic trace/session/proxy tooling;
- prefix-cache and L2 probes;
- trace analyzers;
- OpenCode/pi trace harnesses.

These tools are useful for diagnosis, but their outputs are not public claims
unless the run directory records enough context to reproduce the command and
environment.

Rules:

- use an explicit prompt or real runtime suite for performance claims;
- use lab harnesses to answer one specific mechanism question;
- do not compare numbers from different harnesses as if they were one protocol;
- do not use full cycle tracing for performance claims;
- benchmark sequentially, never with two heavy model loads in parallel.

## Good Command Patterns

Canonical fixed-prompt run:

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

HumanEval-style runtime prompts:

```bash
dflash benchmark --suite humaneval --limit 10 --model Qwen/Qwen3.5-4B
```

GSM8K-style runtime prompts:

```bash
dflash benchmark --suite gsm8k --limit 10 --model Qwen/Qwen3.5-4B
```

Math500-style runtime prompts:

```bash
dflash benchmark --suite math500 --limit 10 --model Qwen/Qwen3.5-4B
```

AIME25 speed + score prompts:

```bash
dflash benchmark \
  --suite aime25 \
  --limit 30 \
  --shuffle \
  --seed 42 \
  --model mlx-community/Qwen3.6-27B-4bit \
  --draft z-lab/Qwen3.6-27B-DFlash
```

DFlash-only AIME25 run:

```bash
dflash benchmark \
  --suite aime25 \
  --limit 30 \
  --shuffle \
  --seed 42 \
  --model mlx-community/Qwen3.6-27B-4bit \
  --draft z-lab/Qwen3.6-27B-DFlash \
  --only-dflash
```

Fixed-length decode:

```bash
dflash benchmark \
  --model Qwen/Qwen3.5-4B \
  --prompt "$PROMPT" \
  --max-tokens 128 \
  --no-eos
```

Synthetic context stress:

```bash
dflash benchmark \
  --suite longctx \
  --model Qwen/Qwen3.5-4B \
  --ctx-tokens 65536 \
  --max-tokens 64 \
  --out .artifacts/dflash/benchmarks/manual-64k
```

High-context memory sweep with bounded MLX cache:

```bash
dflash benchmark \
  --suite longctx \
  --model Qwen/Qwen3.5-9B \
  --ctx-tokens 32768 \
  --max-tokens 64 \
  --cache-limit 4GB \
  --out .artifacts/dflash/benchmarks/manual-32k-cache4gb
```

External prompt file:

```bash
dflash benchmark \
  --suite gsm8k \
  --prompt-file data/gsm8k_sample.jsonl \
  --limit 100 \
  --model Qwen/Qwen3.5-4B
```

`--prompt-file` is the offline/custom path and bypasses Hugging Face dataset
loading even when `--suite` is `humaneval`, `gsm8k`, `math500`, or `aime25`.

Low-memory runtime check:

```bash
dflash benchmark \
  --model Qwen/Qwen3.5-4B \
  --prompt "$PROMPT" \
  --draft-sink-size 64 \
  --draft-window-size 1024 \
  --verify-len-cap 0 \
  --target-fa-window 0
```

## Reading Results

Look at:

- baseline tokens/sec;
- DFlash tokens/sec;
- speedup versus baseline;
- TTFT;
- acceptance rate;
- tokens per cycle;
- peak memory if enabled;
- prompt token count and whether chat template was enabled.

Do not interpret DFlash speed without acceptance and tokenization regime. A raw
prompt and a chat-template prompt are different benchmark inputs.
