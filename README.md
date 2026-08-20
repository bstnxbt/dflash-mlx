<p align="center">
  <h1 align="center">dflash-mlx</h1>
  <p align="center">DFlash speculative decoding for Apple Silicon (MLX)</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License">
  <img src="https://img.shields.io/badge/MLX-stock-red" alt="Stock MLX">
</p>

Paper: [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036) (Chen et al., 2026)

Block-diffusion draft generates 16 tokens in one pass. Target verifies in one pass. Output is lossless — every emitted token is verified against the target model before it is committed.

https://github.com/user-attachments/assets/a9be2b48-3264-4970-b836-c876b0b7fdda

## How it works

- A small draft model (~1B params) generates 16 tokens in parallel with block diffusion.
- The target model verifies those 16 tokens in a single forward pass.
- Greedy acceptance keeps the correct prefix and rejects the rest.
- Lossless: every emitted token is the target model's greedy argmax at verification time. Output can still differ from pure AR because of MLX dispatch divergence, but no unverified token is ever emitted.
- Built on stock MLX with a small number of targeted Metal kernels where rollback and long-context verify need tighter numerical control.

## Technical details

- **Tape-replay rollback** — instead of snapshotting and restoring the full GatedDeltaNet state, dflash-mlx records an innovation tape during verify and replays only the accepted steps through a custom Metal kernel. Keeps rollback cost low and preserves acceptance over long generations.
- **Target-owned long-context attention routing** — Qwen and Gemma adapters route verify blocks through the appropriate MLX or GQA-reshape SDPA path internally, keeping the public CLI free of attention-kernel switches.
- **Verify-specialized int4 qmm** (`verify_qmm`) — custom Metal kernels for the small-M quantized matmuls that dominate the target verify step. M5-class Apple GPUs (`applegpu_g17*`) use the Metal 4 NAX M=16 path automatically when the shape is supported; older Apple GPUs keep the steel simdgroup-MMA fallbacks. Auto-enabled on MoE targets and dense models with ≥40 layers.
- **Numerical coherence** — bf16-sensitive paths, including recurrent state replay and small projections, are stabilized across speculative cycles so accepted tokens stay consistent.
- **Prefix cache (L1+L2)** — RAM snapshots of target KV + GDN recurrent state + captured hidden + last logits, with optional SSD spill, byte/entry budgets, and automatic eviction. Hits skip prefill on revisited prompts. This hot/cold cache hierarchy is inspired by [oMLX](https://github.com/jundot/omlx)'s tiered KV cache work, but dflash-mlx stores DFlash prefix snapshots rather than active paged-KV blocks.
- **Positional sparse prefill** (`prompt_token_positions`) — `stream_dflash_generate` accepts an optional list of original positions alongside a compacted `prompt_tokens_override`, so a caller (e.g. oMLX specprefill) can prefill the target on a selected token subset placed at its true positions. Full-attention layers RoPE each token at its original position (contiguous runs delegate to the native rope and stay bitwise-exact); GDN layers run over the compacted sequence; gemma4 sliding-window layers rebuild their window mask from the true positions. Mutually exclusive with the prefix cache. Use chunk-based selection — incoherent (e.g. every-other-token) selection shreds local structure and collapses draft acceptance.

## Benchmarks

Apple M5 Max, 64 GB unified memory, MLX 0.31.1. Protocol: stock `mlx_lm.stream_generate` baseline vs DFlash, sequential, 3 repeats, median, 60s cooldown. Generation prompt: `"The function $f$ satisfies the functional equation \[ f(x) + f(y) = f(x + y) - xy - 1 \] for all real numbers $x$ and $y$. If $f(1) = 1$, then find all integers $n$ such that $f(n) = n$. Enter all such integers, separated by commas. Please reason step by step, and put your final answer within \boxed{}."`

| Model | Tokens | Baseline | DFlash | Speedup | Acceptance |
|-------|--------|----------|--------|---------|------------|
| Qwen3.5-4B | 1024 | 53.80 tok/s | 182.87 tok/s | 3.40x | 86.43% |
| Qwen3.5-4B | 2048 | 53.90 tok/s | 188.70 tok/s | 3.49x | 87.70% |
| Qwen3.5-4B | 4096 | 53.49 tok/s | 195.84 tok/s | 3.66x | 88.35% |
| Qwen3.5-4B | 8192 | 53.28 tok/s | 160.51 tok/s | 3.02x | 87.30% |
| Qwen3.5-9B | 1024 | 30.95 tok/s | 135.34 tok/s | 4.37x | 89.55% |
| Qwen3.5-9B | 2048 | 30.70 tok/s | 113.00 tok/s | 3.65x | 89.16% |
| Qwen3.5-9B | 4096 | 30.56 tok/s | 94.59 tok/s | 3.06x | 88.31% |
| Qwen3.5-9B | 8192 | 29.43 tok/s | 66.94 tok/s | 2.22x | 86.67% |
| Qwen3.5-27B-4bit | 1024 | 33.55 tok/s | 79.02 tok/s | 2.37x | 90.04% |
| Qwen3.5-27B-4bit | 2048 | 33.10 tok/s | 70.21 tok/s | 2.12x | 89.60% |
| Qwen3.5-27B-4bit | 4096 | 31.47 tok/s | 55.68 tok/s | 1.77x | 88.38% |
| Qwen3.5-27B-4bit | 8192 | 33.88 tok/s | 45.29 tok/s | 1.34x | 85.97% |
| Qwen3.6-27B-4bit | 1024 | 33.26 tok/s | 98.05 tok/s | 2.95x | 84.67% |
| Qwen3.6-27B-4bit | 2048 | 32.34 tok/s | 90.67 tok/s | 2.81x | 84.62% |
| Qwen3.6-27B-4bit | 4096 | 30.58 tok/s | 93.55 tok/s | 3.06x | 87.04% |
| Qwen3.6-27B-4bit | 8192 | 26.03 tok/s | 79.12 tok/s | 3.04x | 83.45% |
| Qwen3.6-27B-4bit | 16384 | 21.50 tok/s | 60.77 tok/s | 2.78x | 84.40% |
| Qwen3.5-35B-A3B-4bit | 1024 | 143.03 tok/s | 248.85 tok/s | 1.76x | 89.26% |
| Qwen3.5-35B-A3B-4bit | 2048 | 141.43 tok/s | 255.01 tok/s | 1.81x | 89.75% |
| Qwen3.5-35B-A3B-4bit | 4096 | 141.49 tok/s | 216.47 tok/s | 1.53x | 88.50% |
| Qwen3.5-35B-A3B-4bit | 8192 | 138.59 tok/s | 170.39 tok/s | 1.22x | 86.41% |
| Qwen3.6-35B-A3B-4bit | 1024 | 138.26 tok/s | 300.33 tok/s | 2.20x | 91.02% |
| Qwen3.6-35B-A3B-4bit | 2048 | 139.03 tok/s | 252.93 tok/s | 1.82x | 89.60% |
| Qwen3.6-35B-A3B-4bit | 4096 | 134.50 tok/s | 208.40 tok/s | 1.56x | 88.43% |
| Qwen3.6-35B-A3B-4bit | 8192 | 133.20 tok/s | 177.45 tok/s | 1.33x | 87.01% |

Per-run JSON: [`benchmark/results/`](benchmark/results/). Reproduce on your hardware with `dflash benchmark`.

## Install

```bash
pip install dflash-mlx
```

Optional benchmark dataset support:

```bash
pip install "dflash-mlx[bench]"
```

## Quick start

```bash
PROMPT='The function $f$ satisfies the functional equation \[ f(x) + f(y) = f(x + y) - xy - 1 \] for all real numbers $x$ and $y$. If $f(1) = 1$, then find all integers $n$ such that $f(n) = n$. Enter all such integers, separated by commas. Please reason step by step, and put your final answer within \boxed{}.'

# One-shot generation, draft auto-resolved
dflash generate --model Qwen/Qwen3.5-9B --prompt "$PROMPT"

# Server (OpenAI-compatible)
dflash serve \
  --model mlx-community/Qwen3.6-27B-4bit \
  --draft z-lab/Qwen3.6-27B-DFlash \
  --port 8000

# Canonical local benchmark
dflash benchmark \
  --model Qwen/Qwen3.5-9B \
  --prompt "$PROMPT" \
  --max-tokens 1024 \
  --repeat 3 \
  --cooldown 60 \
  --no-eos

# AIME25 dataset run, with baseline-vs-DFlash speed and exact-answer score
dflash benchmark \
  --suite aime25 \
  --limit 30 \
  --shuffle \
  --seed 42 \
  --model mlx-community/Qwen3.6-27B-4bit \
  --draft z-lab/Qwen3.6-27B-DFlash
```

Send a request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"mlx-community/Qwen3.6-27B-4bit\",
    \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
    \"max_tokens\": 1024,
    \"stream\": true
  }"
```

Compatible with OpenCode, aider, Continue, Open WebUI, LM Studio through its
OpenAI-compatible adapter, and any other OpenAI-compatible client. Chat
Completions tool calls stream as OpenAI
`delta.tool_calls` for Qwen3-Coder XML, Gemma4, and JSON tool-call payloads
inside model tool spans; malformed or undeclared tool calls fail at the server
boundary instead of leaking raw XML/JSON as assistant content. Chat Completions
accepts `tool_choice: "auto"` and `tool_choice: "none"`; function-specific
`tool_choice` and `parallel_tool_calls: false` are rejected because the server
does not implement serial tool enforcement. Greedy requests use DFlash by
default; pass a positive `--fastpath-max-tokens` value to opt into the
target-only short-response fast path.

`POST /v1/responses` is available as a minimal non-streaming compatibility
adapter for text input and function-call tools. Streaming Responses,
multimodal input, reasoning/text/truncation controls, `tool_choice`,
`parallel_tool_calls`, and persistent `previous_response_id` / `store`
behavior are not implemented. Greedy requests use DFlash; requests that need
sampling, logits processing, or log probabilities use exact target-only AR so
their OpenAI-compatible generation parameters are honored.

Inspect live server metrics:

```bash
curl http://127.0.0.1:8000/metrics
```

`prefill_tok_s_physical` counts only tokens actually computed after prefix-cache
restore. `prefill_tok_s_restored` counts restored prefix tokens over the same
wall time. `prefill_tok_s_apparent` uses the full logical prompt length over
the same user-visible prefill wall time. `rates.average_decode_tok_s` is the
weighted decode-only average since server startup: total generated tokens
divided by cumulative decode seconds. `current_request` shows an in-flight
prefill/decode, `recent_requests` keeps the last 32 completed requests, and
`cache_status` is `WARM` when a request restored prefix tokens and `COLD`
otherwise. In-flight and completed DFlash requests also report
`tokens_per_cycle`, `cycles`, adaptive block counters, and CopySpec counters so
long-context sessions show when speculative progress collapses. `rss_gb`
reports process resident memory. `wired_gb` stays `null` unless a true
per-process wired-memory source is available.
The endpoint is for live debugging and benchmark visibility; it does not create
benchmark artifacts.

Chat-template thinking follows the tokenizer default. For Qwen thinking models,
that means the thinking template path is enabled unless a request or CLI
template override explicitly disables it:

```bash
dflash serve --model mlx-community/Qwen3.6-27B-4bit --chat-template-args '{"enable_thinking":false}'
```

## Tested models

Optimized for Qwen3.5 / Qwen3.6 hybrid GatedDeltaNet + attention targets. Qwen3
(pure attention) targets work but skip the tape-replay rollback path. Gemma4
targets use the Gemma4 adapter. Prefix snapshots are enabled only for Gemma4
configs with known non-shared KV (`num_kv_shared_layers == 0`); shared-KV or
unknown configs fail closed. Local `dflash serve` diagnostics have verified
Gemma4 31B and 26B-A4B exact repeated-prompt restore and long-chat continuation
restore, but those diagnostics are cache-latency evidence, not public benchmark
throughput claims.

Validated large DFlash drafts default to `w4` in memory. Current Qwen3.5,
Qwen3.6, and Gemma4 probes showed this is the best practical memory/throughput
tradeoff; pass `--draft-quant none` when you need a bf16/non-quant draft A/B.

For Gemma4 long-context memory pressure, set `--prefill-step-size 1024`
explicitly. For Gemma4 31B under tighter long-context memory limits,
`--prefill-step-size 512` reduced peak memory and TTFT in local `dflash serve`
probes, but it is prompt-sensitive and not a public benchmark throughput claim.
The default is `2048`.

`dflash serve` uses the product session policy by default: prefix cache
enabled, L2 snapshots enabled, boundary cache clears enabled, and a `4GB` MLX
cache limit. Pass explicit flags such as `--no-prefix-cache-l2`,
`--no-clear-cache-boundaries`, or `--cache-limit auto` only when you want to
override that policy.

| Target | Draft |
|--------|-------|
| [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | [z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash) |
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | [z-lab/Qwen3.5-9B-DFlash](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash) |
| [mlx-community/Qwen3.5-27B-4bit](https://huggingface.co/mlx-community/Qwen3.5-27B-4bit) | [z-lab/Qwen3.5-27B-DFlash](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash) |
| [mlx-community/Qwen3.5-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit) | [z-lab/Qwen3.5-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash) |
| [mlx-community/Qwen3.6-27B-4bit](https://huggingface.co/mlx-community/Qwen3.6-27B-4bit) | [z-lab/Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) |
| [mlx-community/Qwen3.6-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit) | [z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash) |
| [mlx-community/Qwen3.8-27B-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) | [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) |
| [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | [z-lab/Qwen3-4B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16) |
| [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | [z-lab/Qwen3-8B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16) |
| [mlx-community/gemma-4-31b-it-4bit](https://huggingface.co/mlx-community/gemma-4-31b-it-4bit) | [z-lab/gemma-4-31B-it-DFlash](https://huggingface.co/z-lab/gemma-4-31B-it-DFlash) |
| [mlx-community/gemma-4-26b-a4b-it-4bit](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit) | [z-lab/gemma-4-26B-A4B-it-DFlash](https://huggingface.co/z-lab/gemma-4-26B-A4B-it-DFlash) |

```bash
dflash models
```

Models without a matching DFlash draft are rejected. Pass `--draft` explicitly to override the registry.

## CLI

```
dflash serve      # OpenAI-compatible server
dflash generate   # one-shot local generation
dflash benchmark  # baseline-vs-DFlash runtime benchmark
dflash doctor     # environment and config checks
dflash models     # list supported target/draft pairs
```

## Common server controls

```bash
# Opt into target-only AR for very short responses
dflash serve --model Qwen/Qwen3.5-9B --fastpath-max-tokens 64

# Tune prefill batching
dflash serve --model Qwen/Qwen3.5-9B --prefill-step-size 8192

# Diagnostics
dflash serve --model Qwen/Qwen3.5-9B --diagnostics basic   # request + cache events
dflash serve --model Qwen/Qwen3.5-9B --diagnostics full    # + memory waterfall + cycle timings

# Bound L1 prefix snapshots
dflash serve --model Qwen/Qwen3.5-9B \
  --prefix-cache-max-entries 2 \
  --prefix-cache-max-bytes 2GB

# Enable SSD L2 spill
dflash serve --model Qwen/Qwen3.5-9B \
  --prefix-cache-l2 \
  --prefix-cache-l2-dir .artifacts/dflash/l2 \
  --prefix-cache-l2-max-bytes 50GB
```

Diagnostics artifacts land in `.artifacts/dflash/diagnostics/<timestamp>-serve-<mode>/`. `basic` writes request and cache events; `full` adds the memory waterfall and per-cycle timings. Use `full` for diagnosis, not for throughput claims.

## Features

- **Auto draft resolution** — no manual `--draft` flag needed for registered targets
- **Streaming** — token-by-token output (CLI + SSE)
- **Chat templates** — enabled by default
- **Recurrent rollback** — `RecurrentRollbackCache` keeps GatedDeltaNet state coherent across speculative verify and rollback
- **Verify-specialized int4 qmm** — custom M=16 Metal kernel auto-enabled on MoE and dense ≥40-layer targets; falls back to stock `mx.quantized_matmul` everywhere else
- **Adaptive verify policy** — default verify mode adjusts the verify block from observed acceptance and long-context pressure; fixed DFlash verification is still available with `--verify-mode dflash`
- **Prefix cache L1+L2** — RAM snapshots with optional SSD spill, budget-based eviction, and hybrid-architecture support
- **Diagnostics** — opt-in structured artifacts under `.artifacts/dflash/diagnostics/`

## Roadmap

- **More architecture backends** — add new target families only with
  family-specific cache layout, attention masks, logits post-processing, hidden
  capture, rollback/trim behavior, and parity tests.
- **Kernel work where it matters** — optimize family-specific hot paths only
  after the backend contract and parity tests are stable.
- **Tool-call regime auto-fallback** — switch to target-only AR when speculative surplus goes negative on structured outputs
- **Sustained acceptance at long context** — draft KV cache window scaling and long-context verify optimization

## Citation

```bibtex
@misc{chen2026dflash,
  title={DFlash: Block Diffusion for Flash Speculative Decoding},
  author={Jian Chen and Yesheng Liang and Zhijian Liu},
  year={2026},
  eprint={2602.06036},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2602.06036}
}
```

## License

Apache-2.0
