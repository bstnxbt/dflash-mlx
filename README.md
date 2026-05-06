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
- **JIT SDPA 2-pass** — long-context verify (`N >= 1024`) uses a custom Metal attention kernel that stays numerically aligned with stock MLX attention.
- **Verify-specialized int4 qmm** (`verify_qmm`) — custom Metal simdgroup-MMA kernel for the M=16 quantized matmul that dominates the target verify step. Two shape-adaptive variants (`mma2big`, `mma2big_pipe` with K-split + double-buffered staging). Auto-enabled on MoE targets and dense models with ≥40 layers.
- **Numerical coherence** — bf16-sensitive paths, including recurrent state replay and small projections, are stabilized across speculative cycles so accepted tokens stay consistent.
- **Prefix cache (L1+L2)** — RAM snapshots of target KV + GDN recurrent state + captured hidden + last logits, with optional SSD spill, byte/entry budgets, and automatic eviction. Hits skip prefill on revisited prompts. This hot/cold cache hierarchy is inspired by [oMLX](https://github.com/jundot/omlx)'s tiered KV cache work, but dflash-mlx stores DFlash prefix snapshots rather than active paged-KV blocks.

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
| Qwen3.5-35B-A3B-4bit | 1024 | 143.03 tok/s | 248.85 tok/s | 1.76x | 89.26% |
| Qwen3.5-35B-A3B-4bit | 2048 | 141.43 tok/s | 255.01 tok/s | 1.81x | 89.75% |
| Qwen3.5-35B-A3B-4bit | 4096 | 141.49 tok/s | 216.47 tok/s | 1.53x | 88.50% |
| Qwen3.5-35B-A3B-4bit | 8192 | 138.59 tok/s | 170.39 tok/s | 1.22x | 86.41% |
| Qwen3.6-35B-A3B-4bit | 1024 | 138.26 tok/s | 300.33 tok/s | 2.20x | 91.02% |
| Qwen3.6-35B-A3B-4bit | 2048 | 139.03 tok/s | 252.93 tok/s | 1.82x | 89.60% |
| Qwen3.6-35B-A3B-4bit | 4096 | 134.50 tok/s | 208.40 tok/s | 1.56x | 88.43% |
| Qwen3.6-35B-A3B-4bit | 8192 | 133.20 tok/s | 177.45 tok/s | 1.33x | 87.01% |
| LFM2.5-1.2B-Instruct\* | 1024 | 141.94 tok/s | 338.98 tok/s | 2.39x | 86.82% |
| LFM2.5-1.2B-Instruct\* | 2048 | 141.03 tok/s | 209.59 tok/s | 1.49x | 78.76% |

\* Measured on Apple M4 Max, 36 GB unified memory, MLX 0.31.2. The bundled `nathanrchn/LFM2.5-1.2B-Instruct-DFlash` draft was trained on 2k-token sequences, so acceptance and speedup degrade past that horizon; 4k/8k rows are omitted.

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

Compatible with OpenCode, aider, Continue, Open WebUI, and any OpenAI-compatible client. Tool calls, streaming, and chat templates all flow through. Short responses may take the target-only fast path; pass `--fastpath-max-tokens 0` to force DFlash on every request.

Inspect live server metrics:

```bash
curl http://127.0.0.1:8000/metrics
```

`prefill_tok_s_physical` counts only tokens actually computed after prefix-cache
restore. `prefill_tok_s_apparent` uses the full logical prompt length over the
same user-visible prefill wall time. `current_request` shows an in-flight
prefill/decode, `recent_requests` keeps the last 32 completed requests, and
`rss_gb` reports process resident memory. `wired_gb` stays `null` unless a true
per-process wired-memory source is available.
The endpoint is for live debugging and benchmark visibility; it does not create
benchmark artifacts.

Enable Qwen reasoning mode when needed:

```bash
dflash serve --model mlx-community/Qwen3.6-27B-4bit --enable-thinking
```

## Tested models

Optimized for Qwen3.5 / Qwen3.6 hybrid GatedDeltaNet + attention targets. Qwen3
(pure attention) targets work but skip the tape-replay rollback path. Gemma4
targets use the Gemma4 adapter; prefix snapshots stay disabled for Gemma4 until
snapshot parity is proven.

| Target | Draft |
|--------|-------|
| [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | [z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash) |
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | [z-lab/Qwen3.5-9B-DFlash](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash) |
| [mlx-community/Qwen3.5-27B-4bit](https://huggingface.co/mlx-community/Qwen3.5-27B-4bit) | [z-lab/Qwen3.5-27B-DFlash](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash) |
| [mlx-community/Qwen3.5-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit) | [z-lab/Qwen3.5-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash) |
| [mlx-community/Qwen3.6-27B-4bit](https://huggingface.co/mlx-community/Qwen3.6-27B-4bit) | [z-lab/Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) |
| [mlx-community/Qwen3.6-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit) | [z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash) |
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
dflash profiles   # list runtime presets
dflash models     # list supported target/draft pairs
```

## Profiles

Readable defaults. Explicit CLI flags override them.

| Profile | Prefill | Prefix cache | L1 budget | L2 | Intent |
|---|---:|---|---|---|---|
| `balanced` | 4096 | on | 4 / 8 GiB | off | default coding sessions |
| `fast` | 8192 | on | 4 / 16 GiB | off | throughput first |
| `low-memory` | 1024 | on | 2 / 2 GiB | off | lower memory pressure |
| `long-session` | 4096 | on | 8 / 8 GiB | on / 50 GiB | prefix revisits |

```bash
dflash profiles
dflash serve --profile fast --model Qwen/Qwen3.5-9B
dflash serve --profile long-session --model mlx-community/Qwen3.6-27B-4bit \
  --prefix-cache-l2-dir .artifacts/dflash/l2
```

## Common server controls

```bash
# Force DFlash even for short responses
dflash serve --model Qwen/Qwen3.5-9B --fastpath-max-tokens 0

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
- **Prefix cache L1+L2** — RAM snapshots with optional SSD spill, budget-based eviction, and hybrid-architecture support
- **Diagnostics** — opt-in structured artifacts under `.artifacts/dflash/diagnostics/`

## Roadmap

- **Adaptive block size** — vary draft block length per cycle based on observed acceptance regime instead of a fixed 16
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
