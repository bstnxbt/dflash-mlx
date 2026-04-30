<p align="center">
  <h1 align="center">dflash-mlx</h1>
  <p align="center">DFlash speculative decoding for Apple Silicon</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
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
- Uses stock MLX plus a small number of targeted kernels where rollback and long-context verify need tighter numerical control.

## Technical details

- **Tape-replay rollback**: instead of snapshotting and restoring the full GatedDeltaNet state, dflash-mlx records an innovation tape during verify and replays only the accepted steps through a custom Metal kernel. Keeps rollback cost low and preserves acceptance over long generations.
- **JIT SDPA 2-pass**: long-context verify (`N >= 1024`) uses a custom Metal attention kernel that stays numerically aligned with stock MLX attention.
- **Verify-specialized int4 qmm** (`verify_qmm`): custom Metal simdgroup-MMA kernel for the M=16 quantized matmul that dominates the target verify step. Two shape-adaptive variants (`mma2big`, `mma2big_pipe` with K-split + double-buffered staging). Auto-enabled on MoE targets and dense models with ≥40 layers where the M=16 specialization amortizes over enough layers to beat stock `mx.quantized_matmul` end-to-end. Use `--verify-mode off` only for debug/parity runs.
- **Numerical coherence**: bf16-sensitive paths, including recurrent state replay and small projections, are stabilized across speculative cycles so accepted tokens stay consistent.

## Benchmarks

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

### Methodology

Hardware: Apple M5 Max, 64GB unified memory. MLX 0.31.1 from the stock pip install.

Protocol: stock `mlx_lm.stream_generate` on a pristine target model vs stock MLX plus the local DFlash runtime, measured sequentially. `3` repeats, median reported, `60s` cooldown between measured runs.

Generation: prompt `"The function $f$ satisfies the functional equation \[ f(x) + f(y) = f(x + y) - xy - 1 \] for all real numbers $x$ and $y$. If $f(1) = 1$, then find all integers $n$ such that $f(n) = n$. Enter all such integers, separated by commas. Please reason step by step, and put your final answer within \boxed{}."` with chat templates enabled by default and post-prefill tok/s as the primary metric.

## Install

```bash
pip install dflash-mlx

# or with pipx
pipx install dflash-mlx
```

`dflash serve` wraps `mlx_lm.server` for full OpenAI-compatible serving semantics, including tools, reasoning, and streaming, while using the DFlash runtime as the generation engine.

## Quick start

```bash
PROMPT='The function $f$ satisfies the functional equation \[ f(x) + f(y) = f(x + y) - xy - 1 \] for all real numbers $x$ and $y$. If $f(1) = 1$, then find all integers $n$ such that $f(n) = n$. Enter all such integers, separated by commas. Please reason step by step, and put your final answer within \boxed{}.'

# Generate — draft auto-resolved
dflash generate --model Qwen/Qwen3.5-9B --prompt "$PROMPT"

# Explicit draft model
dflash generate --model Qwen/Qwen3.5-9B --draft z-lab/Qwen3.5-9B-DFlash --prompt "$PROMPT"

# Server
dflash serve --model Qwen/Qwen3.5-9B --port 8000

# Raise the DFlash fallback threshold for longer prompts
dflash serve --model mlx-community/Qwen3.5-35B-A3B-4bit --port 8000 \
  --dflash-max-ctx 16384

# Benchmark
dflash benchmark --model Qwen/Qwen3.5-9B --draft z-lab/Qwen3.5-9B-DFlash \
  --prompt "$PROMPT" --max-tokens 1024 --repeat 3

# Live demo — baseline vs DFlash side-by-side
PYTHONPATH=. python3 -m examples.demo --mode dflash \
  --target-model Qwen/Qwen3.5-9B --draft-model z-lab/Qwen3.5-9B-DFlash \
  --prompt "$PROMPT" --max-tokens 2048
```

- Compatible with Open WebUI, Continue, OpenCode, aider, and other OpenAI-compatible clients
- Streaming SSE support
- `dflash serve` requires a supported DFlash draft model (auto-detected from the registry or passed explicitly with `--draft`)

## Tested models

Any model with a DFlash draft on HuggingFace should work.

Optimized for Qwen3.5 / Qwen3.6 models (hybrid GatedDeltaNet + attention architecture). Qwen3 (pure attention) models work, but without the precision benefits of tape-replay rollback.

| Target | Draft |
|--------|-------|
| [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | [z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash) |
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | [z-lab/Qwen3.5-9B-DFlash](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash) |
| [mlx-community/Qwen3.5-27B-4bit](https://huggingface.co/mlx-community/Qwen3.5-27B-4bit) | [z-lab/Qwen3.5-27B-DFlash](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash) |
| [mlx-community/Qwen3.5-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit) | [z-lab/Qwen3.5-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash) |
| [mlx-community/Qwen3.6-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit) | [z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash) |

Models without a matching DFlash draft are rejected. Pass `--draft` explicitly if you want to override the registry.

## Features

- **Auto draft resolution** — no manual `--draft` flag needed
- **Streaming** — token-by-token output in the CLI and server
- **Chat templates** — enabled by default
- **Recurrent rollback** — `RecurrentRollbackCache` keeps GatedDeltaNet state coherent across speculative verify and rollback
- **Verify-specialized int4 qmm kernel** — custom M=16 Metal kernel auto-enabled on MoE and dense ≥40-layer targets; falls back to stock `mx.quantized_matmul` everywhere else

## Roadmap

- Sustained acceptance at 4096+ tokens — draft KV cache window scaling and long-context verify optimization
- Draft model distillation and compression for faster draft forward

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

MIT
