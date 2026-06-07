# oMLX Integration Benchmarks on M1 Max

This report documents DFlash performance through the
[oMLX](https://github.com/jundot/omlx) integration on an earlier Apple Silicon
generation than the M5 Max results in the main README.

These are integration measurements, not canonical `dflash benchmark` artifacts.
The request path, prompt construction, and cache orchestration came from oMLX.
Use the main README results when comparing the standalone dflash-mlx CLI.

## Environment

| Item | Value |
|---|---|
| Hardware | MacBook Pro, Apple M1 Max |
| Unified memory | 64 GB |
| Operating system | macOS 26.5 |
| oMLX revision | `21d5b1e707c59d2da744787e9fe2a04f8a0363d3` |
| dflash-mlx revision | `ff14fc80acaad6f3d4360a0fb675d524e5042e5c` |
| Merged dflash-mlx equivalent | PR #37, merge commit `b7f192b62bc5a59cad41fda888c1118c60fc58b1` |
| Thinking | Disabled |
| Request concurrency | One |

Each target used oQ6 quantized matrix weights. Floating target tensors such as
scales, norms, and biases were FP16. DFlash drafter weights remained BF16. The
drafter was bundled with its matching target. Native MTP tensors were removed
from the dedicated Qwen DFlash bundles so only one speculative path was active.

Performance requests generated 256 tokens after a warmup request. Cache probes
generated 128 tokens with deterministic sampling.

## Generation and Cold Prefill

`gen` and `pre` are tokens per second. TTFT is milliseconds. Prompt lengths are
token targets produced by the same benchmark harness for every model.

| Family | gen/1k | gen/4k | gen/8k | gen/16k | pre/1k | pre/4k | TTFT/1k | TTFT/4k | TTFT/8k | TTFT/16k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen 3.6 27B | 53.1 | 51.7 | 50.8 | 49.1 | 134.9 | 137.3 | 7,588 | 29,835 | 61,756 | 130,143 |
| Qwen 3.6 35B-A3B | 190.1 | 187.8 | 180.2 | 174.3 | 723.9 | 931.1 | 1,415 | 4,399 | 8,970 | 19,077 |
| Gemma 4 31B | 42.9 | 41.2 | 38.9 | 35.2 | 110.6 | 108.0 | 9,261 | 37,942 | 77,653 | 164,754 |
| Gemma 4 26B-A4B | 141.0 | 136.1 | 137.8 | 134.4 | 691.5 | 745.8 | 1,481 | 5,492 | 11,163 | 23,497 |

The strongest controlled decode comparisons were:

| Target artifact | Control | DFlash | Relative throughput |
|---|---:|---:|---:|
| Qwen 3.6 27B FP16-oQ6 | 17.1 tok/s native MTP | 52.5 tok/s | 3.07x |
| Gemma 4 31B BF16-oQ6 (earlier run) | 9.4 tok/s target-only | 29.2 tok/s at 93.8% acceptance | 3.11x |

The Qwen comparison used the same FP16-oQ6 target weights for both rows. The
control used native MTP heads; the DFlash row used the BF16
`z-lab/Qwen3.6-27B-DFlash` drafter.

The Gemma control was a separate, earlier matched run using the original
BF16-oQ6 target for both legs. It should not be compared directly with the
42.9 tok/s result above: that later result uses the converted FP16-oQ6 target.
No matched target-only measurement was collected for the FP16 Gemma bundle, so
the report does not claim a speedup ratio for its 42.9 tok/s result.

## Prefix Cache Restoration

The tables report user-visible TTFT in milliseconds. `L1 exact` repeats the
prompt against the in-memory snapshot. `L2 exact` restarts the server and
restores from the SSD snapshot.

### Qwen 3.6 27B

| Context | Cold | L1 exact | L2 exact | Restored / computed |
|---:|---:|---:|---:|---:|
| 8,192 | 63,298 | 494 | 3,738 | 8,185 / 7 |
| 32,768 | 273,439 | 675 | 5,026 | 32,761 / 7 |
| 65,536 | 642,125 | 919 | 5,905 | 65,529 / 7 |

### Qwen 3.6 35B-A3B

| Context | Cold | L1 exact | L2 exact | Restored / computed |
|---:|---:|---:|---:|---:|
| 8,192 | 12,402 | 266 | 5,341 | 8,185 / 7 |
| 32,768 | 46,721 | 389 | 5,745 | 32,761 / 7 |
| 65,536 | 130,792 | 576 | 6,102 | 65,529 / 7 |

### Gemma 4 31B

| Context | Cold | L1 exact | L2 exact | Restored / computed |
|---:|---:|---:|---:|---:|
| 8,192 | 83,024 | 581 | 6,465 | 8,188 / 4 |
| 32,768 | 351,475 | 707 | 7,661 | 32,764 / 4 |
| 65,536 | 802,474 | 947 | 8,441 | 65,532 / 4 |

### Gemma 4 26B-A4B

| Context | Cold | L1 exact | L2 exact | Restored / computed |
|---:|---:|---:|---:|---:|
| 8,192 | 14,227 | 293 | 3,764 | 8,188 / 4 |
| 32,768 | 54,507 | 457 | 4,135 | 32,764 / 4 |
| 65,536 | 130,694 | 638 | 5,017 | 65,532 / 4 |

For every 8K, 32K, and 64K probe, the restored response SHA-256 matched its
deterministic cold control. The FP16 Gemma 31B 64K runs measured approximately
45.1-52.3 GB peak DFlash memory depending on cache phase; exact L1 and L2
restores were approximately 45.6 GB.

Cache settings:

```text
L1 maximum entries: 4
L1 byte budget: 8 GiB
maximum snapshot tokens: 65,536
L2 byte budget: 20 GiB
L2 frontier stride: 32,768 tokens
```

## Accuracy Guardrail

Accuracy used the same DFlash bundles with cache disabled and thinking off.
These samples are included to detect obvious regressions, not to define an
official benchmark leaderboard.

| Family | MMLU-Pro | TruthfulQA | ARC Challenge | GSM8K | HumanEval | LiveCodeBench |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 3.6 27B | 69.0% | 88.0% | 95.3% | 93.0% | 92.1% | 60.0% |
| Qwen 3.6 35B-A3B | 60.0% | 86.5% | 95.3% | 92.0% | 71.3% | 53.0% |
| Gemma 4 31B | 79.7% | 89.6% | 96.7% | 95.0% | 95.1% | 70.0% |
| Gemma 4 26B-A4B | 75.3% | 85.7% | 95.7% | 93.0% | 95.7% | 58.0% |

Sample counts were 300 MMLU-Pro, 817 TruthfulQA, 300 ARC Challenge, 100 GSM8K,
164 HumanEval, and 100 LiveCodeBench tasks.

## Interpretation

- DFlash provided roughly 3.1x decode throughput in two matched controls: the
  current Qwen FP16 artifact and an earlier Gemma BF16 artifact.
- Cold TTFT remains dominated by target prefill and scales with model
  architecture and context length.
- Exact L1 restoration reduced 64K TTFT to less than one second on all four
  families.
- Exact L2 restoration reduced 64K TTFT to approximately 5-8 seconds after a
  server restart.
- FP16 in this report describes the target's floating tensors; it does not mean
  the oQ6 matrices or BF16 drafter were converted to full FP16.

Results are specific to these artifacts, software revisions, benchmark
settings, and the tested M1 Max system.
