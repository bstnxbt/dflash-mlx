## 2026-04-16 — Iteration 1

- Attempted: fixed the three quartet-GQA P0 regressions in [dflash_mlx/kernels.py](/Users/bastienbouge/Documents/dev/dflash-mlx-quartet-gqa/dflash_mlx/kernels.py:575), re-enabled the runtime hook path behind an instance flag in [dflash_mlx/runtime.py](/Users/bastienbouge/Documents/dev/dflash-mlx-quartet-gqa/dflash_mlx/runtime.py:591), and added an isolated microbench in [benchmark/quartet_gqa_microbench.py](/Users/bastienbouge/Documents/dev/dflash-mlx-quartet-gqa/benchmark/quartet_gqa_microbench.py:1).
- Observed: `python3 -m benchmark.quartet_gqa_validate` now passes on `N={256,1024,4096,8192}` with `allclose=True`, `fallback_m1=True`, and `fallback_fp16=True`.
- Observed: isolated microbench (`python3 -m benchmark.quartet_gqa_microbench`) reports stock/quartet ratios of `1.1865` at `N=1024`, `1.1422` at `N=2048`, `1.4286` at `N=4096`, and `1.3263` at `N=8192`.
- Changed: restored the shared batched reduce kernel to `BN=32`, switched quartet pass-1 back to bf16 IO with fp32 stats, fixed the quartet grid to MLX's total-thread convention, and added silent fallback from quartet to batched 2-pass in the FA hook when `_dflash_quartet_enabled` is true.
- Next blocking issue: run the hook-level and end-to-end numerical checks on real Qwen3.5-9B FA verify with `_dflash_quartet_enabled=True` before considering a merge.

## 2026-04-16 — Iteration 2

- Attempted: added [benchmark/quartet_gqa_hook_check.py](/Users/bastienbouge/Documents/dev/dflash-mlx-quartet-gqa/benchmark/quartet_gqa_hook_check.py:1), [benchmark/quartet_gqa_e2e_check.py](/Users/bastienbouge/Documents/dev/dflash-mlx-quartet-gqa/benchmark/quartet_gqa_e2e_check.py:1), a runtime helper [configure_quartet_gqa()](/Users/bastienbouge/Documents/dev/dflash-mlx-quartet-gqa/dflash_mlx/runtime.py:677), and an opt-in benchmark flag in [benchmark/benchmark.py](/Users/bastienbouge/Documents/dev/dflash-mlx-quartet-gqa/benchmark/benchmark.py:629).
- Observed: `python3 -m benchmark.quartet_gqa_hook_check` passed on all 8 FA layers of Qwen3.5-9B (`[3, 7, 11, 15, 19, 23, 27, 31]`) with `max_abs_diff=0.0` and `allclose=True` for each layer at `ctx_len=4096`, `verify_len=16`.
- Observed: `python3 -m benchmark.quartet_gqa_e2e_check` passed on 3 long prompts (`code`, `math`, `narrative`) with `token_match=True` in all cases. Acceptance matched exactly off/on: `0.93359375`, `0.9296875`, `0.92578125`.
- Observed: V3 single-run benchmark comparison against current `main`, using exact prompt lengths `1024/2048/4096`, repeated README math prompt, `max_new_tokens=256`, `block_tokens=16`, `--no-eos` semantics:
- Observed: `ctx=1024` main `baseline_tps=30.8161`, `dflash_tps=29.1141`, `acceptance=0.01171875`; quartet `baseline_tps=30.8574`, `dflash_tps=160.9654`, `acceptance=0.921875`.
- Observed: `ctx=2048` main `baseline_tps=30.9132`, `dflash_tps=28.1646`, `acceptance=0.0`; quartet `baseline_tps=28.9968`, `dflash_tps=117.0376`, `acceptance=0.92578125`.
- Observed: `ctx=4096` main `baseline_tps=30.4248`, `dflash_tps=23.3831`, `acceptance=0.0078125`; quartet `baseline_tps=26.2662`, `dflash_tps=103.4183`, `acceptance=0.9296875`.
- Changed: benchmark control path can now enable quartet with `--quartet-gqa` while keeping the runtime default OFF.
- Next blocking issue: none for numerical/perf validation on this branch; the remaining merge step is packaging the branch state into a PR while keeping quartet disabled by default.
