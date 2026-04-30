# Internal Benchmark Tools

This directory contains lab harnesses. They are useful for diagnosis, but they
are not the public benchmark contract.

Use `dflash benchmark` first for public baseline-vs-DFlash claims.

## Surfaces

```bash
PYTHONPATH=$PWD python -m tools.benchmarks.agentic_trace --help
PYTHONPATH=$PWD python -m tools.benchmarks.prefix_cache_probe --help
PYTHONPATH=$PWD python -m tools.benchmarks.analyze_trace --help
PYTHONPATH=$PWD python tools/benchmarks/bench_long_context_decode.py --help
```

`agentic_trace.py`
: Agentic session/proxy trace tooling. Use it to study server behavior under a
real client shape, not as a public speed claim by default.

`prefix_cache_probe.py`
: Prefix-cache and L2 mechanism probes.

`analyze_trace.py`
: Trace and prompt/memory analyzers.

`bench_long_context_decode.py`
: Long-context decode canary.

## Cross-runtime opencode comparison (dflash vs mlxlm)

Run baseline first, then dflash with `--compare-to`:

```bash
python -m tools.benchmarks.agentic_trace run --backend mlxlm --target <model> \
    --task-file <task.txt> --label mlx_baseline
python -m tools.benchmarks.agentic_trace run --backend dflash --target <model> \
    --draft <draft> --task-file <task.txt> --label dflash_run \
    --enable-prefix-cache --enable-prefix-cache-l2 \
    --compare-to .artifacts/dflash/traces/<mlx_baseline_dir>
```

The dflash run emits `compare.md` with:

- **Trajectory-invariant metrics** (`decode_tps_avg`,
  `prefix_tokens_saved`, `weighted_acceptance`,
  `post_prefill_ms_per_token`) — the only metrics that are mathematically
  valid for cross-runtime comparison.
- **Trajectory-dependent metrics** (`wall_s`, POST count) shown for reference
  with caveats — runtimes take different paths to converge under greedy 4-bit
  decoding because dflash's `split_sdpa` attention path and Q4 fused matmul
  variants can flip borderline argmax tokens versus `mlx_lm.server`.
- **Per-POST gap table** emitted only when trajectories align: same POST
  count and decode tokens within ±5%. Otherwise it is omitted with an explicit
  `TRAJECTORY DIVERGED` warning.

For deterministic per-token A/B with no trajectory divergence by construction,
use `dflash benchmark` on a fixed prompt. That benchmark runs both runtimes
in-process on identical input.

Private `_*.py` files are implementation modules for these wrappers.

## Output Policy

New lab outputs should go under `.artifacts/dflash/...` or an explicit local
`--out` path. Do not add new benchmark outputs to Git.

Some harnesses are intentionally narrower than `dflash benchmark` and may not
write a full public manifest. If a result will be quoted outside local
debugging, record the exact command, model refs, git hash, profile/flags, prompt
tokenization mode, and output directory.

## Rules

- Do not compare numbers across harnesses as if they were one protocol.
- Do not use full tracing for throughput claims.
- Do not overlap heavy model loads.
- Keep mechanism probes separate from product benchmark results.
- Prefer the smallest harness that answers the question.
