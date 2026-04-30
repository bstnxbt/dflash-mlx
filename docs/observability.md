# Observability

DFlash has two observability surfaces:

1. compact stderr lines that help interactive server use;
2. structured diagnostics artifacts for debugging and benchmarking.

Structured diagnostics are opt-in. The compact server memory line is currently
always printed after DFlash requests.

## Diagnostics Modes

`dflash serve` accepts:

```bash
--diagnostics off
--diagnostics basic
--diagnostics full
--diagnostics-dir PATH
```

`off`
: No structured JSONL artifacts.

`basic`
: Request/post events and cache summaries.

`full`
: Basic diagnostics plus memory waterfall and per-cycle profiling.

Default output:

```text
.artifacts/dflash/diagnostics/<timestamp>-serve-<mode>/
```

Advanced aliases:

```bash
--bench-log-dir PATH
--memory-waterfall
```

Use the `--diagnostics` form for normal operation. The direct aliases exist for
bench and internal harness wiring.

## Files

When diagnostics are enabled, the runtime may write:

```text
cycle_events.jsonl
post_events.jsonl
summary.md
```

`cycle_events.jsonl`
: Per-cycle engine events. In full mode this includes cycle timing and memory
waterfall samples.

`post_events.jsonl`
: Per-request summary events. These include request ids, prompt length, token
counts, acceptance/cycle fields when applicable, cache stats, and memory summary
fields.

`summary.md`
: Human-readable append-only request summary.

## Memory Waterfall

Memory waterfall records named buckets at selected runtime boundaries. It is
used to separate active model memory, MLX cache memory, prefix snapshots, draft
cache, target KV/state, target hidden chunks, rollback tape, and untracked RSS.

Enable it with:

```bash
dflash serve --diagnostics full
```

or:

```bash
dflash serve --memory-waterfall --bench-log-dir .artifacts/dflash/diagnostics/manual
```

Waterfall is for diagnosis. Do not enable full diagnostics for throughput
claims.

## Always-On Server Line

After a DFlash request, the server prints a compact memory line on stderr. This
is separate from JSONL diagnostics. It exists so a normal server run still gives
basic memory visibility without a diagnostics directory.

If you need reproducible analysis, use `--diagnostics basic` or
`--diagnostics full` instead of scraping stderr.

## Trace Discipline

For performance claims:

- diagnostics should be `off` unless the claim is specifically about
  diagnostics;
- full cycle tracing should not be used;
- record the exact command, model refs, profile, prompt-token count, and output
  directory;
- prefer `dflash benchmark` for public baseline-vs-DFlash numbers.

For debugging:

- use `--diagnostics basic` for request/cache issues;
- use `--diagnostics full` for memory or cycle timing issues;
- keep the run directory local under `.artifacts/`;
- attach the whole diagnostics directory to bug reports rather than copying
  selected lines.
