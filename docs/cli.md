# CLI

The public command is `dflash`.

```bash
dflash --help
```

Commands:

```text
dflash serve
dflash generate
dflash benchmark
dflash doctor
dflash profiles
dflash models
```

Legacy top-level commands are not part of the product surface.

## Serve

Start the OpenAI-compatible server:

```bash
dflash serve \
  --model Qwen/Qwen3.5-4B \
  --profile balanced
```

Useful profiles:

```bash
dflash profiles
dflash serve --profile fast
dflash serve --profile low-memory
dflash serve --profile long-session
```

Expert overrides stay explicit:

```bash
dflash serve \
  --profile balanced \
  --prefill-step-size 8192 \
  --prefix-cache-max-bytes 17179869184
```

See [runtime-flags.md](runtime-flags.md) for the full flag surface.

## Generate

One prompt, no server:

```bash
dflash generate \
  --model Qwen/Qwen3.5-4B \
  --prompt "Explain speculative decoding in two paragraphs."
```

This path is for smoke checks. It does not enable cross-request prefix cache and
should not be used for public performance claims.

## Benchmark

Public local baseline-vs-DFlash smoke benchmark:

```bash
dflash benchmark \
  --model Qwen/Qwen3.5-4B \
  --max-tokens 64
```

Outputs default to:

```text
.artifacts/dflash/benchmarks/<timestamp>-<mode>-<model>/
```

See [benchmarking.md](benchmarking.md) for protocol and artifact details.

## Doctor

Check local runtime state:

```bash
dflash doctor
dflash doctor --json
dflash doctor --strict
```

Validate an effective runtime profile:

```bash
dflash doctor --profile low-memory
dflash doctor --profile long-session --prefix-cache-l2 --json
```

Check model/draft resolution:

```bash
dflash doctor --model Qwen/Qwen3.5-4B
dflash doctor --model Qwen/Qwen3.5-4B --load-model
```

`doctor` accepts the same runtime config flags as the server for validation:
profile, prefill size, draft sink/window, verify cap, prefix cache, L2, target
FA window, and max context.

## Models

List the current built-in target-to-draft registry:

```bash
dflash models
```

Only listed families are supported by the automatic draft resolver. Passing a
different target without a compatible `--draft` is a load error, not a silent
fallback to a generic server.

## Diagnostics

Basic request/cache logs:

```bash
dflash serve --diagnostics basic
```

Full memory/cycle diagnostics:

```bash
dflash serve --diagnostics full
```

Custom directory:

```bash
dflash serve --diagnostics full --diagnostics-dir .artifacts/dflash/diagnostics/manual
```

See [observability.md](observability.md).

## Common Examples

Normal coding server:

```bash
dflash serve --model Qwen/Qwen3.5-27B --profile balanced
```

Throughput-oriented server:

```bash
dflash serve --model Qwen/Qwen3.5-27B --profile fast
```

Lower-memory server:

```bash
dflash serve --model Qwen/Qwen3.5-27B --profile low-memory
```

Long-session cache experiment:

```bash
dflash serve \
  --model Qwen/Qwen3.5-27B \
  --profile long-session \
  --prefix-cache-l2-dir .artifacts/dflash/prefix-l2
```

Synthetic 64k public smoke:

```bash
dflash benchmark \
  --model Qwen/Qwen3.5-4B \
  --ctx 64000 \
  --max-tokens 64
```
