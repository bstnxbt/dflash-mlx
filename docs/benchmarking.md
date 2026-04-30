# Benchmarking

Use `dflash benchmark` for public local performance claims. Keep lab harnesses
separate from product claims.

## Public Smoke Benchmark

```bash
PYTHONPATH=$PWD dflash benchmark \
  --model Qwen/Qwen3.5-4B \
  --max-tokens 64
```

Default protocol:

1. load target and matching DFlash draft;
2. render/tokenize one prompt once;
3. run baseline MLX first;
4. run DFlash second using the same prompt token ids;
5. repeat according to `--repeat`;
6. write artifacts under `.artifacts/dflash/benchmarks/...`.

This command is for local smoke numbers and regression checks. It is not a
substitute for a full server workload benchmark.

## Important Defaults

| Setting | Default |
| --- | --- |
| prompt | `Explain speculative decoding in two paragraphs.` |
| chat template | enabled |
| generated tokens | `64` |
| block tokens | `16` |
| repeat | `1` |
| cooldown | `10` seconds |
| memory summary | enabled |
| split-SDPA in benchmark | enabled |
| output dir | `.artifacts/dflash/benchmarks/<timestamp>-<mode>-<model>` |

`--ctx INT` builds an approximate synthetic long-context prompt. It is useful for
cheap stress testing, but it is not the same as a real multi-turn coding agent
session.

## Benchmark Flags

| Flag | Meaning |
| --- | --- |
| `--prompt TEXT` | prompt text |
| `--ctx INT` | synthetic long-context prompt size |
| `--max-tokens INT` | generation length |
| `--block-tokens INT` | DFlash verify block size |
| `--repeat INT` | measured runs |
| `--cooldown SECONDS` | sleep between runs |
| `--model REF_OR_PATH` | target model |
| `--draft REF_OR_PATH` | draft override |
| `--no-chat-template` | raw prompt text |
| `--quantize-draft` | quantize draft after load |
| `--no-eos` | suppress EOS for fixed-length runs |
| `--split-sdpa`, `--no-split-sdpa` | benchmark verifier split-SDPA mode |
| `--target-fa-window INT` | experimental target FA rotating window |
| `--draft-sink-size INT` | draft cache sink tokens |
| `--draft-window-size INT` | draft cache rolling window tokens |
| `--verify-len-cap INT` | max tokens per verify forward |
| `--no-memory` | omit memory medians |
| `--out PATH` | artifact directory |

## Artifacts

Each public benchmark run writes:

- `manifest.json` - repo/runtime metadata;
- `invocation.json` - command, model refs, prompt token mode, protocol;
- `runs.jsonl` - per-run measurements;
- `summary.json` - aggregate numbers;
- `summary.md` - human-readable report.

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
- long-context decode canaries;
- OpenCode head-to-head scripts.

These tools are useful for diagnosis, but their outputs are not public claims
unless the run directory records enough context to reproduce the command and
environment.

Rules:

- use `dflash benchmark` first for public smoke checks;
- use lab harnesses to answer one specific mechanism question;
- do not compare numbers from different harnesses as if they were one protocol;
- do not use full cycle tracing for performance claims;
- benchmark sequentially, never with two heavy model loads in parallel.

## Good Command Patterns

Small smoke:

```bash
PYTHONPATH=$PWD dflash benchmark --model Qwen/Qwen3.5-4B --max-tokens 64
```

Fixed-length decode:

```bash
PYTHONPATH=$PWD dflash benchmark \
  --model Qwen/Qwen3.5-4B \
  --max-tokens 128 \
  --no-eos
```

Synthetic context stress:

```bash
PYTHONPATH=$PWD dflash benchmark \
  --model Qwen/Qwen3.5-4B \
  --ctx 64000 \
  --max-tokens 64 \
  --out .artifacts/dflash/benchmarks/manual-64k
```

Low-memory runtime check:

```bash
PYTHONPATH=$PWD dflash benchmark \
  --model Qwen/Qwen3.5-4B \
  --draft-sink-size 64 \
  --draft-window-size 1024 \
  --verify-len-cap 0 \
  --target-fa-window 0
```

## Reading Results

Look at:

- baseline tokens/sec;
- DFlash tokens/sec;
- acceptance rate;
- tokens per cycle;
- peak memory if enabled;
- prompt token count and whether chat template was enabled.

Do not interpret DFlash speed without acceptance and tokenization regime. A raw
prompt and a chat-template prompt are different benchmark inputs.
