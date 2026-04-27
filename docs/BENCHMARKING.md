# Benchmarking

This page documents every benchmark harness shipped with `dflash-mlx`. Five
runners are available: a single-prompt performance bench (`dflash-benchmark`),
two agentic session runners that drive an external CLI agent against a running
model server (`dflash-opencode-bench`, `dflash-pi-bench`), an in-process
multi-turn prefix-cache bench (`bench_prefix_cache_multiturn`), and a full
HTTP+SSE+server orchestrator that records every byte of an agentic OpenAI-API
session (`bench_agentic_trace`).

Agentic benches have meaningful run-to-run variance driven by sampler entropy,
agent path length, and thermal state. For any comparison statement, run at
least three iterations per cell, alternate the order of backends, and insert a
thermal cooldown of around 100 seconds between runs. Single-prompt runs
(`dflash-benchmark`) are repeatable enough that `--repeat 3` with a 10s cooldown
is usually sufficient.

All commands below assume your working directory is the repository root and
that you have installed the package in editable mode (`pip install -e .`).

## `dflash-benchmark`

**Purpose**: single-prompt head-to-head between the stock MLX baseline (via
`mlx_lm.stream_generate`) and the DFlash speculative-decoding runtime, on the
same target model and same prompt tokens. Loads each backend in turn, generates
once, and writes a JSON report.

**Command**:
```bash
dflash-benchmark --prompt "Explain speculative decoding in two paragraphs." \
    --model mlx-community/Qwen3.6-27B-4bit --max-tokens 256 --repeat 3
```

**Key flags**:

| Flag | Default | Effect |
|---|---|---|
| `--prompt` | (required) | Prompt string fed to both backends. |
| `--model` | auto-resolved | Target model HF reference. |
| `--draft` | auto-resolved | Draft model HF reference. Override only if your target is not in the auto-mapped registry. |
| `--max-tokens` | `64` | Tokens to generate. |
| `--block-tokens` | `16` | DFlash speculative block size. |
| `--repeat` | `1` (or `3` with `--matrix`) | Number of measured runs. Triggers matrix mode automatically when greater than 1. |
| `--matrix` | off | Force matrix mode (multi-run with summary statistics). |
| `--cooldown` | `10` | Seconds of `time.sleep` between matrix runs for thermal stability. |
| `--no-chat-template` | off | Feed the raw prompt instead of applying the tokenizer chat template. |
| `--quantize-draft` | off | 4-bit quantize the draft model after load. |
| `--no-eos` | off | Disable EOS so generation always reaches `--max-tokens`. |
| `--split-sdpa / --no-split-sdpa` | on | Toggle the split full-attention SDPA kernel on the target. |

**Output layout**:
```
benchmark/results/
    <model-slug>-<max-tokens>.json
```
The filename is derived from the resolved model reference and the `--max-tokens`
value, so reruns at the same configuration overwrite the same file.

**How to read**: each report contains `hardware`, `config`, a `runs` array, and
a `summary` block. The `summary` is the headline:

- `baseline_tps_median`, `dflash_tps_median` — generation throughput in
  tokens/second (excludes prefill).
- `speedup_median` — `dflash_tps / baseline_tps`. Greater than 1.0 means DFlash
  is faster.
- `acceptance_ratio_median` — fraction of speculatively drafted tokens accepted
  by the target.
- `dflash_tps_min` / `dflash_tps_max` — spread across runs; check this before
  trusting the median.

Each entry in `runs` also carries `thermal_pressure` (one of `nominal`, `fair`,
`serious`, `critical`, `unknown`) so you can spot a throttled run after the
fact.

## `dflash-opencode-bench`

**Purpose**: drive an entire `opencode` agentic session against a model server
you have already started, then capture stdout, stderr, the final workspace,
and a STATUS-friendly Markdown snippet. The runner does not start a model
server; you must launch one yourself.

Start a server first, in a separate shell:
```bash
# DFlash backend
dflash-serve --model mlx-community/Qwen3.6-27B-4bit \
    --draft z-lab/Qwen3.6-27B-DFlash --port 8090

# Or the mlx_lm baseline
python -m mlx_lm.server --model mlx-community/Qwen3.6-27B-4bit --port 8091
```
Configure the matching opencode provider so the `--model` argument resolves
through opencode's `provider/model` convention.

**Command**:
```bash
dflash-opencode-bench --model dflash/mlx-community/Qwen3.6-27B-4bit \
    --task-file tasks/brick_breaker.txt --label dflash_run1
```

**Key flags**:

| Flag | Default | Effect |
|---|---|---|
| `--model` | (required) | Model name in opencode's `provider/model` form. |
| `--task` | brick-breaker default | Inline task string. |
| `--task-file` | — | Read the task from a file. Use this for any comparison so the prompt is identical across runs. |
| `--label` | `opencode_<stamp>` | Run label. Used in the run-directory name and in the opencode session title. |
| `--out-root` | `benchmark/opencode_runs` | Parent directory for run folders. |
| `--workspace` | `<run-dir>/workspace` | Override the workspace passed to `opencode run --dir`. |
| `--timeout-s` | `1800` | Subprocess timeout. |
| `--opencode-bin` | autodetected | Path to the `opencode` binary. |
| `--thinking / --no-thinking` | on | Pass opencode's `--thinking` flag. |
| `--dangerously-skip-permissions / --no-dangerously-skip-permissions` | on | Non-interactive permission acks. |
| `--append-status` | off | Append the STATUS snippet to a top-level `STATUS.md`. |

**Output layout**:
```
benchmark/opencode_runs/<stamp>_<label>/
    metadata.json          # cwd, label, model, git, command, env
    command.txt            # exact argv that was executed
    task.txt               # the task as passed to opencode
    stdout.log             # opencode stdout (often newline-delimited JSON)
    stderr.log             # opencode stderr
    summary.json           # metadata + exit code + wall_s + stdout/event counts
    STATUS_SNIPPET.md      # ready-to-paste run summary
    workspace/             # everything opencode created or modified
```

**How to read**: `summary.json` is the canonical record. The fields most useful
for comparing runs are `wall_s` (end-to-end latency of the agentic session),
`exit_code`, `stdout.event_counts` (a histogram of opencode JSON event types,
which proxies for tool-call activity), and `workspace_files` (a manifest of
what the agent produced). The `STATUS_SNIPPET.md` is a self-contained Markdown
record of the run with the command, paths, and observed metrics, suitable for
pasting into a project log or PR description.

## `dflash-pi-bench`

**Purpose**: same shape as the opencode bench, but drives the `pi` CLI
(<https://pi.dev>) instead. The two runners produce directly comparable
outputs.

As with opencode, you must start a model server yourself before running this
bench. The `--model` argument follows pi's `<provider>/<id>` form; for example
`dflash/mlx-community/Qwen3.6-27B-4bit` selects a custom provider named
`dflash` declared in `~/.pi/agent/models.json`. If that provider is not
registered, the bench will fail with a pi-side error.

**Command**:
```bash
dflash-pi-bench --model dflash/mlx-community/Qwen3.6-27B-4bit \
    --task-file tasks/brick_breaker.txt --label dflash_run1
```

**Key flags**:

| Flag | Default | Effect |
|---|---|---|
| `--model` | (required) | pi model spec, `<provider>/<id>`. |
| `--task` | brick-breaker default | Inline task string. |
| `--task-file` | — | Read the task from a file. |
| `--label` | `pi_<stamp>` | Run label. |
| `--out-root` | `benchmark/pi_runs` | Parent directory for run folders. |
| `--workspace` | `<run-dir>/workspace` | Override the agent's working directory. |
| `--timeout-s` | `1800` | Subprocess timeout. |
| `--pi-bin` | autodetected | Path to the `pi` binary. |
| `--thinking` | `high` | One of `off`, `minimal`, `low`, `medium`, `high`, `xhigh`. `off` omits the flag entirely. |
| `--extensions / --no-extensions` | off | Whether pi auto-discovers extensions. |
| `--skills / --no-skills` | off | Whether pi auto-discovers skills. |
| `--prompt-templates / --no-prompt-templates` | off | Prompt-template discovery. |
| `--themes / --no-themes` | off | Theme discovery. |
| `--context-files / --no-context-files` | off | `AGENTS.md` / `CLAUDE.md` autoload. |
| `--append-status` | off | Append STATUS snippet to `STATUS.md`. |

The `--no-*` defaults are deliberate: extensions, skills, prompt templates,
themes, and ambient context files all change pi's behaviour run-to-run, so they
are turned off so the bench is reproducible. Turn them back on only when you
want to measure your local pi configuration in particular.

**Output layout**:
```
benchmark/pi_runs/<stamp>_<label>/
    metadata.json
    command.txt
    task.txt
    stdout.log
    stderr.log
    summary.json
    STATUS_SNIPPET.md
    workspace/
```

**How to read**: identical to the opencode bench. `summary.json -> wall_s` is
the headline; `summary.json -> stdout.event_counts` shows the JSON-mode event
histogram pi emitted; `workspace_files` is the produced artefact list.

## `bench_prefix_cache_multiturn`

**Purpose**: in-process multi-turn microbench that compares three configurations
on a synthetic agentic conversation: pure `mlx_lm` autoregressive baseline with
a fresh KV cache each turn, DFlash with the prefix cache disabled, and DFlash
with the prefix cache enabled. Each turn extends the previous turn's prompt by
a small user message, modelling growing-context agent traffic.

**Command**:
```bash
PYTHONPATH=$PWD .venv/bin/python -m benchmark.bench_prefix_cache_multiturn \
    --target mlx-community/Qwen3.6-27B-4bit \
    --draft z-lab/Qwen3.6-27B-DFlash \
    --turns 10 --system-tokens 800 --max-tokens 48
```

**Key flags**:

| Flag / env | Default | Effect |
|---|---|---|
| `--target` / `BENCH_TARGET` | `mlx-community/Qwen3.6-27B-4bit` | Target model HF reference. |
| `--draft` / `BENCH_DRAFT` | `z-lab/Qwen3.6-27B-DFlash` | Draft model HF reference. |
| `--turns` / `BENCH_TURNS` | `10` | Number of simulated chat turns. |
| `--system-tokens` / `BENCH_SYSTEM_TOKENS` | `800` | Approximate token budget for the synthetic system preamble. Larger values stress the prefix cache more. |
| `--max-tokens` / `BENCH_MAX_TOKENS` | `48` | Tokens to generate per turn. |
| `BENCH_OUT` env | `benchmark/results/apple-m5-max/prefix_cache_multiturn_<utc>.json` | Where to write the JSON record. |

**Output layout**:
```
stdout                    # per-turn table (mlx_lm AR vs DFlash-off vs DFlash+cache)
benchmark/results/apple-m5-max/
    prefix_cache_multiturn_<utc-stamp>.json
```

**How to read**: stdout shows three sections (one per configuration) followed
by a side-by-side summary with per-turn TTFT, total wall, and the
`speedup vs mlx_lm (tot)` ratio. The bottom of the run prints session totals
plus two headline ratios: DFlash+cache vs mlx_lm AR, and DFlash+cache vs
DFlash without the cache. The JSON file mirrors stdout: it contains the raw
`mlxlm`, `baseline`, and `cached` per-turn arrays plus a `session_totals`
block and a `cache_stats` block from the prefix cache. The `from_snapshot`
and `matched_lookup` fields on cached turns tell you when the prefix cache
fired and how many tokens it covered.

## `bench_agentic_trace`

**Purpose**: full end-to-end orchestrator for an OpenAI-API agentic run.
Spawns the chosen model server (`dflash` or `mlxlm`), spawns a recording
proxy, patches an `opencode` provider entry to point at the proxy, runs an
opencode session, and tears everything down. Every HTTP request body, every
SSE chunk, every server stderr line, and every server-side bench event is
captured verbatim, then post-processed into a per-POST table.

**Command**:
```bash
# DFlash side
python -m benchmark.bench_agentic_trace \
    --backend dflash \
    --target mlx-community/Qwen3.6-27B-4bit \
    --draft z-lab/Qwen3.6-27B-DFlash \
    --task-file tasks/brick_breaker.txt \
    --label dflash_brick

# mlx_lm side, on the same task
python -m benchmark.bench_agentic_trace \
    --backend mlxlm \
    --target mlx-community/Qwen3.6-27B-4bit \
    --task-file tasks/brick_breaker.txt \
    --label mlxlm_brick \
    --compare-to benchmark/results/agentic-trace-<earlier-stamp>-dflash_brick
```

**Key flags**:

| Flag | Default | Effect |
|---|---|---|
| `--backend` | (required) | `dflash` or `mlxlm`. |
| `--target` | (required) | Target model HF reference. |
| `--draft` | — | Required when `--backend dflash`. |
| `--task-file` | (required) | File containing the task. |
| `--label` | `<backend>_<target-name>` | Run label. |
| `--out-root` | `benchmark/results` | Parent directory. |
| `--proxy-port` | `9788` | Proxy listen port. |
| `--dflash-port` | `8090` | DFlash server port. |
| `--mlxlm-port` | `8091` | mlx_lm server port. |
| `--server-ready-timeout-s` | `300` | Server health-check window. |
| `--proxy-ready-timeout-s` | `30` | Proxy health-check window. |
| `--opencode-timeout-s` | `1800` | opencode subprocess timeout. |
| `--opencode-bin` | autodetected | opencode binary path. |
| `--thinking / --no-thinking` | on | opencode `--thinking`. |
| `--dangerously-skip-permissions / --no-...` | on | Non-interactive perms. |
| `--enable-prefix-cache / --no-enable-prefix-cache` | on | DFlash only: sets `DFLASH_PREFIX_CACHE=1` plus default sizing env vars on the server. |
| `--compare-to` | — | Path to a prior `agentic-trace-...` run directory; emits a head-to-head section in `compare.md`. |

**Output layout**:
```
benchmark/results/agentic-trace-<stamp>-<label>/
    metadata.json
    task.txt
    config_snapshot.json     # opencode.jsonc as it was before patching
    requests/NNN.json        # full HTTP request body, one file per POST
    sse/NNN.jsonl            # per-chunk SSE log with monotonic t_ms
    server/
        cmd.txt
        stdout.log
        stderr.log
        metrics.jsonl        # one JSON row per POST, server-side metrics
    proxy/
        cmd.txt
        stdout.log
        stderr.log
    opencode/
        cmd.txt
        stdout.jsonl
        stderr.log
    events/                  # DFlash bench_logger events (DFlash backend only)
        post_events.jsonl
        cycle_events.jsonl
        cache_events.jsonl
    workspace/               # opencode workspace
    summary.json             # aggregated record
    compare.md               # human-readable per-POST table + verdict
```

**How to read**: open `compare.md` first. The header gives the backend, target,
draft, commit, opencode wall time, total POST count, total prompt tokens,
total decode tokens, average decode tps across POSTs, and an acceptance
ratio weighted by decoded tokens (DFlash only). The per-POST table lists
prompt and decode counts, decode tps, acceptance, prefix-cache hit tokens,
server-side timing landmarks (TTFT, prefill, decode), and trace-side
landmarks (`first_byte_ms`, `first_content_token_ms`, `first_tool_call_ms`).
The `src` column shows whether each row came from server metrics or from the
SSE `usage` chunk emitted by `mlx_lm.server`, so blank rows are still
populated for baseline runs.

When you pass `--compare-to`, `compare.md` adds a verdict block: per-POST
`first_tool_call_ms` deltas, the summed gap, and the wall-time ratio between
the two runs. `summary.json` contains the same data in machine-readable form
(the `totals` block has `wall_s`, `decode_tps_avg`, `weighted_acceptance`,
`total_cache_hit_tokens`, and `first_tool_call_ms_avg`).

Two helpers ship alongside this orchestrator: `benchmark/agentic_proxy.py` is
the recording proxy invoked by the orchestrator (it can also be run standalone
if you want to record a session driven by some other client), and
`benchmark/aggregate_replay.py` collapses a directory of trace runs into a
single comparison record.

## Comparing dflash vs a baseline

Use this protocol whenever you publish a numeric claim:

1. Pin the same task. For agentic benches, pass `--task-file` so the prompt is
   byte-identical across runs. For `dflash-benchmark`, keep `--prompt`,
   `--max-tokens`, `--block-tokens`, and `--no-eos` constant.
2. Run on the same hardware in alternating order to balance thermal drift.
   For example, run the sequence `mlx_lm, dflash, mlx_lm, dflash, mlx_lm,
   dflash` rather than three of each in a row.
3. Insert a thermal cooldown between runs. 100 seconds is a reasonable
   default on M-series Macs; the `dflash-benchmark` runner exposes
   `--cooldown` for this. For agentic runners, sleep manually between
   invocations.
4. Report n>=3 per cell and quote the median. Single runs of an agentic
   bench can swing by tens of percent purely from agent path variance.
5. Compute `wall_dflash / wall_baseline` (lower is better) or
   `tps_dflash / tps_baseline` (higher is better). Be explicit about which.
6. Compare equal work. If the agent makes 6 tool calls on one backend and 3
   on the other, the wall-time difference is partly an agent-path
   difference, not a runtime difference. Eyeball the per-POST count in
   `summary.json` before drawing conclusions.

Worked example (illustrative numbers, not measured):

| target | backend | wall_s | POSTs | tps avg | ratio vs mlx_lm |
|---|---|---|---|---|---|
| Qwen3.6-27B-4bit | mlx_lm | 312.0 | 4 | 33.1 | 1.00 |
| Qwen3.6-27B-4bit | dflash | 248.0 | 4 | 51.4 | 0.79 |
| Qwen3.6-35B-A3B-4bit | mlx_lm | 198.0 | 4 | 130.5 | 1.00 |
| Qwen3.6-35B-A3B-4bit | dflash | 158.0 | 4 | 162.0 | 0.80 |

Treat this table as a shape, not as a measurement.

## Result interpretation gotchas

- `mlx_lm.server` ships with a built-in LRU prompt cache that fires on every
  request. If you compare DFlash with `--no-prefix-cache` against
  `mlx_lm.server` on a multi-turn workload, you are racing a cached baseline
  against an uncached candidate; the comparison is unfair. The DFlash default
  is `--enable-prefix-cache`. State the prefix-cache configuration of both
  sides whenever you publish a number.
- Different agent CLIs use different system prompts and different tool
  catalogs. Wall times from `dflash-opencode-bench` and `dflash-pi-bench` are
  not directly comparable across CLI families even on the same task and the
  same model server, because the agents are sending different bytes.
- Single-turn benches do not exercise the prefix cache. If you want to see
  cache effects, use `bench_prefix_cache_multiturn` or one of the agentic
  runners; `dflash-benchmark` will not show them.

## Local override of the model registry

`dflash-benchmark` and `bench_prefix_cache_multiturn` auto-resolve a draft
model from the target via the built-in registry. If your target is not in the
registry, or you want to test a different draft, pass `--draft <hf-ref>`
explicitly. For the agentic trace bench, the `--draft` flag is required when
`--backend=dflash` regardless of the registry.

For the full list of runtime environment variables that affect DFlash
behaviour during a bench (prefix-cache sizing, probation steps, schedule
limits), see `docs/RUNTIME_FLAGS.md`. For background on how the runtime
chooses between speculative and autoregressive paths, see
`docs/ARCHITECTURE.md`.
