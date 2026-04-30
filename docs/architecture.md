# DFlash MLX Architecture

This repository is an Apple Silicon runtime for long coding sessions on MLX.
DFlash is one acceleration path inside that runtime. The public product surface
is the CLI, the OpenAI-compatible server, the benchmark command, and the cache /
diagnostics behavior around real requests.

`mlx_lm.generate_step` remains the model-truth reference. `mlx_lm.server` remains
the serving baseline.

## Public Entrypoints

`dflash serve`
: OpenAI-compatible server. It loads the target, resolves a supported DFlash
draft, applies the runtime profile, and serves requests through `mlx_lm.server`
with a DFlash request path for supported requests.

`dflash generate`
: One-shot offline generation. It uses the same target/draft runtime core as the
server, but disables the cross-request prefix cache.

`dflash benchmark`
: Public local smoke benchmark. It pre-tokenizes one prompt, runs baseline MLX
first, runs DFlash second with the same token ids, and writes self-contained
artifacts.

`dflash doctor`
: Local environment and config check. It resolves the same runtime profile and
can optionally load a model.

`dflash profiles`
: Prints the runtime presets.

`dflash models`
: Prints the currently registered target-to-draft mappings.

## Supported Model Shape

The live registry is explicit and Qwen-focused. A target must resolve to a
matching DFlash draft through `DRAFT_REGISTRY` or through `--draft`.

If no draft resolves, loading fails. This runtime does not silently become a
generic no-draft MLX server. The server does have a short-output AR fast path for
small `max_tokens` requests, but that is a serving policy after a valid runtime
has already loaded.

The current DFlash draft class is `DFlashDraftModel`, a Qwen-style draft model.
The target verifier is also shaped around the cache and hidden-state behavior
used by the current Qwen targets. Adding Gemma or another architecture is a real
adapter project, not just a new registry row.

The required future seam for more architectures is:

- target adapter: embeddings, layer loop, hidden capture points, logits
  post-processing, masks, and cache construction;
- draft adapter: draft class, feature projection, mask token, vocab/shape
  validation;
- cache adapter: active cache type, rollback behavior, and snapshot codec;
- parity tests: adapter logits must match the model's own MLX forward path.

## Runtime Config

The runtime config is resolved once at startup into `RuntimeContext`.

For server/doctor config:

1. explicit CLI flags win;
2. startup env vars win over profiles;
3. profile values win over product defaults.

After resolution, engine code consumes typed config instead of rereading product
env vars. A small set of `DFLASH_VERIFY_*` env vars remains internal kernel-debug
surface only.

The main runtime presets are:

| Profile | Intent |
| --- | --- |
| `balanced` | default for normal coding sessions |
| `fast` | larger prefill chunks and larger L1 budget |
| `low-memory` | smaller prefill chunks and smaller L1 budget |
| `long-session` | larger L1 plus SSD L2 for revisited prefixes |

Use `dflash profiles` for the exact active numbers.

## Server Request Flow

The server wraps `mlx_lm.server` rather than replacing it.

1. `dflash_mlx.server.config` parses CLI flags, resolves profiles, diagnostics,
   and Metal limits.
2. `dflash_mlx.server.model_provider` loads the target and draft bundles.
3. `dflash_mlx.serve.DFlashServer` receives OpenAI-compatible requests through
   the upstream `mlx_lm.server` machinery.
4. For short-output requests, the server may use a target-only upstream path.
5. For DFlash requests, `server.prefix_cache_flow` performs prefix lookup and
   computes the stable prefix.
6. `server.request_loop` drives `engine.spec_epoch.stream_dflash_generate_impl`
   and forwards token, cache, timing, and snapshot events.
7. At snapshot-ready events, `PrefixCacheFlow` builds and inserts prefix
   snapshots.

The prefix cache is process-local. The current implementation uses a documented
server singleton keyed by relevant runtime config so repeated requests can share
state.

## Offline Generate Flow

`dflash generate` uses `build_offline_runtime_context()`.

Differences from server mode:

- no cross-request prefix cache;
- no OpenAI-compatible request wrapper;
- direct prompt string input;
- same verifier, draft backend, cache construction, and Metal limit machinery.

This command is useful for a small correctness smoke test. It is not the public
performance benchmark.

## Public Benchmark Flow

`dflash benchmark` is the public performance smoke surface.

It:

- resolves target and draft once;
- renders/tokenizes the prompt once;
- reuses identical prompt token ids for baseline and DFlash;
- runs baseline first, then DFlash;
- records invocation, manifest, runs, summary JSON, and summary markdown;
- writes new artifacts under `.artifacts/dflash/benchmarks/...` unless `--out`
  is provided.

`benchmark/results/*.json` files in this repo are pinned legacy evidence, not
the default destination for new runs.

## DFlash Cycle

The live DFlash cycle is not a classic autoregressive draft loop.

1. Target prefill runs in chunks and initializes target cache/state.
2. The target's next-token argmax is staged.
3. The draft receives a block that starts with that staged token followed by
   mask tokens.
4. The draft emits candidate tail tokens in one block pass.
5. The target verifies the staged token plus draft tail.
6. Acceptance is computed by comparing draft tail tokens with the target
   posterior.
7. The runtime commits `1 + acceptance_len` tokens.
8. The target posterior token after the accepted span becomes the staged first
   token for the next cycle.
9. Attention KV can be truncated by length; GDN recurrent state uses rollback
   cache/tape mechanics.

This shape matters because it is why the runtime captures target hidden
features for the draft and why rollback/cache behavior is DFlash-specific.

## Caches

There are several distinct memory buckets. They should not be conflated.

Target active cache
: The live verifier cache/state for the current request. This includes full
attention KV and recurrent state where applicable. Prefix L2 does not reduce
this bucket.

Draft cache
: A bounded draft context cache using a sink plus rolling window.

Prefix L1
: RAM snapshots of stable DFlash prefixes. A snapshot includes token ids, target
attention cache, recurrent state, hidden-state chunks, and last logits.

Prefix L2
: SSD spill for prefix snapshots evicted from L1. It is for revisits and parked
prefixes, not active hot-path KV paging.

mlx_lm prompt cache
: Upstream prompt cache controlled by `--prompt-cache-size`. `--prompt-cache-bytes`
is parsed as a public flag but should be treated cautiously until the
upstream byte-cap wiring is verified.

## Prefix Snapshot Shape

`DFlashPrefixSnapshot` stores:

- token ids;
- full-attention cache state;
- recurrent/GDN state;
- `target_hidden_chunks`;
- `target_hidden_chunk_spans`;
- `target_hidden_total_len`;
- last logits;
- a `DFlashPrefixKey`;
- snapshot kind and timestamp.

The key includes target id, draft id, capture layer ids, draft sink/window,
target FA window, and format version. This is what prevents incompatible
snapshots from being reused across different runtime shapes.

## Diagnostics

Diagnostics are opt-in for structured JSONL artifacts:

- `--diagnostics basic` enables request/cache post events;
- `--diagnostics full` adds per-cycle profiling and memory waterfall events;
- `--diagnostics-dir` controls output location;
- `--bench-log-dir` and `--memory-waterfall` are advanced direct aliases.

The server also prints a compact per-request memory line on stderr after DFlash
requests. That line is separate from structured diagnostics artifacts.

## Module Map

Public surface:

- `dflash_mlx/cli.py` - root command dispatcher;
- `dflash_mlx/serve.py` - OpenAI-compatible server wrapper;
- `dflash_mlx/generate.py` - one-shot generation and draft registry;
- `dflash_mlx/benchmark.py` - public baseline-vs-DFlash benchmark;
- `dflash_mlx/doctor.py` - local environment/config checks.

Runtime/config:

- `dflash_mlx/runtime.py` - model loading, target cache construction, verify
  config;
- `dflash_mlx/runtime_profiles.py` - profiles and effective runtime config;
- `dflash_mlx/runtime_context.py` - typed runtime context carrier;
- `dflash_mlx/diagnostics.py` - diagnostics config;
- `dflash_mlx/metal_limits.py` - MLX wired/cache limit application;
- `dflash_mlx/internal_debug.py` - private kernel-debug env surface.

Engine:

- `dflash_mlx/engine/spec_epoch.py` - DFlash request/cycle driver;
- `dflash_mlx/engine/prefill.py` - prefill helpers;
- `dflash_mlx/engine/target_verifier.py` - target verification forward;
- `dflash_mlx/engine/rollback.py` - recurrent rollback helpers;
- `dflash_mlx/engine/acceptance.py` - acceptance length logic;
- `dflash_mlx/engine/fallback.py` - baseline fallback helpers;
- `dflash_mlx/engine/memory_waterfall.py` - memory bucket snapshots.

Draft/model/kernels:

- `dflash_mlx/model.py` - DFlash draft model;
- `dflash_mlx/draft_backend.py` - masked-block draft call;
- `dflash_mlx/kernels.py` - local kernel helpers;
- `dflash_mlx/verify_linear.py` and `dflash_mlx/verify_qmm.py` - verify
  implementation paths;
- `dflash_mlx/recurrent_rollback_cache.py` - GDN rollback cache.

Prefix cache:

- `dflash_mlx/cache/snapshot.py` - snapshot dataclasses;
- `dflash_mlx/cache/fingerprints.py` - key/fingerprint helpers;
- `dflash_mlx/cache/codecs.py` - snapshot serialization helpers;
- `dflash_mlx/cache/prefix_l1.py` - RAM prefix cache;
- `dflash_mlx/cache/prefix_l2.py` - SSD prefix cache;
- `dflash_mlx/server/prefix_cache_flow.py` - request-time prefix flow and
  singleton boundary;
- `dflash_mlx/server/prefix_cache_manager.py` - cache construction.

Server package:

- `dflash_mlx/server/config.py` - server CLI/config;
- `dflash_mlx/server/model_provider.py` - model provider;
- `dflash_mlx/server/request_loop.py` - event/token bridge;
- `dflash_mlx/server/metrics.py` - structured event logging;
- `dflash_mlx/server/protocol.py` - lightweight protocol types.

Lab tools:

- `tools/benchmarks/` - private/lab harnesses. They are not the public
  benchmark contract.

