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
draft, applies the runtime config, and serves requests through `mlx_lm.server`
with a DFlash request path for supported requests.

`dflash generate`
: One-shot offline generation. It uses the same target/draft runtime core as the
server, but disables the cross-request prefix cache.

`dflash benchmark`
: Public local smoke benchmark. It pre-tokenizes one prompt, runs baseline MLX
first, runs DFlash second with the same token ids, and writes self-contained
artifacts.

`dflash doctor`
: Local environment and config check. It resolves the same runtime config and
can optionally load a model.

`dflash models`
: Prints the currently registered target-to-draft mappings.

## Supported Model Shape

The live registry is explicit. A target must resolve to a matching DFlash draft
through `runtime.registry.DRAFT_REGISTRY` or through `--draft`.

If no draft resolves, loading fails. This runtime does not silently become a
generic no-draft MLX server. The server has an opt-in short-output AR fast path
for small `max_tokens` requests, controlled by `--fastpath-max-tokens` (default
`0`, disabled), but that is a serving policy after a valid runtime has already
loaded.

The current DFlash draft class is `DFlashDraftModel`. Target-family behavior is
owned by `engine.target_ops`; the registered concrete implementations are
`engine.target_qwen_gdn` and `engine.target_gemma4`.

Adding another architecture is a real adapter project, not just a new registry
row. The required seam for more architectures is:

- target adapter: embeddings, layer loop, hidden capture points, logits
  post-processing, masks, and cache construction;
- draft adapter: draft class, feature projection, mask token, vocab/shape
  validation;
- cache adapter: active cache type, rollback behavior, and snapshot codec;
- parity tests: adapter logits must match the model's own MLX forward path.

Gemma4 is supported through its own adapter. Prefix snapshots are enabled for
Gemma4 configs only when the adapter can prove known non-shared KV
(`num_kv_shared_layers == 0`); shared-KV, missing, or malformed capability data
fails closed. Product `dflash serve` diagnostics have verified exact repeated
prompt restore and stable long-chat continuation restore for the current Gemma4
31B and 26B-A4B targets, without turning those diagnostics into public
throughput claims. Future adapters must prove load, cache policy, and parity
before becoming product-supported targets.

Contributor workflow for a future target family is documented in
[`docs/model_backend.md`](model_backend.md).

## Runtime Config

The runtime config is resolved once at startup into `RuntimeContext`.

For server/doctor config:

1. explicit CLI flags win;
2. startup env vars win;
3. product defaults fill the rest.

After resolution, engine code consumes typed config instead of rereading product
env vars. A small set of `DFLASH_VERIFY_*` env vars remains internal kernel-debug
surface only.

`dflash serve` has one product default policy. Expert behavior is expressed with
explicit flags such as `--prefill-step-size`, `--no-prefix-cache-l2`, and
`--cache-limit auto`.

## Server Request Flow

The server wraps `mlx_lm.server` rather than replacing it.

1. `dflash_mlx.server.config` parses CLI flags, resolves runtime config, diagnostics,
   and Metal limits.
2. `dflash_mlx.server.runtime.ServerRuntime` owns the server lifecycle, live
   metrics setup, DFlash request orchestration, and shutdown.
3. `dflash_mlx.server.model_provider` loads a shared `RuntimeBundle`.
4. `dflash_mlx.serve.DFlashAPIHandler` and
   `dflash_mlx.serve.DFlashResponseGenerator` receive OpenAI-compatible
   requests through the upstream `mlx_lm.server` machinery.
5. For short-output requests, the server may use a target-only upstream path.
6. For DFlash requests, `server.prefix_cache_flow` computes the stable prefix
   and delegates lookup to `cache.manager`.
7. `runtime.stream_dflash_generate` enters `engine.spec_epoch.SpeculativeSession`.
8. `server.request_loop` consumes typed token, timing, memory, cycle, and
   snapshot-publication events. Snapshot-publication events carry metadata only;
   live cache arrays stay inside the engine/cache layer.
9. `cache.snapshot_service.SnapshotService` builds and admits snapshots through
   the runtime cache manager.

The prefix cache is process-local. The runtime cache manager owns the singleton
lifecycle keyed by relevant runtime config so repeated requests can share state.

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
: Upstream prompt cache controlled by `--prompt-cache-size`.

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

## Snapshot Lifecycle

- Prefill publishes are skipped when an end-of-request generation snapshot
  will follow; that snapshot adopts the live cache arrays instead of cloning
  them. Adoption is safe because GDN entries are replaced on every update
  (never mutated in place) and exact-length FA arrays force the growth path
  on the next update.
- Generation snapshots carry a GDN boundary sidecar: recurrent state and
  logits captured at the stable-prefix boundary, the last point where that
  state exists — GDN cannot be rewound. The sidecar lets a later request
  restore at a point inside the snapshot instead of missing.
- L1 serves a generation snapshot once, then consumes it: the request it
  serves publishes a strictly dominating successor at its end, so keeping
  the served entry would pin multi-GB arrays for hits that can never come.
- L2 disk writes are paused while a request is served and flush between
  requests. At shutdown, resident L1 entries spill to L2 so the next session
  warm-starts; generation snapshots reach disk only through this spill.
- L2 eviction is LRU by file mtime and lookups touch served files. A
  full-coverage snapshot gets a distinct fingerprint so a trimmed twin of
  the same tokens cannot shadow it, and full-context draft requests reject
  non-covering snapshots at lookup.

## Diagnostics

Diagnostics are opt-in for structured JSONL artifacts:

- `--diagnostics basic` enables request/cache post events;
- `--diagnostics full` adds per-cycle profiling and memory waterfall events;
- `--diagnostics-dir` controls output location;

The server also prints a compact per-request memory line on stderr after DFlash
requests. That line is separate from structured diagnostics artifacts.

## Module Map

Public surface:

- `dflash_mlx/cli.py` - root command dispatcher;
- `dflash_mlx/serve.py` - OpenAI-compatible server wrapper;
- `dflash_mlx/generate.py` - one-shot generation command;
- `dflash_mlx/benchmark.py` - public baseline-vs-DFlash benchmark;
- `dflash_mlx/doctor.py` - local environment/config checks.

Runtime/config:

- `dflash_mlx/runtime/__init__.py` - verify config and stream entry;
- `dflash_mlx/runtime/registry.py` - supported target-to-draft registry;
- `dflash_mlx/runtime/loading.py` - target/draft loading and load-time
  target optimization;
- `dflash_mlx/runtime/bundle.py` - shared target/draft/TargetOps/draft-backend
  binding;
- `dflash_mlx/runtime/config.py` - `RuntimeConfigSpec`, product defaults,
  resolver, validation, source reporting, and offline command projections;
- `dflash_mlx/runtime/context.py` - typed runtime context carrier;
- `dflash_mlx/runtime/chip_detect.py` - Apple Silicon generation/tier and BF16
  capability detection;
- `dflash_mlx/diagnostics.py` - diagnostics config;
- `dflash_mlx/metal_limits.py` - MLX wired/cache limit application;
- `dflash_mlx/internal_debug.py` - private kernel-debug env surface.

Server:

- `dflash_mlx/server/runtime.py` - server lifecycle, live metrics setup, DFlash
  request orchestration, startup banner, and shutdown;
- `dflash_mlx/server/model_provider.py` - shared runtime bundle publication for
  `mlx_lm.server`;
- `dflash_mlx/server/prefix_cache_flow.py` - per-request prefix cache lookup
  and snapshot-service setup;
- `dflash_mlx/server/protocol.py` - response protocol helpers;
- `dflash_mlx/server/tool_calls.py` - tokenizer tool-parser install, OpenAI
  tool-call normalization, and fail-fast tool-choice validation;
- `dflash_mlx/server/responses_adapter.py` - minimal non-streaming
  `/v1/responses` compatibility adapter;
- `dflash_mlx/server/metrics.py` - request accounting, live `/metrics`, and
  diagnostics event projection;
- `dflash_mlx/server/request_loop.py` - DFlash event-to-response loop.

Engine:

- `dflash_mlx/engine/config.py` - engine-local config dataclasses;
- `dflash_mlx/engine/spec_epoch.py` - DFlash request/cycle driver;
- `dflash_mlx/engine/events.py` - typed engine event ABI;
- `dflash_mlx/engine/prefill.py` - prefill helpers;
- `dflash_mlx/engine/sampling.py` - prompt token prep, suppress-token masks,
  greedy token selection, and eval/timing helpers;
- `dflash_mlx/engine/copyspec.py` - exact repeated-token candidate reuse;
- `dflash_mlx/engine/ddtree.py` - DDTree candidate planning helpers;
- `dflash_mlx/engine/gqa_sdpa.py` - long-context GQA SDPA routing helpers;
- `dflash_mlx/engine/target_features.py` - target hidden feature store;
- `dflash_mlx/engine/target_ops.py` - target architecture seam and resolver;
- `dflash_mlx/engine/target_qwen_gdn.py` - Qwen target cache, hidden capture,
  logits, rollback/tape, and hook policy;
- `dflash_mlx/engine/target_gemma4.py` - Gemma4 target adapter;
- `dflash_mlx/engine/acceptance.py` - acceptance length logic;
- `dflash_mlx/engine/fallback.py` - baseline fallback helpers;
- `dflash_mlx/engine/memory_waterfall.py` - memory bucket snapshots.

Draft/model/kernels:

- `dflash_mlx/model.py` - DFlash draft model;
- `dflash_mlx/draft/` - draft architecture dispatch and DFlash2-specific
  draft modules;
- `dflash_mlx/draft_backend.py` - masked-block draft call;
- `dflash_mlx/kernels.py` - local kernel helpers;
- `dflash_mlx/verify_linear.py` and `dflash_mlx/verify_qmm.py` - verify
  implementation paths, including Apple G17 NAX and steel fallback kernels;
- `dflash_mlx/recurrent_rollback_cache.py` - GDN rollback cache.

Prefix cache:

- `dflash_mlx/cache/snapshot.py` - snapshot dataclasses;
- `dflash_mlx/cache/snapshot_service.py` - snapshot build/admission service;
- `dflash_mlx/cache/store.py` - L1/L2 prefix snapshot store;
- `dflash_mlx/cache/fingerprints.py` - key/fingerprint helpers;
- `dflash_mlx/cache/codecs.py` - snapshot serialization helpers;
- `dflash_mlx/cache/manager.py` - runtime prefix cache lifecycle, lookup, and
  snapshot admission;
- `dflash_mlx/cache/prefix_l1.py` - RAM prefix cache;
- `dflash_mlx/cache/prefix_l2.py` - SSD prefix cache;
- `dflash_mlx/server/prefix_cache_flow.py` - request-time stable-prefix flow;
- `dflash_mlx/server/prefix_cache_manager.py` - server request key helpers.

Server package:

- `dflash_mlx/server/config.py` - server CLI/config;
- `dflash_mlx/server/model_provider.py` - model provider;
- `dflash_mlx/server/request_loop.py` - event/token bridge;
- `dflash_mlx/server/metrics.py` - structured event logging;
- `dflash_mlx/server/protocol.py` - lightweight protocol types;
- `dflash_mlx/server/tool_calls.py` - tool parser install, tool-call
  normalization, and tool-choice validation;
- `dflash_mlx/server/responses_adapter.py` - minimal non-streaming
  `/v1/responses` compatibility adapter.

Observability:

- `dflash_mlx/observability/cache.py` - cache telemetry projection;
- `dflash_mlx/observability/memory.py` - process/system/MLX memory reporting;
- `dflash_mlx/observability/writer.py` - diagnostics JSONL writer.

Lab tools:

- `tools/benchmarks/` - private/lab harnesses. They are not the public
  benchmark contract.
