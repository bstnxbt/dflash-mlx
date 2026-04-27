# Architecture

dflash-mlx is a speculative-decoding runtime for Apple Silicon. It pairs a
target Qwen language model with a small DFlash draft model and runs them in a
verify-and-accept loop on top of MLX. The result is a server you can drop into
the same place you would call `mlx_lm.server`, with the same OpenAI-compatible
HTTP surface, that delivers multiple committed tokens per target forward when
the draft is well aligned with the target.

The runtime is organized around a single decode cycle (prefill, draft block,
target verify, acceptance, commit, rollback) plus a cross-request prefix cache
that recycles KV state across turns of an agentic conversation. Output is
distribution-equivalent to running the target alone under greedy decoding: the
draft only proposes tokens; the target's argmax is the source of truth for
every committed token.

This document covers the conceptual model. A companion module map describes the
file-by-file layout of the package.

## 1. What is dflash-mlx

dflash-mlx is a Python package that exposes two entry points:

- `dflash-serve` — an OpenAI-compatible HTTP server. It speaks the same
  endpoints as `mlx_lm.server` (`/v1/chat/completions`,
  `/v1/completions`), so existing clients (chat UIs, agentic frameworks like
  opencode and pi, evaluation harnesses) work unchanged.
- `dflash-generate` — a CLI for one-shot generation, useful for benchmarks and
  smoke tests.

Both entry points run the same speculative-decoding core. The server adds a
prefix cache, a fast-path for short autoregressive requests, and the usual
streaming SSE protocol.

The runtime is built specifically for the Qwen3.5 / Qwen3.6 family of hybrid
models — architectures that interleave full attention with gated delta-net
(GDN) recurrent layers. Supported targets cover the dense and MoE members of
the family: 4B and 9B dense, 27B (a hybrid full-attention plus GDN architecture
with a dense FFN), and the 35B-A3B mixture-of-experts. A draft registry
maps each target to a matching DFlash draft (see section 5).

The fact that the target is hybrid matters for the runtime: full-attention
layers and GDN layers have different KV-state shapes and rollback semantics,
and both need to be snapshotted to make the prefix cache and target verify
correct. The cache and verifier modules treat both kinds of layer as
first-class.

## 2. Speculative decoding refresher

Speculative decoding gets more committed tokens per target forward by running
two models cooperatively:

1. A small, cheap draft model proposes K candidate tokens autoregressively.
2. The large target model is run once on those K candidates in a single
   forward pass and produces its own argmax for each position.
3. The runtime accepts the longest prefix of the draft's proposal that
   matches the target's argmax token-by-token. Whatever the target picks at
   the first mismatch is appended as a "free" extra token.
4. Cycle repeats from the new committed position.

If the draft and target agree often, every target forward commits more than
one token, and end-to-end throughput goes up. If they rarely agree, you fall
back to roughly one committed token per target forward — the same as plain
autoregressive generation, plus the (small) overhead of running the draft.

In dflash-mlx the verify pass is implemented in
`dflash_mlx/engine/target_verifier.py` (`verify_target_block`,
`target_forward_with_hidden_states`) backed by lower-level kernels in
`dflash_mlx/verify_backend.py`. Acceptance matching — comparing draft tokens
with the target's argmax and computing the longest common prefix length — is
in `dflash_mlx/engine/acceptance.py` (`match_acceptance_length`, a vectorized
`cumprod` over the equality mask).

Because acceptance is greedy-argmax-equality, a token is committed only if the
target itself would have produced it. The output distribution is identical to
running the target alone under greedy decoding. Speculative decoding here is a
pure throughput optimization, not a sampling-quality knob.

## 3. The DFlash cycle

The decode loop is a small state machine that repeatedly does:

```
prefill -> draft block -> target verify -> acceptance match -> commit accepted -> rollback rejected -> repeat
```

The driver is `stream_dflash_generate_impl` in
`dflash_mlx/engine/spec_epoch.py`, which yields events (prefill, token, cycle
metrics, summary) so the server can stream them to the client.

### Prefill

When a request comes in, the prompt is run once through the target to populate
its KV state. If the prefix cache (section 4) had a hit, prefill only runs
over the suffix of the prompt that wasn't cached — the cached prefix is
hydrated directly into the target's KV state without any forward computation.

Prefill bookkeeping (boundary computation, hidden-state initialization from a
restored snapshot) lives in `dflash_mlx/engine/prefill.py`
(`compute_snapshot_boundary`, `init_target_hidden_from_snapshot`).

### Draft block

The draft proposes a window of K candidate tokens autoregressively, conditioned
on the last committed target hidden state and on a small "context tape" of
recently committed tokens. The draft backend is in
`dflash_mlx/draft_backend.py`. K is the draft window size (configurable via
`--num-draft-tokens` and tuning environment variables); the loop picks an
effective K every cycle based on remaining budget and stop-token horizon.

The draft is much smaller than the target and has its own KV cache, so this
phase is a small fraction of the cycle wall time even at K = 8.

### Target verify

The K draft tokens, plus the previously committed token, are packed into one
verify input and run through the target in a single forward pass.
`verify_target_block` in `dflash_mlx/engine/target_verifier.py` returns the
target's logits for every position and the captured hidden states from the
layers the draft conditions on. The single-forward shape is the whole point:
we pay one target forward to verify K candidates instead of K target forwards.

### Acceptance match

`match_acceptance_length` compares the draft's proposed tokens with the
target's argmax at each position and returns the length of the longest
matching prefix. The token at the first mismatch — the target's argmax there —
is appended as the "free" extra token. So a cycle commits between 1 and K + 1
tokens.

### Commit accepted

The runtime advances its committed-token state, updates the live token count,
flushes any newly-decoded text fragment to the streaming response, and updates
the draft's context tape so the next draft block has fresh conditioning.

### Rollback rejected

Both the target and draft KV caches were extended with K positions during the
verify pass, but only `accepted + 1` of them are real. The runtime trims the
caches back to the committed length so the next cycle starts in a clean state.
Rollback helpers — including the prefix-aware rollback used to make the prefix
cache safe across requests — live in `dflash_mlx/engine/rollback.py`
(`arm_target_rollback_with_prefix`, `restore_target_cache_after_acceptance`,
`cleanup_generation_caches`). For hybrid targets this is non-trivial: GDN
layers need a full state snapshot rather than a positional trim.

### Repeat, or fallback

The cycle repeats until a stop token, the max-token budget, or an explicit
client disconnect. There is also an autoregressive fallback path
(`dflash_mlx/engine/fallback.py`, `stream_baseline_generate`) used when
speculative decoding is not viable. The server picks the fallback in two
distinct cases:

- **Short generations.** In `dflash_mlx/serve.py` (`_serve_single`), if
  `args.max_tokens <= 256` the server short-circuits to plain autoregressive
  decoding via the upstream `mlx_lm.server` path. At small budgets the
  fixed cost of priming the draft and sequencing verify cycles outweighs any
  acceptance gain, so plain AR is faster.
- **Draft unavailable or context too long.** The spec-epoch driver itself
  hands off to `stream_baseline_generate` when the prompt length exceeds
  `DFLASH_MAX_CTX`, when the draft model is missing, or when a runtime check
  detects that the draft is incompatible with the target's captured
  features. The fallback returns the same event shape as the speculative
  path so the server's streaming logic is identical.

## 4. Prefix cache

The prefix cache turns multi-turn agentic workloads into something close to a
single long stream. It snapshots the target's KV state at the end of a
request, keyed by the prompt prefix that's stable across turns, and restores
that snapshot directly into the next request that shares the same prefix.
Effectively, a turn that would have re-prefilled 2,800 tokens prefills only
the new suffix.

### Where it lives

- `dflash_mlx/cache/prefix_l1.py` — `DFlashPrefixCache`, the in-process LRU
  store with byte and entry budgets. Methods: `lookup`, `insert`, `stats`,
  `clear`. Eviction is LRU within the byte budget.
- `dflash_mlx/server/prefix_cache_manager.py` — `make_prefix_cache` (process
  singleton constructor, reads budgets from environment), `build_prefix_key`
  (combines target id, draft id, captured layer ids, and draft window
  parameters into the cache key), `chat_template_marker_ids` (resolves the
  `<|im_start|>` and `assistant` token ids used by the stable-prefix policy),
  and `format_stats_line` (the periodic `prefix-cache-stats` log line).

### What is stored

A `DFlashPrefixSnapshot` (`dflash_mlx/cache/snapshot.py`) holds:

- `token_ids` — the exact prefix the snapshot was taken at.
- `fa_states` — per-layer full-attention K and V tensors.
- `gdn_states` — per-layer GDN recurrent state tuples (None for non-GDN
  layers).
- `target_hidden` — the last hidden state at the snapshot boundary, which
  the draft needs to seed conditioning at the next request.
- `last_logits` — optional, used to skip the first prefill forward when the
  next request's prompt length equals the snapshot boundary exactly.
- `key` — the `DFlashPrefixKey` (target id, draft id, captured layer ids,
  draft window) so we never restore a snapshot built under different shapes.

Serialization and hydration of the target KV state are in
`dflash_mlx/cache/codecs.py` (`serialize_target_cache`, `hydrate_target_cache`,
`build_snapshot`, `target_cache_is_serializable`). Hydration deep-copies the
underlying mx arrays so that ongoing generation cannot mutate stored buffers.

### When it fires

At request entry, the server (in `serve.py`):

1. Builds the prefix key from the loaded models.
2. Computes `stable_prefix_len` over the prompt tokens with
   `compute_stable_prefix_len` (`dflash_mlx/cache/policies.py`). This trims
   the trailing `<|im_start|>assistant` boilerplate that opens the new turn —
   the suffix that would change every turn and break exact-prefix matching.
3. Calls `prefix_cache.lookup(prompt[:stable_prefix_len], prefix_key)`. The
   cache returns `(matched_len, snapshot)` for the longest stored prefix that
   matches token-for-token under the same key.
4. On a non-zero match, the snapshot is passed to the speculative driver
   which hydrates the target KV cache, jumps prefill ahead by `matched_len`
   tokens, and prefills only the suffix `prompt[matched_len:]`.
5. At end of request, `serve.py` builds a fresh snapshot from the post-decode
   target state and inserts it into the cache. Old, dominated entries are
   pruned.

A typical hit logs:

```
[dflash] prefix cache hit 2710/2800 tokens (stable prefix 2795)
```

### How to enable, disable, tune

Defined in `dflash_mlx/server/config.py:build_parser`:

- `--prefix-cache` / `--no-prefix-cache` — enable or disable the cache.
  Default: enabled.
- `--prefix-cache-max-entries N` — maximum number of cached snapshots.
  Default: 4.
- `--prefix-cache-max-bytes B` — total byte budget for the cache. Default:
  8 GiB.

These flags map to environment variables (`DFLASH_PREFIX_CACHE`,
`DFLASH_PREFIX_CACHE_MAX_ENTRIES`, `DFLASH_PREFIX_CACHE_MAX_BYTES`) which the
cache modules read at startup. See `RUNTIME_FLAGS.md` for the full table of
runtime knobs.

## 5. Draft auto-detection

`dflash-serve --model <ref>` does not require you to specify a draft. The
runtime ships a registry mapping a target to its matching DFlash draft
(`dflash_mlx/generate.py:DRAFT_REGISTRY`):

| Target base name      | Draft repository                       |
|-----------------------|----------------------------------------|
| Qwen3.5-4B            | z-lab/Qwen3.5-4B-DFlash                |
| Qwen3.5-9B            | z-lab/Qwen3.5-9B-DFlash                |
| Qwen3.5-27B           | z-lab/Qwen3.5-27B-DFlash               |
| Qwen3.5-35B-A3B       | z-lab/Qwen3.5-35B-A3B-DFlash           |
| Qwen3.6-27B           | z-lab/Qwen3.6-27B-DFlash               |
| Qwen3.6-35B-A3B       | z-lab/Qwen3.6-35B-A3B-DFlash           |
| Qwen3-4B              | z-lab/Qwen3-4B-DFlash-b16              |
| Qwen3-8B              | z-lab/Qwen3-8B-DFlash-b16              |

Resolution (`resolve_optional_draft_ref`) strips the org prefix from the
target reference (so `mlx-community/Qwen3.6-35B-A3B-4bit` resolves to the
`Qwen3.6-35B-A3B` row), normalizes case, and picks the longest matching base
name. Quantization suffixes like `-4bit` are tolerated by the longest-prefix
match.

You can override at any time with `--draft <hf-ref>` (or
`--draft-model <hf-ref>`). If neither the registry nor `--draft` produces a
match, the runtime starts in autoregressive-only mode and the speculative
path is bypassed for every request.

## 6. Fallback to autoregressive

Speculative decoding is a throughput optimization, not a correctness primitive,
so the runtime is designed to fall back to a plain target-only loop whenever
speculation cannot pay for itself or cannot run at all.

The fallback implementation is `stream_baseline_generate` in
`dflash_mlx/engine/fallback.py`. It runs the same event protocol as the
speculative driver — prefill event, per-token events, cycle metrics, summary —
so the streaming layer in the server does not need to know which path it is
talking to.

Fallback fires in three places:

- **Short-budget fast path.** `_serve_single` in `dflash_mlx/serve.py`
  routes any request with `max_tokens <= 256` to the upstream
  `mlx_lm.server` AR loop. Below this threshold, the per-request fixed
  costs of speculative decoding (draft warm-up, verify framing) outweigh the
  per-token gain.
- **Hard limits.** `stream_dflash_generate_impl` falls back when the prompt
  length exceeds `DFLASH_MAX_CTX`.
- **No usable draft.** If no draft is registered for the target, if `--draft`
  was not supplied, or if the loaded draft fails a compatibility check
  against the target's captured features, the server runs the request on the
  fallback path with an explanatory `fallback_reason` in the event stream.

The fast path is intentionally conservative: it is far better to match the
baseline on workloads where speculation cannot win than to ship a wrapper that
is slightly slower than `mlx_lm` on small generations.

## 7. Where DFlash wins

Speculative decoding is workload-sensitive. The dflash-mlx defaults are tuned
for the workloads where it has the most room to win, and the fallback paths
are tuned to stay close to baseline elsewhere.

DFlash wins on:

- **Long, predictable code generation.** Code has high local entropy at the
  token level (long literal copies, repeated identifiers, boilerplate
  scaffolding) that a small DFlash draft tracks well. Copy, refactor, and
  boilerplate workloads see acceptance ratios that turn into multi-token
  commits per target forward.
- **Multi-turn agentic loops.** Tools like opencode, pi, and similar harnesses
  send a long, mostly-stable prefix on every turn. The prefix cache amortizes
  prefill across turns; combined with speculative decoding on the new tail,
  this is the highest-leverage workload for the runtime.

DFlash is roughly neutral on:

- **Short single-turn requests.** Below the 256-token threshold the server
  takes the autoregressive fast path on purpose.
- **Highly creative or divergent prose.** Low draft acceptance means cycles
  commit close to one token each, which is the same as plain AR plus a small
  draft overhead.

These shapes are qualitative on purpose. Speedup ratios depend heavily on the
target model, the draft, the prompt distribution, the temperature, and the
hardware (M-series generation, memory bandwidth, thermal state). Run the
provided benches on your own workload before drawing conclusions: see
`BENCHMARKING.md` for the protocol and the scripts in `benchmark/` for the
reference workloads.
## Module map

This half of the architecture document is a structural reference: it lists every Python module in the `dflash_mlx` package, says what each module owns, and names the public symbols you would import from it. Pair it with the concepts half above for the reasoning behind these boundaries.

### Top-level layout

```
dflash_mlx/
├── __init__.py
├── adapter.py                     # picks FullAttention vs HybridGDN engine for a target model
├── bench_logger.py                # JSONL bench-event logger gated on DFLASH_BENCH_LOG_DIR
├── draft_backend.py               # eager draft-model wrapper used inside the spec cycle
├── generate.py                    # the `dflash` CLI (one-shot generation entry point)
├── kernels.py                     # custom Metal kernels (gated-delta tape, tape replay, SDPA 2-pass)
├── model.py                       # DFlashDraftModel, DFlashDraftModelArgs, ContextOnlyDraftKVCache
├── prefix_cache.py                # backward-compat re-export shim for dflash_mlx.cache.*
├── recurrent_rollback_cache.py    # GDN/SSM rollback-aware KV cache entry
├── runtime.py                     # orchestration: prefill -> draft -> verify -> accept -> rollback
├── serve.py                       # the `dflash-serve` HTTP server (mlx_lm.server-compatible)
├── verify_backend.py              # FullAttentionEngine / HybridGDNEngine verify dispatchers
├── verify_linear.py               # VerifyQuantizedLinear: install/uninstall verify-fast linears
├── verify_qmm.py                  # quantized-matmul Metal kernels for the verify path
│
├── cache/                         # prefix-cache primitives (snapshots, hashing, policies, L1)
│   ├── __init__.py
│   ├── codecs.py                  # serialize/hydrate target KV state across snapshot boundaries
│   ├── fingerprints.py            # DFlashPrefixKey: model + capture + window fingerprint
│   ├── policies.py                # stable-prefix detection + env-driven budget knobs
│   ├── prefix_l1.py               # DFlashPrefixCache: in-memory LRU L1 with stats
│   └── snapshot.py                # DFlashPrefixSnapshot dataclass + validator
│
├── engine/                        # speculative-decoding cycle building blocks
│   ├── __init__.py
│   ├── acceptance.py              # accepted-prefix length from draft vs verifier
│   ├── config.py                  # env-driven runtime knobs (DFLASH_MAX_CTX, DFLASH_VERIFY_LEN, etc.)
│   ├── events.py                  # typed event-name constants emitted on the stream
│   ├── fallback.py                # plain autoregressive fallback (`stream_baseline_generate`)
│   ├── prefill.py                 # chunked prefill helpers + snapshot-boundary math
│   ├── rollback.py                # arm/restore/clear KV-cache rollback after partial accept
│   ├── spec_epoch.py              # `stream_dflash_generate_impl` cycle loop
│   ├── target_verifier.py         # `verify_target_block` + lm-head and hidden-state plumbing
│   └── types.py                   # TypedDict schemas for stream events
│
└── server/                        # HTTP server pieces composed by serve.py
    ├── __init__.py
    ├── config.py                  # argparse parser + Metal/log configuration
    ├── metrics.py                 # bench POST logging + human summary line
    ├── model_provider.py          # DFlashModelProvider: loads target+draft on startup
    ├── prefix_cache_manager.py    # request-handler glue around DFlashPrefixCache
    └── protocol.py                # adapters across mlx_lm.server API versions
```

### `dflash_mlx/runtime.py`

The orchestration entry point that ties prefill, draft, verify, acceptance, rollback, and prefix cache together. At ~737 lines it is the largest single file and the place to start when tracing what happens to a request after the server hands it off.

Key public symbols:

- `stream_dflash_generate(**kwargs) -> Iterator[dict]` — the main streaming generator. Thin wrapper that defers to `engine.spec_epoch.stream_dflash_generate_impl`; this is what `serve.py` and `generate.py` call.
- `make_target_cache(target_model, *, quantize_kv_cache, ...)` — builds the per-layer KV cache list for the target, mixing `KVCache` and `RecurrentRollbackCache` according to the architecture detected by `adapter.detect_engine`.
- `load_target_bundle(model_ref, *, draft_quant=None, ...)` — loads the target model and tokenizer, applies optional verify-linear / verify-qmm hooks, returns a bundle consumed by the runtime.
- `load_draft_bundle(draft_ref, *, target_config, ...)` — loads a `DFlashDraftModel` and binds it to the target's `target_layer_ids` capture indices.
- `_prepare_prompt_tokens(tokenizer, prompt, *, use_chat_template)` — tokenization helper that handles chat-template formatting.
- `build_suppress_token_mask(vocab_size, suppress_token_ids)` — converts a list of forbidden token ids into a `(vocab,)` bool/float mask.
- `greedy_tokens_with_mask(logits, suppress_mask)` — argmax with suppression mask applied.
- `detect_engine(target_model)` — re-export of `adapter.detect_engine`.
- `configure_full_attention_split(target_model, *, enabled)` — installs split-SDPA hooks on full-attention layers when the env knob is set.
- `make_draft_backend()` — re-export of `draft_backend.make_draft_backend`.

Other notable internals: `pack_target_model_weights_selective`, `_install_target_speculative_hooks`, `_install_speculative_linear_cache_hook`, `_install_split_full_attention_hook`, `parse_draft_quant_spec`, `detect_target_family`.

### `dflash_mlx/engine/`

Per-file breakdown of the speculative-decoding cycle.

#### `events.py`

Typed event-name constants emitted to the streaming consumer. Keeps event identifiers in one place so producers and consumers agree.

- `PREFILL`, `PREFILL_PROGRESS`, `PREFILL_SNAPSHOT_READY`, `GENERATION_SNAPSHOT_READY`, `TOKEN`, `CYCLE_COMPLETE`, `SUMMARY`
- `ALL_EVENT_NAMES` — frozenset of all event names for validation.

#### `types.py`

`TypedDict` schemas for events emitted on the stream. Source of truth for what fields a consumer can read off each event payload.

- `PhaseTimingsUs` — per-phase timings (`prefill`, `draft`, `draft_prefill`, `draft_incremental`, `verify`, `replay`, `commit`).
- `StreamSummary` — final summary event (acceptance ratio, generation tokens, cycles, peak memory, fallback flags).

#### `prefill.py`

Helpers used while running chunked prefill or while bootstrapping the target hidden state from a cached snapshot.

- `compute_snapshot_boundary(prompt_len, stable_prefix_len)` — clamps the snapshot cut point.
- `init_target_hidden_from_snapshot(prefix_snapshot, snap_prefix_len, prompt_len)` — pre-allocates the target hidden tensor and copies the cached prefix region in.

#### `fallback.py`

Plain autoregressive fallback used when speculative decoding is disabled or aborts. Emits the same event shape as the speculative path so the consumer is identical.

- `stream_baseline_generate(*, target_model, tokenizer, prompt, max_new_tokens, ...)` — generator yielding the same event types as `stream_dflash_generate`.

#### `acceptance.py`

Acceptance-length matching against draft proposals.

- `match_acceptance_length(drafted_tokens, posterior_tokens) -> mx.array` — returns the length of the longest prefix where the draft and verifier agree, computed via `cumprod` on equality.

#### `rollback.py`

KV cache rollback after a partial accept. Centralises the difference between full-attention `KVCache` (no rollback needed past the prefix) and `RecurrentRollbackCache` (must replay the tape).

- `arm_target_rollback_with_prefix(cache_entries, *, prefix_len)` — calls `arm_rollback` on each rollback-aware entry.
- `clear_rollback_state(cache_entry)` — drops tape/snapshot transients on a cache entry.
- `cleanup_generation_caches(target_cache, draft_cache, ...)` — end-of-request cleanup.
- `restore_target_cache_after_acceptance(...)` — trims the target cache to the accepted length.

#### `target_verifier.py`

Forward pass plumbing for the verify step. Handles both the wrapped (`model.model.layers`) and unwrapped target shapes, runs the target on a verify block, and exposes the captured per-layer hidden states needed by the draft model.

- `verify_target_block(*, target_model, target_cache, tokens, ...)` — runs the target on `tokens`, returns logits and captured features.
- `target_forward_with_hidden_states(target_model, tokens, cache, ...)` — lower-level forward that captures named hidden states.
- `extract_context_feature_from_dict(captured_dict, target_layer_ids)` — concatenates the captured layers along the feature axis for the draft.
- `_target_text_wrapper`, `_target_text_model`, `_lm_head_logits` — model-shape helpers.

#### `spec_epoch.py`

The speculative cycle loop itself. This is the function `runtime.stream_dflash_generate` defers to, and the one that drives the prefill -> draft -> verify -> accept -> rollback rhythm and emits events.

- `stream_dflash_generate_impl(*, target_model, draft_model, tokenizer, ...)` — the cycle generator.

#### `config.py`

Runtime knobs read from the environment. Centralises every `DFLASH_*` env variable consulted on the hot path so the cycle code can stay free of `os.environ` reads.

- `_resolve_verify_len_cap(target_model, block_tokens)` — `DFLASH_VERIFY_LEN` cap.
- `_resolve_dflash_max_ctx()` — `DFLASH_MAX_CTX`.
- `_resolve_draft_window()` — `DFLASH_DRAFT_SINK`, `DFLASH_DRAFT_WINDOW`.
- `_draft_window_override_enabled()`, `_is_unwindowed_full_attention_draft(draft_model)`, `_effective_draft_window_size(...)`.
- `_profile_dflash_cycles_enabled()` — gates per-cycle profiling.

### `dflash_mlx/cache/`

The prefix cache. Concept-side: snapshot a target's KV state at a stable boundary so the next request that shares that prefix can hydrate instead of re-prefilling.

#### `snapshot.py`

The dataclass that defines the on-memory shape of a cached prefix and a validator for it.

- `DFlashPrefixSnapshot` — fields: `token_ids`, `fa_states` (per-layer full-attention K/V/offset triples), `gdn_states` (per-layer recurrent state tuples), `target_hidden`, `last_logits`, `key`, `kind`, `created_at`. Properties: `prefix_len`, `nbytes`.
- `validate_prefix_snapshot(snapshot, ...)` — sanity-check helper.

#### `codecs.py`

Encode and decode of target KV state for snapshot serialisation. Knows about both `KVCache` and `RecurrentRollbackCache` and refuses to serialise other kinds.

- `target_cache_is_serializable(target_cache)` — predicate.
- `serialize_target_cache(target_cache)` — extracts `(fa_states, gdn_states)` tuples from a live target cache list.
- `hydrate_target_cache(target_cache, fa_states, gdn_states)` — inverse: writes cached state back into freshly built cache entries.
- `build_snapshot(*, key, token_ids, target_cache, target_hidden, last_logits, kind)` — constructor that bundles `serialize_target_cache` output with the rest of the snapshot fields.

#### `fingerprints.py`

The hash key used to identify a cache entry. Two entries collide iff their fingerprint is equal, so anything that changes the captured-feature layout must change the key.

- `DFlashPrefixKey` — frozen dataclass with `target_model_id`, `draft_model_id`, `capture_layer_ids`, `draft_sink_size`, `draft_window_size`, `format_version`.

#### `policies.py`

The "should I cache this, and where do I cut" policy. Pure functions, no state.

- `compute_stable_prefix_len(tokens, *, im_start_id=None, assistant_id=None)` — finds the last `<|im_start|>assistant` boundary; returns `len(tokens)` if the markers are not configured (cache-disabled fallback).
- `prefix_cache_enabled()` — reads `DFLASH_PREFIX_CACHE`.
- `read_budget_env(*, default_entries=4, default_bytes=8 GiB)` — reads `DFLASH_PREFIX_CACHE_MAX_ENTRIES` and `DFLASH_PREFIX_CACHE_MAX_BYTES`.

#### `prefix_l1.py`

The in-memory L1 cache itself.

- `DFlashPrefixCache` — thread-safe LRU keyed by `DFlashPrefixKey`, byte-budget and entry-budget. Methods: `lookup(req_tokens, key) -> (matched_len, snapshot)`, `insert(snapshot)`, `stats()`, plus internal eviction. Tracks `exact_hits`, `prefix_hits`, `misses`, `insertions`, `evictions`, `prefix_prunes`, `prefill_tokens_saved`, `fingerprint_rejects`.

### `dflash_mlx/server/`

The HTTP server pieces. `serve.py` composes them; this directory holds the implementation.

#### `config.py`

`dflash-serve` argparse parser plus Metal and log setup helpers.

- `build_parser() -> argparse.ArgumentParser` — defines `--model`, `--draft-model`, `--dflash-max-ctx`, `--num-draft-tokens`, `--draft-quant`, `--host`, `--port`, etc.
- `normalize_cli_args(args)` — applies defaults and env overrides.
- `configure_metal_limits()` — sets the Metal wired-memory and cache limits.
- `configure_logging(log_level)` — sets up the root logger.

#### `protocol.py`

Adapter layer over `mlx_lm.server` so the same code works against multiple `mlx_lm` versions. Probes for the stateful API and switches based on `STATEFUL_SERVER_API`.

- `STATEFUL_SERVER_API` — boolean, true on newer mlx_lm.
- `build_generation_context(tokenizer, prompt, stop_words=None, sequences=None)` — constructs the right `GenerationContext` shape.
- `make_response(...)` — constructs the right `Response` payload.
- `match_stream_token(...)` — adapts stream-token matching across versions.

#### `metrics.py`

Bench logging helpers and the human-readable per-request summary line.

- `write_summary_line(*, summary_event, prompt_token_count)` — writes the `[dflash] X tok/s | Y% accepted | ...` line to stderr.
- `log_bench_post(...)` — emits the JSONL POST record (forwards to `bench_logger.log_post`).

#### `model_provider.py`

Subclass of `mlx_lm.server.ModelProvider` that loads target + draft on startup using `dflash_mlx.generate.load_runtime_components`.

- `DFlashModelProvider` — overrides `load(model_path, adapter_path=None, draft_model_path=None)` to populate `target_model`, `draft_model`, tokenizer, and the engine descriptor.
- `wait_for_initial_model_load(provider)` — busy-waits until the provider has its target loaded; used by the startup banner.

#### `prefix_cache_manager.py`

Glue between the request handler and `DFlashPrefixCache`. Owns the question "does this request share a usable prefix with the last one?".

- `make_prefix_cache() -> DFlashPrefixCache` — reads the budget env and constructs the L1.
- `format_stats_line(cache, label="")` — writes a `[dflash] cache stats ...` line to stderr.
- `build_prefix_key(model_provider, draft_model) -> DFlashPrefixKey` — composes the fingerprint from model ids, capture layer ids, and the draft window.
- `chat_template_marker_ids(tokenizer)` — extracts `im_start_id` and `assistant_id` for the stable-prefix policy.

### Other top-level modules

#### `model.py`

Definition of the DFlash draft model. Exposes the `from_dict` / `sanitize` surface that `mlx_lm.utils.load_model` relies on, so the draft can be loaded the same way as any other mlx_lm model.

- `DFlashDraftModel` — the draft `nn.Module`.
- `DFlashDraftModelArgs` — config dataclass; supplies `from_dict` and `sanitize`.
- `DFlashAttention`, `DFlashDecoderLayer` — internal blocks.
- `ContextOnlyDraftKVCache` — the draft-side KV cache, anchored to the target's prompt sink.
- `build_target_layer_ids(num_target_layers, num_draft_layers)` — chooses which target layers to capture features from.

#### `draft_backend.py`

Draft model wrapper used during the speculative cycle. Owns the per-cycle "draft K tokens" call.

- `EagerDraftBackend` — methods: `make_cache(*, draft_model, sink_size, window_size)`, plus the draft-greedy entry used by `spec_epoch`.
- `make_draft_backend() -> EagerDraftBackend` — constructor.

#### `verify_backend.py`

Engine dispatcher used on the verify path. Picks between full-attention and hybrid-GDN behaviour based on the target architecture.

- `_BaseEngine` — shared `arm_rollback` / `verify` shell.
- `FullAttentionEngine` — for pure-attention targets.
- `HybridGDNEngine` — for hybrid full-attention + gated-delta targets.

#### `kernels.py`

Custom Metal kernel helpers used by the verify and rollback paths. Each kernel has a `_make_*` builder (compiled once) and an `_ops` wrapper.

- `gated_delta_kernel_with_tape(...)` — gated-delta forward that records a tape for rollback.
- `tape_replay_kernel(...)` — replays the tape after a partial accept.
- `batched_sdpa_2pass_exact(...)` — batched 2-pass SDPA used inside the verify block.

#### `verify_linear.py` and `verify_qmm.py`

Alternate verify primitives that swap in a custom quantized-matmul kernel for the lm-head and projection layers when the target is quantised. Selected by `DFLASH_VERIFY_QMM=1` and tuned by `DFLASH_VERIFY_VARIANT`.

`verify_linear.py`:

- `VerifyQuantizedLinear` — drop-in replacement for `nn.QuantizedLinear` on the verify path.
- `is_verify_eligible(ql, path="")` — predicate for which linears get rewritten.
- `install_verify_linears(model, ...)` / `uninstall_verify_linears(model)` — patch and restore.
- `prewarm_verify_kernels(model)` — first-call kernel compilation.

`verify_qmm.py`:

- `is_enabled()`, `_variant()`, `_auto_variant(K, N)` — env-driven dispatch.
- `verify_matmul(...)` — the qmm entry point.
- `_build_kernel_mma2big`, `_build_kernel_mma2big_8bit`, `_build_kernel_mma2big_pipe`, `_build_kernel_mma2big_pipe_8bit` — kernel builders.
- `_should_use_verify(...)` — shape-based gate.

#### `recurrent_rollback_cache.py`

GDN/SSM-specific cache entry. Same interface as `mlx_lm`'s `_BaseCache`, plus an `arm_rollback`/`clear_transients` pair and a tape that `kernels.tape_replay_kernel` consumes.

- `RecurrentRollbackCache` — the cache class. Holds `_armed`, `_tape`, `_tape_k`, `_tape_g`, `_tape_qkv`, `_snapshot`, plus the `_BaseCache` fields.

#### `prefix_cache.py`

Slim re-exports of `dflash_mlx.cache.*` so older import paths keep working. New code should import from the subpackages.

- Re-exports `DFlashPrefixCache`, `DFlashPrefixKey`, `DFlashPrefixSnapshot`, `build_snapshot`, `compute_stable_prefix_len`, `hydrate_target_cache`, `prefix_cache_enabled`, `read_budget_env`, `serialize_target_cache`, `target_cache_is_serializable`.

#### `bench_logger.py`

Lazy JSONL logger gated on `DFLASH_BENCH_LOG_DIR`. When the env var is set, the runtime emits a record for each POST, each cycle, and each cache event.

- `_BenchLogger` — internal lazy-init class.
- `log_post(**fields)`, `log_cycle(**fields)`, `log_cache(**fields)` — emit one JSONL line per call.
- `enabled() -> bool` — whether logging is on.

#### `adapter.py`

Picks between `FullAttentionEngine` and `HybridGDNEngine` for a loaded target by sniffing layer attributes (`fa_idx`/`ssm_idx` markers, presence of `linear_attn`).

- `detect_engine(target_model) -> FullAttentionEngine | HybridGDNEngine`.

### User entry points

#### `generate.py`

The `dflash` CLI for one-shot generation. Reads a prompt, runs `stream_dflash_generate`, prints the decoded tokens.

The `DRAFT_REGISTRY` mapping documents which targets have a published draft:

```python
DRAFT_REGISTRY = {
    "Qwen3.5-4B": "z-lab/Qwen3.5-4B-DFlash",
    "Qwen3.5-9B": "z-lab/Qwen3.5-9B-DFlash",
    "Qwen3.5-27B": "z-lab/Qwen3.5-27B-DFlash",
    "Qwen3.5-35B-A3B": "z-lab/Qwen3.5-35B-A3B-DFlash",
    "Qwen3.6-27B": "z-lab/Qwen3.6-27B-DFlash",
    "Qwen3.6-35B-A3B": "z-lab/Qwen3.6-35B-A3B-DFlash",
    "Qwen3-4B": "z-lab/Qwen3-4B-DFlash-b16",
    "Qwen3-8B": "z-lab/Qwen3-8B-DFlash-b16",
}
```

If the user passes a target whose stripped name is in this registry, the matching draft repo is picked automatically.

Key symbols:

- `main()` — argparse entry point.
- `run_generate(...)` — the actual work; returns the summary event.
- `load_runtime_components(target_ref, draft_ref, ...)` — loads target + draft + tokenizer; reused by the server's `DFlashModelProvider`.
- `resolve_optional_draft_ref(model_ref, draft_ref)` — applies `DRAFT_REGISTRY` if `draft_ref` is None.
- `get_stop_token_ids(tokenizer)`, `decode_token(tokenizer, token_id)`, `generation_tps_from_summary(summary)` — small helpers.

#### `serve.py`

The `dflash-serve` HTTP server. Subclasses `mlx_lm.server.ResponseGenerator` and `mlx_lm.server.APIHandler` to keep full OpenAI compatibility — clients hit the same routes as plain mlx_lm:

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET /v1/models`

For short requests with `max_tokens <= 256`, the server delegates to `mlx_lm.server`'s plain autoregressive path; speculative decoding only kicks in for longer generations where the cycle overhead pays off.

Key symbols:

- `main()` — argparse + server bootstrap.
- `DFlashResponseGenerator` — overrides `_serve_single` to read target+draft from `DFlashModelProvider`, compute the stable-prefix key, hit the prefix cache, and call `dflash_mlx.runtime.stream_dflash_generate`.
- `DFlashAPIHandler` — overrides `handle_completion` to route through `DFlashResponseGenerator`.
- `_print_startup_banner(...)` — prints the boot banner indicating speculative-decoding mode.
- `_run_with_dflash_server(host, port, model_provider)` — starts the HTTP listener.
- `log_prefix_cache_stats(label="")` — convenience around `prefix_cache_manager.format_stats_line`.

### Data flow on a streaming chat completion request

1. Client POSTs to `/v1/chat/completions` with `stream=true`.
2. `DFlashAPIHandler.handle_completion` calls `mlx_lm.server.APIHandler.handle_chat_completions`, which builds the `CompletionRequest` from the JSON body and applies the chat template.
3. `DFlashResponseGenerator._serve_single` reads `target_model` and `draft_model` from `DFlashModelProvider`, computes the stable prefix length via `cache.policies.compute_stable_prefix_len`, and looks the prefix up in the `DFlashPrefixCache` keyed by `server.prefix_cache_manager.build_prefix_key`.
4. The handler calls `dflash_mlx.runtime.stream_dflash_generate(...)`, passing the prompt tokens, the prefix snapshot (if any), the suppress-token mask, and stop-token ids.
5. `stream_dflash_generate` enters `engine.spec_epoch.stream_dflash_generate_impl`, which runs the cycle: prefill (chunked, possibly hydrated from snapshot) -> draft (`draft_backend.EagerDraftBackend`) -> verify (`engine.target_verifier.verify_target_block`) -> acceptance (`engine.acceptance.match_acceptance_length`) -> rollback (`engine.rollback`) -> emit accepted tokens -> repeat.
6. Per-token `TOKEN` events flow back into `_serve_single`, which converts each one to an OpenAI-shaped SSE chunk via `server.protocol.make_response` and pushes it to the response queue. `CYCLE_COMPLETE` events feed `bench_logger.log_cycle` when bench logging is enabled.
7. On end of request, the handler builds a fresh `DFlashPrefixSnapshot` via `cache.codecs.build_snapshot` and inserts it into the prefix cache. The final SSE chunk's `usage` object reports `cached_tokens` if the cache was hit, and `metrics.write_summary_line` prints the human-readable tok/s line to stderr.
