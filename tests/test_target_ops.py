# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.models.cache import KVCache, QuantizedKVCache, RotatingKVCache

from dflash_mlx.engine import target_qwen_gdn as qwen_gdn
from dflash_mlx.engine.target_ops import bind_draft_to_target, resolve_target_ops
from dflash_mlx.engine.target_gemma4 import Gemma4TargetOps
from dflash_mlx.engine.target_qwen_gdn import QwenGdnTargetOps

class _FakeLinearAttn:
    conv_kernel_size = 4

class _FakeGdnLayer:
    is_linear = True
    linear_attn = _FakeLinearAttn()

class _FakeFaLayer:
    is_linear = False
    self_attn = object()

class _FakeEmbed:
    pass

class _FakeTarget:
    def __init__(
        self,
        *,
        model_type: str = "qwen3_5",
        num_hidden_layers: int = 64,
        hidden_size: int | None = 5120,
        num_attention_heads: int | None = 40,
        num_key_value_heads: int | None = 8,
        num_experts: int = 0,
    ) -> None:
        self.args = SimpleNamespace(
            model_type=model_type,
            tie_word_embeddings=True,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            num_experts=num_experts,
        )
        self.model = SimpleNamespace(
            layers=[_FakeFaLayer(), _FakeGdnLayer(), _FakeFaLayer()],
            embed_tokens=_FakeEmbed(),
        )

class _WrongOuterEmbed:
    def as_linear(self, hidden: mx.array) -> mx.array:
        del hidden
        return mx.array([[[1.0, 0.0]]])

class _RealQwenLmHead:
    def __call__(self, hidden: mx.array) -> mx.array:
        del hidden
        return mx.array([[[0.0, 1.0]]])

def _fake_qwen_full_attention(*, q_heads: int = 4, kv_heads: int = 1, head_dim: int = 256):
    class _Proj:
        def __init__(self, out_dim: int) -> None:
            self.out_dim = int(out_dim)
            self.weight = mx.zeros((self.out_dim, 1), dtype=mx.float32)

        def __call__(self, x: mx.array) -> mx.array:
            return mx.zeros((x.shape[0], x.shape[1], self.out_dim), dtype=x.dtype)

    class _Norm:
        def __init__(self) -> None:
            self.weight = mx.zeros((head_dim,), dtype=mx.float32)

        def __call__(self, x: mx.array) -> mx.array:
            return x

    class _Cache:
        offset = 4096

        def update_and_fetch(self, keys: mx.array, values: mx.array):
            return keys, values

    class _Attention:
        num_attention_heads = q_heads
        num_key_value_heads = kv_heads
        scale = 1.0

        def __init__(self) -> None:
            self.q_proj = _Proj(2 * q_heads * head_dim)
            self.k_proj = _Proj(kv_heads * head_dim)
            self.v_proj = _Proj(kv_heads * head_dim)
            self.q_norm = _Norm()
            self.k_norm = _Norm()
            self.original_calls = 0

        def rope(self, x: mx.array, offset=None) -> mx.array:
            del offset
            return x

        def o_proj(self, x: mx.array) -> mx.array:
            return x

        def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
            del mask, cache
            self.original_calls += 1
            return mx.full(
                (x.shape[0], x.shape[1], q_heads * head_dim),
                7.0,
                dtype=x.dtype,
            )

    return _Attention(), _Cache()

class _FakePureAttentionTarget:
    def __init__(self, *, model_type: str = "qwen3") -> None:
        self.args = SimpleNamespace(model_type=model_type, tie_word_embeddings=True)
        self.model = SimpleNamespace(
            layers=[_FakeFaLayer(), _FakeFaLayer()],
            embed_tokens=_FakeEmbed(),
        )

class _FakeGemmaInner:
    embed_tokens = _FakeEmbed()

    def __init__(self, layer_types: list[str]) -> None:
        self.layers = [SimpleNamespace(layer_type=layer_type) for layer_type in layer_types]

class _FakeGemmaWrapper:
    def __init__(self, *, layer_types: list[str] | None = None) -> None:
        layer_types = layer_types or [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]
        self.args = SimpleNamespace(
            model_type="gemma4_text",
            layer_types=layer_types,
            sliding_window=1024,
            num_kv_shared_layers=0,
            tie_word_embeddings=True,
        )
        self.model = _FakeGemmaInner(layer_types)

    def make_cache(self):
        caches = []
        for layer_type in self.args.layer_types:
            if layer_type == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window, keep=0))
        return caches

class _FakeGemmaTarget:
    def __init__(self) -> None:
        self.args = SimpleNamespace(model_type="gemma4")
        self.language_model = _FakeGemmaWrapper()

class _FakeDraft:
    def __init__(self) -> None:
        self.bound = []

    def bind_target_model(self, target_model, *, target_ops):
        self.bound.append((target_model, target_ops))

class _FakeAsLinearEmbed:
    def as_linear(self, hidden):
        return hidden + 2.0

class _FakeLmHead:
    def __call__(self, hidden):
        return (hidden * 3.0) - 1.0

class _FakeGemmaLogitInner:
    embed_tokens = _FakeAsLinearEmbed()

class _FakeGemmaLogitWrapper:
    def __init__(
        self,
        *,
        tied: bool,
        softcap: float | None = None,
        args_softcap: float | None = None,
    ) -> None:
        self.args = SimpleNamespace(
            model_type="gemma4_text",
            layer_types=["full_attention"],
            final_logit_softcapping=args_softcap,
        )
        self.model = _FakeGemmaLogitInner()
        self.tie_word_embeddings = tied
        self.lm_head = _FakeLmHead()
        if softcap is not None:
            self.final_logit_softcapping = softcap

class _FakeGemmaLogitTarget:
    def __init__(
        self,
        *,
        tied: bool,
        softcap: float | None = None,
        args_softcap: float | None = None,
    ) -> None:
        self.args = SimpleNamespace(model_type="gemma4")
        self.language_model = _FakeGemmaLogitWrapper(
            tied=tied,
            softcap=softcap,
            args_softcap=args_softcap,
        )

def _assert_close(actual, expected, *, atol: float = 1e-6) -> None:
    mx.eval(actual, expected)
    assert float(mx.max(mx.abs(actual - expected)).item()) <= atol

def test_resolver_selects_qwen_ops_for_current_target_shape():
    ops = resolve_target_ops(_FakeTarget())

    assert isinstance(ops, QwenGdnTargetOps)
    assert ops.family(_FakeTarget()) == "hybrid_gdn"

def test_resolver_selects_gemma4_ops_for_gemma4_text_shape():
    ops = resolve_target_ops(_FakeGemmaTarget())

    assert isinstance(ops, Gemma4TargetOps)
    assert ops.family(_FakeGemmaTarget()) == "gemma4_swa"

def test_bind_draft_to_target_invokes_optional_target_binding():
    target = _FakeGemmaTarget()
    target_ops = Gemma4TargetOps()
    draft = _FakeDraft()

    returned = bind_draft_to_target(draft, target, target_ops=target_ops)

    assert returned is target_ops
    assert draft.bound == [(target, target_ops)]

def test_resolver_rejects_known_unsupported_family_markers():
    for model_type in ("llama", "mistral", "olmo"):
        target = _FakeTarget(model_type=model_type)

        try:
            resolve_target_ops(target)
        except NotImplementedError as exc:
            message = str(exc)
            assert f"model_type={model_type}" in message
            assert "model_class=_FakeTarget" in message
            assert "supported target backends: qwen_gdn, gemma4" in message
        else:
            raise AssertionError(f"{model_type} marker must not resolve to Qwen ops")

def test_resolver_rejects_unknown_model_type():
    try:
        resolve_target_ops(_FakeTarget(model_type="unknown_arch"))
    except NotImplementedError as exc:
        assert "model_type=unknown_arch" in str(exc)
    else:
        raise AssertionError("unknown non-empty model_type must not resolve to Qwen ops")

def test_resolver_rejects_empty_model_type_even_with_qwen_like_shape():
    try:
        resolve_target_ops(_FakeTarget(model_type=""))
    except NotImplementedError as exc:
        assert "model_type=unknown" in str(exc)
    else:
        raise AssertionError("empty model_type must not resolve to Qwen ops")

def test_qwen_text_wrapper_prefers_language_model_over_outer_model():
    real_text_wrapper = SimpleNamespace(
        args=SimpleNamespace(tie_word_embeddings=False),
        model=SimpleNamespace(layers=[_FakeFaLayer()], embed_tokens=_FakeEmbed()),
        lm_head=_RealQwenLmHead(),
    )
    outer_wrapper = SimpleNamespace(
        model=SimpleNamespace(
            layers=[_FakeFaLayer()],
            embed_tokens=_WrongOuterEmbed(),
        ),
        language_model=real_text_wrapper,
    )

    ops = QwenGdnTargetOps()
    hidden = mx.zeros((1, 1, 1))
    logits = ops.logits_from_hidden(outer_wrapper, hidden)
    expected = real_text_wrapper.lm_head(hidden)
    mx.eval(logits, expected)

    assert ops.text_wrapper(outer_wrapper) is real_text_wrapper
    assert int(mx.argmax(logits[:, -1, :], axis=-1).item()) == int(
        mx.argmax(expected[:, -1, :], axis=-1).item()
    )

def test_capabilities_for_distinguishes_hybrid_and_pure_attention():
    ops = QwenGdnTargetOps()

    hybrid = ops.capabilities_for(_FakeTarget())
    pure = ops.capabilities_for(_FakePureAttentionTarget())

    assert hybrid.supports_recurrent_rollback is True
    assert hybrid.supports_dflash is True
    assert hybrid.supports_prefix_snapshot is True
    assert hybrid.supports_tree_verify is True
    assert hybrid.supports_full_context_draft_layers is True
    assert pure.supports_full_context_draft_layers is True
    assert pure.supports_recurrent_rollback is False
    assert pure.supports_dflash is True
    assert pure.supports_kv_trim is True
    assert pure.supports_rotating_cache_snapshot is False
    assert pure.supports_shared_kv is False
    assert pure.supports_tree_verify is True

def test_qwen_verify_gqa_route_is_internal_default(monkeypatch):
    attn, cache = _fake_qwen_full_attention()
    qwen_gdn._install_full_attention_gqa_hook(attn)
    calls: list[tuple[int, int, int]] = []

    def _fake_gqa(queries, keys, values, *, cache, scale, mask):
        del values, cache, scale, mask
        calls.append((int(queries.shape[2]), int(keys.shape[2]), int(queries.shape[-1])))
        return mx.ones(queries.shape, dtype=queries.dtype)

    monkeypatch.setattr(qwen_gdn, "_gqa_reshape_sdpa", _fake_gqa)

    out = attn(mx.zeros((1, 4, 32), dtype=mx.bfloat16), mask="causal", cache=cache)
    mx.eval(out)

    assert calls == [(4, 4, 256)]
    assert attn.original_calls == 0

def test_qwen_prefill_uses_original_attention(monkeypatch):
    attn, cache = _fake_qwen_full_attention()
    qwen_gdn._install_full_attention_gqa_hook(attn)

    def _fail_gqa(*args, **kwargs):
        del args, kwargs
        raise AssertionError("q_len > 16 must not enter verify GQA route")

    monkeypatch.setattr(qwen_gdn, "_gqa_reshape_sdpa", _fail_gqa)

    out = attn(mx.zeros((1, 32, 32), dtype=mx.bfloat16), mask="causal", cache=cache)
    mx.eval(out)

    assert attn.original_calls == 1
    assert out.shape == (1, 32, 1024)

def test_qwen_quantized_kv_cache_uses_original_attention(monkeypatch):
    attn, _cache = _fake_qwen_full_attention()
    quantized_cache = QuantizedKVCache(group_size=64, bits=8)
    quantized_cache.offset = 4096
    qwen_gdn._install_full_attention_gqa_hook(attn)

    def _fail_gqa(*args, **kwargs):
        del args, kwargs
        raise AssertionError("quantized target KV must not enter verify GQA route")

    monkeypatch.setattr(qwen_gdn, "_gqa_reshape_sdpa", _fail_gqa)

    out = attn(mx.zeros((1, 4, 32), dtype=mx.bfloat16), mask="causal", cache=quantized_cache)
    mx.eval(out)

    assert attn.original_calls == 1
    assert out.shape == (1, 4, 1024)

def test_qwen_tree_cache_support_rejects_rotating_and_quantized_kv():
    ops = QwenGdnTargetOps()

    assert ops.supports_tree_cache([KVCache()]) is True
    assert ops.supports_tree_cache([RotatingKVCache(max_size=4)]) is False
    assert ops.supports_tree_cache([QuantizedKVCache(group_size=64, bits=8)]) is False


def test_qwen_capabilities_own_qwen_shape_policy():
    ops = QwenGdnTargetOps()

    excluded_moe = ops.capabilities_for(
        _FakeTarget(
            num_hidden_layers=40,
            hidden_size=2048,
            num_attention_heads=16,
            num_key_value_heads=2,
            num_experts=128,
        )
    )
    supported_moe = ops.capabilities_for(
        _FakeTarget(
            num_hidden_layers=40,
            hidden_size=5120,
            num_attention_heads=40,
            num_key_value_heads=8,
            num_experts=128,
        )
    )
    small_dense = ops.capabilities_for(_FakeTarget(num_hidden_layers=32))

    assert excluded_moe.supports_verify_linear is False
    assert supported_moe.supports_verify_linear is True
    assert small_dense.supports_verify_linear is False


def test_qwen_verify_linear_capability_rejects_incomplete_moe_shape():
    ops = QwenGdnTargetOps()

    try:
        ops.capabilities_for(
            _FakeTarget(
                num_hidden_layers=40,
                hidden_size=None,
                num_attention_heads=16,
                num_key_value_heads=2,
                num_experts=128,
            )
        )
    except ValueError as exc:
        assert "Missing Qwen target config field hidden_size" in str(exc)
    else:
        raise AssertionError("incomplete Qwen MoE verify config must fail closed")

def test_gemma4_capabilities_enable_prefix_snapshot_without_shared_kv():
    caps = Gemma4TargetOps().capabilities_for(_FakeGemmaTarget())

    assert caps.supports_dflash is True
    assert caps.supports_recurrent_rollback is False
    assert caps.supports_kv_trim is True
    assert caps.supports_prefix_snapshot is True
    assert caps.supports_rotating_cache_snapshot is True
    assert caps.supports_shared_kv is False
    assert caps.supports_target_hidden_capture is True
    assert caps.supports_verify_linear is True
    assert caps.supports_tree_verify is False


def test_gemma4_capabilities_disable_prefix_snapshot_with_shared_kv():
    target = _FakeGemmaTarget()
    target.language_model.args.num_kv_shared_layers = 2
    caps = Gemma4TargetOps().capabilities_for(target)

    assert caps.supports_prefix_snapshot is False
    assert caps.supports_rotating_cache_snapshot is False
    assert caps.supports_shared_kv is True


def test_gemma4_capabilities_disable_prefix_snapshot_when_shared_kv_unknown():
    target = _FakeGemmaTarget()
    delattr(target.language_model.args, "num_kv_shared_layers")
    caps = Gemma4TargetOps().capabilities_for(target)

    assert caps.supports_prefix_snapshot is False
    assert caps.supports_rotating_cache_snapshot is False
    assert caps.supports_shared_kv is False


def test_gemma4_capabilities_disable_prefix_snapshot_when_shared_kv_malformed():
    for raw in ("unknown", "0", 0.0, False):
        target = _FakeGemmaTarget()
        target.language_model.args.num_kv_shared_layers = raw
        caps = Gemma4TargetOps().capabilities_for(target)

        assert caps.supports_prefix_snapshot is False
        assert caps.supports_rotating_cache_snapshot is False
        assert caps.supports_shared_kv is False


def test_gemma4_make_cache_matches_official_shared_kv_cache_types():
    from mlx_lm.models import gemma4_text

    args = gemma4_text.ModelArgs(
        hidden_size=16,
        num_hidden_layers=6,
        intermediate_size=32,
        num_attention_heads=2,
        head_dim=8,
        global_head_dim=8,
        num_key_value_heads=1,
        num_kv_shared_layers=2,
        vocab_size=128,
        vocab_size_per_layer_input=128,
        hidden_size_per_layer_input=0,
        sliding_window=16,
        sliding_window_pattern=3,
        use_double_wide_mlp=False,
        attention_k_eq_v=False,
        final_logit_softcapping=None,
        tie_word_embeddings=True,
    )
    official = gemma4_text.Model(args)
    target = SimpleNamespace(
        args=SimpleNamespace(model_type="gemma4"),
        language_model=official,
    )

    actual = Gemma4TargetOps().make_cache(
        target,
        enable_speculative_linear_cache=True,
        quantize_kv_cache=False,
        target_fa_window=0,
    )
    expected = official.make_cache()

    assert len(actual) == len(expected) == 4
    assert [type(cache) for cache in actual] == [type(cache) for cache in expected]
    assert [type(cache).__name__ for cache in actual] == [
        "RotatingKVCache",
        "RotatingKVCache",
        "KVCache",
        "RotatingKVCache",
    ]


def test_gemma4_full_attention_gqa_hook_is_internal():
    from mlx_lm.models import gemma4_text

    args = gemma4_text.ModelArgs(
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=2,
        head_dim=8,
        global_head_dim=8,
        num_key_value_heads=1,
        num_kv_shared_layers=0,
        vocab_size=128,
        vocab_size_per_layer_input=128,
        hidden_size_per_layer_input=0,
        sliding_window=16,
        layer_types=("sliding_attention", "full_attention"),
        use_double_wide_mlp=False,
        attention_k_eq_v=False,
        final_logit_softcapping=None,
        tie_word_embeddings=True,
    )
    model = gemma4_text.Model(args)
    target = SimpleNamespace(
        args=SimpleNamespace(model_type="gemma4"),
        language_model=model,
    )

    ops = Gemma4TargetOps()
    ops.install_speculative_hooks(target)

    sliding_attn = model.model.layers[0].self_attn
    full_attn = model.model.layers[1].self_attn
    assert not hasattr(sliding_attn, "_dflash_split_sdpa_enabled")
    assert not hasattr(full_attn, "_dflash_split_sdpa_enabled")
    assert getattr(type(full_attn), "_dflash_full_attention_gqa_installed") == "gemma4"

def test_gemma4_logits_use_tied_embedding_head():
    target = _FakeGemmaLogitTarget(tied=True)
    hidden = mx.array([[[1.0, -2.0, 3.0]]])

    logits = Gemma4TargetOps().logits_from_hidden(target, hidden)
    expected = target.language_model.model.embed_tokens.as_linear(hidden)

    _assert_close(logits, expected)

def test_gemma4_logits_use_untied_lm_head_for_a4b_style_target():
    target = _FakeGemmaLogitTarget(tied=False)
    hidden = mx.array([[[1.0, -2.0, 3.0]]])

    logits = Gemma4TargetOps().logits_from_hidden(target, hidden)
    expected = target.language_model.lm_head(hidden)

    _assert_close(logits, expected)

def test_gemma4_logits_apply_final_softcap_after_native_projection():
    target = _FakeGemmaLogitTarget(tied=False, softcap=2.5)
    hidden = mx.array([[[4.0, -4.0, 0.25]]])

    native_logits = target.language_model.lm_head(hidden)
    logits = Gemma4TargetOps().logits_from_hidden(target, hidden)
    expected = mx.tanh(native_logits / 2.5) * 2.5

    _assert_close(logits, expected)

def test_gemma4_logits_fall_back_to_args_final_softcap():
    target = _FakeGemmaLogitTarget(tied=False, args_softcap=2.5)
    hidden = mx.array([[[4.0, -4.0, 0.25]]])

    native_logits = target.language_model.lm_head(hidden)
    logits = Gemma4TargetOps().logits_from_hidden(target, hidden)
    expected = mx.tanh(native_logits / 2.5) * 2.5

    _assert_close(logits, expected)
