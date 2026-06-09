# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

"""Position-aware RoPE for sparse prefill.

When the target/draft are prefilled on a *selected subset* of prompt tokens
(positional sparse prefill), the full-attention layers must still RoPE each
token at its **original** position so relative-position semantics are preserved.
mlx's ``nn.RoPE`` only rotates contiguous positions ``offset + arange(L)``; these
helpers rotate at arbitrary, non-contiguous positions while matching mlx's
non-traditional (GPT-NeoX / half-split) convention exactly.

Two wrappers are installed onto an attention module's ``.rope`` for the duration
of generation:

* ``PositionMappedRoPE`` — during prefill, maps the incoming (compacted) cache
  offset to the original positions of the selected tokens.
* ``OffsetAdjustedRoPE`` — during decode, adds a constant ``adjustment`` so the
  generated tokens continue from the true last prompt position.

Wrapping ``.rope`` (rather than the attention ``__call__``) catches every code
path that rotates queries/keys, including mlx-lm fallbacks.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

__all__ = [
    "manual_rope",
    "manual_rope_with_freqs",
    "PositionMappedRoPE",
    "OffsetAdjustedRoPE",
    "iter_attention_modules",
    "install_position_mapped_rope",
    "switch_to_offset_adjusted_rope",
    "restore_ropes",
    "decode_position_adjustment",
]


def _scalar_offset(offset: Any) -> int:
    """Coerce a scalar RoPE offset to int (some backends wrap it in mx.array)."""
    if isinstance(offset, mx.array):
        return int(offset.item())
    return int(offset)


def manual_rope(
    x: mx.array,
    positions: mx.array,
    dims: int,
    base: float = 10000.0,
    scale: float = 1.0,
) -> mx.array:
    """Apply non-traditional RoPE at arbitrary positions.

    Args:
        x: (B, n_heads, L, head_dim). Only the first ``dims`` channels are
            rotated; any tail channels pass through.
        positions: (L,) integer position indices.
        dims: number of channels to rotate (``head_dim`` for full rotary).
        base: RoPE base frequency (``rope_theta``).
        scale: position scale divisor (1.0 for un-extended context).

    Trig is computed in float32 for precision; the result is cast back to the
    input dtype so the downstream SDPA dtype contract (bf16/fp16) is preserved.
    """
    out_dtype = x.dtype
    half = dims // 2
    inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
    scaled_pos = positions.astype(mx.float32) / scale
    angles = scaled_pos[:, None] * inv_freq[None, :]
    cos_a = mx.cos(angles)[None, None, :, :]
    sin_a = mx.sin(angles)[None, None, :, :]
    x_rot, x_pass = x[..., :dims].astype(mx.float32), x[..., dims:]
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = mx.concatenate([x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], axis=-1)
    return mx.concatenate([rotated.astype(out_dtype), x_pass], axis=-1)


def manual_rope_with_freqs(
    x: mx.array,
    positions: mx.array,
    dims: int,
    freqs: mx.array,
    pre_scale: float = 1.0,
) -> mx.array:
    """Apply non-traditional RoPE at arbitrary positions using stored ``_freqs``.

    For custom RoPE variants (Llama3, Yarn, SuScaled) that precompute and store
    per-channel frequencies on the module rather than deriving them from ``base``.
    """
    out_dtype = x.dtype
    half = dims // 2
    inv_freq = (1.0 / freqs).astype(mx.float32)
    angles = positions[:, None].astype(mx.float32) * inv_freq[None, :]
    cos_a = mx.cos(angles)[None, None, :, :]
    sin_a = mx.sin(angles)[None, None, :, :]
    x_rot, x_pass = x[..., :dims].astype(mx.float32), x[..., dims:]
    if pre_scale != 1.0:
        x_rot = pre_scale * x_rot
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = mx.concatenate([x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], axis=-1)
    return mx.concatenate([rotated.astype(out_dtype), x_pass], axis=-1)


def _get_dims(rope_module: Any) -> int:
    for attr in ("_dims", "dims", "dim"):
        value = getattr(rope_module, attr, None)
        if value is not None:
            return int(value)
    raise ValueError(f"cannot determine rotary dims from {type(rope_module)!r}")


def _get_pre_scale(rope_module: Any) -> float:
    mscale = getattr(rope_module, "mscale", None)
    if mscale is not None:
        return float(mscale)
    if hasattr(rope_module, "_scale") and hasattr(rope_module, "dim"):
        return float(rope_module._scale)
    return 1.0


def _reject_traditional(rope_module: Any) -> None:
    if bool(getattr(rope_module, "traditional", False)):
        raise NotImplementedError(
            "sparse prefill does not support traditional (interleaved) RoPE; "
            "only the non-traditional half-split convention is implemented"
        )


class PositionMappedRoPE:
    """Rotate tokens at their original positions during sparse prefill.

    The wrapped attention passes ``offset = cache.offset`` (the *compacted* index
    of the first token in the current forward). That offset indexes into the full
    ``positions`` array to recover the original positions of the ``L`` tokens
    being processed::

        positions[(offset - cache_start) : (offset - cache_start) + L]
    """

    def __init__(self, original_rope: Any, positions: mx.array, cache_start: int = 0):
        _reject_traditional(original_rope)
        self._original = original_rope
        self._positions = positions
        # A python copy for cheap per-call slicing and contiguity checks.
        self._pos_list = [int(p) for p in positions.tolist()]
        self._cache_start = _scalar_offset(cache_start)
        self._has_custom_freqs = getattr(original_rope, "_freqs", None) is not None
        if self._has_custom_freqs:
            self._freqs = original_rope._freqs
            self._dims = _get_dims(original_rope)
            self._pre_scale = _get_pre_scale(original_rope)
        else:
            self._dims = _get_dims(original_rope)
            self._base = float(getattr(original_rope, "base", 10000.0))
            self._scale = float(getattr(original_rope, "scale", 1.0))

    def __call__(self, x: mx.array, offset: Any = 0) -> mx.array:
        length = int(x.shape[2])
        idx = _scalar_offset(offset) - self._cache_start
        window = self._pos_list[idx : idx + length]
        # A contiguous run [start, start+1, ...] is exactly native RoPE at
        # offset=start. Delegate so contiguous segments (including select-all)
        # are bitwise-identical to the model's own rope; manual_rope's float32
        # intermediates would otherwise drift ~1 bf16 ULP per layer and amplify.
        if window and window[-1] - window[0] == length - 1:
            return self._original(x, offset=window[0])
        positions = self._positions[idx : idx + length]
        if self._has_custom_freqs:
            return manual_rope_with_freqs(
                x, positions, self._dims, self._freqs, pre_scale=self._pre_scale
            )
        return manual_rope(
            x, positions, self._dims, base=self._base, scale=self._scale
        )


class OffsetAdjustedRoPE:
    """Shift decode positions after sparse prefill.

    After sparse prefill of ``N`` selected tokens whose last original position is
    ``P - 1`` (so the first generated token is at position ``P``), the cache holds
    ``N`` entries and the attention passes ``offset = N + i`` for decode step
    ``i``. Adding ``adjustment = P - N`` yields the true position ``P + i``.
    """

    def __init__(self, original_rope: Any, adjustment: int):
        self._original = original_rope
        self._adjustment = int(adjustment)

    def __call__(self, x: mx.array, offset: Any = 0) -> mx.array:
        return self._original(x, offset=_scalar_offset(offset) + self._adjustment)


def decode_position_adjustment(positions: "tuple[int, ...]") -> int:
    """Constant RoPE shift for decode after sparse prefill.

    ``adjustment = (last_position + 1) - num_selected`` so the first generated
    token lands at ``last_position + 1`` while the cache holds ``num_selected``
    entries. Zero when positions are dense (select-all), making the decode
    wrapper a no-op.
    """
    if not positions:
        return 0
    return (int(positions[-1]) + 1) - len(positions)


def iter_attention_modules(text_model: Any):
    """Yield the full-attention submodule of each rope-bearing layer.

    GDN / linear-attention layers (``layer.is_linear``) have no rope and are
    skipped — their position is implicit in the recurrence over the compacted
    sequence.
    """
    for layer in getattr(text_model, "layers", []):
        if getattr(layer, "is_linear", False):
            continue
        attn = getattr(layer, "self_attn", None)
        if attn is not None and getattr(attn, "rope", None) is not None:
            yield attn


def install_position_mapped_rope(
    text_model: Any,
    positions: mx.array,
    cache_start: int = 0,
) -> list[tuple[Any, Any]]:
    """Swap each attention rope to a PositionMappedRoPE; return originals."""
    saved: list[tuple[Any, Any]] = []
    for attn in iter_attention_modules(text_model):
        saved.append((attn, attn.rope))
        attn.rope = PositionMappedRoPE(attn.rope, positions, cache_start=cache_start)
    return saved


def switch_to_offset_adjusted_rope(
    saved: list[tuple[Any, Any]],
    adjustment: int,
) -> None:
    """Replace the prefill wrappers with decode OffsetAdjustedRoPE wrappers.

    ``saved`` holds the *original* ropes captured at install time, so the decode
    wrapper composes over the original rather than the prefill wrapper.
    """
    for attn, original in saved:
        attn.rope = OffsetAdjustedRoPE(original, adjustment)


def restore_ropes(saved: list[tuple[Any, Any]]) -> None:
    """Restore the original rope modules captured at install time."""
    for attn, original in saved:
        attn.rope = original
