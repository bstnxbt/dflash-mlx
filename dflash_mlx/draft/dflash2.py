# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3 import MLP

from dflash_mlx.model import (
    DFlashAttention,
    DFlashDraftModel,
    DFlashDraftModelArgs,
    DraftRuntimeCapabilities,
)

_DFLASH2_ARCHITECTURE = "DFlash2DraftModel"
_DFLASH2_REQUIRED_DFLASH_FIELDS = frozenset(
    (
        "block_size",
        "conv_group_size",
        "conv_kernel_size",
        "mask_token_id",
        "selector_rank",
        "selector_top_k",
        "target_layer_ids",
    )
)


@dataclass(frozen=True)
class DraftProposal:
    token_ids: mx.array
    q_token_ids: Optional[mx.array] = None
    q_probs: Optional[mx.array] = None


def has_dflash2_architecture(config: dict[str, Any]) -> bool:
    return _DFLASH2_ARCHITECTURE in tuple(str(v) for v in config.get("architectures") or ())


def normalize_dflash2_config(config: dict[str, Any]) -> dict[str, Any]:
    data = dict(config)
    if not has_dflash2_architecture(data):
        raise ValueError("DFlash2 draft config must declare architectures=['DFlash2DraftModel']")
    dflash_config = data.get("dflash_config")
    if not isinstance(dflash_config, dict):
        raise ValueError("DFlash2 draft config requires object dflash_config")
    missing = sorted(_DFLASH2_REQUIRED_DFLASH_FIELDS - set(dflash_config))
    if missing:
        raise ValueError(
            "DFlash2 draft config missing dflash_config field(s): "
            + ", ".join(missing)
        )
    if data.get("is_causal") is not False:
        raise ValueError("DFlash2 draft config requires is_causal=false")
    layer_types = tuple(data.get("layer_types") or ())
    if not layer_types:
        raise ValueError("DFlash2 draft config requires layer_types")
    if len(layer_types) != int(data["num_hidden_layers"]):
        raise ValueError(
            "DFlash2 draft layer_types length must match num_hidden_layers: "
            f"{len(layer_types)} != {int(data['num_hidden_layers'])}"
        )
    if set(layer_types) != {"sliding_attention"}:
        raise ValueError("DFlash2 MLX support requires all draft layers to be sliding_attention")
    rope_parameters = data.get("rope_parameters")
    rope_scaling = data.get("rope_scaling")
    rope = rope_parameters or rope_scaling or {}
    if "rope_theta" not in data:
        data["rope_theta"] = rope.get("rope_theta", 10000.0)
    if rope_scaling is None and isinstance(rope_parameters, dict):
        rope_type = (
            rope_parameters.get("type")
            or rope_parameters.get("rope_type")
            or "default"
        )
        data["rope_scaling"] = (
            None
            if rope_type == "default"
            else {
                key: value
                for key, value in rope_parameters.items()
                if key != "rope_theta"
            }
        )
    for key, value in dflash_config.items():
        data[key] = value
    return data


def remap_dflash2_codebook_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    remapped = dict(weights)
    for name in ("predecessor_codebook", "successor_codebook"):
        old_key = f"candidate_selector.{name}"
        new_key = f"{old_key}.weight"
        if old_key not in remapped:
            continue
        if new_key in remapped:
            raise ValueError(
                f"DFlash2 safetensors contain both {old_key!r} and {new_key!r}"
            )
        remapped[new_key] = remapped.pop(old_key)
    return remapped


@dataclass
class DFlash2DraftModelArgs(DFlashDraftModelArgs):
    conv_kernel_size: int = 2
    conv_group_size: int = 16
    selector_rank: int = 256
    selector_top_k: int = 16
    is_causal: bool = False
    final_logit_softcapping: Optional[float] = None
    input_embedding_scale: float = 1.0
    output_multiplier: float = 1.0

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "DFlash2DraftModelArgs":
        data = normalize_dflash2_config(params)
        names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in names})

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.is_causal is not False:
            raise ValueError("DFlash2 draft attention must be non-causal")
        if int(self.conv_kernel_size) != 2:
            raise ValueError("DFlash2 MLX support requires two-tap dynamic conv")
        if int(self.conv_group_size) != 16:
            raise ValueError("DFlash2 MLX support requires dynamic conv group size 16")
        if int(self.selector_rank) != 256:
            raise ValueError("DFlash2 MLX support requires selector rank 256")
        if int(self.selector_top_k) != 16:
            raise ValueError("DFlash2 MLX support requires selector_top_k 16")
        if int(self.hidden_size) % int(self.conv_group_size) != 0:
            raise ValueError("DFlash2 hidden_size must be divisible by conv_group_size")
        if set(self.layer_types) != {"sliding_attention"}:
            raise ValueError("DFlash2 MLX support requires all draft layers to be sliding_attention")


def _grouped_dynamic_convolve(
    hidden: mx.array,
    dynamic: mx.array,
    base: mx.array,
    group_size: int,
) -> mx.array:
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // int(group_size)
    blocks = hidden.reshape(batch, length, groups, int(group_size))
    dynamic = dynamic.reshape(batch, length, int(base.shape[0]), groups, 1)
    output = mx.zeros_like(blocks)
    for offset in range(int(base.shape[0])):
        if offset == 0:
            values = blocks
        else:
            values = mx.concatenate(
                (mx.zeros_like(blocks[:, :offset]), blocks[:, :-offset]),
                axis=1,
            )
        kernel = base[offset].reshape(1, 1, groups, int(group_size)).astype(hidden.dtype)
        output = output + kernel * values
        output = output + dynamic[:, :, offset] * values
    return output.reshape(hidden.shape)


class GroupedDynamicCausalConv(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, group_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.group_size = int(group_size)
        groups = int(hidden_size) // self.group_size
        self.base_kernel = mx.zeros((2, self.kernel_size, int(hidden_size)))
        self.kernel_projection = nn.Linear(
            int(hidden_size),
            2 * self.kernel_size * groups,
            bias=False,
        )

    def prepare(self, hidden: mx.array) -> tuple[mx.array, mx.array]:
        groups = int(hidden.shape[-1]) // self.group_size
        dynamic = self.kernel_projection(hidden).reshape(
            *hidden.shape[:-1],
            2,
            self.kernel_size,
            groups,
        )
        return (
            _grouped_dynamic_convolve(
                hidden,
                dynamic[..., 0, :, :],
                self.base_kernel[0],
                self.group_size,
            ),
            dynamic[..., 1, :, :],
        )

    def finish(self, hidden: mx.array, dynamic: mx.array) -> mx.array:
        return _grouped_dynamic_convolve(
            hidden,
            dynamic,
            self.base_kernel[1],
            self.group_size,
        )


class DFlash2Attention(DFlashAttention):
    def __init__(self, args: DFlash2DraftModelArgs, layer_idx: int):
        super().__init__(args, layer_idx)
        self.is_causal = bool(args.is_causal)

    def _attention_mask(
        self,
        *,
        block_len: int,
        query_offset: int,
        key_len: int,
        key_positions: Optional[mx.array] = None,
    ) -> Optional[mx.array]:
        if self.sliding_window is None and not self.is_causal:
            return None
        query_positions = mx.arange(
            query_offset,
            query_offset + int(block_len),
            dtype=mx.int32,
        )
        if key_positions is None:
            key_start = query_offset + int(block_len) - int(key_len)
            key_positions = mx.arange(key_start, key_start + int(key_len), dtype=mx.int32)
        if self.sliding_window is None:
            return key_positions[None, :] <= query_positions[:, None]
        context = (key_positions[None, :] < int(query_offset)) & (
            query_positions[:, None] - key_positions[None, :] < int(self.sliding_window)
        )
        block = key_positions[None, :] >= int(query_offset)
        if self.is_causal:
            block = block & (key_positions[None, :] <= query_positions[:, None])
        return context | block


class DFlash2DecoderLayer(nn.Module):
    def __init__(self, args: DFlash2DraftModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = DFlash2Attention(args, layer_idx)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attention_conv = GroupedDynamicCausalConv(
            args.hidden_size,
            args.conv_kernel_size,
            args.conv_group_size,
        )
        self.mlp_conv = GroupedDynamicCausalConv(
            args.hidden_size,
            args.conv_kernel_size,
            args.conv_group_size,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        target_hidden: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states, dynamic = self.attention_conv.prepare(
            self.input_layernorm(hidden_states)
        )
        hidden_states = residual + self.attention_conv.finish(
            self.self_attn(
                hidden_states,
                target_hidden=target_hidden,
                cache=cache,
            ),
            dynamic,
        )
        residual = hidden_states
        hidden_states, dynamic = self.mlp_conv.prepare(
            self.post_attention_layernorm(hidden_states)
        )
        return residual + self.mlp_conv.finish(self.mlp(hidden_states), dynamic)

    def advance_projected_context_cache(
        self,
        *,
        target_hidden: mx.array,
        cache: Any,
    ) -> None:
        self.self_attn.append_projected_context_cache(
            target_hidden=target_hidden,
            cache=cache,
        )


def _sample_probs(probs: mx.array) -> mx.array:
    return mx.random.categorical(mx.log(probs))


def _selector_probabilities(scores: mx.array, temperature: float) -> mx.array:
    if float(temperature) <= 0:
        raise ValueError("DFlash2 selector temperature must be positive")
    return mx.softmax(scores.astype(mx.float32) / float(temperature), axis=-1)


@dataclass(frozen=True)
class _CandidateSelection:
    token_ids: mx.array
    candidate_ids: mx.array
    probabilities: Optional[mx.array]


class CandidateSelector(nn.Module):
    def __init__(self, args: DFlash2DraftModelArgs):
        super().__init__()
        self.top_k = int(args.selector_top_k)
        self.predecessor_codebook = nn.Embedding(args.vocab_size, int(args.selector_rank))
        self.successor_codebook = nn.Embedding(args.vocab_size, int(args.selector_rank))
        self.hidden_projection = nn.Linear(args.hidden_size, int(args.selector_rank), bias=False)

    def _edge_scores(
        self,
        predecessor_ids: mx.array,
        successor_vectors: mx.array,
        hidden: mx.array,
    ) -> mx.array:
        return mx.sum(
            self.predecessor_codebook(predecessor_ids)[:, :, None]
            * hidden[:, None, None]
            * successor_vectors[:, None],
            axis=-1,
        )

    def _select_ancestral(
        self,
        *,
        candidates: mx.array,
        unary: mx.array,
        hidden: mx.array,
        anchor_ids: mx.array,
        successors: mx.array,
        temperature: float,
        capture_q: bool,
    ) -> _CandidateSelection:
        predecessor = anchor_ids
        path = []
        q_rows = []
        for position in range(int(hidden.shape[1])):
            edges = self._edge_scores(
                predecessor[:, None],
                successors[:, position],
                hidden[:, position],
            )[:, 0]
            scores = unary[:, position] + edges
            if float(temperature) > 0 or capture_q:
                q = _selector_probabilities(
                    scores,
                    float(temperature) if float(temperature) > 0 else 1.0,
                )
                q_rows.append(q)
            if float(temperature) > 0:
                selected = _sample_probs(q)
            else:
                selected = mx.argmax(scores, axis=-1)
            predecessor = mx.take_along_axis(
                candidates[:, position],
                selected[:, None],
                axis=-1,
            )[:, 0]
            path.append(predecessor)
        sparse_q = mx.stack(q_rows, axis=1) if q_rows else None
        return _CandidateSelection(
            token_ids=mx.stack(path, axis=1),
            candidate_ids=candidates,
            probabilities=sparse_q,
        )

    def select(
        self,
        hidden: mx.array,
        logits: mx.array,
        anchor_ids: mx.array,
        temperature: float,
        capture_q: bool = False,
    ) -> _CandidateSelection:
        if len(hidden.shape) != 3 or len(logits.shape) != 3:
            raise ValueError("DFlash2 selector requires 3D hidden and logits")
        if tuple(hidden.shape[:2]) != tuple(logits.shape[:2]):
            raise ValueError("DFlash2 selector hidden/logits batch and length must match")
        if int(logits.shape[-1]) < self.top_k:
            raise ValueError("DFlash2 selector vocabulary must be at least top_k")
        anchor_ids = anchor_ids.reshape(-1)
        if int(anchor_ids.shape[0]) != int(hidden.shape[0]):
            raise ValueError("DFlash2 selector anchor batch must match hidden batch")

        candidates = mx.argpartition(logits, -self.top_k, axis=-1)[..., -self.top_k :]
        unary = mx.take_along_axis(logits, candidates, axis=-1)
        hidden = self.hidden_projection(hidden)
        successors = self.successor_codebook(candidates)
        return self._select_ancestral(
            candidates=candidates,
            unary=unary,
            hidden=hidden,
            anchor_ids=anchor_ids,
            successors=successors,
            temperature=float(temperature),
            capture_q=bool(capture_q),
        )


class DFlash2DraftModel(DFlashDraftModel):
    def __init__(self, args: DFlash2DraftModelArgs):
        super().__init__(args)
        self.model_type = "dflash2_qwen3"
        self.args = args
        self.candidate_selector = CandidateSelector(args)
        self.capabilities = DraftRuntimeCapabilities(
            default_block_tokens=5,
            max_block_tokens=min(5, int(args.block_size)),
            supports_copyspec=False,
            supports_ddtree=False,
            supports_early_rollback_launch=True,
        )

    def _build_layers(self, args: DFlashDraftModelArgs) -> list[nn.Module]:
        if not isinstance(args, DFlash2DraftModelArgs):
            raise TypeError("DFlash2 layer construction requires DFlash2 arguments")
        return [
            DFlash2DecoderLayer(args, layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]

    def bind_target_model(self, target_model: Any, *, target_ops: Any) -> None:
        super().bind_target_model(target_model, target_ops=target_ops)
        self.embed_scale *= float(self.args.input_embedding_scale)

    def compute_logits(self, logits: mx.array) -> mx.array:
        logits = logits * float(self.args.output_multiplier)
        cap = self.args.final_logit_softcapping
        if cap is not None and float(cap) > 0:
            cap_value = float(cap)
            logits = mx.tanh(logits / cap_value) * cap_value
        return logits

    def select_proposal(
        self,
        *,
        draft_hidden: mx.array,
        logits: mx.array,
        anchor_ids: mx.array,
        temperature: float = 0.0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        capture_q: bool = False,
    ) -> DraftProposal:
        del top_p, min_p, top_k
        proposal = self.candidate_selector.select(
            draft_hidden,
            self.compute_logits(logits),
            anchor_ids,
            temperature,
            capture_q=capture_q,
        )
        return DraftProposal(
            token_ids=proposal.token_ids.squeeze(0),
            q_token_ids=(
                proposal.candidate_ids.squeeze(0)
                if proposal.probabilities is not None
                else None
            ),
            q_probs=(
                proposal.probabilities.squeeze(0)
                if proposal.probabilities is not None
                else None
            ),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        return remap_dflash2_codebook_weights(weights)
