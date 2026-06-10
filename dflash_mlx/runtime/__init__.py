# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot
    from dflash_mlx.cache.snapshot_service import SnapshotService
    from dflash_mlx.draft_backend import DraftBackend
    from dflash_mlx.engine.events import EngineEvent
    from dflash_mlx.engine.target_ops import TargetOps
    from dflash_mlx.model import DFlashDraftModel

__all__ = ["VerifyConfig", "get_stop_token_ids", "stream_dflash_generate"]


def get_stop_token_ids(tokenizer: Any) -> list[int]:
    eos_token_ids = list(getattr(tokenizer, "eos_token_ids", None) or [])
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and eos_token_id not in eos_token_ids:
        eos_token_ids.append(int(eos_token_id))
    return eos_token_ids


@dataclass(frozen=True)
class VerifyConfig:
    mode: str = "dflash"
    enable_qmm: bool = True

    @classmethod
    def from_mode(cls, mode: str | None) -> "VerifyConfig":
        resolved = (mode or "dflash").strip().lower()
        if resolved not in ("dflash", "adaptive", "ddtree", "off"):
            raise ValueError("verify mode must be dflash, adaptive, ddtree, or off")
        return cls(mode=resolved)


def stream_dflash_generate(
    *,
    target_model: Any = None,
    target_ops: TargetOps | None = None,
    tokenizer: Any = None,
    draft_model: DFlashDraftModel | None = None,
    draft_backend: DraftBackend | None = None,
    prompt: str = "",
    max_new_tokens: int = 0,
    use_chat_template: bool = False,
    block_tokens: int | None = None,
    stop_token_ids: list[int] | None = None,
    suppress_token_ids: list[int] | None = None,
    prompt_tokens_override: list[int] | None = None,
    prompt_token_positions: list[int] | None = None,
    quantize_kv_cache: bool = False,
    prefix_snapshot: DFlashPrefixSnapshot | None = None,
    snapshot_service: SnapshotService | None = None,
    stable_prefix_len: int | None = None,
    prefix_cache_active: bool = False,
    publish_generation_snapshot: bool = True,
    prefix_hit_kind: str = "miss",
    runtime_context: Any = None,
) -> Iterator[EngineEvent]:
    if runtime_context is None:
        raise ValueError("runtime_context is required")
    if target_ops is None:
        raise ValueError("target_ops is required")
    if draft_backend is None:
        raise ValueError("draft_backend is required")
    import mlx.core as mx

    from dflash_mlx.engine.spec_epoch import (
        stream_dflash_generate_impl as _stream_dflash_generate_impl,
    )

    gen_stream = mx.default_stream(mx.default_device())
    with mx.stream(gen_stream):
        inner = _stream_dflash_generate_impl(
            target_model=target_model,
            target_ops=target_ops,
            tokenizer=tokenizer,
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            use_chat_template=use_chat_template,
            block_tokens=block_tokens,
            stop_token_ids=stop_token_ids,
            suppress_token_ids=suppress_token_ids,
            prompt_tokens_override=prompt_tokens_override,
            prompt_token_positions=prompt_token_positions,
            quantize_kv_cache=quantize_kv_cache,
            prefix_snapshot=prefix_snapshot,
            snapshot_service=snapshot_service,
            stable_prefix_len=stable_prefix_len,
            prefix_cache_active=prefix_cache_active,
            publish_generation_snapshot=publish_generation_snapshot,
            prefix_hit_kind=prefix_hit_kind,
            runtime_context=runtime_context,
        )
        # This wrapper frame lives for the whole generation; the snapshot is
        # bound into the inner generator, so drop the local ref here too.
        prefix_snapshot = None
        yield from inner
