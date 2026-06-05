# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import mlx.core as mx

from dflash_mlx.cache.snapshot import TargetHiddenChunks
from dflash_mlx.engine.prefill import init_target_hidden_from_snapshot


@dataclass
class TargetFeatureStore:
    prompt_len: int
    project_context: Callable[[mx.array], mx.array] | None = None
    _current_hidden: mx.array | TargetHiddenChunks | None = None
    _prefill_hidden_for_snapshot: mx.array | TargetHiddenChunks | None = None
    _generation_chunks: list[mx.array] = field(default_factory=list)

    @property
    def current_hidden(self) -> mx.array | TargetHiddenChunks | None:
        return self._current_hidden

    def require_current_hidden(self) -> mx.array | TargetHiddenChunks:
        if self._current_hidden is None:
            raise RuntimeError("target hidden features are unavailable")
        return self._current_hidden

    @property
    def generation_chunks(self) -> tuple[mx.array, ...]:
        return tuple(self._generation_chunks)

    def hydrate_from_snapshot(
        self,
        prefix_snapshot: Any,
        *,
        snap_prefix_len: int,
    ) -> mx.array | TargetHiddenChunks:
        if int(snap_prefix_len) == int(self.prompt_len):
            self._current_hidden = TargetHiddenChunks(
                total_len=int(snap_prefix_len),
                chunks=tuple(prefix_snapshot.target_hidden_chunks),
                spans=tuple(prefix_snapshot.target_hidden_chunk_spans),
            )
        else:
            self._current_hidden = init_target_hidden_from_snapshot(
                prefix_snapshot,
                snap_prefix_len=int(snap_prefix_len),
                prompt_len=int(self.prompt_len),
            )
        return self._current_hidden

    def write_prompt_slice(
        self,
        *,
        start: int,
        end: int,
        features: mx.array,
    ) -> mx.array:
        features = self._project(features)
        if self._current_hidden is None:
            self._current_hidden = mx.zeros(
                (features.shape[0], int(self.prompt_len), features.shape[-1]),
                dtype=features.dtype,
            )
        elif isinstance(self._current_hidden, TargetHiddenChunks):
            self._current_hidden = self._current_hidden.materialize()
        self._current_hidden[:, int(start):int(end), :] = features
        mx.eval(self._current_hidden)
        return self._current_hidden

    def prefix_view(self, boundary: int) -> mx.array | None:
        if self._current_hidden is None:
            return None
        if isinstance(self._current_hidden, TargetHiddenChunks):
            return self._current_hidden.slice(0, int(boundary))
        return self._current_hidden[:, :int(boundary), :]

    def freeze_prefill_for_snapshot(self, *, enabled: bool) -> None:
        self._prefill_hidden_for_snapshot = self._current_hidden if enabled else None

    def commit_generation(
        self,
        committed_hidden: mx.array,
        *,
        collect_snapshot: bool,
    ) -> None:
        committed_hidden = self._project(committed_hidden)
        self._current_hidden = committed_hidden
        if collect_snapshot:
            self._generation_chunks.append(committed_hidden)

    def generation_snapshot_hidden(self) -> mx.array | TargetHiddenChunks | None:
        if self._prefill_hidden_for_snapshot is None or not self._generation_chunks:
            return None
        gen_hidden = (
            self._generation_chunks[0]
            if len(self._generation_chunks) == 1
            else mx.concatenate(self._generation_chunks, axis=1)
        )
        if isinstance(self._prefill_hidden_for_snapshot, TargetHiddenChunks):
            prefill = self._prefill_hidden_for_snapshot
            generated_len = int(gen_hidden.shape[1])
            return TargetHiddenChunks(
                total_len=int(prefill.total_len) + generated_len,
                chunks=(*prefill.chunks, gen_hidden),
                spans=(
                    *prefill.spans,
                    (
                        int(prefill.total_len),
                        int(prefill.total_len) + generated_len,
                    ),
                ),
            )
        hidden = mx.concatenate([self._prefill_hidden_for_snapshot, gen_hidden], axis=1)
        mx.eval(hidden)
        return hidden

    def _project(self, features: mx.array) -> mx.array:
        if self.project_context is None:
            return features
        projected = self.project_context(features)
        mx.eval(projected)
        return projected
