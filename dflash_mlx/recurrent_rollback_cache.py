# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import _BaseCache

from dflash_mlx.kernels import tape_replay_kernel


class RecurrentRollbackCache(_BaseCache):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.left_padding = None
        instance.lengths = None
        instance._armed = False
        instance._tape = None
        instance._tape_k = None
        instance._tape_g = None
        instance._tape_qkv = None
        instance._snapshot = None
        return instance

    def __init__(self, size: int, *, conv_kernel_size: int = 4):
        self.cache = [None] * size
        self.conv_kernel_size = int(conv_kernel_size)

    def __getitem__(self, idx: int):
        return self.cache[idx]

    def __setitem__(self, idx: int, value: Any) -> None:
        self.cache[idx] = value

    @property
    def state(self):
        return self.cache

    @state.setter
    def state(self, value) -> None:
        self.cache = value

    def filter(self, batch_indices):
        self.cache = [c[batch_indices] if c is not None else None for c in self.cache]
        if self.lengths is not None:
            self.lengths = self.lengths[batch_indices]

    def extend(self, other):
        def cat(lhs, rhs):
            if lhs is None:
                return rhs
            if rhs is None:
                return lhs
            return mx.concatenate([lhs, rhs])

        self.cache = [cat(lhs, rhs) for lhs, rhs in zip(self.cache, other.cache, strict=True)]

    def extract(self, idx):
        cache = RecurrentRollbackCache(len(self.cache), conv_kernel_size=self.conv_kernel_size)
        cache.cache = [c[idx : idx + 1] if c is not None else None for c in self.cache]
        return cache

    def prepare(self, lengths=None, **kwargs):
        self.lengths = None if lengths is None else mx.array(lengths)

    def finalize(self):
        self.lengths = None
        self.left_padding = None
        self.clear_transients()

    def advance(self, n: int):
        if self.lengths is not None:
            self.lengths -= n
        if self.left_padding is not None:
            self.left_padding -= n

    def make_mask(self, n: int):
        if self.left_padding is not None:
            pos = mx.arange(n)
            return pos >= self.left_padding[:, None]
        if self.lengths is not None:
            pos = mx.arange(n)
            return pos < self.lengths[:, None]
        return None

    def empty(self):
        return self.cache[0] is None

    @property
    def nbytes(self):
        return sum(c.nbytes for c in self.cache if c is not None)

    def checkpoint(self) -> None:
        self._snapshot = list(self.cache)

    def clear_transients(self) -> None:
        self._armed = False
        self._tape = None
        self._tape_k = None
        self._tape_g = None
        self._tape_qkv = None
        self._snapshot = None

    def arm_rollback(self, prefix_len: int = 0) -> None:
        del prefix_len
        self._armed = True
        self._tape = None
        self._tape_k = None
        self._tape_g = None
        self._tape_qkv = None
        self.checkpoint()

    def record_tape(
        self,
        *,
        tape: mx.array,
        k: mx.array,
        g: mx.array,
        qkv: mx.array,
    ) -> None:
        self._tape = mx.contiguous(tape)
        self._tape_k = mx.contiguous(k)
        self._tape_g = mx.contiguous(g)
        self._tape_qkv = mx.contiguous(qkv)

    def _rebuild_conv_state(self, accepted_steps: int) -> mx.array | None:
        if self._tape_qkv is None:
            return self.cache[0]
        keep = self.conv_kernel_size - 1
        if keep <= 0:
            return None
        conv_state = self._snapshot[0] if self._snapshot is not None else None
        if conv_state is None:
            prefix = mx.zeros(
                (self._tape_qkv.shape[0], keep, self._tape_qkv.shape[-1]),
                dtype=self._tape_qkv.dtype,
            )
        else:
            prefix = conv_state
        conv_input = mx.concatenate([prefix, self._tape_qkv], axis=1)
        start = accepted_steps
        end = min(start + keep, int(conv_input.shape[1]))
        return mx.contiguous(conv_input[:, start:end, :])

    def rollback(self, n_accepted: int) -> None:
        if self._snapshot is None:
            self.clear_transients()
            return

        self.cache = list(self._snapshot)
        if (
            self._tape is not None
            and self._tape_k is not None
            and self._tape_g is not None
            and self.cache[1] is not None
        ):
            accepted_steps = int(n_accepted) + 1
            state = tape_replay_kernel(
                self._tape[:, :accepted_steps],
                self._tape_k[:, :accepted_steps],
                self._tape_g[:, :accepted_steps],
                self.cache[1],
                None,
            )
            self.cache[1] = state
            self.cache[0] = self._rebuild_conv_state(accepted_steps)

        self.clear_transients()
