# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import _BaseCache


class ShortConvRollbackCache(_BaseCache):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.left_padding = None
        instance.lengths = None
        instance._armed = False
        instance._tape = None
        instance._snapshot = None
        return instance

    def __init__(self, kernel_size: int):
        self.cache = [None]
        self.kernel_size = int(kernel_size)

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
        cache = ShortConvRollbackCache(self.kernel_size)
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

    def clear_transients(self) -> None:
        self._armed = False
        self._tape = None
        self._snapshot = None

    def arm_rollback(self, prefix_len: int = 0) -> None:
        del prefix_len
        self._armed = True
        self._tape = None
        self._snapshot = list(self.cache)

    def record_tape_if_armed(self, bx_extended: mx.array) -> None:
        if self._armed:
            self._tape = mx.contiguous(bx_extended)

    def rollback(self, n_accepted: int) -> None:
        if self._snapshot is None:
            self.clear_transients()
            return
        n_keep = self.kernel_size - 1
        if self._tape is None or n_keep <= 0:
            self.cache = list(self._snapshot)
            self.clear_transients()
            return
        accepted_steps = int(n_accepted) + 1
        start = accepted_steps
        end = start + n_keep
        tape_len = int(self._tape.shape[1])
        if end > tape_len:
            self.cache = list(self._snapshot)
        else:
            self.cache[0] = mx.contiguous(self._tape[:, start:end, :])
        self.clear_transients()
