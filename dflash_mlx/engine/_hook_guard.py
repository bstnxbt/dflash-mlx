# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

"""Guard for class-level target hooks shared across backends.

Target backends install speculative hooks by replacing ``cls.__call__`` on a
model's attention class and tagging the class with a marker so a second install
is a no-op. The marker is keyed by the class *object*, which is safe only while
each attention class resolves to exactly one target backend. Should two model
families that resolve to different backends ever share an attention class, a
bare boolean marker would let whichever backend installed first silently run
its attention for the other model. This guard makes that case fail loudly.
"""
from __future__ import annotations


class DFlashHookConflict(RuntimeError):
    """Two target backends tried to patch the same attention class."""


def claim_class_hook(cls: type, attr: str, owner: str) -> bool:
    """Claim the class-level hook slot ``attr`` on ``cls`` for ``owner``.

    Returns ``True`` when the slot is free (the caller should install and then
    set ``cls.attr = owner``) and ``False`` when ``owner`` already installed it
    (the caller should no-op). Raises :class:`DFlashHookConflict` when a
    *different* backend already owns the slot.
    """
    existing = getattr(cls, attr, None)
    if existing is None:
        return True
    if existing == owner:
        return False
    raise DFlashHookConflict(
        f"{cls.__module__}.{cls.__qualname__} is already patched by DFlash target "
        f"backend {existing!r}; backend {owner!r} cannot reuse the same attention "
        f"class. Two model families that share an attention class must resolve to "
        f"the same target backend."
    )
