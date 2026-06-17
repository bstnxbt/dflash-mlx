# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

"""Cross-backend safety for class-level target hooks.

Target backends patch the attention class (``cls.__call__``) and tag it with a
per-class marker so a second install is a no-op. The marker is keyed by class
*object*, which is safe only while each attention class maps to exactly one
target backend. If a future model routed to a different backend shared an
attention class with an existing one, a bare boolean marker would let the
first-installed backend's attention silently run for the second model. These
tests pin the fail-loud contract: a different backend claiming an already
patched class raises instead of silently skipping.
"""
from __future__ import annotations

import mlx.nn as nn
import pytest

from dflash_mlx.engine._hook_guard import DFlashHookConflict, claim_class_hook


def test_free_slot_is_claimable():
    class A(nn.Module):
        pass

    assert claim_class_hook(A, "_marker", "gemma4") is True


def test_same_owner_reclaim_is_noop():
    class A(nn.Module):
        pass

    assert claim_class_hook(A, "_marker", "gemma4") is True
    A._marker = "gemma4"  # installer sets the marker after patching
    assert claim_class_hook(A, "_marker", "gemma4") is False


def test_different_owner_raises():
    class A(nn.Module):
        pass

    A._marker = "gemma4"
    with pytest.raises(DFlashHookConflict, match="already patched"):
        claim_class_hook(A, "_marker", "qwen_gdn")


def test_installers_fail_loud_on_shared_attention_class():
    # Real installers must wire the guard: a gemma4-claimed class cannot then be
    # claimed by the qwen backend. (The claim runs before any attention math, so
    # this needs no real weights.)
    from dflash_mlx.engine.target_gemma4 import (
        _install_full_attention_gqa_hook as gemma_install,
    )
    from dflash_mlx.engine.target_qwen_gdn import (
        _install_full_attention_gqa_hook as qwen_install,
    )

    class SharedAttn(nn.Module):
        def __call__(self, *args, **kwargs):  # patched by the installer
            return None

    inst = SharedAttn()
    gemma_install(inst)  # claims the class for gemma4
    # A second gemma install on the same class is a no-op.
    gemma_install(inst)
    with pytest.raises(DFlashHookConflict):
        qwen_install(inst)  # different backend, same class -> loud failure


def test_same_backend_two_models_sharing_attention_class_is_safe():
    # Two different model architectures that resolve to the SAME backend may
    # share an attention class (e.g. plain qwen3 and qwen3-next both use
    # qwen3_next.Qwen3NextAttention). Installing the backend's hook for the
    # second model on the already-patched class must be a silent no-op, not a
    # conflict — the shared call is stateless and reads per-instance weights.
    from dflash_mlx.engine.target_qwen_gdn import (
        _install_full_attention_gqa_hook as qwen_install,
    )

    class SharedQwenAttn(nn.Module):
        def __call__(self, *args, **kwargs):
            return None

    model_a_attn = SharedQwenAttn()
    model_b_attn = SharedQwenAttn()  # distinct instance, same class
    qwen_install(model_a_attn)  # model A claims qwen_gdn
    qwen_install(model_b_attn)  # model B: same class + backend -> no-op, no raise
    assert (
        getattr(SharedQwenAttn, "_dflash_full_attention_gqa_installed") == "qwen_gdn"
    )
