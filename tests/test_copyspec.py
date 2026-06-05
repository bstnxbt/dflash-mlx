from __future__ import annotations

import pytest

from dflash_mlx.engine.copyspec import CopySpecIndex


def test_copyspec_draft_after_returns_full_prompt_copy() -> None:
    index = CopySpecIndex(
        [1, 2, 3, 4, 5, 0, 7, 8, 1, 2, 3, 4, 5],
        window_size=6,
    )

    draft = index.draft_after(0, max_tokens=2)

    assert draft == (7, 8)


def test_copyspec_rejects_partial_copy() -> None:
    index = CopySpecIndex([1, 2, 3, 4, 1, 2], window_size=3)

    assert index.draft_after(3, max_tokens=4) is None


def test_copyspec_append_committed_indexes_generated_history() -> None:
    index = CopySpecIndex([9, 9, 9, 1, 2], window_size=3)
    index.append_committed([3, 4, 1, 2])

    draft = index.draft_after(3, max_tokens=1)

    assert draft == (4,)


def test_copyspec_short_initial_prompt_stays_disabled() -> None:
    index = CopySpecIndex([1, 2], window_size=3)
    index.append_committed([3, 4, 1, 2])

    assert index.draft_after(3, max_tokens=1) is None


def test_copyspec_skips_forbidden_tokens() -> None:
    index = CopySpecIndex([1, 2, 3, 4, 5, 0, 7, 1, 2, 3, 4, 5], window_size=6)

    assert index.draft_after(0, max_tokens=1, forbidden_tokens={7}) is None


def test_copyspec_rejects_invalid_window_size() -> None:
    with pytest.raises(ValueError, match="window_size"):
        CopySpecIndex([1, 2, 3], window_size=0)
