from __future__ import annotations

from mlx_lm.generate import SequenceStateMachine

from dflash_mlx.serve import _match_stream_token


def test_match_stream_token_terminal_state_is_not_reused():
    sm = SequenceStateMachine(
        transitions={"normal": [((42,), None)]},
        initial="normal",
    )
    state = sm.make_state()

    state, match, current, terminal = _match_stream_token(sm, state, 42)

    assert match == (42,)
    assert current is None
    assert terminal is True

    state2, match2, current2, terminal2 = _match_stream_token(sm, state, 7)

    assert state2 is state
    assert match2 is None
    assert current2 is None
    assert terminal2 is True


def test_match_stream_token_absent_state_machine_is_normal():
    state, match, current, terminal = _match_stream_token(None, None, 42)

    assert state is None
    assert match is None
    assert current == "normal"
    assert terminal is False
