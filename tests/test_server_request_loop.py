# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from queue import SimpleQueue

from dflash_mlx.server.request_loop import consume_dflash_events

class FakeDetokenizer:
    def __init__(self):
        self.last_segment = ""
        self.reset_calls = 0
        self.tokens = []

    def reset(self):
        self.reset_calls += 1
        self.last_segment = ""
        self.tokens = []

    def add_token(self, token):
        self.tokens.append(int(token))
        self.last_segment = f"T{int(token)}"

    def finalize(self):
        self.last_segment = ""

class FakeTokenizer:
    def __init__(self):
        self.detokenizer = FakeDetokenizer()

class FakeContext:
    _should_stop = False

class ClosableEvents:
    def __init__(self, events):
        self.events = list(events)
        self.closed = False

    def __iter__(self):
        return iter(self.events)

    def close(self):
        self.closed = True

def test_consume_dflash_events_streams_pending_token_and_summary():
    rqueue = SimpleQueue()
    events = ClosableEvents(
        [
            {"event": "prefill", "prompt_token_count": 3},
            {"event": "token", "token_id": 10, "acceptance_ratio": 0.5},
            {"event": "token", "token_id": 11, "acceptance_ratio": 0.5},
            {
                "event": "summary",
                "generated_token_ids": [10, 11],
                "generation_tokens": 2,
            },
        ]
    )

    result = consume_dflash_events(
        event_iter=events,
        rqueue=rqueue,
        ctx=FakeContext(),
        tokenizer=FakeTokenizer(),
        prompt=[1, 2, 3],
        max_tokens=16,
        eos_token_ids=set(),
        request_start_ns=0,
    )

    responses = []
    while not rqueue.empty():
        responses.append(rqueue.get())

    assert events.closed is True
    assert result.summary_event is not None
    assert result.live_token_count == 2
    assert result.finish_reason == "stop"
    assert len(responses) == 2
    assert responses[0].token == 10
    assert responses[0].text == "T10"
    assert responses[1].token == 11
    assert responses[1].text == "T11"
