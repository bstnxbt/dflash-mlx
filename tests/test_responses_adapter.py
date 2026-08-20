# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import json
from io import BytesIO
from types import SimpleNamespace

import pytest

from dflash_mlx import serve
from dflash_mlx.serve import DFlashAPIHandler
from dflash_mlx.server.responses_adapter import (
    ResponsesAdapterError,
    chat_response_to_responses,
    responses_to_chat_body,
)


def test_responses_string_input_maps_to_chat_user_message():
    chat = responses_to_chat_body(
        {
            "model": "target",
            "input": "hello",
        }
    )

    assert chat["model"] == "target"
    assert chat["messages"] == [{"role": "user", "content": "hello"}]


def test_responses_instructions_prepend_system_message():
    chat = responses_to_chat_body(
        {
            "model": "target",
            "instructions": "Be terse.",
            "input": "hello",
        }
    )

    assert chat["messages"] == [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "hello"},
    ]


def test_responses_max_output_tokens_maps_to_max_tokens():
    chat = responses_to_chat_body(
        {
            "model": "target",
            "input": "hello",
            "max_output_tokens": 123,
        }
    )

    assert chat["max_tokens"] == 123


def test_responses_message_list_and_text_blocks_map_to_chat_messages():
    chat = responses_to_chat_body(
        {
            "model": "target",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "hello "},
                        {"type": "text", "text": "world"},
                    ],
                },
                {"role": "assistant", "content": "previous"},
            ],
        }
    )

    assert chat["messages"] == [
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "previous"},
    ]


def test_responses_metadata_is_ignored():
    chat = responses_to_chat_body(
        {
            "model": "target",
            "input": "hello",
            "metadata": {"trace": "x"},
        }
    )

    assert "metadata" not in chat


def test_responses_sampling_fields_passthrough():
    fields = {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 12,
        "min_p": 0.05,
        "xtc_probability": 0.1,
        "xtc_threshold": 0.2,
        "logit_bias": {"42": 1.5},
        "repetition_penalty": 1.1,
        "repetition_context_size": 32,
        "presence_penalty": 0.3,
        "presence_context_size": 48,
        "frequency_penalty": 0.4,
        "frequency_context_size": 64,
        "logprobs": True,
        "top_logprobs": 4,
        "stop": ["END"],
    }
    chat = responses_to_chat_body(
        {
            "model": "target",
            "input": "hello",
            **fields,
        }
    )

    assert {key: chat[key] for key in fields} == fields


def test_responses_function_tools_map_to_chat_tools():
    chat = responses_to_chat_body(
        {
            "model": "target",
            "input": "hello",
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Lookup a value.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                    "strict": True,
                }
            ],
        }
    )

    assert chat["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup a value.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
                "strict": True,
            },
        }
    ]


def test_responses_unsupported_request_fields_are_rejected():
    cases = [
        ("tool_choice", "auto", "tool_choice is not implemented"),
        ("parallel_tool_calls", True, "parallel_tool_calls is not implemented"),
        ("previous_response_id", "resp_old", "previous_response_id is not implemented"),
        ("store", False, "store is not implemented"),
        ("reasoning", {"effort": "low"}, "reasoning is not implemented"),
        ("text", {"format": {"type": "text"}}, "text is not implemented"),
        ("truncation", "auto", "truncation is not implemented"),
    ]
    for field, value, error in cases:
        body = {"model": "target", "input": "hello", field: value}
        with pytest.raises(ResponsesAdapterError, match=error):
            responses_to_chat_body(body)


def test_responses_unsupported_tool_type_is_rejected():
    with pytest.raises(ResponsesAdapterError, match="tool type is not supported"):
        responses_to_chat_body(
            {
                "model": "target",
                "input": "hello",
                "tools": [{"type": "web_search_preview"}],
            }
        )


def test_responses_rejects_chat_completions_tool_schema():
    with pytest.raises(ResponsesAdapterError, match="must use Responses schema"):
        responses_to_chat_body(
            {
                "model": "target",
                "input": "hello",
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "lookup"},
                    }
                ],
            }
        )


def test_responses_function_call_inputs_map_to_chat_tool_messages():
    chat = responses_to_chat_body(
        {
            "model": "target",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": "{\"query\":\"x\"}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "result",
                },
            ],
        }
    )

    assert chat["messages"] == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "{\"query\":\"x\"}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "result",
        },
    ]


def test_responses_missing_input_raises_clear_error():
    with pytest.raises(ResponsesAdapterError, match="requires an input"):
        responses_to_chat_body({"model": "target"})


def test_responses_streaming_is_rejected_for_now():
    with pytest.raises(ResponsesAdapterError, match="streaming is not implemented"):
        responses_to_chat_body({"model": "target", "input": "hello", "stream": True})


def test_chat_response_maps_to_responses_output_text():
    response = chat_response_to_responses(
        {
            "id": "chatcmpl_x",
            "created": 123,
            "model": "target",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        },
        request_id="resp_test",
    )

    assert response["id"] == "resp_test"
    assert response["object"] == "response"
    assert response["created_at"] == 123
    assert response["model"] == "target"
    assert response["status"] == "completed"
    assert response["output_text"] == "done"
    assert response["output"][0]["content"] == [
        {"type": "output_text", "text": "done"}
    ]
    assert response["usage"]["total_tokens"] == 3
    assert response["finish_reason"] == "stop"


def test_chat_tool_calls_map_to_responses_function_call_output_items():
    response = chat_response_to_responses(
        {
            "id": "chatcmpl_x",
            "created": 123,
            "model": "target",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": "{\"query\":\"x\"}",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        },
        request_id="resp_test",
    )

    assert response["output_text"] == ""
    assert response["output"][0]["id"].startswith("fc_")
    assert response["output"][0] | {"id": "fc_test"} == {
        "id": "fc_test",
        "type": "function_call",
        "status": "completed",
        "call_id": "call_1",
        "name": "lookup",
        "arguments": "{\"query\":\"x\"}",
    }
    assert response["finish_reason"] == "tool_calls"


def test_chat_content_and_tool_calls_preserve_output_order():
    response = chat_response_to_responses(
        {
            "created": 123,
            "model": "target",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I will call a tool.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": "{}",
                                },
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": "{\"q\":\"x\"}",
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
        request_id="resp_test",
    )

    assert [item["type"] for item in response["output"]] == [
        "message",
        "function_call",
        "function_call",
    ]
    assert response["output"][1]["call_id"] == "call_1"
    assert response["output"][2]["call_id"] == "call_2"


def test_chat_tool_call_non_function_type_is_rejected():
    with pytest.raises(ResponsesAdapterError, match="type is not supported"):
        chat_response_to_responses(
            {
                "created": 123,
                "model": "target",
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "custom",
                                    "function": {"name": "lookup"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }
        )


def test_responses_route_uses_adapter_and_chat_completion_path(monkeypatch):
    seen = {}

    def fake_handle_completion(self, request, stop_words):
        seen["path"] = self.path
        seen["body"] = dict(self.body)
        seen["request"] = request
        seen["stop_words"] = stop_words
        payload = self.generate_response(
            "ok",
            "stop",
            prompt_token_count=3,
            completion_token_count=1,
            prompt_cache_count=0,
        )
        self._write_json_response(200, payload)

    monkeypatch.setattr(DFlashAPIHandler, "handle_completion", fake_handle_completion)
    handler = _make_handler(
        path="/v1/responses",
        body={
            "model": "target",
            "instructions": "System",
            "input": "Hello",
            "max_output_tokens": 16,
            "tools": [{"type": "function", "name": "lookup"}],
        },
    )

    DFlashAPIHandler.do_POST(handler)

    payload = json.loads(handler.wfile.getvalue().decode())
    assert seen["path"] == "/v1/chat/completions"
    assert seen["body"]["messages"] == [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Hello"},
    ]
    assert seen["body"]["max_tokens"] == 16
    assert seen["body"]["tools"] == [
        {"type": "function", "function": {"name": "lookup"}}
    ]
    assert seen["request"].request_type == "chat"
    assert seen["stop_words"] == []
    assert payload["object"] == "response"
    assert payload["output_text"] == "ok"


def test_responses_route_returns_function_call_output(monkeypatch):
    def fake_handle_completion(self, request, stop_words):
        payload = self.generate_response(
            "",
            "tool_calls",
            prompt_token_count=3,
            completion_token_count=1,
            prompt_cache_count=0,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "{\"query\":\"x\"}",
                    },
                }
            ],
        )
        self._write_json_response(200, payload)

    monkeypatch.setattr(DFlashAPIHandler, "handle_completion", fake_handle_completion)
    handler = _make_handler(
        path="/v1/responses",
        body={
            "model": "target",
            "input": "Call lookup.",
            "tools": [{"type": "function", "name": "lookup"}],
        },
    )

    DFlashAPIHandler.do_POST(handler)

    payload = json.loads(handler.wfile.getvalue().decode())
    assert payload["output_text"] == ""
    assert payload["output"][0]["type"] == "function_call"
    assert payload["output"][0]["call_id"] == "call_1"
    assert payload["output"][0]["name"] == "lookup"


def test_chat_completions_route_delegates_to_upstream(monkeypatch):
    called = []

    def fake_upstream_do_post(self):
        called.append(self.path)

    monkeypatch.setattr(serve.mlx_server.APIHandler, "do_POST", fake_upstream_do_post)
    handler = object.__new__(DFlashAPIHandler)
    handler.path = "/v1/chat/completions"

    DFlashAPIHandler.do_POST(handler)

    assert called == ["/v1/chat/completions"]


def test_responses_route_missing_input_returns_json_error():
    handler = _make_handler(path="/v1/responses", body={"model": "target"})

    DFlashAPIHandler.do_POST(handler)

    payload = json.loads(handler.wfile.getvalue().decode())
    assert handler.statuses == [400]
    assert "requires an input" in payload["error"]


def test_responses_route_unsupported_field_returns_json_error():
    handler = _make_handler(
        path="/v1/responses",
        body={"model": "target", "input": "hello", "previous_response_id": "resp_old"},
    )

    DFlashAPIHandler.do_POST(handler)

    payload = json.loads(handler.wfile.getvalue().decode())
    assert handler.statuses == [400]
    assert "previous_response_id is not implemented" in payload["error"]


def _make_handler(*, path: str, body: dict) -> DFlashAPIHandler:
    raw = json.dumps(body).encode()
    handler = object.__new__(DFlashAPIHandler)
    handler.path = path
    handler.headers = {"Content-Length": str(len(raw))}
    handler.rfile = BytesIO(raw)
    handler.wfile = BytesIO()
    handler.statuses = []
    handler.headers_sent = []
    handler._set_completion_headers = lambda status=200: handler.statuses.append(status)
    handler.send_header = lambda *args: handler.headers_sent.append(args)
    handler.end_headers = lambda: None
    handler.response_generator = SimpleNamespace(
        cli_args=SimpleNamespace(
            num_draft_tokens=0,
            max_tokens=64,
            temp=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
        ),
        model_provider=SimpleNamespace(model_key=("served-target", None, "draft")),
    )
    handler.created = 123
    handler.system_fingerprint = "test-fingerprint"
    return handler
