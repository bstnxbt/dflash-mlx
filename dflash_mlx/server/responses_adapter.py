# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import time
import uuid
from typing import Any, Optional


class ResponsesAdapterError(ValueError):
    pass

_UNSUPPORTED_REQUEST_FIELDS = (
    "previous_response_id",
    "store",
    "reasoning",
    "text",
    "truncation",
)


def responses_to_chat_body(body: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ResponsesAdapterError("Request should be a JSON dictionary")
    if body.get("stream", False):
        raise ResponsesAdapterError("/v1/responses streaming is not implemented yet")
    if body.get("tool_choice") is not None:
        raise ResponsesAdapterError("/v1/responses tool_choice is not implemented yet")
    if body.get("parallel_tool_calls") is not None:
        raise ResponsesAdapterError(
            "/v1/responses parallel_tool_calls is not implemented yet"
        )
    for key in _UNSUPPORTED_REQUEST_FIELDS:
        if key in body:
            raise ResponsesAdapterError(f"/v1/responses {key} is not implemented yet")
    if "input" not in body:
        raise ResponsesAdapterError("/v1/responses requires an input field")

    messages = _input_to_messages(body["input"])
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages = [{"role": "system", "content": instructions}] + messages

    chat: dict[str, Any] = {
        "model": body.get("model", "default_model"),
        "messages": messages,
    }

    if "max_output_tokens" in body:
        chat["max_tokens"] = body["max_output_tokens"]
    if "tools" in body:
        chat["tools"] = _tools_to_chat_tools(body["tools"])

    passthrough = (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "xtc_probability",
        "xtc_threshold",
        "logit_bias",
        "repetition_penalty",
        "repetition_context_size",
        "presence_penalty",
        "presence_context_size",
        "frequency_penalty",
        "frequency_context_size",
        "logprobs",
        "top_logprobs",
        "stop",
        "draft_model",
        "num_draft_tokens",
        "seed",
        "chat_template_kwargs",
    )
    for key in passthrough:
        if key in body:
            chat[key] = body[key]

    return chat


def chat_response_to_responses(
    body: dict[str, Any],
    *,
    request_id: Optional[str] = None,
) -> dict[str, Any]:
    choice = _first_choice(body)
    message = dict(choice.get("message") or {})
    content = _message_text(message)
    tool_calls = _chat_tool_calls_to_response_items(message.get("tool_calls"))
    response_id = request_id or _response_id(body.get("id"))
    created_at = body.get("created")
    if created_at is None:
        created_at = int(time.time())

    output = []
    if content:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                    }
                ],
            }
        )
    output.extend(tool_calls)

    response = {
        "id": response_id,
        "object": "response",
        "created_at": int(created_at),
        "model": body.get("model"),
        "status": "completed",
        "output": output,
        "output_text": content,
    }
    if "usage" in body:
        response["usage"] = body["usage"]
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None:
        response["finish_reason"] = finish_reason
    return response


def _input_to_messages(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if not isinstance(value, list):
        raise ResponsesAdapterError("/v1/responses input must be a string or list")

    messages: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            raise ResponsesAdapterError("/v1/responses input list items must be messages")
        item_type = item.get("type")
        if item_type == "function_call":
            messages.append(_function_call_input_to_chat_message(item))
            continue
        if item_type == "function_call_output":
            messages.append(_function_call_output_to_chat_message(item))
            continue
        role = item.get("role", "user")
        if not isinstance(role, str):
            raise ResponsesAdapterError("/v1/responses message role must be a string")
        messages.append(
            {
                "role": role,
                "content": _content_to_text(item.get("content", "")),
            }
        )
    if not messages:
        raise ResponsesAdapterError("/v1/responses input must not be empty")
    return messages


def _function_call_input_to_chat_message(item: dict[str, Any]) -> dict[str, Any]:
    name = item.get("name")
    call_id = item.get("call_id") or item.get("id")
    arguments = item.get("arguments", "{}")
    if not isinstance(name, str) or not name:
        raise ResponsesAdapterError("/v1/responses function_call requires a name")
    if not isinstance(call_id, str) or not call_id:
        raise ResponsesAdapterError("/v1/responses function_call requires a call_id")
    if not isinstance(arguments, str):
        raise ResponsesAdapterError("/v1/responses function_call arguments must be a string")
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        ],
    }


def _function_call_output_to_chat_message(item: dict[str, Any]) -> dict[str, Any]:
    call_id = item.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        raise ResponsesAdapterError(
            "/v1/responses function_call_output requires a call_id"
        )
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": _content_to_text(item.get("output", "")),
    }


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return _content_item_to_text(content)
    if isinstance(content, list):
        return "".join(_content_item_to_text(item) for item in content)
    raise ResponsesAdapterError("/v1/responses message content must be text")


def _content_item_to_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        raise ResponsesAdapterError("/v1/responses only text input content is supported")
    item_type = item.get("type")
    if item_type in ("input_text", "output_text", "text"):
        text = item.get("text", "")
        if not isinstance(text, str):
            raise ResponsesAdapterError("/v1/responses text content must be a string")
        return text
    if "text" in item and item_type is None:
        text = item.get("text", "")
        if isinstance(text, str):
            return text
    raise ResponsesAdapterError(
        f"/v1/responses content item type is not supported: {item_type!r}"
    )


def _tools_to_chat_tools(tools: Any) -> list[dict[str, Any]]:
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise ResponsesAdapterError("/v1/responses tools must be a list")
    return [_tool_to_chat_tool(tool) for tool in tools]


def _tool_to_chat_tool(tool: Any) -> dict[str, Any]:
    if not isinstance(tool, dict):
        raise ResponsesAdapterError("/v1/responses tool entries must be objects")
    tool_type = tool.get("type")
    if tool_type != "function":
        raise ResponsesAdapterError(
            f"/v1/responses tool type is not supported: {tool_type!r}"
        )
    if "function" in tool:
        raise ResponsesAdapterError(
            "/v1/responses function tools must use Responses schema"
        )
    name = tool.get("name")
    if not isinstance(name, str) or not name:
        raise ResponsesAdapterError("/v1/responses function tools require a name")
    function = {"name": name}
    for key in ("description", "parameters", "strict"):
        if key in tool:
            function[key] = tool[key]
    return {"type": "function", "function": function}


def _chat_tool_calls_to_response_items(tool_calls: Any) -> list[dict[str, Any]]:
    if not tool_calls:
        return []
    if not isinstance(tool_calls, list):
        raise ResponsesAdapterError("Chat completion tool_calls must be a list")
    return [_chat_tool_call_to_response_item(tool_call) for tool_call in tool_calls]


def _chat_tool_call_to_response_item(tool_call: Any) -> dict[str, Any]:
    if not isinstance(tool_call, dict):
        raise ResponsesAdapterError("Chat completion tool_call must be an object")
    tool_type = tool_call.get("type")
    if tool_type != "function":
        raise ResponsesAdapterError(
            f"Chat completion tool_call type is not supported: {tool_type!r}"
        )
    function = tool_call.get("function")
    if not isinstance(function, dict):
        raise ResponsesAdapterError("Chat completion tool_call missing function")
    name = function.get("name")
    arguments = function.get("arguments", "{}")
    if not isinstance(name, str) or not name:
        raise ResponsesAdapterError("Chat completion tool_call function missing name")
    if not isinstance(arguments, str):
        raise ResponsesAdapterError("Chat completion tool_call arguments must be a string")
    call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex}"
    return {
        "id": f"fc_{uuid.uuid4().hex}",
        "type": "function_call",
        "status": "completed",
        "call_id": str(call_id),
        "name": name,
        "arguments": arguments,
    }


def _first_choice(body: dict[str, Any]) -> dict[str, Any]:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ResponsesAdapterError("Chat completion response did not contain choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ResponsesAdapterError("Chat completion choice must be an object")
    return choice


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _content_to_text(content)
    return ""


def _response_id(chat_id: Any) -> str:
    if isinstance(chat_id, str) and chat_id.startswith("resp_"):
        return chat_id
    return f"resp_{uuid.uuid4().hex}"
