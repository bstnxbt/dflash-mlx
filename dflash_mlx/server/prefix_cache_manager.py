# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import hashlib
import json
from typing import Any

from dflash_mlx.cache.fingerprints import DFlashPrefixKey


def build_runtime_cache_identity(model_provider: Any) -> tuple[str, str]:
    model_key = getattr(model_provider, "model_key", None)
    if not isinstance(model_key, (list, tuple)) or len(model_key) < 3:
        raise ValueError("model provider is missing a loaded model_key")
    target_id = str(model_key[0]) if model_key[0] is not None else ""
    draft_id = str(model_key[2]) if model_key[2] is not None else ""
    if not target_id or not draft_id:
        raise ValueError("prefix cache requires loaded target and draft model ids")
    return target_id, draft_id


def build_prefix_key(
    model_provider: Any,
    draft_model: Any,
    runtime_context: Any,
) -> DFlashPrefixKey:
    target_id, draft_id = build_runtime_cache_identity(model_provider)

    raw_capture_ids = getattr(draft_model, "target_layer_ids", None)
    if raw_capture_ids is None:
        raise ValueError("draft model is missing target_layer_ids")
    try:
        capture_ids = tuple(int(x) for x in raw_capture_ids)
    except (TypeError, ValueError) as exc:
        raise ValueError("draft model target_layer_ids must be integers") from exc
    if not capture_ids:
        raise ValueError("draft model target_layer_ids must not be empty")

    try:
        runtime_config = runtime_context.runtime
        sink = int(runtime_config.draft_sink_size)
        window = int(runtime_config.draft_window_size)
        target_fa_window = int(runtime_config.target_fa_window)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("runtime context is missing prefix cache identity fields") from exc

    tokenizer = getattr(model_provider, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("prefix cache requires loaded tokenizer")

    cli_args = getattr(model_provider, "cli_args", None)
    chat_template_args = getattr(cli_args, "chat_template_args", None)
    if chat_template_args is None:
        chat_template_args = {}
    if not isinstance(chat_template_args, dict):
        raise ValueError("chat_template_args must be a dictionary")

    return DFlashPrefixKey(
        target_model_id=target_id,
        draft_model_id=draft_id,
        capture_layer_ids=capture_ids,
        draft_sink_size=int(sink),
        draft_window_size=int(window),
        template_hash=_hash_text(_effective_chat_template(tokenizer)),
        prompt_policy_hash=_prompt_policy_hash(tokenizer, chat_template_args),
        target_fa_window=int(target_fa_window),
    )

def chat_template_stable_marker(
    tokenizer: Any,
) -> tuple[int | None, int | None, int]:
    im_start = None
    assistant = None
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return im_start, assistant, 0
    try:
        ids = convert(["<|im_start|>", "assistant"])
    except Exception as exc:
        raise RuntimeError("tokenizer failed to resolve chat template marker ids") from exc
    if not isinstance(ids, (list, tuple)):
        return im_start, assistant, 0
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if ids and ids[0] is not None and ids[0] != unk_token_id:
        im_start = int(ids[0])
    if ids and len(ids) > 1 and ids[1] is not None and ids[1] != unk_token_id:
        assistant = int(ids[1])
    if im_start is not None and assistant is not None:
        return im_start, assistant, 0

    gemma_start = None
    gemma_model = None
    if im_start is None:
        try:
            ids = convert(["<|turn>", "model"])
        except Exception as exc:
            raise RuntimeError("tokenizer failed to resolve chat template marker ids") from exc
        if isinstance(ids, (list, tuple)):
            if ids and ids[0] is not None and ids[0] != unk_token_id:
                gemma_start = int(ids[0])
            if ids and len(ids) > 1 and ids[1] is not None and ids[1] != unk_token_id:
                gemma_model = int(ids[1])
    if gemma_start is None or gemma_model is None:
        return im_start, assistant, 0

    boundary_offset = 2
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            role_prefix = encode("<|turn>model\n")
        except Exception as exc:
            raise RuntimeError("tokenizer failed to resolve chat template marker ids") from exc
        if isinstance(role_prefix, (list, tuple)) and len(role_prefix) >= 2:
            role_prefix_ids = tuple(int(x) for x in role_prefix)
            if role_prefix_ids[:2] == (gemma_start, gemma_model):
                boundary_offset = len(role_prefix_ids)
    return gemma_start, gemma_model, int(boundary_offset)

def _effective_chat_template(tokenizer: Any) -> str:
    for attr in ("chat_template", "default_chat_template"):
        template = getattr(tokenizer, attr, None)
        if template is not None:
            return str(template)
    return ""

def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {
            str(k): _json_safe(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)

def _text_attr(tokenizer: Any, name: str) -> str | None:
    value = getattr(tokenizer, name, None)
    if value is None:
        return None
    return str(value)

def _parser_name(tokenizer: Any) -> str | None:
    parser = getattr(tokenizer, "tool_parser", None)
    if parser is None:
        return None
    parser_type = type(parser)
    return f"{parser_type.__module__}.{parser_type.__qualname__}"

def _prompt_policy_hash(tokenizer: Any, chat_template_args: dict[str, Any]) -> str:
    payload = {
        "chat_template_args": _json_safe(chat_template_args),
        "has_tool_calling": bool(getattr(tokenizer, "has_tool_calling", False)),
        "tool_call_start": _text_attr(tokenizer, "tool_call_start"),
        "tool_call_end": _text_attr(tokenizer, "tool_call_end"),
        "tool_parser": _parser_name(tokenizer),
        "has_thinking": bool(getattr(tokenizer, "has_thinking", False)),
        "think_start": _text_attr(tokenizer, "think_start"),
        "think_end": _text_attr(tokenizer, "think_end"),
        "stable_marker": list(chat_template_stable_marker(tokenizer)),
    }
    return _hash_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )
