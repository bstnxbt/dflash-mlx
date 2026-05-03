# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PROMPT = "Explain speculative decoding in two paragraphs."
DEFAULT_CONTEXT_SEED = (
    "DFlash benchmark long-context filler. The task is to preserve distant "
    "context while generating a concise answer at the end.\n"
)
SUITE_CHOICES = ("smoke", "humaneval", "gsm8k", "math500", "longctx")
DEFAULT_SUITE_LIMITS = {
    "smoke": 1,
    "humaneval": 10,
    "gsm8k": 10,
    "math500": 10,
    "longctx": 1,
}
DEFAULT_CTX_TOKENS = 8192
HF_DATASET_SUITES = {
    "humaneval": {
        "name": "openai_humaneval",
        "config": None,
        "split": "test",
    },
    "gsm8k": {
        "name": "gsm8k",
        "config": "main",
        "split": "test",
    },
    "math500": {
        "name": "HuggingFaceH4/MATH-500",
        "config": None,
        "split": "test",
    },
}

@dataclass(frozen=True)
class BenchmarkPrompt:
    id: str
    suite: str
    prompt: str
    source: str
    row_index: int | None = None
    hf_dataset_name: str | None = None
    hf_dataset_config: str | None = None
    hf_dataset_split: str | None = None

def ctx_tokens(args: argparse.Namespace) -> int:
    value = getattr(args, "ctx_tokens", None)
    if value is not None:
        return int(value)
    return int(getattr(args, "ctx", 0) or 0)

def default_limit_for_suite(suite: str) -> int:
    return int(DEFAULT_SUITE_LIMITS.get(suite, 1))

def build_long_context_prompt(base: str, ctx_token_count: int) -> str:
    seed = base + "\n\n" + DEFAULT_CONTEXT_SEED
    target_chars = max(int(ctx_token_count) * 4, len(seed))
    repeats = (target_chars // len(seed)) + 1
    body = (seed * repeats)[:target_chars]
    return (
        body
        + "\n\nFinal task: answer the user request using the context above. "
        "Keep the answer concise."
    )

def resolve_benchmark_prompts(args: argparse.Namespace) -> list[BenchmarkPrompt]:
    suite = str(args.suite)
    limit = int(args.limit) if args.limit is not None else default_limit_for_suite(suite)
    if limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.prompt_file:
        prompts = load_prompt_file(args.prompt_file)
    elif args.prompt:
        prompt = str(args.prompt)
        if suite == "longctx":
            prompt = build_long_context_prompt(prompt, ctx_tokens(args) or DEFAULT_CTX_TOKENS)
        source = "synthetic" if suite == "longctx" else "smoke"
        prompts = (BenchmarkPrompt(f"{suite}-custom-{slugify_prompt_id(prompt)}", suite, prompt, source),)
    elif suite == "longctx":
        ctx_token_count = ctx_tokens(args) or DEFAULT_CTX_TOKENS
        prompts = (
            BenchmarkPrompt(
                f"longctx-{ctx_token_count}",
                "longctx",
                build_long_context_prompt(DEFAULT_PROMPT, ctx_token_count),
                "synthetic",
            ),
        )
    elif suite in HF_DATASET_SUITES:
        prompts = load_hf_prompts(args)
    else:
        prompts = (BenchmarkPrompt("smoke-default", "smoke", DEFAULT_PROMPT, "smoke"),)
    return list(prompts[:limit])

def load_prompt_file(path: str | Path) -> list[BenchmarkPrompt]:
    prompts: list[BenchmarkPrompt] = []
    prompt_path = Path(path)
    with prompt_path.open() as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{prompt_path}:{line_no}: invalid JSONL row") from exc
            prompt_id = str(payload.get("id") or "").strip()
            prompt_text = str(payload.get("prompt") or "")
            if not prompt_id:
                raise ValueError(f"{prompt_path}:{line_no}: missing id")
            if not prompt_text:
                raise ValueError(f"{prompt_path}:{line_no}: missing prompt")
            prompts.append(
                BenchmarkPrompt(
                    id=prompt_id,
                    suite=str(payload.get("suite") or "custom"),
                    prompt=prompt_text,
                    source="jsonl",
                )
            )
    if not prompts:
        raise ValueError(f"{prompt_path}: no prompts found")
    return prompts

def datasets_load_dataset():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install datasets to use --suite humaneval/gsm8k/math500, "
            "or use --prompt-file PATH."
        ) from exc
    return load_dataset

def load_hf_dataset(name: str, config: str | None, split: str):
    try:
        load_dataset = datasets_load_dataset()
    except ImportError as exc:
        raise RuntimeError(
            "Install datasets to use --suite humaneval/gsm8k/math500, "
            "or use --prompt-file PATH."
        ) from exc
    if config is None:
        return load_dataset(name, split=split)
    return load_dataset(name, config, split=split)

def load_hf_prompts(args: argparse.Namespace) -> list[BenchmarkPrompt]:
    suite = str(args.suite)
    meta = HF_DATASET_SUITES[suite]
    dataset = load_hf_dataset(str(meta["name"]), meta["config"], str(meta["split"]))
    rows = _dataset_rows(dataset)
    if args.shuffle:
        rng = random.Random(int(args.seed))
        rng.shuffle(rows)
    rows = rows[: int(args.limit)]
    return [_format_hf_prompt(suite, row_index, row) for row_index, row in rows]

def _dataset_rows(dataset: Any) -> list[tuple[int, dict[str, Any]]]:
    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        return [(idx, dict(dataset[idx])) for idx in range(len(dataset))]
    return [(idx, dict(row)) for idx, row in enumerate(dataset)]

def _format_hf_prompt(suite: str, row_index: int, row: dict[str, Any]) -> BenchmarkPrompt:
    meta = HF_DATASET_SUITES[suite]
    if suite == "humaneval":
        task_id = str(row.get("task_id") or row_index)
        prompt = str(row["prompt"])
        prompt_id = f"humaneval:{task_id}"
    elif suite == "gsm8k":
        prompt = f"Question: {row['question']}\nAnswer: "
        prompt_id = f"gsm8k:{row_index}"
    elif suite == "math500":
        stable_id = str(row.get("problem_id") or row_index)
        prompt = f"Problem: {row['problem']}\nSolution: "
        prompt_id = f"math500:{stable_id}"
    else:
        raise ValueError(f"unsupported HF benchmark suite: {suite}")
    return BenchmarkPrompt(
        id=prompt_id,
        suite=suite,
        prompt=prompt,
        source="hf",
        row_index=row_index,
        hf_dataset_name=str(meta["name"]),
        hf_dataset_config=meta["config"],
        hf_dataset_split=str(meta["split"]),
    )

def slugify_prompt_id(prompt: str) -> str:
    prompt_text = str(prompt)
    head = re.sub(r"[^a-z0-9]+", "-", prompt_text[:48].lower())
    head = re.sub(r"-+", "-", head).strip("-")
    digest = hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:8]
    return f"{head}-{digest}" if head else digest
