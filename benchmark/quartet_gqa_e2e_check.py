# Copyright 2026 bstnxbt
# MIT License — see LICENSE file

from __future__ import annotations

import argparse

from dflash_mlx.generate import get_stop_token_ids
from dflash_mlx.runtime import configure_quartet_gqa, generate_dflash_once, load_draft_bundle, load_target_bundle

MATH_PROMPT = (
    "The function $f$ satisfies the functional equation "
    "\\[ f(x) + f(y) = f(x + y) - xy - 1 \\] for all real numbers $x$ and $y$. "
    "If $f(1) = 1$, then find all integers $n$ such that $f(n) = n$. "
    "Enter all such integers, separated by commas. Please reason step by step, "
    "and put your final answer within \\boxed{}."
)

CODE_PROMPT = (
    "Write a Python implementation of Kahn's algorithm for topological sorting, "
    "include cycle detection, explain the invariants, and discuss the time and "
    "space complexity in detail."
)

NARRATIVE_PROMPT = (
    "Write the opening scene of a literary science-fiction novella about a cartographer "
    "mapping the weather inside a rotating habitat. Keep the prose precise, vivid, "
    "and grounded in concrete sensory detail."
)


def _build_long_token_stream(tokenizer, *, seed_text: str, total_tokens: int) -> list[int]:
    chunk = list(tokenizer.encode(seed_text + "\n\n"))
    if not chunk:
        raise ValueError("Tokenizer produced no tokens for e2e-check seed prompt")
    tokens: list[int] = []
    while len(tokens) < total_tokens:
        tokens.extend(chunk)
    return tokens[:total_tokens]


def _run_once(
    *,
    target_model,
    tokenizer,
    draft_model,
    prompt_label: str,
    prompt_tokens: list[int],
    max_new_tokens: int,
    block_tokens: int,
    quartet_enabled: bool,
) -> dict:
    configure_quartet_gqa(target_model, enabled=quartet_enabled)
    eos_token_ids = get_stop_token_ids(tokenizer)
    return generate_dflash_once(
        target_model=target_model,
        tokenizer=tokenizer,
        draft_model=draft_model,
        prompt=prompt_label,
        prompt_tokens_override=prompt_tokens,
        max_new_tokens=max_new_tokens,
        use_chat_template=False,
        block_tokens=block_tokens,
        verify_chunk_tokens=None,
        stop_token_ids=[],
        suppress_token_ids=eos_token_ids,
        quantize_kv_cache=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end quartet-GQA token-match check on 3 long prompts.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--draft", default="z-lab/Qwen3.5-9B-DFlash")
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--block-tokens", type=int, default=16)
    args = parser.parse_args()

    target_model, tokenizer, _ = load_target_bundle(
        args.model,
        lazy=True,
        split_full_attention_sdpa=True,
    )
    draft_model, _ = load_draft_bundle(args.draft, lazy=True)

    prompts = {
        "code": _build_long_token_stream(
            tokenizer,
            seed_text=CODE_PROMPT,
            total_tokens=int(args.context_tokens),
        ),
        "math": _build_long_token_stream(
            tokenizer,
            seed_text=MATH_PROMPT,
            total_tokens=int(args.context_tokens),
        ),
        "narrative": _build_long_token_stream(
            tokenizer,
            seed_text=NARRATIVE_PROMPT,
            total_tokens=int(args.context_tokens),
        ),
    }

    failures: list[dict[str, object]] = []
    for prompt_name, prompt_tokens in prompts.items():
        ref = _run_once(
            target_model=target_model,
            tokenizer=tokenizer,
            draft_model=draft_model,
            prompt_label=prompt_name,
            prompt_tokens=prompt_tokens,
            max_new_tokens=int(args.max_new_tokens),
            block_tokens=int(args.block_tokens),
            quartet_enabled=False,
        )
        test = _run_once(
            target_model=target_model,
            tokenizer=tokenizer,
            draft_model=draft_model,
            prompt_label=prompt_name,
            prompt_tokens=prompt_tokens,
            max_new_tokens=int(args.max_new_tokens),
            block_tokens=int(args.block_tokens),
            quartet_enabled=True,
        )
        token_match = ref["generated_token_ids"] == test["generated_token_ids"]
        result = {
            "prompt": prompt_name,
            "token_match": token_match,
            "acceptance_ref": float(ref.get("acceptance_ratio", 0.0)),
            "acceptance_test": float(test.get("acceptance_ratio", 0.0)),
            "elapsed_us_ref": float(ref.get("elapsed_us", 0.0)),
            "elapsed_us_test": float(test.get("elapsed_us", 0.0)),
        }
        print(result)
        if not token_match:
            failures.append(result)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
