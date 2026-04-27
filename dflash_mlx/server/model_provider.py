# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import sys
import time

import mlx.core as mx
import mlx_lm.server as mlx_server

from dflash_mlx.generate import (
    load_runtime_components,
    resolve_optional_draft_ref,
)


class DFlashModelProvider(mlx_server.ModelProvider):
    def load(self, model_path, adapter_path=None, draft_model_path=None):
        default_map = getattr(self, "_model_map", None)
        if default_map is None:
            default_map = getattr(self, "default_model_map", {})
        requested_model = default_map.get(model_path, model_path)
        if self.cli_args.model is not None:
            model_ref = self.cli_args.model
        elif requested_model == "default_model":
            raise ValueError(
                "A model path has to be given as a CLI argument or in the HTTP request"
            )
        else:
            model_ref = requested_model

        if draft_model_path == "default_model":
            draft_ref = self.cli_args.draft_model
        elif draft_model_path is not None:
            draft_ref = draft_model_path
        else:
            draft_ref = None
        resolved_draft_ref = resolve_optional_draft_ref(model_ref, draft_ref)

        if self.model_key == (model_ref, None, resolved_draft_ref):
            return self.model, self.tokenizer

        self.model = None
        self.tokenizer = None
        self.model_key = None
        self.draft_model = None

        draft_quant = getattr(self.cli_args, "draft_quant", None)
        if not draft_quant and getattr(self.cli_args, "quantize_draft", False):
            draft_quant = "w4a16"
        model, tokenizer, draft_model, resolved_draft_ref = load_runtime_components(
            model_ref=model_ref,
            draft_ref=draft_ref,
            draft_quant=draft_quant or None,
        )

        if self.cli_args.chat_template:
            tokenizer.chat_template = self.cli_args.chat_template
        if self.cli_args.use_default_chat_template and tokenizer.chat_template is None:
            tokenizer.chat_template = tokenizer.default_chat_template

        self.model = model
        self.tokenizer = tokenizer
        self.draft_model = draft_model
        self.model_key = (model_ref, None, resolved_draft_ref)

        try:
            mx.eval(model.parameters())
            if draft_model is not None:
                mx.eval(draft_model.parameters())
        except Exception as _eval_err:
            sys.stderr.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] weight "
                f"materialize failed (non-fatal): "
                f"{type(_eval_err).__name__}: {_eval_err}\n"
            )
            sys.stderr.flush()

        return self.model, self.tokenizer


def wait_for_initial_model_load(
    model_provider: DFlashModelProvider,
    *,
    timeout_s: float = 300.0,
    poll_interval_s: float = 0.2,
) -> None:
    start = time.perf_counter()
    announced = False
    while model_provider.model_key is None:
        if not announced:
            sys.stderr.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] loading model "
                f"on generation worker thread...\n"
            )
            sys.stderr.flush()
            announced = True
        if time.perf_counter() - start > timeout_s:
            raise RuntimeError(
                f"DFlash generation worker failed to load model within "
                f"{timeout_s}s; check earlier log lines for the underlying error."
            )
        time.sleep(poll_interval_s)
