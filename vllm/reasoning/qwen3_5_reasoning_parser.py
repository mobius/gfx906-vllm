# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ResponsesRequest
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser


class Qwen3_5ReasoningParser(BaseThinkingReasoningParser):
    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self.thinking_enabled = chat_kwargs.get("enable_thinking", False)

    @staticmethod
    def _strip_stray_end_tokens(text: str) -> str:
        return re.sub(r'^(?:</think>\s*)+', '', text)

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        model_output = self._strip_stray_end_tokens(model_output)

        if self.start_token in model_output:
            _, _, after_start = model_output.partition(self.start_token)
            if self.end_token in after_start:
                reasoning, _, content = after_start.partition(self.end_token)
                if self.thinking_enabled and getattr(request, "include_reasoning", False):
                    return reasoning or None, content or None
                final_content = content if content else reasoning
                return None, final_content or None

            if self.thinking_enabled and getattr(request, "include_reasoning", False):
                return after_start or None, None
            return None, after_start or None

        if self.end_token in model_output:
            reasoning, _, content = model_output.partition(self.end_token)
            if self.thinking_enabled and getattr(request, "include_reasoning", False):
                return reasoning or None, content or None
            final_content = content if content else reasoning
            return None, final_content or None

        return None, model_output or None
