# SPDX-License-Identifier: Apache-2.0
# gfx906-vllm: Qwen2/Qwen2.5 EAGLE draft model for speculative decoding.
# Pattern mirrors llama_eagle.py, adapted for Qwen2 decoder layers.

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM
from vllm.model_executor.models.utils import (
    get_draft_quant_config,
    maybe_prefix,
)


class EagleQwen2Model(nn.Module):
    """EAGLE draft backbone using Qwen2 decoder layers.

    The EAGLE draft model receives hidden_states from the target model,
    concatenates them with token embeddings via a fc projection, then
    runs through the single-layer draft transformer.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        start_layer_id: int = 0,
    ) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        self.quant_config = get_draft_quant_config(vllm_config)

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(
                    config=self.config,
                    cache_config=vllm_config.cache_config,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, f"layers.{i + start_layer_id}"),
                )
                for i in range(self.config.num_hidden_layers)
            ]
        )

        # fc: cat(embed, hidden) → hidden  (EAGLE core fusion)
        self.fc = ReplicatedLinear(
            input_size=self.config.hidden_size * 2,
            output_size=self.config.hidden_size,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "fc"),
            return_bias=False,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_embeds = self.embed_tokens(input_ids)
        hidden_states = self.fc(torch.cat((input_embeds, hidden_states), dim=-1))
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        if residual is not None:
            hidden_states = hidden_states + residual
        return hidden_states, hidden_states


class EagleQwen2ForCausalLM(Qwen2ForCausalLM):
    """EAGLE draft model for Qwen2/Qwen2.5 target models.

    Registered as EagleQwen2ForCausalLM (and EagleQwen25ForCausalLM) in the
    model registry. vLLM's EAGLEConfig renames Qwen2ForCausalLM →
    EagleQwen2ForCausalLM when method='eagle'.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)

        self._start_layer_id = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = EagleQwen2Model(
            vllm_config=vllm_config,
            prefix="model",
            start_layer_id=self._start_layer_id,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size, scale=logit_scale
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support multimodal inputs yet."
            )
        return self.model(input_ids, positions, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        import re as _re
        start_id = self._start_layer_id
        params = dict(self.named_parameters())

        for name, loaded_weight in weights:
            # Prefix with "model." (draft weights don't have this)
            if "lm_head" not in name:
                mapped = "model." + name
            else:
                mapped = name
            # Remap layer indices: file has layers.0..N, model has layers.S..S+N
            if start_id > 0:
                mapped = _re.sub(
                    r"(model\.layers\.)(\d+)\.",
                    lambda m: f"{m.group(1)}{int(m.group(2)) + start_id}.",
                    mapped,
                )
            if mapped in params:
                param = params[mapped]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            # skip weights not present in model (e.g. input_layernorm missing in draft)
