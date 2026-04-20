# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# gfx906-vllm: Attention lives in vllm.attention.layer (not attention.attention)
# Only import what Gemma4 (and other models) actually need for gfx906.
from vllm.attention.layer import Attention

__all__ = ["Attention"]
