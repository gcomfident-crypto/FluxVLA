# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ONNX-exportable partitions for GR00T-N1.5.

The wrappers follow NVIDIA Isaac-GR00T's seven-engine partition while adapting
the model boundaries to FluxVLA's Eagle backbone and action head.
"""

from __future__ import annotations

import torch
from torch import nn


class GR00TVisionExportWrapper(nn.Module):
    """Eagle SigLIP vision transformer with explicit position IDs.

    The explicit-position-ID export interface follows the design described by
    NVIDIA's N1.5 deployment, but this wrapper is independently implemented
    against FluxVLA's vendored Eagle model object.
    """

    def __init__(self, eagle_backbone: nn.Module) -> None:
        super().__init__()
        self.vision = eagle_backbone.vlm.vision_model.vision_model
        self.eval()

    def forward(self, pixel_values: torch.Tensor,
                position_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.vision.embeddings
        patch = embeddings.patch_embedding(
            pixel_values.to(embeddings.patch_embedding.weight.dtype))
        hidden = patch.flatten(2).transpose(1, 2)
        hidden = hidden + embeddings.position_embedding(position_ids)
        output = self.vision.encoder(
            inputs_embeds=hidden,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True)
        return self.vision.post_layernorm(output.last_hidden_state)


class GR00TLLMExportWrapper(nn.Module):
    """Selected Eagle language layers and final feature projection."""

    def __init__(self, eagle_backbone: nn.Module) -> None:
        super().__init__()
        self.language_model = eagle_backbone.vlm.language_model
        self.eagle_linear = eagle_backbone.eagle_linear
        self.select_layer = int(eagle_backbone.select_layer)
        self.eval()

    def forward(self, inputs_embeds: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True)
        return self.eagle_linear(output.hidden_states[self.select_layer])


class GR00TBackboneOutputExportWrapper(nn.Module):
    """VLLN and VL self-attention partition from the GR00T action head."""

    def __init__(self, action_head: nn.Module) -> None:
        super().__init__()
        self.vlln = action_head.vlln
        self.vl_self_attention = action_head.vl_self_attention
        self.eval()

    def forward(self, backbone_features: torch.Tensor) -> torch.Tensor:
        return self.vl_self_attention(self.vlln(backbone_features))


__all__ = [
    'GR00TVisionExportWrapper',
    'GR00TLLMExportWrapper',
    'GR00TBackboneOutputExportWrapper',
]
