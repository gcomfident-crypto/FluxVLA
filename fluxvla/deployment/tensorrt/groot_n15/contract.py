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
"""Static TensorRT deployment contract for GR00T-N1.5."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class GR00TN15TensorRTContract:
    """Static deployment contract shared by all seven TensorRT plans."""

    batch: int = 1
    image_views: int = 2
    image_size: int = 224
    vision_patches: int = 256
    projected_image_tokens_per_view: int = 256
    sequence_length: int = 600
    state_dim: int = 64
    action_horizon: int = 10
    action_dim: int = 32
    future_tokens: int = 32
    backbone_dim: int = 2048
    vision_hidden: int = 1152
    input_embedding_dim: int = 1536
    hidden_size: int = 1024
    denoise_steps: int = 4

    @property
    def sa_length(self) -> int:
        return 1 + self.future_tokens + self.action_horizon

    def __post_init__(self) -> None:
        invalid = {
            name: value
            for name, value in vars(self).items()
            if not isinstance(value, int) or value <= 0
        }
        if invalid:
            raise ValueError(f'GR00T contract values must be positive: '
                             f'{invalid}')
        if self.batch != 1:
            raise ValueError('GR00T-N1.5 TensorRT v1 supports batch size 1')
        image_tokens = (
            self.image_views * self.projected_image_tokens_per_view)
        if self.sequence_length < image_tokens:
            raise ValueError(
                'GR00T sequence_length must hold every projected image '
                f'token: {self.sequence_length} < {image_tokens}')


__all__ = ['GR00TN15TensorRTContract']
