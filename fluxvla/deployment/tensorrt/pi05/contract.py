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
"""Static TensorRT deployment contract for PI0.5."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class PI05ExportContract:
    """Static PI0.5 deployment profile used by the first implementation."""

    batch: int = 1
    image_views: int = 2
    image_height: int = 224
    image_width: int = 224
    image_tokens_per_view: int = 256
    language_tokens: int = 180
    kv_layers: int = 18
    kv_heads: int = 1
    head_dim: int = 256
    action_horizon: int = 10
    action_dim: int = 32
    denoise_steps: int = 10
    prefix_hidden: int = 2048
    expert_hidden: int = 1024

    @property
    def image_tokens(self) -> int:
        return self.image_views * self.image_tokens_per_view

    @property
    def prefix_tokens(self) -> int:
        return self.image_tokens + self.language_tokens

    @property
    def cache_shape(self) -> tuple[int, ...]:
        return (self.batch, self.kv_heads, self.prefix_tokens, self.head_dim)

    def __post_init__(self) -> None:
        invalid = {
            name: value
            for name, value in vars(self).items()
            if not isinstance(value, int) or value <= 0
        }
        if invalid:
            raise ValueError(f'PI0.5 contract values must be positive: '
                             f'{invalid}')
        if self.batch != 1:
            raise ValueError('PI0.5 TensorRT v1 supports batch size 1')
        if self.action_horizon != 10 or self.denoise_steps != 10:
            raise ValueError(
                'PI0.5 TensorRT v1 requires action horizon 10 and exactly '
                '10 Euler steps')


def pi05_cache_names(contract: PI05ExportContract) -> tuple[str, ...]:
    """Return the ordered K/V binding names for every prefix layer."""
    return tuple(
        name for layer in range(contract.kv_layers)
        for name in (f'prefix_k_{layer:02d}', f'prefix_v_{layer:02d}'))


__all__ = ['PI05ExportContract', 'pi05_cache_names']
