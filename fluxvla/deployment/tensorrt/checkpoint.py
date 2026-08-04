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
"""Checkpoint helpers shared by TensorRT export and runtime adapters."""

from __future__ import annotations
from typing import Mapping

import torch


def checkpoint_state_for_model(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Resolve native or configured source keys to one exact model state."""
    target = model.state_dict()
    if set(state) == set(target) and all(
            tuple(state[name].shape) == tuple(value.shape)
            for name, value in target.items()):
        return dict(state)

    candidate_fn = getattr(model, '_mapped_name_candidates', None)
    if not callable(candidate_fn) or not getattr(model, 'name_mapping', None):
        missing = sorted(set(target) - set(state))
        unexpected = sorted(set(state) - set(target))
        raise RuntimeError(
            'Checkpoint does not exactly match the export model; '
            f'missing={missing[:8]}, unexpected={unexpected[:8]}')

    remapped = {}
    used = set()
    for name, target_value in target.items():
        candidates = [name]
        candidates.extend(mapped for _, mapped in candidate_fn(name))
        matches = [
            candidate for candidate in dict.fromkeys(candidates)
            if candidate in state
            and tuple(state[candidate].shape) == tuple(target_value.shape)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f'Checkpoint tensor {name!r} resolved to {matches}; '
                'expected exactly one shape-compatible source tensor')
        source = matches[0]
        if source in used:
            raise RuntimeError(
                f'Checkpoint source tensor {source!r} resolves to more than '
                'one model tensor; name mapping must be one-to-one')
        remapped[name] = state[source]
        used.add(source)
    unexpected = sorted(set(state) - used)
    if unexpected:
        raise RuntimeError(
            'Checkpoint contains tensors unused by the export model: '
            f'{unexpected[:8]}')
    return remapped


__all__ = ['checkpoint_state_for_model']
