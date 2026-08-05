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
"""Root LlavaVLA adapter for checkpoint-safe TensorRT deployment."""

from __future__ import annotations
import threading
from typing import Mapping

import torch

from fluxvla.deployment.tensorrt.checkpoint import checkpoint_state_for_model
from fluxvla.engines import VLAS
from .llava_vla import LlavaVLA


@VLAS.register_module()
class LlavaTensorRT(LlavaVLA):
    """Apply the exporter's one-to-one checkpoint mapping before eval load."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The backbone and head own separate TensorRT context groups. Hold one
        # root transaction lock so concurrent callers cannot overwrite a
        # persistent backbone output before the matching head consumes it.
        self._tensorrt_predict_lock = threading.RLock()

    def prepare_eval_state_dict(
            self,
            state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return checkpoint_state_for_model(self, state_dict)

    def predict_action(self, *args, **kwargs):
        with self._tensorrt_predict_lock:
            return super().predict_action(*args, **kwargs)
