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
"""Registry adapter for the TensorRT-backed PI0.5 V/P/D runtime."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from fluxvla.deployment.tensorrt.pi05 import (PI05ExportContract,
                                              PI05TensorRTInference)
from fluxvla.engines import VLAS


@VLAS.register_module()
class PI05FlowMatchingTensorRTInference(nn.Module):
    """Load and execute static PI0.5 TensorRT engines.

    The module intentionally owns no neural parameters.  FluxVLA's evaluation
    runner verifies the requested checkpoint against the engine manifest and
    skips state-dict loading for this adapter.  The actual weights are
    exclusively those serialized into the V/P/D plans.
    """

    def __init__(
        self,
        engine_dir: str | Path,
        image_views: int = 2,
        language_tokens: int = 180,
        n_action_steps: int = 10,
        max_action_dim: int = 32,
        num_steps: int = 10,
        profile_index: int = 0,
        strict_manifest: bool = True,
        device: Optional[str] = None,
        torch_dtype: Optional[Any] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            raise TypeError(
                f'Unsupported PI0.5 TensorRT arguments: {sorted(kwargs)}')
        if torch_dtype not in (None, torch.bfloat16, 'bfloat16', 'bf16'):
            raise ValueError('PI0.5 TensorRT plans require BF16 neural I/O')
        if strict_manifest is not True:
            raise ValueError('strict_manifest must be true in production')
        contract = PI05ExportContract(
            image_views=image_views,
            language_tokens=language_tokens,
            action_horizon=n_action_steps,
            action_dim=max_action_dim,
            denoise_steps=num_steps)
        self.runtime = PI05TensorRTInference(
            engine_dir,
            contract=contract,
            profile_index=profile_index,
            strict_manifest=strict_manifest)
        self.configured_device = device
        self.n_action_steps = contract.action_horizon
        self.max_action_dim = contract.action_dim
        self.num_steps = contract.denoise_steps
        self.norm_stats = None
        self.checkpoint_loading_mode = 'engine_only'
        # Compatibility attributes used by FluxVLA runners.
        self.freeze_vision_backbone = True
        self.freeze_llm_backbone = True
        self.freeze_projector = True
        self.freeze_vlm_backbone = True

    @property
    def device(self) -> torch.device:
        if self.runtime.device is not None:
            return self.runtime.device
        if self.configured_device is not None:
            return torch.device(self.configured_device)
        return torch.device('cuda', torch.cuda.current_device())

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    def load_state_dict(self, state_dict, strict: bool = True, assign=False):
        """Reject state dicts because weights live in verified plans."""
        del state_dict, strict, assign
        raise RuntimeError(
            'PI05FlowMatchingTensorRTInference is engine-only; verify the '
            'checkpoint with bind_eval_checkpoint() instead of loading a '
            'PyTorch state dict')

    def bind_eval_checkpoint(self, checkpoint_path) -> None:
        """Fail closed if the eval checkpoint differs from engine weights."""
        self.runtime.bind_checkpoint(checkpoint_path)

    def prepare_inference_runtime(self, device_id=None) -> None:
        if device_id is None:
            device_id = self.configured_device
        if device_id is None:
            device_id = torch.cuda.current_device()
        if isinstance(device_id, int):
            device_id = torch.device('cuda', device_id)
        self.runtime.prepare(device_id)

    def close_inference_runtime(self) -> None:
        self.runtime.close()

    def configure_inference_profile(self, enabled: bool) -> None:
        self.runtime.configure_inference_profile(enabled)

    def resolve_inference_profile(self):
        return self.runtime.resolve_inference_profile()

    def __del__(self):
        # The runner lifecycle hook performs normal teardown.  Keep this
        # best-effort fallback for direct use or partially constructed models.
        runtime = getattr(self, 'runtime', None)
        if runtime is not None:
            try:
                runtime.close()
            except Exception:
                pass

    def predict_action(
        self,
        images: torch.Tensor,
        lang_tokens: torch.Tensor,
        states: torch.Tensor,
        img_masks: torch.Tensor,
        lang_masks: torch.Tensor,
        past_key_values=None,
        noise=None,
        prev_actions=None,
        prefix_len: int = 0,
        rtc_config: Optional[dict] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if past_key_values is not None:
            raise ValueError('External past_key_values are unsupported')
        if prev_actions is not None or prefix_len != 0 or rtc_config:
            raise ValueError('PI0.5 TensorRT v1 does not support RTC')
        if args:
            raise TypeError('Additional positional arguments are unsupported')
        allowed = {
            'unnorm_key', 'num_inference_steps', 'seed', 'reset_history',
            'stream'
        }
        unknown = set(kwargs) - allowed
        if unknown:
            raise TypeError(
                f'Unsupported predict_action arguments: {sorted(unknown)}')
        requested_steps = kwargs.get('num_inference_steps')
        if requested_steps is not None and int(requested_steps) != 10:
            raise ValueError('PI0.5 TensorRT v1 always uses 10 Euler steps')
        reset_history = kwargs.get('reset_history')
        if reset_history is not None and not isinstance(reset_history, bool):
            raise TypeError('reset_history must be bool')
        return self.runtime.predict_action(
            images,
            lang_tokens,
            states,
            img_masks,
            lang_masks,
            noise=noise,
            seed=kwargs.get('seed'),
            stream=kwargs.get('stream'))

    def forward(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            'PI05FlowMatchingTensorRTInference has no PyTorch forward path; '
            'call predict_action()')
