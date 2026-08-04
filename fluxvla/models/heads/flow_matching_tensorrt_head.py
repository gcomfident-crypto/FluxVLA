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
"""TensorRT-only GR00T-N1.5 flow-matching inference head."""

from __future__ import annotations
import threading
from pathlib import Path
from typing import Optional

import torch

from fluxvla.deployment.tensorrt.groot_n15 import (
    GR00TN15ActionHeadTensorRTInference, GR00TN15TensorRTContract)
from fluxvla.deployment.tensorrt.groot_n15.runtime import \
    _validate_embodiment_metadata
from fluxvla.engines import HEADS
from .flow_matching_head import FlowMatchingHead


@HEADS.register_module()
class FlowMatchingTensorRTHead(FlowMatchingHead):
    """Execute the five action-side GR00T plans and Euler integration.

    ``future_tokens`` and the action position embedding remain ordinary
    PyTorch operations, matching NVIDIA's official seven-engine partition.
    Every learned neural module covered by an engine is released immediately
    before plan deserialization to bound peak memory.  RTC is rejected because
    the exported action encoder has
    one global timestep per sample rather than the per-position RTC schedule.
    """

    supports_predict_action_controls = True
    predict_action_control_names = frozenset(
        ('noise', 'seed', 'stream', 'num_inference_steps'))

    def __init__(
        self,
        engine_dir: str | Path,
        image_views: int = 2,
        sequence_length: int = 600,
        image_size: int = 224,
        vision_patches: int = 256,
        projected_image_tokens_per_view: int = 256,
        vision_hidden: int = 1152,
        profile_index: int = 0,
        strict_manifest: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        state_dim = int(self.state_encoder.layer1.W.shape[1])
        self.max_num_embodiments = int(self.action_encoder.num_embodiments)
        future_tokens = int(self.future_tokens.num_embeddings)
        backbone_dim = int(self.vlln.normalized_shape[0])
        contract = GR00TN15TensorRTContract(
            image_views=image_views,
            image_size=image_size,
            vision_patches=vision_patches,
            projected_image_tokens_per_view=projected_image_tokens_per_view,
            sequence_length=sequence_length,
            state_dim=state_dim,
            action_horizon=int(self.num_steps),
            action_dim=int(self.action_dim),
            future_tokens=future_tokens,
            backbone_dim=backbone_dim,
            vision_hidden=vision_hidden,
            input_embedding_dim=int(self.input_embedding_dim),
            hidden_size=int(self.hidden_size),
            denoise_steps=int(self.num_inference_timesteps))
        if not self.add_positional_embeddings:
            raise ValueError('GR00T-N1.5 TensorRT export requires action '
                             'position embeddings')
        self.trt_runtime = GR00TN15ActionHeadTensorRTInference(
            engine_dir,
            contract=contract,
            profile_index=profile_index,
            strict_manifest=strict_manifest)
        self.trt_contract = contract
        self._trt_ready = False
        self._trt_release_started = False
        self._inference_lock = threading.Lock()

    def _prepare_runtime(self, device: torch.device) -> None:
        if self._trt_ready:
            return
        if self._trt_release_started:
            raise RuntimeError(
                'GR00T TensorRT head preparation previously failed after '
                'releasing engine-covered PyTorch modules')
        retained_parameters = (
            self.future_tokens.weight,
            self.position_embedding.weight,
        )
        if any(parameter.device != device
               for parameter in retained_parameters):
            raise RuntimeError(
                'Move the GR00T model to its CUDA device before preparing '
                'TensorRT')
        if any(parameter.dtype != torch.float16
               for parameter in retained_parameters):
            raise RuntimeError(
                'GR00T-N1.5 TensorRT retained layers must use FP16')
        # NVIDIA's official setup releases the PyTorch partitions before
        # loading their plans to keep peak device memory bounded.
        self._trt_release_started = True
        del self.vlln
        del self.vl_self_attention
        del self.state_encoder
        del self.action_encoder
        del self.model
        del self.action_decoder
        torch.cuda.empty_cache()
        self.trt_runtime.prepare(device)
        self._trt_ready = True

    def _validate_inputs(self, input_features: torch.Tensor,
                         states: torch.Tensor, attention_mask: torch.Tensor,
                         embodiment_ids: torch.Tensor,
                         noise: Optional[torch.Tensor]) -> torch.device:
        tensors = {
            'input_features': input_features,
            'states': states,
            'attention_mask': attention_mask,
            'embodiment_ids': embodiment_ids,
        }
        if not all(
                isinstance(value, torch.Tensor) for value in tensors.values()):
            raise TypeError('GR00T TensorRT head inputs must be tensors')
        device = input_features.device
        if device.type != 'cuda':
            raise ValueError('GR00T TensorRT head inputs must be on CUDA')
        if any(value.device != device for value in tensors.values()):
            raise ValueError(
                'GR00T TensorRT inputs must share one CUDA device')
        c = self.trt_contract
        expected = {
            'input_features': (c.batch, c.sequence_length, c.backbone_dim),
            'states': (c.batch, c.state_dim),
            'attention_mask': (c.batch, c.sequence_length),
            'embodiment_ids': (c.batch, ),
        }
        for name, shape in expected.items():
            if tuple(tensors[name].shape) != shape:
                raise ValueError(f'{name} shape must be {shape}, got '
                                 f'{tuple(tensors[name].shape)}')
        if embodiment_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError('embodiment_ids must be INT32 or INT64')
        _validate_embodiment_metadata(embodiment_ids, self.max_num_embodiments)
        if noise is not None:
            expected_noise = (c.batch, c.action_horizon, c.action_dim)
            if tuple(noise.shape) != expected_noise:
                raise ValueError(f'noise shape must be {expected_noise}')
            if noise.device != device:
                raise ValueError('noise must share the input CUDA device')
        return device

    def predict_action(
        self,
        input_features: torch.Tensor,
        states: torch.Tensor,
        attention_mask: torch.Tensor,
        embodiment_ids: torch.Tensor,
        prev_actions=None,
        prefix_len: int = 0,
        rtc_config: dict = None,
        noise: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        stream: Optional[torch.cuda.Stream] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        with self._inference_lock:
            return self._predict_action_locked(
                input_features=input_features,
                states=states,
                attention_mask=attention_mask,
                embodiment_ids=embodiment_ids,
                prev_actions=prev_actions,
                prefix_len=prefix_len,
                rtc_config=rtc_config,
                noise=noise,
                seed=seed,
                stream=stream,
                args=args,
                kwargs=kwargs)

    def _predict_action_locked(
        self,
        input_features: torch.Tensor,
        states: torch.Tensor,
        attention_mask: torch.Tensor,
        embodiment_ids: torch.Tensor,
        prev_actions,
        prefix_len: int,
        rtc_config: Optional[dict],
        noise: Optional[torch.Tensor],
        seed: Optional[int],
        stream: Optional[torch.cuda.Stream],
        args: tuple,
        kwargs: dict,
    ) -> torch.Tensor:
        if (prev_actions is not None or prefix_len != 0
                or rtc_config is not None):
            raise ValueError('GR00T-N1.5 TensorRT does not support RTC')
        if args:
            raise TypeError('Additional positional arguments are unsupported')
        requested_steps = kwargs.pop('num_inference_steps', None)
        if kwargs:
            raise TypeError(f'Unsupported TensorRT head arguments: '
                            f'{sorted(kwargs)}')
        if (requested_steps is not None
                and int(requested_steps) != self.num_inference_timesteps):
            raise ValueError('TensorRT plans use a fixed number of denoising '
                             'steps')

        device = self._validate_inputs(input_features, states, attention_mask,
                                       embodiment_ids, noise)
        self._prepare_runtime(device)
        c = self.trt_contract
        caller_stream = torch.cuda.current_stream(device)
        selected_stream = stream or caller_stream
        if torch.device(selected_stream.device) != device:
            raise ValueError('GR00T execution stream is on the wrong CUDA '
                             'device')
        if (int(caller_stream.cuda_stream) != int(
                selected_stream.cuda_stream)):
            selected_stream.wait_stream(caller_stream)
        for source in (input_features, states, attention_mask, embodiment_ids,
                       noise):
            if source is not None:
                source.record_stream(selected_stream)
        with torch.cuda.stream(selected_stream):
            features = input_features.to(torch.float16).contiguous()
            embodiment = embodiment_ids.to(torch.int64).contiguous()
            state = states.to(torch.float16).unsqueeze(1).contiguous()
            features = self.trt_runtime.process_backbone(
                features, selected_stream)
            state_features = self.trt_runtime.encode_state(
                state, embodiment, selected_stream)

            if noise is None:
                generator = None
                if seed is not None:
                    if type(seed) is not int or seed < 0:
                        raise ValueError('seed must be a non-negative integer')
                    generator = torch.Generator(device=device)
                    generator.manual_seed(seed)
                actions = torch.randn(
                    c.batch,
                    c.action_horizon,
                    c.action_dim,
                    dtype=torch.float16,
                    device=device,
                    generator=generator)
            else:
                actions = noise.to(torch.float16).contiguous()

            timestep = torch.empty((c.batch, ),
                                   dtype=torch.int64,
                                   device=device)
            position_ids = torch.arange(
                c.action_horizon, dtype=torch.long, device=device)
            position = self.position_embedding(position_ids).unsqueeze(0)
            position = position.to(torch.float16)
            future = self.future_tokens.weight.unsqueeze(0).to(torch.float16)
            dt = 1.0 / c.denoise_steps

            for step in range(c.denoise_steps):
                discretized = int((step / float(c.denoise_steps)) *
                                  self.num_timestep_buckets)
                timestep.fill_(discretized)
                action_features = self.trt_runtime.encode_action(
                    actions.contiguous(), timestep, embodiment,
                    selected_stream)
                action_features = action_features + position
                sa_embs = torch.cat((state_features, future, action_features),
                                    dim=1)
                model_output = self.trt_runtime.run_dit(
                    sa_embs.contiguous(), features, timestep, selected_stream)
                prediction = self.trt_runtime.decode_action(
                    model_output, embodiment, selected_stream)
                velocity = prediction[:, -c.action_horizon:]
                actions = actions + dt * velocity

            result = actions.clone()
            if self.ori_action_dim is not None:
                result = result[:, :, :self.ori_action_dim]
        if (int(caller_stream.cuda_stream) != int(
                selected_stream.cuda_stream)):
            caller_stream.wait_stream(selected_stream)
            result.record_stream(caller_stream)
        return result

    def bind_eval_checkpoint(self, checkpoint_path) -> None:
        self.trt_runtime.bind_checkpoint(checkpoint_path)

    def prepare_inference_runtime(self, device_id=None) -> None:
        if device_id is None:
            device_id = torch.cuda.current_device()
        if isinstance(device_id, int):
            device_id = torch.device('cuda', device_id)
        device = torch.device(device_id)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        self._prepare_runtime(device)

    def forward(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            'FlowMatchingTensorRTHead has no PyTorch forward path; call '
            'predict_action()')

    def close_inference_runtime(self) -> None:
        self.trt_runtime.close()

    def __del__(self):
        runtime = getattr(self, 'trt_runtime', None)
        if runtime is not None:
            try:
                runtime.close()
            except Exception:
                pass
