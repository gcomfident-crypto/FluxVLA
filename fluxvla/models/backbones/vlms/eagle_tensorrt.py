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
"""TensorRT-backed Eagle backbone for GR00T-N1.5 inference."""

from __future__ import annotations
import threading
from pathlib import Path
from typing import Dict, Optional

import torch

from fluxvla.deployment.tensorrt.groot_n15 import (
    GR00TN15BackboneTensorRTInference, GR00TN15TensorRTContract,
    groot_dtype_precision)
from fluxvla.engines import VLM_BACKBONES, str_to_dtype
# yapf: disable
from fluxvla.engines.utils.inference_profile import (begin_module_profile,
                                                     configure_module_profile,
                                                     module_stage,
                                                     resolve_module_profile,
                                                     start_module_stage,
                                                     stop_module_stage)
# yapf: enable
from .eagle import EagleBackbone


@VLM_BACKBONES.register_module()
class EagleTensorRTBackbone(EagleBackbone):
    """Run Eagle's ViT and selected LLM layers with TensorRT plans.

    The token embedding, image-token splice, and Eagle ``mlp1`` connector are
    the lightweight PyTorch operations retained by NVIDIA's official seven-plan
    decomposition.  The checkpoint is loaded into the ordinary Eagle module by
    existing FluxVLA runners before the first call.  As in NVIDIA's official
    deployment setup, engine-covered PyTorch modules are released immediately
    before the TensorRT plans are deserialized to control peak GPU memory.
    """

    def __init__(
        self,
        vlm_path: str,
        engine_dir: str | Path,
        vlm_config: Dict = None,
        project_to_dim: Optional[int] = None,
        select_layer: int = 12,
        dtype: str = 'float16',
        image_views: int = 2,
        sequence_length: int = 600,
        state_dim: int = 64,
        action_horizon: int = 10,
        action_dim: int = 32,
        future_tokens: int = 32,
        input_embedding_dim: int = 1536,
        hidden_size: int = 1024,
        denoise_steps: int = 4,
        profile_index: int = 0,
        strict_manifest: bool = True,
    ) -> None:
        trt_dtype = str_to_dtype(dtype)
        trt_precision = groot_dtype_precision(trt_dtype)
        super().__init__(
            vlm_path=vlm_path,
            vlm_config=vlm_config,
            project_to_dim=project_to_dim,
            select_layer=select_layer,
            dtype=dtype,
            attention_implementation='eager')
        self._trt_dtype = trt_dtype
        self._trt_precision = trt_precision
        config = self.vlm.config
        image_size = int(config.force_image_size
                         or config.vision_config.image_size)
        patch_size = int(config.vision_config.patch_size)
        vision_patches = (image_size // patch_size)**2
        contract = GR00TN15TensorRTContract(
            image_views=image_views,
            image_size=image_size,
            vision_patches=vision_patches,
            projected_image_tokens_per_view=int(self.vlm.num_image_token),
            sequence_length=sequence_length,
            state_dim=state_dim,
            action_horizon=action_horizon,
            action_dim=action_dim,
            future_tokens=future_tokens,
            backbone_dim=int(config.text_config.hidden_size),
            vision_hidden=int(config.vision_config.hidden_size),
            input_embedding_dim=input_embedding_dim,
            hidden_size=hidden_size,
            denoise_steps=denoise_steps)
        if contract.backbone_dim != 2048:
            raise ValueError('GR00T-N1.5 TensorRT expects Eagle hidden size '
                             '2048')
        if project_to_dim not in (None, contract.backbone_dim):
            raise ValueError('TensorRT GR00T action plans require 2048-wide '
                             'Eagle output')
        self.trt_runtime = GR00TN15BackboneTensorRTInference(
            engine_dir,
            contract=contract,
            precision=self._trt_precision,
            profile_index=profile_index,
            strict_manifest=strict_manifest)
        self.trt_contract = contract
        self._trt_ready = False
        self._trt_release_started = False
        self._inference_lock = threading.Lock()
        self._embedding_layer = None
        self._image_token_index = int(self.vlm.image_token_index)
        position_ids = torch.arange(
            contract.vision_patches,
            dtype=torch.int64)[None, :].expand(contract.image_views,
                                               -1).contiguous()
        self.register_buffer(
            '_trt_position_ids', position_ids, persistent=False)

    def _prepare_runtime(self, device: torch.device) -> None:
        if self._trt_ready:
            return
        if self._trt_release_started:
            raise RuntimeError(
                'GR00T TensorRT backbone preparation previously failed '
                'after releasing engine-covered PyTorch modules')
        embedding_layer = self.vlm.language_model.get_input_embeddings()
        retained_parameters = (
            embedding_layer.weight,
            next(self.vlm.mlp1.parameters()),
        )
        if any(parameter.device != device
               for parameter in retained_parameters):
            raise RuntimeError(
                'Move the GR00T model to its CUDA device before preparing '
                'TensorRT')
        if any(parameter.dtype != self._trt_dtype
               for parameter in retained_parameters):
            raise RuntimeError(
                'GR00T-N1.5 TensorRT retained layers must use '
                f'{self._trt_precision.upper()}')
        # Follow NVIDIA's official deployment order: retain the lightweight
        # embedding/connector, release engine-covered modules, return their
        # cached allocations, and only then deserialize the plans.  This
        # avoids holding the full PyTorch backbone and TensorRT weights at the
        # same time. This adapter has no PyTorch neural fallback, so a failed
        # deserialize is reported as unrecoverable.
        self._embedding_layer = embedding_layer
        self._trt_release_started = True
        del self.vlm.vision_model
        del self.vlm.language_model
        del self.eagle_linear
        torch.cuda.empty_cache()
        self.trt_runtime.prepare(device)
        self._trt_ready = True

    def _validate_inputs(self, images: torch.Tensor, lang_tokens: torch.Tensor,
                         img_masks: Optional[torch.Tensor],
                         lang_masks: torch.Tensor) -> torch.device:
        tensors = {
            'images': images,
            'lang_tokens': lang_tokens,
            'lang_masks': lang_masks,
        }
        if not all(
                isinstance(value, torch.Tensor) for value in tensors.values()):
            raise TypeError('GR00T TensorRT backbone inputs must be tensors')
        device = images.device
        if device.type != 'cuda':
            raise ValueError('GR00T TensorRT backbone inputs must be on CUDA')
        if any(value.device != device for value in tensors.values()):
            raise ValueError(
                'GR00T TensorRT inputs must share one CUDA device')
        c = self.trt_contract
        expected = {
            'images': (c.batch, c.image_views * 3, c.image_size, c.image_size),
            'lang_tokens': (c.batch, c.sequence_length),
            'lang_masks': (c.batch, c.sequence_length),
        }
        for name, shape in expected.items():
            if tuple(tensors[name].shape) != shape:
                raise ValueError(f'{name} shape must be {shape}, got '
                                 f'{tuple(tensors[name].shape)}')
        if lang_tokens.dtype not in (torch.int32, torch.int64):
            raise TypeError('lang_tokens must be INT32 or INT64')
        if lang_masks.dtype not in (torch.bool, torch.int32, torch.int64):
            raise TypeError('lang_masks must be BOOL, INT32, or INT64')
        if img_masks is not None:
            if not isinstance(img_masks, torch.Tensor):
                raise TypeError('img_masks must be a tensor')
            if img_masks.device != device:
                raise ValueError('img_masks must share the input CUDA device')
            if tuple(img_masks.shape) not in ((c.image_views, ),
                                              (c.batch, c.image_views)):
                raise ValueError('img_masks has an invalid shape')
            # The supported GR00T transform emits two valid views.  Avoid a
            # CUDA ``.item()`` value check here because it would synchronize
            # every inference before TensorRT can be enqueued.
        return device

    def forward(
        self,
        images: torch.Tensor,
        lang_tokens: torch.Tensor,
        img_masks: torch.Tensor,
        lang_masks: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        del args, kwargs
        with self._inference_lock:
            return self._forward_locked(images, lang_tokens, img_masks,
                                        lang_masks)

    def _forward_locked(
        self,
        images: torch.Tensor,
        lang_tokens: torch.Tensor,
        img_masks: torch.Tensor,
        lang_masks: Optional[torch.Tensor],
    ):
        begin_module_profile(self)
        if lang_masks is None:
            raise ValueError('lang_masks is required by the TensorRT LLM plan')
        device = self._validate_inputs(images, lang_tokens, img_masks,
                                       lang_masks)
        self._prepare_runtime(device)
        assert self._embedding_layer is not None
        c = self.trt_contract
        stream = torch.cuda.current_stream(device)
        with torch.cuda.stream(stream):
            start_module_stage(self, 'backbone_total', stream)
            pixel_values = images.reshape(c.image_views, 3, c.image_size,
                                          c.image_size).to(
                                              self._trt_dtype).contiguous()
            position_ids = self._trt_position_ids.contiguous()
            with module_stage(self, 'vit', stream):
                vit_embeds = self.trt_runtime.run_vision(
                    pixel_values, position_ids, stream)

            if self.vlm.use_pixel_shuffle:
                height = width = int(vit_embeds.shape[1]**0.5)
                if height * width != vit_embeds.shape[1]:
                    raise ValueError('Pixel-shuffle Eagle plans require a '
                                     'square token grid')
                vit_embeds = vit_embeds.reshape(c.image_views, height, width,
                                                c.vision_hidden)
                vit_embeds = self.vlm.pixel_shuffle(
                    vit_embeds, scale_factor=self.vlm.downsample_ratio)
                vit_embeds = vit_embeds.reshape(c.image_views, -1,
                                                vit_embeds.shape[-1])
            vit_embeds = self.vlm.mlp1(vit_embeds).to(
                self._trt_dtype).contiguous()
            if vit_embeds.shape[1] != c.projected_image_tokens_per_view:
                raise ValueError(
                    'Eagle projected image-token count does not match the '
                    'TensorRT contract')
            vit_embeds = vit_embeds.reshape(c.batch, -1, c.backbone_dim)

            tokens = lang_tokens.to(torch.int64).contiguous()
            input_embeds = self._embedding_layer(tokens).to(self._trt_dtype)
            flat_embeds = input_embeds.reshape(-1, c.backbone_dim).contiguous()
            selected = tokens.reshape(-1) == self._image_token_index
            # Boolean assignment already raises on a source/selection shape
            # mismatch.  Avoid materializing ``selected.sum().item()`` on the
            # host, which introduced a synchronization in every VLM call.
            flat_embeds[selected] = vit_embeds.reshape(-1, c.backbone_dim)
            input_embeds = flat_embeds.reshape(c.batch, c.sequence_length,
                                               c.backbone_dim).contiguous()
            attention_mask = lang_masks.to(torch.int64).contiguous()
            with module_stage(self, 'llm', stream):
                embeddings = self.trt_runtime.run_llm(input_embeds,
                                                      attention_mask, stream)
            stop_module_stage(self, 'backbone_total', stream)
        return embeddings, lang_masks, None

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

    def close_inference_runtime(self) -> None:
        self.trt_runtime.close()

    def configure_inference_profile(self, enabled: bool) -> None:
        configure_module_profile(self, enabled)

    def resolve_inference_profile(self):
        return resolve_module_profile(self)

    def __del__(self):
        runtime = getattr(self, 'trt_runtime', None)
        if runtime is not None:
            try:
                runtime.close()
            except Exception:
                pass
