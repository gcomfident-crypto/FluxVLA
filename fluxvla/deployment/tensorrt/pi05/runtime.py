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
"""TensorRT-backed PI0.5 V/P/D inference orchestration.

The exported denoise partition includes one Euler update.  Consequently this
runtime enqueues V once, P once, and D exactly ten times for every action
prediction.  It deliberately has no PyTorch model or accelerated fallback.
"""

from __future__ import annotations
import hashlib
import json
import struct
import threading
from pathlib import Path
from typing import Callable, Mapping, Optional

import torch

from fluxvla.engines.utils.inference_profile import (begin_module_profile,
                                                     configure_module_profile,
                                                     module_stage,
                                                     resolve_module_profile)
from ..runtime import TensorRTRuntime, TensorSpec
from .contract import PI05ExportContract, pi05_cache_names

_EngineSpecs = Mapping[str, tuple[TensorSpec, ...]]


def _float32(value: float) -> float:
    return struct.unpack('f', struct.pack('f', value))[0]


def _euler_timesteps(steps: int) -> tuple[float, ...]:
    """Match PI0.5's iterative FP32 ``time += -1 / steps`` schedule."""
    value = _float32(1.0)
    delta = _float32(-1.0 / steps)
    result = []
    for _ in range(steps):
        result.append(value)
        value = _float32(value + delta)
    return tuple(result)


def _expected_specs(contract: PI05ExportContract) -> _EngineSpecs:
    cache_names = pi05_cache_names(contract)
    cache_shape = contract.cache_shape
    vision = (
        TensorSpec('images', 'input',
                   (contract.image_views, 3, contract.image_height,
                    contract.image_width), torch.bfloat16),
        TensorSpec(
            'image_prefix', 'output',
            (contract.batch, contract.image_tokens, contract.prefix_hidden),
            torch.bfloat16),
    )
    prefix = [
        TensorSpec(
            'image_prefix', 'input',
            (contract.batch, contract.image_tokens, contract.prefix_hidden),
            torch.bfloat16),
        TensorSpec('lang_tokens', 'input',
                   (contract.batch, contract.language_tokens), torch.int32),
        TensorSpec('lang_mask', 'input',
                   (contract.batch, contract.language_tokens), torch.bool),
    ]
    prefix.extend(
        TensorSpec(name, 'output', cache_shape, torch.bfloat16)
        for name in cache_names)
    denoise = [
        TensorSpec(
            'x_t', 'input',
            (contract.batch, contract.action_horizon, contract.action_dim),
            torch.float32),
        TensorSpec('timestep', 'input', (contract.batch, ), torch.float32),
        TensorSpec('lang_mask', 'input',
                   (contract.batch, contract.language_tokens), torch.bool),
    ]
    denoise.extend(
        TensorSpec(name, 'input', cache_shape, torch.bfloat16)
        for name in cache_names)
    denoise.append(
        TensorSpec(
            'x_next', 'output',
            (contract.batch, contract.action_horizon, contract.action_dim),
            torch.float32))
    return {
        'vision': vision,
        'prefix': tuple(prefix),
        'denoise': tuple(denoise),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class PI05TensorRTInference:
    """Own and orchestrate the three static PI0.5 TensorRT engines."""

    def __init__(
        self,
        engine_dir: str | Path,
        *,
        contract: PI05ExportContract = PI05ExportContract(),
        profile_index: int = 0,
        strict_manifest: bool = True,
        runtime_factory: Callable[..., TensorRTRuntime] = TensorRTRuntime,
    ) -> None:
        if contract.action_horizon != 10 or contract.denoise_steps != 10:
            raise ValueError(
                'PI0.5 TensorRT production inference requires H=10 and '
                'exactly 10 Euler steps')
        self.contract = contract
        self.engine_dir = Path(engine_dir).expanduser()
        self.strict_manifest = bool(strict_manifest)
        self._manifest = self._validate_manifest() \
            if self.strict_manifest else None
        specs = _expected_specs(contract)
        self._runtimes = {
            name: runtime_factory(
                self.engine_dir / f'{name}.engine',
                expected_specs=specs[name],
                expected_tensorrt_version=self._manifest['tensorrt_version']
                if self._manifest else None,
                profile_index=profile_index)
            for name in ('vision', 'prefix', 'denoise')
        }
        self._cache_names = pi05_cache_names(contract)
        self._timesteps = _euler_timesteps(contract.denoise_steps)
        self._device: Optional[torch.device] = None
        self._x_buffers: tuple[torch.Tensor, torch.Tensor] | tuple[()] = ()
        self._timestep: Optional[torch.Tensor] = None
        self._stream_handle: Optional[int] = None
        self._lock = threading.Lock()
        self._closed = False
        self._checkpoint_path: Optional[Path] = None

    @property
    def prepared(self) -> bool:
        return self._device is not None

    @property
    def device(self) -> Optional[torch.device]:
        return self._device

    @property
    def manifest(self) -> Optional[Mapping]:
        return self._manifest

    @property
    def checkpoint_path(self) -> Optional[Path]:
        return self._checkpoint_path

    def _validate_manifest(self) -> Mapping:
        path = self.engine_dir / 'engine_manifest.json'
        if not path.is_file():
            raise FileNotFoundError(f'PI0.5 engine manifest not found: {path}')
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f'Invalid PI0.5 engine manifest: {path}') from exc
        if value.get('model') != 'pi05' or value.get('engine_model') != 'pi05':
            raise ValueError('Engine manifest is not for PI0.5')
        if value.get('shape_mode') != 'static':
            raise ValueError('PI0.5 engine manifest must use static shapes')
        if value.get('cuda_graph') is not False:
            raise ValueError(
                'PI0.5 production runtime forbids CUDA Graph plans')
        trt_version = value.get('tensorrt_version')
        if not isinstance(trt_version, str) or not trt_version:
            raise ValueError(
                'PI0.5 engine manifest has no TensorRT version identity')
        manifest_contract = value.get('contract')
        expected_contract = vars(self.contract)
        if not isinstance(manifest_contract, dict) or any(
                manifest_contract.get(name) != expected
                for name, expected in expected_contract.items()):
            raise ValueError('Engine manifest PI0.5 contract mismatch')
        engines = value.get('engines')
        if not isinstance(engines, dict):
            raise ValueError('Engine manifest has no engine identities')
        for name in ('vision', 'prefix', 'denoise'):
            identity = engines.get(name)
            if not isinstance(identity, dict):
                raise ValueError(f'Engine manifest is missing {name}')
            if identity.get('file') != f'{name}.engine':
                raise ValueError(
                    f'Engine manifest has invalid {name} filename')
            engine_path = self.engine_dir / identity['file']
            if not engine_path.is_file():
                raise FileNotFoundError(engine_path)
            if identity.get('bytes') != engine_path.stat().st_size:
                raise ValueError(f'{name} engine size does not match manifest')
            if identity.get('sha256') != _sha256(engine_path):
                raise ValueError(f'{name} engine hash does not match manifest')
        return value

    def bind_checkpoint(self, checkpoint: str | Path) -> Path:
        """Verify the runner checkpoint against the serialized plans."""
        path = Path(checkpoint).expanduser().resolve()
        if path.suffix == '.pt':
            safetensors = path.with_suffix('.safetensors')
            if safetensors.is_file():
                path = safetensors
        if not path.is_file():
            raise FileNotFoundError(f'PI0.5 checkpoint not found: {path}')
        if self._checkpoint_path == path:
            return path
        self._checkpoint_path = None
        if self._manifest is None:
            raise RuntimeError('Checkpoint binding requires strict_manifest')
        expected = self._manifest.get('checkpoint_sha256')
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(
                'PI0.5 engine manifest has no checkpoint SHA256 identity')
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                'PI0.5 checkpoint does not match the weights serialized in '
                f'the TensorRT engines: {path}')
        self._checkpoint_path = path
        return path

    def prepare(self, device: str | torch.device) -> None:
        if self._closed:
            raise RuntimeError('PI0.5 TensorRT inference is closed')
        if self._checkpoint_path is None:
            raise RuntimeError(
                'bind_checkpoint() must verify the checkpoint before PI0.5 '
                'TensorRT plans are prepared')
        device = torch.device(device)
        if device.type != 'cuda':
            raise ValueError(f'PI0.5 TensorRT requires CUDA, got {device}')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        if self.prepared:
            if device != self._device:
                raise RuntimeError(
                    f'PI0.5 runtime is on {self._device}, not {device}')
            return
        for runtime in self._runtimes.values():
            runtime.prepare(device)
        shape = (self.contract.batch, self.contract.action_horizon,
                 self.contract.action_dim)
        self._x_buffers = (
            torch.empty(shape, dtype=torch.float32, device=device),
            torch.empty(shape, dtype=torch.float32, device=device),
        )
        self._timestep = torch.empty((self.contract.batch, ),
                                     dtype=torch.float32,
                                     device=device)
        self._device = device

    def _validate_inputs(self, images: torch.Tensor, lang_tokens: torch.Tensor,
                         states: torch.Tensor, img_masks: torch.Tensor,
                         lang_masks: torch.Tensor,
                         noise: Optional[torch.Tensor]) -> torch.device:
        tensors = {
            'images': images,
            'lang_tokens': lang_tokens,
            'states': states,
            'img_masks': img_masks,
            'lang_masks': lang_masks,
        }
        if not all(
                isinstance(value, torch.Tensor) for value in tensors.values()):
            raise TypeError('All PI0.5 inputs must be torch.Tensor instances')
        device = images.device
        if device.type != 'cuda':
            raise ValueError('PI0.5 TensorRT inputs must be on CUDA')
        if any(value.device != device for value in tensors.values()):
            raise ValueError(
                'PI0.5 TensorRT inputs must share one CUDA device')
        expected = {
            'images': (1, self.contract.image_views * 3,
                       self.contract.image_height, self.contract.image_width),
            'lang_tokens':
            (self.contract.batch, self.contract.language_tokens),
            'states': (self.contract.batch, self.contract.action_dim),
            'img_masks': (self.contract.batch, self.contract.image_views),
            'lang_masks': (self.contract.batch, self.contract.language_tokens),
        }
        for name, shape in expected.items():
            if tuple(tensors[name].shape) != shape:
                raise ValueError(f'{name} shape must be {shape}, got '
                                 f'{tuple(tensors[name].shape)}')
        if images.dtype != torch.float32:
            raise TypeError('images must be normalized FP32 model inputs')
        if lang_tokens.dtype not in (torch.int32, torch.int64):
            raise TypeError('lang_tokens must be INT32 or INT64')
        if img_masks.dtype != torch.bool or lang_masks.dtype != torch.bool:
            raise TypeError('img_masks and lang_masks must be BOOL')
        # Token IDs and image-mask values are produced by FluxVLA's supported
        # PI0.5 preprocessing pipeline.  Do not scalarize CUDA tensors here:
        # ``.item()`` would synchronize the device before every prediction and
        # defeat the asynchronous TensorRT execution path.  The static shape
        # and dtype contract above remains fail-closed.
        if noise is not None:
            if noise.device != device or noise.dtype != torch.float32:
                raise TypeError('noise must be FP32 on the input CUDA device')
            action_shape = (self.contract.batch, self.contract.action_horizon,
                            self.contract.action_dim)
            if tuple(noise.shape) != action_shape:
                raise ValueError(f'noise shape must be {action_shape}')
        return device

    def predict_action(
        self,
        images: torch.Tensor,
        lang_tokens: torch.Tensor,
        states: torch.Tensor,
        img_masks: torch.Tensor,
        lang_masks: torch.Tensor,
        *,
        noise: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        """Execute V once, P once, and D ten times."""
        begin_module_profile(self)
        device = self._validate_inputs(images, lang_tokens, states, img_masks,
                                       lang_masks, noise)
        with self._lock:
            self.prepare(device)
            assert self._timestep is not None
            caller_stream = torch.cuda.current_stream(device)
            selected_stream = stream or caller_stream
            if torch.device(selected_stream.device) != device:
                raise ValueError('PI0.5 execution stream is on the wrong '
                                 'CUDA device')
            handle = int(selected_stream.cuda_stream)
            if self._stream_handle is None:
                self._stream_handle = handle
            elif self._stream_handle != handle:
                raise RuntimeError(
                    'One PI0.5 TensorRT instance cannot execute on multiple '
                    'CUDA streams')
            if int(caller_stream.cuda_stream) != handle:
                selected_stream.wait_stream(caller_stream)
            for source in (images, lang_tokens, lang_masks, noise):
                if source is not None:
                    source.record_stream(selected_stream)
            with torch.cuda.stream(selected_stream):
                images_v = images.reshape(
                    self.contract.image_views, 3, self.contract.image_height,
                    self.contract.image_width).to(torch.bfloat16).contiguous()
                tokens = lang_tokens.to(torch.int32).contiguous()
                mask = lang_masks.contiguous()
                if noise is None:
                    generator = None
                    if seed is not None:
                        if type(seed) is not int or seed < 0:
                            raise ValueError(
                                'seed must be a non-negative integer')
                        generator = torch.Generator(device=device)
                        generator.manual_seed(seed)
                    self._x_buffers[0].normal_(generator=generator)
                else:
                    self._x_buffers[0].copy_(noise)

                with module_stage(self, 'vision', selected_stream):
                    vision = self._runtimes['vision'].enqueue(
                        {'images': images_v}, stream=selected_stream)
                prefix_inputs = {
                    'image_prefix': vision['image_prefix'],
                    'lang_tokens': tokens,
                    'lang_mask': mask,
                }
                with module_stage(self, 'prefix', selected_stream):
                    caches = self._runtimes['prefix'].enqueue(
                        prefix_inputs, stream=selected_stream)

                current = 0
                with module_stage(self, 'denoise', selected_stream):
                    for timestep in self._timesteps:
                        with module_stage(self, 'denoise_step',
                                          selected_stream):
                            next_buffer = 1 - current
                            self._timestep.fill_(timestep)
                            denoise_inputs = {
                                'x_t': self._x_buffers[current],
                                'timestep': self._timestep,
                                'lang_mask': mask,
                            }
                            denoise_inputs.update({
                                name: caches[name]
                                for name in self._cache_names
                            })
                            self._runtimes['denoise'].enqueue(
                                denoise_inputs,
                                outputs={
                                    'x_next': self._x_buffers[next_buffer]
                                },
                                stream=selected_stream)
                            current = next_buffer
                result = self._x_buffers[current].clone()
            if int(caller_stream.cuda_stream) != handle:
                caller_stream.wait_stream(selected_stream)
                result.record_stream(caller_stream)
            return result

    def configure_inference_profile(self, enabled: bool) -> None:
        configure_module_profile(self, enabled)

    def resolve_inference_profile(self):
        return resolve_module_profile(self)

    def close(self) -> None:
        if self._closed:
            return
        for runtime in self._runtimes.values():
            runtime.close(synchronize=True)
        self._x_buffers = ()
        self._timestep = None
        self._stream_handle = None
        self._device = None
        self._closed = True
