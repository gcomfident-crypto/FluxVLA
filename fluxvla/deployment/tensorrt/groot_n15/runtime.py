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
"""TensorRT-only orchestration for the GR00T-N1.5 seven-engine split.

The two public runtime components deliberately mirror FluxVLA's model split:
the Eagle backbone owns the ViT and LLM plans, while the flow-matching head
owns the remaining five plans.  Neural layers serialized into a plan are
never executed through PyTorch by this module.
"""

from __future__ import annotations
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping, Optional

import torch

from ..runtime import TensorRTRuntime, TensorSpec
from .contract import GR00TN15TensorRTContract


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _validate_embodiment_metadata(embodiment_ids: torch.Tensor,
                                  count: int) -> None:
    """Validate metadata without synchronizing CUDA in the inference path.

    FluxVLA's transforms own the actual embodiment IDs.  Reading their values
    with ``.item()`` here would force a host/device synchronization on every
    denoising call, so production validation is limited to invariants that can
    be checked from tensor metadata without touching device data.
    """
    if count <= 0:
        raise ValueError('Embodiment category count must be positive')
    if embodiment_ids.numel() == 0:
        raise ValueError('embodiment_ids must not be empty')


@lru_cache(maxsize=8)
def _checkpoint_sha256_identity(path_text: str, size: int,
                                mtime_ns: int) -> str:
    del size, mtime_ns
    return _sha256(Path(path_text))


def _checkpoint_sha256(path: Path) -> str:
    stat = path.stat()
    return _checkpoint_sha256_identity(
        str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def _manifest_contract(contract: GR00TN15TensorRTContract) -> dict:
    value = vars(contract).copy()
    value['sa_length'] = contract.sa_length
    # The exporter calls this width ``llm_hidden`` as well as
    # ``backbone_dim``.  N1.5 uses 2048 for both.
    value['llm_hidden'] = contract.backbone_dim
    return value


def _validate_manifest(engine_dir: Path, contract: GR00TN15TensorRTContract,
                       engine_names: tuple[str, ...]) -> Mapping:
    path = engine_dir / 'engine_manifest.json'
    if not path.is_file():
        raise FileNotFoundError(f'GR00T engine manifest not found: {path}')
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Invalid GR00T engine manifest: {path}') from exc
    if (value.get('model') != 'groot-n15'
            or value.get('engine_model') != 'groot-n15'):
        raise ValueError('Engine manifest is not for GR00T-N1.5')
    if value.get('shape_mode') != 'static':
        raise ValueError('GR00T-N1.5 engine manifest must use static shapes')
    if value.get('precision') != 'fp16':
        raise ValueError('GR00T-N1.5 production plans must use FP16')
    if value.get('cuda_graph') is not False:
        raise ValueError('GR00T-N1.5 production runtime forbids CUDA Graphs')
    trt_version = value.get('tensorrt_version')
    if not isinstance(trt_version, str) or not trt_version:
        raise ValueError(
            'GR00T engine manifest has no TensorRT version identity')

    manifest_contract = value.get('contract')
    expected = _manifest_contract(contract)
    if not isinstance(manifest_contract, dict):
        raise ValueError('Engine manifest has no GR00T contract')
    mismatches = {
        name: (manifest_contract.get(name), wanted)
        for name, wanted in expected.items()
        if manifest_contract.get(name) != wanted
    }
    if mismatches:
        raise ValueError(f'Engine manifest GR00T contract mismatch: '
                         f'{mismatches}')

    engines = value.get('engines')
    if not isinstance(engines, dict):
        raise ValueError('Engine manifest has no engine identities')
    for name in engine_names:
        identity = engines.get(name)
        if not isinstance(identity, dict):
            raise ValueError(f'Engine manifest is missing {name}')
        if identity.get('file') != f'{name}.engine':
            raise ValueError(f'Engine manifest has invalid {name} filename')
        engine_path = engine_dir / identity['file']
        if not engine_path.is_file():
            raise FileNotFoundError(engine_path)
        if identity.get('bytes') != engine_path.stat().st_size:
            raise ValueError(f'{name} engine size does not match manifest')
        if identity.get('sha256') != _sha256(engine_path):
            raise ValueError(f'{name} engine hash does not match manifest')
    return value


def _backbone_specs(
    contract: GR00TN15TensorRTContract,
) -> Mapping[str, tuple[TensorSpec, ...]]:
    return {
        'vit_fp16': (
            TensorSpec('pixel_values', 'input',
                       (contract.image_views, 3, contract.image_size,
                        contract.image_size), torch.float16),
            TensorSpec('position_ids', 'input',
                       (contract.image_views, contract.vision_patches),
                       torch.int64),
            TensorSpec('vit_embeds', 'output',
                       (contract.image_views, contract.vision_patches,
                        contract.vision_hidden), torch.float16),
        ),
        'llm_fp16': (
            TensorSpec('inputs_embeds', 'input',
                       (contract.batch, contract.sequence_length,
                        contract.backbone_dim), torch.float16),
            TensorSpec('attention_mask', 'input',
                       (contract.batch, contract.sequence_length),
                       torch.int64),
            TensorSpec('embeddings', 'output',
                       (contract.batch, contract.sequence_length,
                        contract.backbone_dim), torch.float16),
        ),
    }


def _action_head_specs(
    contract: GR00TN15TensorRTContract,
) -> Mapping[str, tuple[TensorSpec, ...]]:
    batch = contract.batch
    sequence = contract.sequence_length
    sa_length = contract.sa_length
    return {
        'vlln_vl_self_attention': (
            TensorSpec('backbone_features', 'input',
                       (batch, sequence, contract.backbone_dim),
                       torch.float16),
            TensorSpec('output', 'output',
                       (batch, sequence, contract.backbone_dim),
                       torch.float16),
        ),
        'state_encoder': (
            TensorSpec('state', 'input', (batch, 1, contract.state_dim),
                       torch.float16),
            TensorSpec('embodiment_id', 'input', (batch, ), torch.int64),
            TensorSpec('output', 'output',
                       (batch, 1, contract.input_embedding_dim),
                       torch.float16),
        ),
        'action_encoder': (
            TensorSpec('actions', 'input',
                       (batch, contract.action_horizon, contract.action_dim),
                       torch.float16),
            TensorSpec('timesteps_tensor', 'input', (batch, ), torch.int64),
            TensorSpec('embodiment_id', 'input', (batch, ), torch.int64),
            TensorSpec(
                'output', 'output',
                (batch, contract.action_horizon, contract.input_embedding_dim),
                torch.float16),
        ),
        'dit_fp16': (
            TensorSpec('sa_embs', 'input',
                       (batch, sa_length, contract.input_embedding_dim),
                       torch.float16),
            TensorSpec('vl_embs', 'input',
                       (batch, sequence, contract.backbone_dim),
                       torch.float16),
            TensorSpec('timesteps_tensor', 'input', (batch, ), torch.int64),
            TensorSpec('output', 'output',
                       (batch, sa_length, contract.hidden_size),
                       torch.float16),
        ),
        'action_decoder': (
            TensorSpec('model_output', 'input',
                       (batch, sa_length, contract.hidden_size),
                       torch.float16),
            TensorSpec('embodiment_id', 'input', (batch, ), torch.int64),
            TensorSpec('output', 'output',
                       (batch, sa_length, contract.action_dim), torch.float16),
        ),
    }


class _RuntimeGroup:

    def __init__(
        self,
        engine_dir: str | Path,
        engine_names: tuple[str, ...],
        specs: Mapping[str, tuple[TensorSpec, ...]],
        contract: GR00TN15TensorRTContract,
        profile_index: int,
        strict_manifest: bool,
        runtime_factory: Callable[..., TensorRTRuntime],
    ) -> None:
        if strict_manifest is not True:
            raise ValueError('strict_manifest must be true in production')
        self.engine_dir = Path(engine_dir).expanduser()
        self.contract = contract
        self.manifest = _validate_manifest(self.engine_dir, contract,
                                           engine_names)
        self.runtimes = {
            name: runtime_factory(
                self.engine_dir / f'{name}.engine',
                expected_specs=specs[name],
                expected_tensorrt_version=self.manifest['tensorrt_version'],
                profile_index=profile_index)
            for name in engine_names
        }
        self.device: Optional[torch.device] = None
        self.checkpoint_path: Optional[Path] = None
        self.closed = False

    def bind_checkpoint(self, checkpoint: str | Path) -> Path:
        """Verify that all plans were built from the runner checkpoint."""
        path = Path(checkpoint).expanduser().resolve()
        if path.suffix == '.pt':
            safetensors = path.with_suffix('.safetensors')
            if safetensors.is_file():
                path = safetensors
        if not path.is_file():
            raise FileNotFoundError(f'GR00T-N1.5 checkpoint not found: {path}')
        if self.checkpoint_path == path:
            return path
        self.checkpoint_path = None
        expected = self.manifest.get('checkpoint_sha256')
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(
                'GR00T engine manifest has no checkpoint SHA256 identity')
        if _checkpoint_sha256(path) != expected:
            raise ValueError(
                'GR00T checkpoint does not match the weights serialized in '
                f'the TensorRT engines: {path}')
        self.checkpoint_path = path
        return path

    def prepare(self, device: str | torch.device) -> None:
        if self.closed:
            raise RuntimeError('GR00T-N1.5 TensorRT runtime is closed')
        if self.checkpoint_path is None:
            raise RuntimeError(
                'bind_checkpoint() must verify the checkpoint before '
                'GR00T TensorRT plans are prepared')
        device = torch.device(device)
        if device.type != 'cuda':
            raise ValueError(
                f'GR00T-N1.5 TensorRT requires CUDA, got {device}')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        if self.device is not None:
            if self.device != device:
                raise RuntimeError(f'GR00T runtime is on {self.device}, not '
                                   f'{device}')
            return
        for runtime in self.runtimes.values():
            runtime.prepare(device)
        self.device = device

    def close(self) -> None:
        if self.closed:
            return
        for runtime in self.runtimes.values():
            runtime.close(synchronize=True)
        self.device = None
        self.closed = True


class GR00TN15BackboneTensorRTInference(_RuntimeGroup):
    """Own the Eagle ViT and LLM TensorRT plans."""

    def __init__(
        self,
        engine_dir: str | Path,
        *,
        contract: GR00TN15TensorRTContract,
        profile_index: int = 0,
        strict_manifest: bool = True,
        runtime_factory: Callable[...,
                                  TensorRTRuntime] = TensorRTRuntime) -> None:
        names = ('vit_fp16', 'llm_fp16')
        super().__init__(engine_dir, names, _backbone_specs(contract),
                         contract, profile_index, strict_manifest,
                         runtime_factory)

    def run_vision(self, pixel_values: torch.Tensor,
                   position_ids: torch.Tensor,
                   stream: torch.cuda.Stream) -> torch.Tensor:
        output = self.runtimes['vit_fp16'].enqueue(
            {
                'pixel_values': pixel_values,
                'position_ids': position_ids,
            },
            stream=stream)
        return output['vit_embeds']

    def run_llm(self, inputs_embeds: torch.Tensor,
                attention_mask: torch.Tensor,
                stream: torch.cuda.Stream) -> torch.Tensor:
        output = self.runtimes['llm_fp16'].enqueue(
            {
                'inputs_embeds': inputs_embeds,
                'attention_mask': attention_mask,
            },
            stream=stream)
        return output['embeddings']


class GR00TN15ActionHeadTensorRTInference(_RuntimeGroup):
    """Own the five TensorRT plans used by the flow-matching head."""

    def __init__(
        self,
        engine_dir: str | Path,
        *,
        contract: GR00TN15TensorRTContract,
        profile_index: int = 0,
        strict_manifest: bool = True,
        runtime_factory: Callable[...,
                                  TensorRTRuntime] = TensorRTRuntime) -> None:
        names = ('vlln_vl_self_attention', 'state_encoder', 'action_encoder',
                 'dit_fp16', 'action_decoder')
        super().__init__(engine_dir, names, _action_head_specs(contract),
                         contract, profile_index, strict_manifest,
                         runtime_factory)

    def process_backbone(self, features: torch.Tensor,
                         stream: torch.cuda.Stream) -> torch.Tensor:
        return self.runtimes['vlln_vl_self_attention'].enqueue(
            {'backbone_features': features}, stream=stream)['output']

    def encode_state(self, state: torch.Tensor, embodiment_id: torch.Tensor,
                     stream: torch.cuda.Stream) -> torch.Tensor:
        return self.runtimes['state_encoder'].enqueue(
            {
                'state': state,
                'embodiment_id': embodiment_id
            }, stream=stream)['output']

    def encode_action(self, actions: torch.Tensor, timestep: torch.Tensor,
                      embodiment_id: torch.Tensor,
                      stream: torch.cuda.Stream) -> torch.Tensor:
        return self.runtimes['action_encoder'].enqueue(
            {
                'actions': actions,
                'timesteps_tensor': timestep,
                'embodiment_id': embodiment_id,
            },
            stream=stream)['output']

    def run_dit(self, sa_embs: torch.Tensor, vl_embs: torch.Tensor,
                timestep: torch.Tensor,
                stream: torch.cuda.Stream) -> torch.Tensor:
        return self.runtimes['dit_fp16'].enqueue(
            {
                'sa_embs': sa_embs,
                'vl_embs': vl_embs,
                'timesteps_tensor': timestep,
            },
            stream=stream)['output']

    def decode_action(self, model_output: torch.Tensor,
                      embodiment_id: torch.Tensor,
                      stream: torch.cuda.Stream) -> torch.Tensor:
        return self.runtimes['action_decoder'].enqueue(
            {
                'model_output': model_output,
                'embodiment_id': embodiment_id,
            },
            stream=stream)['output']


__all__ = [
    'GR00TN15TensorRTContract', 'GR00TN15BackboneTensorRTInference',
    'GR00TN15ActionHeadTensorRTInference'
]
