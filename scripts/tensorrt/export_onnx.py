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
"""Export FluxVLA PI0.5 or GR00T-N1.5 partitions to ONNX.

The GR00T split mirrors NVIDIA Isaac-GR00T ``n1.5-release``.  The exporter
loads the FluxVLA model/config/checkpoint, so engines are built from the exact
weights used by FluxVLA rather than from a second copy of the upstream model.

Examples:

.. code-block:: bash

   python -m scripts.tensorrt.export_onnx pi05 \
     --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
     --checkpoint checkpoints/pi05_libero/model.safetensors

   python -m scripts.tensorrt.export_onnx groot-n15 \
     --config configs/gr00t/gr00t_eagle_3b_libero_10_full_finetune.py \
     --checkpoint /path/to/fine_tuned.safetensors --sequence-length 600
"""

from __future__ import annotations
import argparse
import copy
import hashlib
import inspect
import json
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping, Sequence

import torch
from mmengine import Config

from fluxvla.deployment.tensorrt.checkpoint import checkpoint_state_for_model
from fluxvla.deployment.tensorrt.pi05 import (PI05ExportContract,
                                              pi05_cache_names)
from .groot_n15 import (GR00TBackboneOutputExportWrapper,
                        GR00TLLMExportWrapper, GR00TVisionExportWrapper)
from .pi05 import (PI05DenoiseStepExportWrapper, PI05PrefixExportWrapper,
                   PI05VisionExportWrapper)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _resolve_checkpoint(path: Path) -> Path:
    """Mirror FluxVLA eval loading when a large ``.pt`` has a sidecar."""
    path = path.expanduser().resolve()
    if path.suffix == '.pt':
        safetensors = path.with_suffix('.safetensors')
        if safetensors.is_file():
            print(f'Using checkpoint sidecar: {safetensors}')
            return safetensors
    return path


def _load_checkpoint_state(checkpoint: Path) -> Mapping[str, torch.Tensor]:
    if checkpoint.suffix == '.safetensors':
        from safetensors.torch import load_file
        return load_file(str(checkpoint), device='cpu')
    value = torch.load(str(checkpoint), map_location='cpu')
    state = value['model'] if isinstance(value, dict) \
        and 'model' in value else value
    if not isinstance(state, Mapping):
        raise TypeError(f'Checkpoint does not contain a state dict: '
                        f'{checkpoint}')
    return state


def _construction_dtype_name(dtype: torch.dtype) -> str:
    names = {
        torch.float16: 'float16',
        torch.bfloat16: 'bfloat16',
        torch.float32: 'float32',
    }
    try:
        return names[dtype]
    except KeyError as exc:
        raise ValueError(f'Unsupported export dtype: {dtype}') from exc


def _load_model(config_path: Path, checkpoint: Path | None,
                device: torch.device, dtype: torch.dtype):
    if checkpoint is None:
        raise ValueError(
            '--checkpoint is required. Refusing to export a randomly '
            'initialized TensorRT engine.')
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')

    # Importing fluxvla.models populates the registries.
    import fluxvla.models  # noqa: F401
    from fluxvla.engines import build_vla_from_cfg

    cfg = Config.fromfile(str(config_path))
    # Eagle constructs its VLM directly in the configured dtype.  Override
    # that construction dtype before loading the checkpoint so GR00T follows
    # the same FP32-checkpoint -> FP16 rounding path as the runtime adapter.
    # Constructing it as BF16 first and converting BF16 -> FP16 later changes
    # most weights and makes the serialized engines disagree with the retained
    # PyTorch embedding/connector weights.
    vlm_backbone = cfg.model.get('vlm_backbone')
    if isinstance(vlm_backbone, Mapping) and 'dtype' in vlm_backbone:
        vlm_backbone['dtype'] = _construction_dtype_name(dtype)
        # FlashAttention is not representable as ordinary ONNX operators.
        # Select the Transformers eager attention implementation before model
        # construction, rather than changing the config after tracing begins.
        vlm_backbone['attention_implementation'] = 'eager'
    model = build_vla_from_cfg(copy.deepcopy(cfg.model)).eval()
    state = checkpoint_state_for_model(model,
                                       _load_checkpoint_state(checkpoint))
    model.load_state_dict(state, strict=True)
    del state
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def _export(
    module: torch.nn.Module,
    args: Sequence[torch.Tensor],
    output: Path,
    input_names: Sequence[str],
    output_names: Sequence[str],
    *,
    opset: int,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        export_params=True,
        do_constant_folding=True,
        input_names=list(input_names),
        output_names=list(output_names),
        opset_version=opset,
        dynamo=False,
    )
    signature = inspect.signature(torch.onnx.export)
    if 'external_data' in signature.parameters:
        kwargs['external_data'] = True
    with torch.inference_mode():
        torch.onnx.export(module, tuple(args), str(output), **kwargs)

    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            'ONNX is required to validate exported models. Install '
            'requirements-tensorrt.txt.') from exc
    onnx.checker.check_model(str(output), full_check=False)
    model_proto = onnx.load(str(output), load_external_data=False)
    locations = set()

    def collect_graph(graph) -> None:
        for tensor in graph.initializer:
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                location = next((item.value for item in tensor.external_data
                                 if item.key == 'location'), None)
                if not location:
                    raise RuntimeError(
                        f'External ONNX tensor has no location: {output}')
                locations.add(location)
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    collect_graph(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for child in attribute.graphs:
                        collect_graph(child)

    collect_graph(model_proto.graph)
    artifact_paths = [output]
    for location in sorted(locations):
        relative = Path(location)
        artifact = (output.parent / relative).resolve()
        if relative.is_absolute(
        ) or artifact.parent != output.parent.resolve():
            raise RuntimeError(
                f'Unsafe ONNX external-data location: {location}')
        if not artifact.is_file():
            raise FileNotFoundError(
                f'ONNX external-data file is missing: {artifact}')
        artifact_paths.append(artifact)
    identity = {
        'graph':
        output.name,
        'files': [{
            'file': path.name,
            'bytes': path.stat().st_size,
            'sha256': _sha256(path),
        } for path in artifact_paths],
    }
    print(f'Exported and checked: {output}')
    return identity


def _write_manifest(output_dir: Path, value: dict) -> None:
    path = output_dir / 'export_manifest.json'
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'Export manifest: {path}')


def _publish_output_dir(staging_dir: Path, output_dir: Path) -> None:
    """Publish a complete export and preserve the old one on error."""
    if not staging_dir.is_dir():
        raise FileNotFoundError(f'Export staging directory is missing: '
                                f'{staging_dir}')
    if output_dir.is_symlink() or (output_dir.exists()
                                   and not output_dir.is_dir()):
        raise NotADirectoryError(
            f'Export destination must be a real directory: {output_dir}')

    backup_dir = None
    if output_dir.exists():
        backup_dir = output_dir.with_name(
            f'.{output_dir.name}.backup-{uuid.uuid4().hex}')
        os.replace(output_dir, backup_dir)

    try:
        os.replace(staging_dir, output_dir)
    except BaseException as publish_error:
        if backup_dir is not None and backup_dir.exists():
            try:
                os.replace(backup_dir, output_dir)
            except BaseException as rollback_error:
                raise RuntimeError(
                    f'Failed to publish {staging_dir} ({publish_error!r}) and '
                    f'restore the previous export from {backup_dir}') \
                    from rollback_error
        raise

    if backup_dir is not None:
        try:
            shutil.rmtree(backup_dir)
        except OSError as exc:
            print(f'Warning: new export was published, but the old backup '
                  f'could not be removed: {backup_dir}: {exc}')
    print(f'Published complete export: {output_dir}')


@contextmanager
def _staged_output_dir(output_dir: Path):
    """Yield a same-filesystem staging directory and publish it on success."""
    output_dir = output_dir.expanduser()
    if output_dir.is_symlink() or (output_dir.exists()
                                   and not output_dir.is_dir()):
        raise NotADirectoryError(
            f'Export destination must be a real directory: {output_dir}')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f'.{output_dir.name}.staging-',
            dir=str(output_dir.parent),
        ))
    try:
        yield staging_dir
        _publish_output_dir(staging_dir, output_dir)
    finally:
        if staging_dir.exists():
            try:
                shutil.rmtree(staging_dir)
            except OSError as exc:
                print(f'Warning: export staging directory could not be '
                      f'removed: {staging_dir}: {exc}')


def export_pi05(args) -> None:
    with _staged_output_dir(args.output_dir / 'pi05') as output_dir:
        _export_pi05(args, output_dir)


def _export_pi05(args, output_dir: Path) -> None:
    device = torch.device(args.device)
    contract = PI05ExportContract(
        image_views=args.image_views,
        language_tokens=args.language_tokens,
        action_horizon=args.action_horizon,
        action_dim=args.action_dim,
        denoise_steps=args.denoise_steps)
    model = _load_model(args.config, args.checkpoint, device, torch.bfloat16)

    vision = PI05VisionExportWrapper(model, contract).to(device)
    images = torch.randn(
        contract.image_views,
        3,
        contract.image_height,
        contract.image_width,
        dtype=torch.bfloat16,
        device=device)
    onnx_artifacts = {}
    onnx_artifacts['vision'] = _export(
        vision, (images, ),
        output_dir / 'vision.onnx', ('images', ), ('image_prefix', ),
        opset=args.opset)
    with torch.inference_mode():
        image_prefix = vision(images)

    prefix = PI05PrefixExportWrapper(model, contract).to(device)
    vocab_size = int(model.llm_backbone.config.vocab_size)
    lang_tokens = torch.randint(
        0,
        vocab_size, (contract.batch, contract.language_tokens),
        dtype=torch.int32,
        device=device)
    lang_mask = torch.ones(
        contract.batch,
        contract.language_tokens,
        dtype=torch.bool,
        device=device)
    cache_names = pi05_cache_names(contract)
    onnx_artifacts['prefix'] = _export(
        prefix, (image_prefix, lang_tokens, lang_mask),
        output_dir / 'prefix.onnx',
        ('image_prefix', 'lang_tokens', 'lang_mask'),
        cache_names,
        opset=args.opset)
    with torch.inference_mode():
        caches = prefix(image_prefix, lang_tokens, lang_mask)

    denoise = PI05DenoiseStepExportWrapper(model, contract).to(device)
    x_t = torch.randn(
        contract.batch,
        contract.action_horizon,
        contract.action_dim,
        dtype=torch.float32,
        device=device)
    timestep = torch.ones(contract.batch, dtype=torch.float32, device=device)
    denoise_inputs = (x_t, timestep, lang_mask, *caches)
    onnx_artifacts['denoise'] = _export(
        denoise,
        denoise_inputs,
        output_dir / 'denoise.onnx',
        ('x_t', 'timestep', 'lang_mask', *cache_names), ('x_next', ),
        opset=args.opset)

    _write_manifest(
        output_dir, {
            'schema_version':
            1,
            'shape_mode':
            'static',
            'model':
            'pi05',
            'config':
            args.config.name,
            'checkpoint': (args.checkpoint.name if args.checkpoint else None),
            'checkpoint_sha256':
            (_sha256(args.checkpoint) if args.checkpoint else None),
            'torch_version':
            torch.__version__,
            'opset':
            args.opset,
            'contract':
            vars(contract),
            'onnx': {
                'vision': 'vision.onnx',
                'prefix': 'prefix.onnx',
                'denoise': 'denoise.onnx',
            },
            'onnx_artifacts':
            onnx_artifacts,
        })


def _set_eager_attention(module: torch.nn.Module) -> None:
    for child in module.modules():
        config = getattr(child, 'config', None)
        if config is not None and hasattr(config, '_attn_implementation'):
            config._attn_implementation = 'eager'


def export_groot(args) -> None:
    with _staged_output_dir(args.output_dir / 'groot_n15') as output_dir:
        _export_groot(args, output_dir)


def _export_groot(args, output_dir: Path) -> None:
    device = torch.device(args.device)
    model = _load_model(args.config, args.checkpoint, device, torch.float16)
    if not hasattr(model, 'vlm_backbone') or not hasattr(model, 'vla_head'):
        raise TypeError('GR00T export requires LlavaVLA with Eagle backbone '
                        'and FlowMatchingHead')
    eagle = model.vlm_backbone
    head = model.vla_head
    if int(head.num_steps) != args.action_horizon:
        raise ValueError(
            f'--action-horizon={args.action_horizon} does not match the '
            f'FlowMatchingHead horizon ({head.num_steps})')
    actual_action_dim = int(head.action_dim)
    if actual_action_dim != args.action_dim:
        raise ValueError(
            f'--action-dim={args.action_dim} does not match the action head '
            f'({actual_action_dim})')
    actual_state_dim = int(head.state_encoder.layer1.W.shape[1])
    if actual_state_dim != args.state_dim:
        raise ValueError(
            f'--state-dim={args.state_dim} does not match the state encoder '
            f'({actual_state_dim})')
    _set_eager_attention(model)

    vision = GR00TVisionExportWrapper(eagle).to(device)
    vision_config = eagle.vlm.vision_model.config
    image_size = int(vision_config.image_size)
    patch_size = int(vision_config.patch_size)
    num_patches = (image_size // patch_size)**2
    projected_image_tokens = int(eagle.vlm.num_image_token)
    pixel_values = torch.randn(
        args.image_views,
        3,
        image_size,
        image_size,
        dtype=torch.float16,
        device=device)
    position_ids = torch.arange(
        num_patches, dtype=torch.int64,
        device=device)[None, :].expand(args.image_views, -1).contiguous()
    onnx_artifacts = {}
    onnx_artifacts['vit'] = _export(
        vision, (pixel_values, position_ids),
        output_dir / 'vit_fp16.onnx', ('pixel_values', 'position_ids'),
        ('vit_embeds', ),
        opset=args.opset)

    llm = GR00TLLMExportWrapper(eagle).to(device)
    llm_hidden = int(eagle.vlm.language_model.config.hidden_size)
    inputs_embeds = torch.randn(
        1,
        args.sequence_length,
        llm_hidden,
        dtype=torch.float16,
        device=device)
    attention_mask = torch.ones(
        1, args.sequence_length, dtype=torch.int64, device=device)
    onnx_artifacts['llm'] = _export(
        llm, (inputs_embeds, attention_mask),
        output_dir / 'llm_fp16.onnx', ('inputs_embeds', 'attention_mask'),
        ('embeddings', ),
        opset=args.opset)

    process = GR00TBackboneOutputExportWrapper(head).to(device)
    backbone_dim = int(head.vlln.normalized_shape[0])
    backbone_features = torch.randn(
        1,
        args.sequence_length,
        backbone_dim,
        dtype=torch.float16,
        device=device)
    onnx_artifacts['backbone_output'] = _export(
        process, (backbone_features, ),
        output_dir / 'vlln_vl_self_attention.onnx', ('backbone_features', ),
        ('output', ),
        opset=args.opset)

    batch = 1
    embodiment = torch.zeros(batch, dtype=torch.int64, device=device)
    state = torch.randn(
        batch, 1, args.state_dim, dtype=torch.float16, device=device)
    onnx_artifacts['state_encoder'] = _export(
        head.state_encoder, (state, embodiment),
        output_dir / 'state_encoder.onnx', ('state', 'embodiment_id'),
        ('output', ),
        opset=args.opset)

    actions = torch.randn(
        batch,
        args.action_horizon,
        args.action_dim,
        dtype=torch.float16,
        device=device)
    timestep = torch.zeros(batch, dtype=torch.int64, device=device)
    onnx_artifacts['action_encoder'] = _export(
        head.action_encoder, (actions, timestep, embodiment),
        output_dir / 'action_encoder.onnx',
        ('actions', 'timesteps_tensor', 'embodiment_id'), ('output', ),
        opset=args.opset)

    future_tokens = int(head.future_tokens.num_embeddings)
    input_embedding_dim = int(head.input_embedding_dim)
    sa_length = 1 + future_tokens + args.action_horizon
    sa_embs = torch.randn(
        batch,
        sa_length,
        input_embedding_dim,
        dtype=torch.float16,
        device=device)
    vl_embs = torch.randn(
        batch,
        args.sequence_length,
        backbone_dim,
        dtype=torch.float16,
        device=device)
    onnx_artifacts['dit'] = _export(
        head.model, (sa_embs, vl_embs, timestep),
        output_dir / 'dit_fp16.onnx',
        ('sa_embs', 'vl_embs', 'timesteps_tensor'), ('output', ),
        opset=args.opset)

    model_output = torch.randn(
        batch,
        sa_length,
        int(head.hidden_size),
        dtype=torch.float16,
        device=device)
    onnx_artifacts['action_decoder'] = _export(
        head.action_decoder, (model_output, embodiment),
        output_dir / 'action_decoder.onnx', ('model_output', 'embodiment_id'),
        ('output', ),
        opset=args.opset)

    _write_manifest(
        output_dir, {
            'schema_version':
            1,
            'shape_mode':
            'static',
            'model':
            'groot-n15',
            'precision':
            'fp16',
            'config':
            args.config.name,
            'checkpoint': (args.checkpoint.name if args.checkpoint else None),
            'checkpoint_sha256':
            (_sha256(args.checkpoint) if args.checkpoint else None),
            'torch_version':
            torch.__version__,
            'opset':
            args.opset,
            'contract': {
                'batch': batch,
                'image_views': args.image_views,
                'image_size': image_size,
                'vision_patches': num_patches,
                'vision_hidden': int(vision_config.hidden_size),
                'projected_image_tokens_per_view': projected_image_tokens,
                'sequence_length': args.sequence_length,
                'state_dim': args.state_dim,
                'action_horizon': args.action_horizon,
                'action_dim': args.action_dim,
                'future_tokens': future_tokens,
                'sa_length': sa_length,
                'backbone_dim': backbone_dim,
                'llm_hidden': llm_hidden,
                'input_embedding_dim': input_embedding_dim,
                'hidden_size': int(head.hidden_size),
                'denoise_steps': int(head.num_inference_timesteps),
            },
            'onnx': {
                'vit': 'vit_fp16.onnx',
                'llm': 'llm_fp16.onnx',
                'backbone_output': 'vlln_vl_self_attention.onnx',
                'state_encoder': 'state_encoder.onnx',
                'action_encoder': 'action_encoder.onnx',
                'dit': 'dit_fp16.onnx',
                'action_decoder': 'action_decoder.onnx',
            },
            'onnx_artifacts':
            onnx_artifacts,
        })


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=('pi05', 'groot-n15'))
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument(
        '--output-dir', type=Path, default=Path('deployment_artifacts/onnx'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--opset', type=int, default=19)
    parser.add_argument('--image-views', type=int, default=2)
    parser.add_argument('--language-tokens', type=int, default=180)
    parser.add_argument('--sequence-length', type=int, default=600)
    parser.add_argument('--state-dim', type=int, default=64)
    parser.add_argument('--action-horizon', type=int, default=10)
    parser.add_argument('--action-dim', type=int, default=32)
    parser.add_argument(
        '--denoise-steps',
        type=int,
        help='PI0.5 Euler steps (fixed to 10); not used by GR00T-N1.5')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.config = args.config.expanduser()
    args.checkpoint = (
        _resolve_checkpoint(args.checkpoint) if args.checkpoint else None)
    args.output_dir = args.output_dir.expanduser()
    if not args.config.is_file():
        raise FileNotFoundError(f'Config not found: {args.config}')
    if args.model == 'pi05':
        args.denoise_steps = (10 if args.denoise_steps is None else
                              args.denoise_steps)
        export_pi05(args)
    else:
        if args.denoise_steps is not None:
            raise ValueError(
                '--denoise-steps applies only to PI0.5; GR00T-N1.5 uses '
                'the action-head value from its config')
        export_groot(args)


if __name__ == '__main__':
    main()
