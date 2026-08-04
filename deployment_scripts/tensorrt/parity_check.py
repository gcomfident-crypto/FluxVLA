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
"""Compare native PyTorch, export-wrapper, and TensorRT inference numerics."""

from __future__ import annotations
import argparse
import copy
import gc
import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Mapping

import torch
from mmengine import Config

from fluxvla.deployment.tensorrt.groot_n15 import GR00TN15TensorRTContract
from fluxvla.deployment.tensorrt.pi05 import PI05ExportContract
from fluxvla.deployment.tensorrt.pi05.runtime import _euler_timesteps
from .export_onnx import (_load_checkpoint_state, _load_model,
                          _resolve_checkpoint)
from .model_forward import (GR00TBackboneOutputExportWrapper,
                            GR00TLLMExportWrapper, GR00TVisionExportWrapper,
                            PI05DenoiseStepExportWrapper,
                            PI05PrefixExportWrapper, PI05VisionExportWrapper)

_DEFAULT_CONFIGS = {
    'pi05': Path('configs/pi05/pi05_paligemma_libero_10_full_finetune.py'),
    'groot-n15':
    Path('configs/gr00t/gr00t_eagle_3b_libero_10_full_finetune.py'),
}
_DEFAULT_TRT_CONFIGS = {
    'pi05': Path('configs/pi05/pi05_paligemma_libero_10_tensorrt.py'),
    'groot-n15': Path('configs/gr00t/gr00t_eagle_3b_libero_10_tensorrt.py'),
}
_DEFAULT_THRESHOLDS = {
    'pi05': {
        'max_abs': 0.05,
        'rmse': 0.010,
        'min_cosine': 0.999,
    },
    'groot-n15': {
        'max_abs': 0.08,
        'rmse': 0.015,
        'min_cosine': 0.999,
    },
}


def _contract_from_manifest(engine_dir: Path, model: str):
    path = engine_dir / 'engine_manifest.json'
    if not path.is_file():
        raise FileNotFoundError(f'Engine manifest not found: {path}')
    value = json.loads(path.read_text(encoding='utf-8'))
    expected_model = 'pi05' if model == 'pi05' else 'groot-n15'
    if (value.get('model') != expected_model
            or value.get('engine_model') != expected_model):
        raise ValueError(f'Engine manifest is not for {expected_model}')
    raw = value.get('contract')
    if not isinstance(raw, Mapping):
        raise ValueError('Engine manifest has no model contract')
    contract_type = (
        PI05ExportContract if model == 'pi05' else GR00TN15TensorRTContract)
    names = {field.name for field in fields(contract_type)}
    return contract_type(**{name: raw[name] for name in names})


def _cpu_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    return generator


def _pi05_inputs(contract: PI05ExportContract, seed: int) -> dict:
    generator = _cpu_generator(seed)
    images = torch.randn(
        contract.batch,
        contract.image_views * 3,
        contract.image_height,
        contract.image_width,
        dtype=torch.float32,
        generator=generator) * 0.5
    tokens = torch.arange(contract.language_tokens, dtype=torch.int64)[None, :]
    tokens = tokens.expand(contract.batch, -1).contiguous() + 7
    lang_masks = torch.ones(
        contract.batch, contract.language_tokens, dtype=torch.bool)
    padding = min(24, contract.language_tokens // 4)
    if padding:
        lang_masks[:, -padding:] = False
    return {
        'images':
        images,
        'lang_tokens':
        tokens,
        'states':
        torch.randn(
            contract.batch,
            contract.action_dim,
            dtype=torch.float32,
            generator=generator),
        'img_masks':
        torch.ones(contract.batch, contract.image_views, dtype=torch.bool),
        'lang_masks':
        lang_masks,
        'noise':
        torch.randn(
            contract.batch,
            contract.action_horizon,
            contract.action_dim,
            dtype=torch.float32,
            generator=generator),
    }


def _groot_inputs(contract: GR00TN15TensorRTContract, image_token: int,
                  seed: int, embodiment_id: int) -> dict:
    generator = _cpu_generator(seed)
    image_tokens = (
        contract.image_views * contract.projected_image_tokens_per_view)
    if image_tokens > contract.sequence_length:
        raise ValueError('Image tokens exceed the GR00T sequence length')
    tokens = torch.arange(contract.sequence_length, dtype=torch.int64)[None, :]
    tokens = (tokens.expand(contract.batch, -1).contiguous() % 30000) + 10
    tokens[:, :image_tokens] = image_token
    lang_masks = torch.ones(
        contract.batch, contract.sequence_length, dtype=torch.bool)
    padding = min(32, max(0, contract.sequence_length - image_tokens))
    if padding:
        lang_masks[:, -padding:] = False
    return {
        'images':
        torch.randn(
            contract.batch,
            contract.image_views * 3,
            contract.image_size,
            contract.image_size,
            dtype=torch.float32,
            generator=generator) * 0.5,
        'lang_tokens':
        tokens,
        'states':
        torch.randn(
            contract.batch,
            contract.state_dim,
            dtype=torch.float32,
            generator=generator),
        'img_masks':
        torch.ones(contract.batch, contract.image_views, dtype=torch.bool),
        'lang_masks':
        lang_masks,
        'embodiment_ids':
        torch.full((contract.batch, ), embodiment_id, dtype=torch.int64),
        'noise':
        torch.randn(
            contract.batch,
            contract.action_horizon,
            contract.action_dim,
            dtype=torch.float32,
            generator=generator),
    }


def _to_device(inputs: Mapping[str, torch.Tensor],
               device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in inputs.items()}


def _pi05_wrapper_action(model: torch.nn.Module, contract: PI05ExportContract,
                         inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    vision = PI05VisionExportWrapper(model,
                                     contract).to(inputs['images'].device)
    prefix = PI05PrefixExportWrapper(model,
                                     contract).to(inputs['images'].device)
    denoise = PI05DenoiseStepExportWrapper(model, contract).to(
        inputs['images'].device)
    images = inputs['images'].reshape(
        contract.image_views, 3, contract.image_height,
        contract.image_width).to(torch.bfloat16).contiguous()
    image_prefix = vision(images)
    lang_tokens = inputs['lang_tokens'].to(torch.int32).contiguous()
    lang_mask = inputs['lang_masks'].contiguous()
    caches = prefix(image_prefix, lang_tokens, lang_mask)
    actions = inputs['noise'].clone()
    timestep = torch.empty((contract.batch, ),
                           dtype=torch.float32,
                           device=actions.device)
    for value in _euler_timesteps(contract.denoise_steps):
        timestep.fill_(value)
        actions = denoise(actions, timestep, lang_mask, *caches)
    return actions


def _run_pi05_references(config: Path, checkpoint: Path,
                         contract: PI05ExportContract, inputs_cpu: dict,
                         device: torch.device) -> dict[str, torch.Tensor]:
    model = _load_model(config, checkpoint, device, torch.bfloat16)
    inputs = _to_device(inputs_cpu, device)
    with torch.inference_mode():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            native = model.predict_action(
                images=inputs['images'],
                lang_tokens=inputs['lang_tokens'],
                states=inputs['states'],
                img_masks=inputs['img_masks'],
                lang_masks=inputs['lang_masks'],
                noise=inputs['noise'].clone())
        wrapper = _pi05_wrapper_action(model, contract, inputs)
    torch.cuda.synchronize(device)
    result = {
        'pytorch': native.float().cpu(),
        'wrapper': wrapper.float().cpu(),
    }
    del wrapper, native, inputs, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _groot_native_action(model: torch.nn.Module,
                         contract: GR00TN15TensorRTContract,
                         inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    features, attention_mask, _ = model.vlm_backbone(
        images=inputs['images'],
        lang_tokens=inputs['lang_tokens'],
        img_masks=inputs['img_masks'],
        lang_masks=inputs['lang_masks'])
    head = model.vla_head
    features = head.vl_self_attention(head.vlln(features))
    state_features = head.state_encoder(inputs['states'].unsqueeze(1),
                                        inputs['embodiment_ids'])
    actions = inputs['noise'].to(features.dtype).clone()
    dt = 1.0 / contract.denoise_steps
    for step in range(contract.denoise_steps):
        discretized = int(
            (step / float(contract.denoise_steps)) * head.num_timestep_buckets)
        timestep = torch.full((contract.batch, ),
                              discretized,
                              device=actions.device)
        velocity = head.denoise_step(actions, features, state_features,
                                     attention_mask, inputs['embodiment_ids'],
                                     timestep)
        actions = actions + dt * velocity
    if head.ori_action_dim is not None:
        actions = actions[:, :, :head.ori_action_dim]
    return actions


def _groot_wrapper_backbone(
    model: torch.nn.Module,
    contract: GR00TN15TensorRTContract,
    inputs: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    backbone = model.vlm_backbone
    vision = GR00TVisionExportWrapper(backbone)
    llm = GR00TLLMExportWrapper(backbone)
    pixel_values = inputs['images'].reshape(
        contract.image_views, 3, contract.image_size,
        contract.image_size).to(torch.float16).contiguous()
    position_ids = torch.arange(
        contract.vision_patches, dtype=torch.int64,
        device=pixel_values.device)[None, :].expand(contract.image_views,
                                                    -1).contiguous()
    vit_embeds = vision(pixel_values, position_ids)
    if backbone.vlm.use_pixel_shuffle:
        height = width = math.isqrt(vit_embeds.shape[1])
        if height * width != vit_embeds.shape[1]:
            raise ValueError('GR00T vision output is not a square token grid')
        vit_embeds = vit_embeds.reshape(contract.image_views, height, width,
                                        contract.vision_hidden)
        vit_embeds = backbone.vlm.pixel_shuffle(
            vit_embeds, scale_factor=backbone.vlm.downsample_ratio)
        vit_embeds = vit_embeds.reshape(contract.image_views, -1,
                                        vit_embeds.shape[-1])
    vit_embeds = backbone.vlm.mlp1(vit_embeds).to(torch.float16).contiguous()
    vit_embeds = vit_embeds.reshape(contract.batch, -1, contract.backbone_dim)

    tokens = inputs['lang_tokens'].to(torch.int64).contiguous()
    embedding = backbone.vlm.language_model.get_input_embeddings()
    input_embeds = embedding(tokens).to(torch.float16)
    flat_embeds = input_embeds.reshape(-1, contract.backbone_dim).contiguous()
    selected = tokens.reshape(-1) == int(backbone.vlm.image_token_index)
    if int(selected.sum().item()) != vit_embeds.shape[1]:
        raise ValueError(
            'Synthetic GR00T prompt has the wrong image-token count')
    flat_embeds[selected] = vit_embeds.reshape(-1, contract.backbone_dim)
    input_embeds = flat_embeds.reshape(contract.batch,
                                       contract.sequence_length,
                                       contract.backbone_dim).contiguous()
    return llm(input_embeds, inputs['lang_masks'].to(torch.int64).contiguous())


def _groot_wrapper_action(model: torch.nn.Module,
                          contract: GR00TN15TensorRTContract,
                          inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    features = _groot_wrapper_backbone(model, contract, inputs)
    head = model.vla_head
    process = GR00TBackboneOutputExportWrapper(head)
    features = process(features)
    embodiment = inputs['embodiment_ids'].to(torch.int64).contiguous()
    state = inputs['states'].to(torch.float16).unsqueeze(1).contiguous()
    state_features = head.state_encoder(state, embodiment)
    actions = inputs['noise'].to(torch.float16).contiguous()
    timestep = torch.empty((contract.batch, ),
                           dtype=torch.int64,
                           device=actions.device)
    position_ids = torch.arange(
        contract.action_horizon, dtype=torch.long, device=actions.device)
    position = head.position_embedding(position_ids).unsqueeze(0).to(
        torch.float16)
    future = head.future_tokens.weight.unsqueeze(0).to(torch.float16)
    dt = 1.0 / contract.denoise_steps
    for step in range(contract.denoise_steps):
        discretized = int(
            (step / float(contract.denoise_steps)) * head.num_timestep_buckets)
        timestep.fill_(discretized)
        action_features = head.action_encoder(actions, timestep, embodiment)
        action_features = action_features + position
        sa_embs = torch.cat((state_features, future, action_features), dim=1)
        model_output = head.model(sa_embs, features, timestep)
        prediction = head.action_decoder(model_output, embodiment)
        actions = actions + dt * prediction[:, -contract.action_horizon:]
    if head.ori_action_dim is not None:
        actions = actions[:, :, :head.ori_action_dim]
    return actions


def _run_groot_references(
        config: Path, checkpoint: Path, contract: GR00TN15TensorRTContract,
        seed: int, embodiment_id: int,
        device: torch.device) -> tuple[dict[str, torch.Tensor], dict]:
    model = _load_model(config, checkpoint, device, torch.float16)
    image_token = int(model.vlm_backbone.vlm.image_token_index)
    inputs_cpu = _groot_inputs(contract, image_token, seed, embodiment_id)
    inputs = _to_device(inputs_cpu, device)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        native = _groot_native_action(model, contract, inputs)
        wrapper = _groot_wrapper_action(model, contract, inputs)
    torch.cuda.synchronize(device)
    result = {
        'pytorch': native.float().cpu(),
        'wrapper': wrapper.float().cpu(),
    }
    del wrapper, native, inputs, model
    gc.collect()
    torch.cuda.empty_cache()
    return result, inputs_cpu


def _run_pi05_tensorrt(config: Path, checkpoint: Path, engine_dir: Path,
                       inputs_cpu: dict, device: torch.device) -> torch.Tensor:
    import fluxvla.models  # noqa: F401
    from fluxvla.engines import build_vla_from_cfg

    cfg = Config.fromfile(str(config))
    model_cfg = copy.deepcopy(cfg.inference_model)
    model_cfg.engine_dir = str(engine_dir)
    model = build_vla_from_cfg(model_cfg).eval()
    model.bind_eval_checkpoint(checkpoint)
    try:
        model.prepare_inference_runtime(device)
        inputs = _to_device(inputs_cpu, device)
        with torch.inference_mode():
            output = model.predict_action(
                images=inputs['images'],
                lang_tokens=inputs['lang_tokens'],
                states=inputs['states'],
                img_masks=inputs['img_masks'],
                lang_masks=inputs['lang_masks'],
                noise=inputs['noise'].clone(),
                num_inference_steps=model.num_steps)
        torch.cuda.synchronize(device)
        return output.float().cpu()
    finally:
        model.close_inference_runtime()
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _run_groot_tensorrt(config: Path, checkpoint: Path, engine_dir: Path,
                        inputs_cpu: dict,
                        device: torch.device) -> torch.Tensor:
    import fluxvla.models  # noqa: F401
    from fluxvla.engines import build_vla_from_cfg

    cfg = Config.fromfile(str(config))
    model_cfg = copy.deepcopy(cfg.inference_model)
    model_cfg.vlm_backbone.engine_dir = str(engine_dir)
    model_cfg.vla_head.engine_dir = str(engine_dir)
    model = build_vla_from_cfg(model_cfg).eval()
    model.vlm_backbone.bind_eval_checkpoint(checkpoint)
    model.vla_head.bind_eval_checkpoint(checkpoint)
    state = _load_checkpoint_state(checkpoint)
    state = model.prepare_eval_state_dict(state)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device=device, dtype=torch.float16)
    try:
        model.vlm_backbone.prepare_inference_runtime(device)
        model.vla_head.prepare_inference_runtime(device)
        inputs = _to_device(inputs_cpu, device)
        with torch.inference_mode():
            output = model.predict_action(
                images=inputs['images'],
                lang_tokens=inputs['lang_tokens'],
                states=inputs['states'],
                img_masks=inputs['img_masks'],
                lang_masks=inputs['lang_masks'],
                embodiment_ids=inputs['embodiment_ids'],
                noise=inputs['noise'].clone(),
                num_inference_steps=model.vla_head.num_inference_timesteps)
        torch.cuda.synchronize(device)
        return output.float().cpu()
    finally:
        model.vla_head.close_inference_runtime()
        model.vlm_backbone.close_inference_runtime()
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    if tuple(reference.shape) != tuple(candidate.shape):
        raise ValueError(f'Parity shapes differ: {tuple(reference.shape)} != '
                         f'{tuple(candidate.shape)}')
    reference = reference.detach().to(torch.float64).reshape(-1)
    candidate = candidate.detach().to(torch.float64).reshape(-1)
    if not (torch.isfinite(reference).all()
            and torch.isfinite(candidate).all()):
        return {
            'max_abs': None,
            'rmse': None,
            'cosine': None,
            'finite': False,
        }
    difference = candidate - reference
    max_abs = difference.abs().max().item()
    rmse = difference.square().mean().sqrt().item()
    reference_norm = torch.linalg.vector_norm(reference).item()
    candidate_norm = torch.linalg.vector_norm(candidate).item()
    if reference_norm == 0.0 or candidate_norm == 0.0:
        cosine = 1.0 if torch.equal(reference, candidate) else 0.0
    else:
        cosine = torch.dot(reference, candidate).item() / (
            reference_norm * candidate_norm)
    return {
        'max_abs': max_abs,
        'rmse': rmse,
        'cosine': cosine,
        'finite': True,
    }


def _evaluate(outputs: Mapping[str, torch.Tensor],
              thresholds: Mapping[str, float]) -> tuple[dict, bool]:
    comparisons = {}
    passed = True
    for left, right in (('pytorch', 'wrapper'), ('pytorch', 'tensorrt'),
                        ('wrapper', 'tensorrt')):
        metrics = _metrics(outputs[left], outputs[right])
        pair_passed = (
            metrics['finite'] and metrics['max_abs'] <= thresholds['max_abs']
            and metrics['rmse'] <= thresholds['rmse']
            and metrics['cosine'] >= thresholds['min_cosine'])
        metrics['passed'] = pair_passed
        comparisons[f'{left}_vs_{right}'] = metrics
        passed = passed and pair_passed
    return comparisons, passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=('pi05', 'groot-n15'))
    parser.add_argument('--engine-dir', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--tensorrt-config', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--embodiment-id', type=int, default=2)
    parser.add_argument('--max-abs', type=float)
    parser.add_argument('--rmse', type=float)
    parser.add_argument('--min-cosine', type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed < 0:
        raise ValueError('--seed must be non-negative')
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('TensorRT parity requires a CUDA device')
    args.engine_dir = args.engine_dir.expanduser()
    args.checkpoint = _resolve_checkpoint(args.checkpoint)
    args.config = (args.config or _DEFAULT_CONFIGS[args.model]).expanduser()
    args.tensorrt_config = (args.tensorrt_config
                            or _DEFAULT_TRT_CONFIGS[args.model]).expanduser()
    for name in ('config', 'tensorrt_config'):
        path = getattr(args, name)
        if not path.is_file():
            raise FileNotFoundError(f'{name} not found: {path}')

    thresholds = dict(_DEFAULT_THRESHOLDS[args.model])
    if args.max_abs is not None:
        thresholds['max_abs'] = args.max_abs
    if args.rmse is not None:
        thresholds['rmse'] = args.rmse
    if args.min_cosine is not None:
        thresholds['min_cosine'] = args.min_cosine
    if (thresholds['max_abs'] < 0 or thresholds['rmse'] < 0
            or not -1.0 <= thresholds['min_cosine'] <= 1.0):
        raise ValueError('Invalid parity threshold')

    contract = _contract_from_manifest(args.engine_dir, args.model)
    if args.model == 'pi05':
        inputs_cpu = _pi05_inputs(contract, args.seed)
        outputs = _run_pi05_references(args.config, args.checkpoint, contract,
                                       inputs_cpu, device)
        outputs['tensorrt'] = _run_pi05_tensorrt(args.tensorrt_config,
                                                 args.checkpoint,
                                                 args.engine_dir, inputs_cpu,
                                                 device)
    else:
        outputs, inputs_cpu = _run_groot_references(args.config,
                                                    args.checkpoint, contract,
                                                    args.seed,
                                                    args.embodiment_id, device)
        outputs['tensorrt'] = _run_groot_tensorrt(args.tensorrt_config,
                                                  args.checkpoint,
                                                  args.engine_dir, inputs_cpu,
                                                  device)

    comparisons, passed = _evaluate(outputs, thresholds)
    result = {
        'model': args.model,
        'checkpoint': args.checkpoint.name,
        'device': str(device),
        'seed': args.seed,
        'shape': list(outputs['pytorch'].shape),
        'thresholds': thresholds,
        'comparisons': comparisons,
        'passed': passed,
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
