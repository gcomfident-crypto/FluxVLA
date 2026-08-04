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
"""Run one non-benchmark TensorRT inference through FluxVLA's public API.

The generated inputs satisfy each static engine contract but are synthetic;
this command validates loading, binding, orchestration, shape, dtype, and
finite outputs.  It does not measure latency or task success.
"""

from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

import torch

from .export_onnx import _load_checkpoint_state, _resolve_checkpoint


def _predict_once(predict, args, device: torch.device) -> torch.Tensor:
    """Run warm-ups and one prediction, optionally inside an nsys range."""
    with torch.inference_mode():
        for _ in range(args.warmup_calls):
            predict()
        if args.warmup_calls:
            torch.cuda.synchronize(device)

        if not args.cuda_profiler_range:
            return predict()

        # Nsight Systems can use the CUDA Profiler API as an exact capture
        # boundary.  Keep model/engine loading and warm-up work outside it so
        # the resulting report contains one public predict_action call.
        torch.cuda.synchronize(device)
        cudart = torch.cuda.cudart()
        start_status = cudart.cudaProfilerStart()
        if start_status != 0:
            raise RuntimeError(
                f'cudaProfilerStart failed with CUDA status {start_status}')
        stop_status = 0
        try:
            output = predict()
            torch.cuda.synchronize(device)
        finally:
            stop_status = cudart.cudaProfilerStop()
        if stop_status != 0:
            raise RuntimeError(
                f'cudaProfilerStop failed with CUDA status {stop_status}')
        return output


def _validate_output(output: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tuple(output.shape) != shape:
        raise RuntimeError(f'Unexpected action shape: {tuple(output.shape)} '
                           f'!= {shape}')
    if output.dtype != torch.float32:
        raise RuntimeError(f'Unexpected action dtype: {output.dtype}')


def _pi05(args, device: torch.device) -> torch.Tensor:
    from mmengine import Config

    import fluxvla.models  # noqa: F401
    from fluxvla.engines import build_vla_from_cfg

    config = args.config or Path(
        'configs/pi05/pi05_paligemma_libero_10_tensorrt.py')
    if not config.is_file():
        raise FileNotFoundError(f'PI0.5 TensorRT config not found: {config}')
    cfg = Config.fromfile(str(config))
    model_cfg = copy.deepcopy(cfg.inference_model)
    model_cfg.engine_dir = str(args.engine_dir)
    model = build_vla_from_cfg(model_cfg).eval()
    model.bind_eval_checkpoint(args.checkpoint)
    try:
        model.prepare_inference_runtime(device)
        contract = model.runtime.contract
        batch = {
            'images':
            torch.zeros(
                contract.batch,
                contract.image_views * 3,
                contract.image_height,
                contract.image_width,
                dtype=torch.float32,
                device=device),
            'lang_tokens':
            torch.zeros(
                contract.batch,
                contract.language_tokens,
                dtype=torch.int64,
                device=device),
            'states':
            torch.zeros(
                contract.batch,
                contract.action_dim,
                dtype=torch.float32,
                device=device),
            'img_masks':
            torch.ones(
                contract.batch,
                contract.image_views,
                dtype=torch.bool,
                device=device),
            'lang_masks':
            torch.ones(
                contract.batch,
                contract.language_tokens,
                dtype=torch.bool,
                device=device),
        }

        def predict() -> torch.Tensor:
            return model.predict_action(
                **batch,
                seed=args.seed,
                num_inference_steps=contract.denoise_steps,
                unnorm_key='tensorrt_smoke')

        output = _predict_once(predict, args, device)
        _validate_output(
            output,
            (contract.batch, contract.action_horizon, contract.action_dim))
        return output
    finally:
        model.close_inference_runtime()


def _groot(args, device: torch.device) -> torch.Tensor:
    from mmengine import Config

    import fluxvla.models  # noqa: F401
    from fluxvla.engines import build_vla_from_cfg

    config = args.config or Path(
        'configs/gr00t/gr00t_eagle_3b_libero_10_tensorrt.py')
    if not config.is_file():
        raise FileNotFoundError(f'GR00T TensorRT config not found: {config}')
    cfg = Config.fromfile(str(config))
    model_cfg = copy.deepcopy(cfg.inference_model)
    model_cfg.vlm_backbone.engine_dir = str(args.engine_dir)
    model_cfg.vla_head.engine_dir = str(args.engine_dir)
    model = build_vla_from_cfg(model_cfg).eval()
    checkpoint = _resolve_checkpoint(args.checkpoint)
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
        contract = model.vlm_backbone.trt_contract
        image_tokens = (
            contract.image_views * contract.projected_image_tokens_per_view)
        if image_tokens > contract.sequence_length:
            raise ValueError('Image tokens exceed the GR00T sequence length')
        lang_tokens = torch.zeros(
            contract.batch,
            contract.sequence_length,
            dtype=torch.int64,
            device=device)
        lang_tokens[:, :image_tokens] = model.vlm_backbone._image_token_index
        batch = {
            'images':
            torch.zeros(
                contract.batch,
                contract.image_views * 3,
                contract.image_size,
                contract.image_size,
                dtype=torch.float32,
                device=device),
            'lang_tokens':
            lang_tokens,
            'states':
            torch.zeros(
                contract.batch,
                contract.state_dim,
                dtype=torch.float32,
                device=device),
            'img_masks':
            torch.ones(
                contract.batch,
                contract.image_views,
                dtype=torch.bool,
                device=device),
            'lang_masks':
            torch.ones(
                contract.batch,
                contract.sequence_length,
                dtype=torch.bool,
                device=device),
            'embodiment_ids':
            torch.full((contract.batch, ),
                       args.embodiment_id,
                       dtype=torch.int64,
                       device=device),
        }

        def predict() -> torch.Tensor:
            return model.predict_action(
                **batch,
                seed=args.seed,
                num_inference_steps=contract.denoise_steps)

        output = _predict_once(predict, args, device)
        action_dim = (
            model.vla_head.ori_action_dim if model.vla_head.ori_action_dim
            is not None else contract.action_dim)
        _validate_output(output,
                         (contract.batch, contract.action_horizon, action_dim))
        return output
    finally:
        model.vla_head.close_inference_runtime()
        model.vlm_backbone.close_inference_runtime()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=('pi05', 'groot-n15'))
    parser.add_argument('--engine-dir', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--embodiment-id', type=int, default=2)
    parser.add_argument(
        '--warmup-calls',
        type=int,
        default=0,
        help='public predict_action calls completed before the checked call')
    parser.add_argument(
        '--cuda-profiler-range',
        action='store_true',
        help=('surround only the checked predict_action call with '
              'cudaProfilerStart/Stop for Nsight Systems capture'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed < 0:
        raise ValueError('--seed must be non-negative')
    if args.warmup_calls < 0:
        raise ValueError('--warmup-calls must be non-negative')
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('TensorRT smoke inference requires a CUDA device')
    args.engine_dir = args.engine_dir.expanduser()
    args.checkpoint = _resolve_checkpoint(args.checkpoint)
    if args.config is not None:
        args.config = args.config.expanduser()
    output = _pi05(args, device) if args.model == 'pi05' else _groot(
        args, device)
    torch.cuda.synchronize(device)
    finite = bool(torch.isfinite(output).all().item())
    result = {
        'model': args.model,
        'shape': list(output.shape),
        'dtype': str(output.dtype),
        'device': str(output.device),
        'finite': finite,
    }
    print(json.dumps(result, sort_keys=True))
    if not finite:
        raise RuntimeError('TensorRT smoke inference produced NaN or Inf')


if __name__ == '__main__':
    main()
