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
"""Build TensorRT engines from the checkpoint produced by training."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


def resolve_tensorrt_profile(model_cfg) -> str:
    """Resolve a supported static TensorRT profile from the training model."""
    if model_cfg.type == 'PI05FlowMatching':
        if (model_cfg.n_action_steps != 10 or model_cfg.max_action_dim != 32
                or ('num_steps' in model_cfg and model_cfg.num_steps != 10)):
            raise ValueError(
                'PI0.5 TensorRT export requires the static horizon-10, '
                'action-dim-32, ten-Euler-step model contract.')
        return 'pi05'

    if model_cfg.type == 'LlavaVLA':
        if (model_cfg.vlm_backbone.type != 'EagleBackbone'
                or model_cfg.vla_head.type != 'FlowMatchingHead'):
            raise ValueError(
                'Automatic TensorRT export supports only PI0.5 and '
                'GR00T-N1.5 training models.')
        if (model_cfg.vla_head.state_dim != 64
                or model_cfg.vla_head.action_dim != 32
                or ('num_steps' in model_cfg.vla_head
                    and model_cfg.vla_head.num_steps != 10)
                or model_cfg.vla_head.num_inference_timesteps != 4):
            raise ValueError(
                'GR00T-N1.5 TensorRT export requires the static state-dim-64, '
                'horizon-10, action-dim-32, four-step model contract.')
        return 'groot-n15'

    raise ValueError(
        'Automatic TensorRT export supports only PI0.5 and GR00T-N1.5 '
        'training models.')


def validate_tensorrt_environment(project_root: Path) -> str:
    """Fail before training if the requested TensorRT toolchain is absent."""
    if importlib.util.find_spec('onnx') is None:
        raise RuntimeError(
            'Automatic TensorRT export requires ONNX. Install '
            'requirements-tensorrt.txt before starting training.')

    trtexec = os.environ['TRTEXEC'] if 'TRTEXEC' in os.environ else 'trtexec'
    resolved_trtexec = shutil.which(trtexec)
    if resolved_trtexec is None:
        raise RuntimeError(
            'Automatic TensorRT export requires trtexec. Add it to PATH or '
            'set TRTEXEC to its absolute path before starting training.')

    required_paths = (
        project_root / 'scripts/tensorrt/export_onnx.py',
        project_root / 'scripts/tensorrt/build_engine.sh',
        project_root / 'scripts/tensorrt/trtexec_version.py',
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f'TensorRT build tool not found: {path}')

    subprocess.run(
        [
            sys.executable,
            str(project_root / 'scripts/tensorrt/trtexec_version.py'),
            resolved_trtexec,
        ],
        cwd=project_root,
        check=True,
    )
    return resolved_trtexec


def _artifact_directory_name(model: str) -> str:
    return {
        'pi05': 'pi05',
        'groot-n15': 'groot_n15',
    }[model]


def build_onnx_export_command(model: str, config_path: Path,
                              checkpoint_path: Path,
                              output_root: Path) -> list[str]:
    return [
        sys.executable,
        '-m',
        'scripts.tensorrt.export_onnx',
        model,
        '--config',
        str(config_path),
        '--checkpoint',
        str(checkpoint_path),
        '--output-dir',
        str(output_root / 'onnx'),
    ]


def build_engine_command(project_root: Path, model: str,
                         output_root: Path) -> tuple[list[str], Path]:
    directory_name = _artifact_directory_name(model)
    onnx_directory = output_root / 'onnx' / directory_name
    engine_directory = output_root / 'engines' / directory_name
    command = [
        'bash',
        str(project_root / 'scripts/tensorrt/build_engine.sh'),
        model,
        str(onnx_directory),
        str(engine_directory),
    ]
    return command, engine_directory


def convert_checkpoint_to_tensorrt(project_root: Path, model: str,
                                   config_path: Path, checkpoint_path: Path,
                                   output_root: Path, trtexec: str) -> Path:
    """Export ONNX partitions and build TensorRT engines synchronously."""
    project_root = project_root.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()

    export_command = build_onnx_export_command(model, config_path,
                                               checkpoint_path, output_root)
    build_command, engine_directory = build_engine_command(
        project_root, model, output_root)

    subprocess.run(export_command, cwd=project_root, check=True)
    build_environment = os.environ.copy()
    build_environment['TRTEXEC'] = trtexec
    subprocess.run(
        build_command,
        cwd=project_root,
        env=build_environment,
        check=True,
    )
    return engine_directory
