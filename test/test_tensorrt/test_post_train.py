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
"""CPU-only tests for automatic post-training TensorRT conversion."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

import pytest

from scripts.tensorrt.post_train import (build_engine_command,
                                         build_onnx_export_command,
                                         convert_checkpoint_to_tensorrt,
                                         get_non_distributed_environment,
                                         resolve_tensorrt_profile,
                                         validate_tensorrt_environment)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _AttributeDict(dict):

    def __getattr__(self, name):
        return self[name]


def test_pi05_profile_uses_static_bf16_contract():
    model_cfg = _AttributeDict(
        type='PI05FlowMatching', n_action_steps=10, max_action_dim=32)
    assert resolve_tensorrt_profile(model_cfg) == 'pi05'


def test_groot_profile_uses_static_fp16_contract():
    model_cfg = SimpleNamespace(
        type='LlavaVLA',
        vlm_backbone=SimpleNamespace(type='EagleBackbone'),
        vla_head=_AttributeDict(
            type='FlowMatchingHead',
            state_dim=64,
            action_dim=32,
            num_inference_timesteps=4),
    )
    assert resolve_tensorrt_profile(model_cfg) == 'groot-n15'


def test_unsupported_models_and_static_contracts_are_rejected():
    with pytest.raises(ValueError, match='horizon-10'):
        resolve_tensorrt_profile(
            _AttributeDict(
                type='PI05FlowMatching', n_action_steps=50, max_action_dim=32))

    with pytest.raises(ValueError, match='ten-Euler-step'):
        resolve_tensorrt_profile(
            _AttributeDict(
                type='PI05FlowMatching',
                n_action_steps=10,
                max_action_dim=32,
                num_steps=5))

    incompatible_groot = SimpleNamespace(
        type='LlavaVLA',
        vlm_backbone=SimpleNamespace(type='EagleBackbone'),
        vla_head=_AttributeDict(
            type='FlowMatchingHead',
            state_dim=64,
            action_dim=64,
            num_inference_timesteps=4),
    )
    with pytest.raises(ValueError, match='action-dim-32'):
        resolve_tensorrt_profile(incompatible_groot)

    with pytest.raises(ValueError, match='supports only'):
        resolve_tensorrt_profile(SimpleNamespace(type='OpenVLA'))


@patch('scripts.tensorrt.post_train.importlib.util.find_spec')
def test_missing_onnx_fails_before_training(find_spec):
    find_spec.return_value = None
    with pytest.raises(RuntimeError, match='requires ONNX'):
        validate_tensorrt_environment(PROJECT_ROOT)


@patch('scripts.tensorrt.post_train.shutil.which')
@patch('scripts.tensorrt.post_train.importlib.util.find_spec')
def test_missing_trtexec_fails_before_training(find_spec, which):
    find_spec.return_value = object()
    which.return_value = None
    with pytest.raises(RuntimeError, match='requires trtexec'):
        validate_tensorrt_environment(PROJECT_ROOT)


@patch('scripts.tensorrt.post_train.subprocess.run')
@patch('scripts.tensorrt.post_train.shutil.which')
@patch('scripts.tensorrt.post_train.importlib.util.find_spec')
def test_preflight_probes_trtexec_version(find_spec, which, run):
    find_spec.return_value = object()
    which.return_value = '/opt/tensorrt/bin/trtexec'
    resolved = validate_tensorrt_environment(PROJECT_ROOT)
    assert resolved == '/opt/tensorrt/bin/trtexec'
    run.assert_called_once_with(
        [
            sys.executable,
            str(PROJECT_ROOT / 'scripts/tensorrt/trtexec_version.py'),
            '/opt/tensorrt/bin/trtexec',
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def test_commands_use_matching_artifact_directories():
    output_root = Path('/tmp/training-run/tensorrt')
    export_command = build_onnx_export_command('groot-n15',
                                               Path('/tmp/config.json'),
                                               Path('/tmp/model.safetensors'),
                                               output_root)
    assert export_command[:4] == [
        sys.executable, '-m', 'scripts.tensorrt.export_onnx', 'groot-n15'
    ]
    assert export_command[-2:] == [
        '--output-dir', '/tmp/training-run/tensorrt/onnx'
    ]

    build_command, engine_directory = build_engine_command(
        PROJECT_ROOT, 'groot-n15', output_root)
    assert build_command[-2] == ('/tmp/training-run/tensorrt/onnx/groot_n15')
    assert engine_directory == Path(
        '/tmp/training-run/tensorrt/engines/groot_n15')


@patch('scripts.tensorrt.post_train.subprocess.run')
def test_conversion_runs_export_before_engine_build(run):
    output_root = Path('/tmp/training-run/tensorrt')
    engine_directory = convert_checkpoint_to_tensorrt(
        project_root=PROJECT_ROOT,
        model='pi05',
        config_path=Path('/tmp/config.json'),
        checkpoint_path=Path('/tmp/model.safetensors'),
        output_root=output_root,
        trtexec='/opt/tensorrt/bin/trtexec',
    )

    export_command = build_onnx_export_command('pi05',
                                               Path('/tmp/config.json'),
                                               Path('/tmp/model.safetensors'),
                                               output_root)
    build_command, expected_directory = build_engine_command(
        PROJECT_ROOT, 'pi05', output_root)
    export_environment = get_non_distributed_environment()
    build_environment = export_environment.copy()
    build_environment['TRTEXEC'] = '/opt/tensorrt/bin/trtexec'
    assert run.call_args_list == [
        call(
            export_command,
            cwd=PROJECT_ROOT,
            env=export_environment,
            check=True,
        ),
        call(
            build_command,
            cwd=PROJECT_ROOT,
            env=build_environment,
            check=True,
        ),
    ]
    assert engine_directory == expected_directory


@patch('scripts.tensorrt.post_train.subprocess.run')
def test_engine_build_failure_is_not_hidden(run):
    run.side_effect = (
        None,
        subprocess.CalledProcessError(1, ['build_engine.sh']),
    )
    with pytest.raises(subprocess.CalledProcessError):
        convert_checkpoint_to_tensorrt(
            project_root=PROJECT_ROOT,
            model='pi05',
            config_path=Path('/tmp/config.json'),
            checkpoint_path=Path('/tmp/model.safetensors'),
            output_root=Path('/tmp/training-run/tensorrt'),
            trtexec='/opt/tensorrt/bin/trtexec',
        )


def test_latest_groot_config_paths_are_used():
    expected = (
        PROJECT_ROOT / 'configs/gr00tn15/'
        'gr00tn15_eagle_3b_libero_10_full_finetune.py')
    assert expected.is_file()
    for relative in ('scripts/tensorrt/export_onnx.py',
                     'scripts/tensorrt/README.md'):
        text = (PROJECT_ROOT / relative).read_text(encoding='utf-8')
        assert 'configs/gr00t/gr00t_eagle_3b_libero_10' not in text
