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
"""CPU-only tests for the strict ``trtexec`` version probe."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.tensorrt.trtexec_version import (parse_trtexec_help,
                                              probe_trtexec)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRTEXEC_111_HELP = ('&&&& RUNNING TensorRT.trtexec [TensorRT v110100] [b106] '
                    '# /usr/src/tensorrt/bin/trtexec --help\n'
                    '=== Model Options ===\n'
                    '  --onnx=<file>\n')


def test_tensorrt_111_help_banner_is_decoded():
    identity = parse_trtexec_help(TRTEXEC_111_HELP)
    assert identity['encoded'] == '110100'
    assert identity['label'] == 'TensorRT v110100'
    assert identity['version'] == '11.1.0'
    assert identity['banner'] == TRTEXEC_111_HELP.splitlines()[0]


@pytest.mark.parametrize('output', (
    '',
    'TensorRT v110100\n',
    TRTEXEC_111_HELP.replace('--help', '--version'),
    TRTEXEC_111_HELP + TRTEXEC_111_HELP,
    TRTEXEC_111_HELP.replace('v110100', 'v11.1.0'),
    TRTEXEC_111_HELP.replace('trtexec --help', 'trtexec --verbose --help'),
))
def test_parser_rejects_non_help_and_ambiguous_banners(output):
    with pytest.raises(ValueError):
        parse_trtexec_help(output)


def test_probe_invokes_only_help(tmp_path, monkeypatch):
    log_path = tmp_path / 'arguments.txt'
    executable = tmp_path / 'trtexec'
    executable.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$*" >"$TRTEXEC_TEST_LOG"\n'
        'if [[ "$#" -eq 1 && "$1" == "--help" ]]; then\n'
        '  printf "%s\\n" "&&&& RUNNING TensorRT.trtexec '
        '[TensorRT v110100] # trtexec --help"\n'
        '  exit 0\n'
        'fi\n'
        'exit 91\n',
        encoding='utf-8')
    executable.chmod(0o755)
    monkeypatch.setenv('TRTEXEC_TEST_LOG', str(log_path))
    identity = probe_trtexec(str(executable))
    assert identity['version'] == '11.1.0'
    assert log_path.read_text(encoding='utf-8') == '--help\n'


def test_probe_rejects_nonzero_help_even_with_banner(tmp_path):
    executable = tmp_path / 'trtexec'
    executable.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "&&&& RUNNING TensorRT.trtexec '
        '[TensorRT v110100] # trtexec --help"\n'
        'exit 2\n',
        encoding='utf-8')
    executable.chmod(0o755)
    with pytest.raises(RuntimeError, match='status 2'):
        probe_trtexec(str(executable))


def test_build_script_probes_help_before_reading_manifest(tmp_path):
    log_path = tmp_path / 'arguments.txt'
    executable = tmp_path / 'trtexec'
    executable.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$*" >>"$TRTEXEC_TEST_LOG"\n'
        'if [[ "$#" -eq 1 && "$1" == "--help" ]]; then\n'
        '  printf "%s\\n" "&&&& RUNNING TensorRT.trtexec '
        '[TensorRT v110100] # trtexec --help"\n'
        '  exit 0\n'
        'fi\n'
        'exit 91\n',
        encoding='utf-8')
    executable.chmod(0o755)
    environment = os.environ.copy()
    environment.update({
        'TRTEXEC': str(executable),
        'TRTEXEC_TEST_LOG': str(log_path),
    })
    script = PROJECT_ROOT / 'scripts/tensorrt/build_engine.sh'
    process = subprocess.run([
        'bash',
        str(script), 'pi05',
        str(tmp_path / 'missing-onnx'),
        str(tmp_path / 'engines')
    ],
                             text=True,
                             capture_output=True,
                             env=environment,
                             check=False)
    assert process.returncode != 0
    assert 'Missing export manifest' in process.stdout
    assert log_path.read_text(encoding='utf-8') == '--help\n'
    assert '--version' not in script.read_text(encoding='utf-8')


def test_build_records_strictly_probed_version_in_engine_manifest(tmp_path):
    onnx_dir = tmp_path / 'onnx'
    engine_dir = tmp_path / 'engines'
    onnx_dir.mkdir()
    artifacts = {}
    filenames = {
        'vision': 'vision.onnx',
        'prefix': 'prefix.onnx',
        'denoise': 'denoise.onnx',
    }
    for graph, filename in filenames.items():
        path = onnx_dir / filename
        path.write_bytes(graph.encode('utf-8'))
        artifacts[graph] = {
            'graph':
            filename,
            'files': [{
                'file': filename,
                'bytes': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            }],
        }
    (onnx_dir / 'export_manifest.json').write_text(
        json.dumps({
            'schema_version': 1,
            'shape_mode': 'static',
            'model': 'pi05',
            'checkpoint_sha256': 'a' * 64,
            'onnx': filenames,
            'onnx_artifacts': artifacts,
        }),
        encoding='utf-8')

    log_path = tmp_path / 'arguments.txt'
    executable = tmp_path / 'trtexec'
    executable.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$*" >>"$TRTEXEC_TEST_LOG"\n'
        'if [[ "$#" -eq 1 && "$1" == "--help" ]]; then\n'
        '  printf "%s\\n" "&&&& RUNNING TensorRT.trtexec '
        '[TensorRT v110100] # trtexec --help"\n'
        '  exit 0\n'
        'fi\n'
        'for argument in "$@"; do\n'
        '  case "$argument" in\n'
        '    --saveEngine=*) output="${argument#--saveEngine=}" ;;\n'
        '  esac\n'
        'done\n'
        'printf "%s" "engine" >"$output"\n',
        encoding='utf-8')
    executable.chmod(0o755)
    environment = os.environ.copy()
    environment.update({
        'TRTEXEC': str(executable),
        'TRTEXEC_TEST_LOG': str(log_path),
    })
    script = PROJECT_ROOT / 'scripts/tensorrt/build_engine.sh'
    subprocess.run(
        ['bash', str(script), 'pi05',
         str(onnx_dir),
         str(engine_dir)],
        text=True,
        capture_output=True,
        env=environment,
        check=True)

    manifest = json.loads(
        (engine_dir / 'engine_manifest.json').read_text(encoding='utf-8'))
    assert manifest['trtexec'] == 'TensorRT v110100'
    assert manifest['tensorrt_version'] == '11.1.0'
    assert set(manifest['engines']) == {'vision', 'prefix', 'denoise'}
    arguments = log_path.read_text(encoding='utf-8').splitlines()
    assert arguments[0] == '--help'
    assert len(arguments) == 4
    assert all('--version' not in line for line in arguments)
