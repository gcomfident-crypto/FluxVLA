#!/usr/bin/env bash
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

# Build FluxVLA TensorRT engines from ONNX without running a benchmark.
#
# Usage:
#   bash scripts/tensorrt/build_engine.sh \
#     pi05 deployment_artifacts/onnx/pi05 deployment_artifacts/engines/pi05
#   bash scripts/tensorrt/build_engine.sh \
#     groot-n15 deployment_artifacts/onnx/groot_n15 \
#     deployment_artifacts/engines/groot_n15

set -euo pipefail

MODEL=${1:?model is required: pi05 or groot-n15}
ONNX_DIR=${2:?ONNX directory is required}
ENGINE_DIR=${3:?engine output directory is required}
TRTEXEC=${TRTEXEC:-trtexec}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if ! command -v "${TRTEXEC}" >/dev/null 2>&1; then
    echo "trtexec was not found. Set TRTEXEC=/absolute/path/to/trtexec."
    exit 1
fi

if [[ "${MODEL}" != "pi05" && "${MODEL}" != "groot-n15" ]]; then
    echo "Unsupported model: ${MODEL}; expected pi05 or groot-n15"
    exit 1
fi

TRTEXEC_IDENTITY=$(
    python3 "${SCRIPT_DIR}/trtexec_version.py" "${TRTEXEC}"
)

MANIFEST="${ONNX_DIR}/export_manifest.json"
if [[ ! -f "${MANIFEST}" ]]; then
    echo "Missing export manifest: ${MANIFEST}"
    exit 1
fi

# Fail closed if an ONNX graph or one of its external-data files was changed
# after export.  This keeps the checkpoint identity in the engine manifest
# tied to the exact graphs that trtexec consumes.
python3 - "${MANIFEST}" "${ONNX_DIR}" "${MODEL}" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
onnx_dir = pathlib.Path(sys.argv[2]).resolve()
model = sys.argv[3]
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
if manifest.get('schema_version') != 1:
    raise ValueError('Unsupported export manifest schema')
if manifest.get('shape_mode') != 'static':
    raise ValueError('TensorRT export manifest must declare static shapes')
if manifest.get('model') != model:
    raise ValueError(
        f'Export manifest model {manifest.get("model")!r} != {model!r}')
checkpoint_sha256 = manifest.get('checkpoint_sha256')
if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
    raise ValueError('Export manifest has no checkpoint SHA256 identity')
expected = {
    'pi05': ('vision', 'prefix', 'denoise'),
    'groot-n15': ('vit', 'llm', 'backbone_output', 'state_encoder',
                  'action_encoder', 'dit', 'action_decoder'),
}[model]
canonical_files = {
    'pi05': {
        'vision': 'vision.onnx',
        'prefix': 'prefix.onnx',
        'denoise': 'denoise.onnx',
    },
    'groot-n15': {
        'vit': 'vit_fp16.onnx',
        'llm': 'llm_fp16.onnx',
        'backbone_output': 'vlln_vl_self_attention.onnx',
        'state_encoder': 'state_encoder.onnx',
        'action_encoder': 'action_encoder.onnx',
        'dit': 'dit_fp16.onnx',
        'action_decoder': 'action_decoder.onnx',
    },
}[model]
artifacts = manifest.get('onnx_artifacts')
if not isinstance(artifacts, dict) or set(artifacts) != set(expected):
    raise ValueError('Export manifest ONNX artifact identities are missing')
onnx_files = manifest.get('onnx')
if not isinstance(onnx_files, dict) or set(onnx_files) != set(expected):
    raise ValueError('Export manifest ONNX file map is missing')
if onnx_files != canonical_files:
    raise ValueError(
        'Export manifest ONNX filenames do not match the files consumed by '
        'this build script')
for graph_name in expected:
    identity = artifacts[graph_name]
    if identity.get('graph') != onnx_files[graph_name]:
        raise ValueError(f'ONNX graph identity mismatch: {graph_name}')
    files = identity.get('files') if isinstance(identity, dict) else None
    if not isinstance(files, list) or not files:
        raise ValueError(f'No files recorded for ONNX graph {graph_name}')
    for item in files:
        relative = pathlib.Path(item['file'])
        path = (onnx_dir / relative).resolve()
        if path.parent != onnx_dir or relative.is_absolute():
            raise ValueError(f'Unsafe ONNX artifact path: {relative}')
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != item.get('bytes'):
            raise ValueError(f'ONNX artifact size mismatch: {path}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != item.get('sha256'):
            raise ValueError(f'ONNX artifact hash mismatch: {path}')
print('Verified exported ONNX artifact identities')
PY

mkdir -p "${ENGINE_DIR}/logs"
TIMING_CACHE="${ENGINE_DIR}/timing.cache"

build_engine() {
    local name=$1
    local onnx_path=$2
    shift 2
    if [[ ! -f "${onnx_path}" ]]; then
        echo "Missing ONNX model: ${onnx_path}"
        exit 1
    fi
    echo "Building ${name}"
    "${TRTEXEC}" \
        --onnx="${onnx_path}" \
        --saveEngine="${ENGINE_DIR}/${name}.engine" \
        --stronglyTyped \
        --skipInference \
        --verbose \
        --profilingVerbosity=detailed \
        --timingCacheFile="${TIMING_CACHE}" \
        "$@" \
        >"${ENGINE_DIR}/logs/${name}.log" 2>&1
}

if [[ "${MODEL}" == "pi05" ]]; then
    build_engine vision "${ONNX_DIR}/vision.onnx"
    build_engine prefix "${ONNX_DIR}/prefix.onnx"
    build_engine denoise "${ONNX_DIR}/denoise.onnx"
elif [[ "${MODEL}" == "groot-n15" ]]; then
    # GR00T-N1.5 v1 is a fully static deployment profile.  Shape information
    # is embedded in each ONNX graph, so no TensorRT optimization profile is
    # supplied here.  This keeps the plans aligned with the runtime contract:
    # batch 1, two image views, sequence length 600, and action horizon 10.
    build_engine vit_fp16 "${ONNX_DIR}/vit_fp16.onnx"
    build_engine llm_fp16 "${ONNX_DIR}/llm_fp16.onnx"
    build_engine vlln_vl_self_attention \
        "${ONNX_DIR}/vlln_vl_self_attention.onnx"
    build_engine state_encoder "${ONNX_DIR}/state_encoder.onnx"
    build_engine action_encoder "${ONNX_DIR}/action_encoder.onnx"
    build_engine dit_fp16 "${ONNX_DIR}/dit_fp16.onnx"
    build_engine action_decoder "${ONNX_DIR}/action_decoder.onnx"
else
    echo "Unsupported model: ${MODEL}; expected pi05 or groot-n15"
    exit 1
fi

python3 - "${MANIFEST}" "${ENGINE_DIR}" "${MODEL}" \
    "${TRTEXEC_IDENTITY}" <<'PY'
import hashlib
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
engine_dir = pathlib.Path(sys.argv[2])
model = sys.argv[3]
trtexec_identity = json.loads(sys.argv[4])
manifest = json.loads(source.read_text(encoding='utf-8'))
expected = {
    'pi05': ('vision', 'prefix', 'denoise'),
    'groot-n15': ('vit_fp16', 'llm_fp16',
                  'vlln_vl_self_attention', 'state_encoder',
                  'action_encoder', 'dit_fp16', 'action_decoder'),
}[model]
engines = {}
for name in expected:
    path = engine_dir / f'{name}.engine'
    if not path.is_file():
        raise FileNotFoundError(f'Expected engine was not built: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    engines[path.stem] = {'file': path.name, 'sha256': digest.hexdigest(),
                          'bytes': path.stat().st_size}
manifest.update({'engine_model': model, 'trtexec': trtexec_identity['label'],
                 'tensorrt_version': trtexec_identity['version'],
                 'engines': engines, 'cuda_graph': False})
(engine_dir / 'engine_manifest.json').write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY

echo "Engines and logs written to ${ENGINE_DIR}"
