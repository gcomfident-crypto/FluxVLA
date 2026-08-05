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

# Official trtexec model-only benchmark. CUDA Graph and data transfers are
# disabled so the result describes the serialized TensorRT plan itself.

set -euo pipefail

MODEL=${1:?model is required: pi05 or groot-n15}
ENGINE_DIR=${2:?engine directory is required}
OUTPUT_DIR=${3:?benchmark output directory is required}
TRTEXEC=${TRTEXEC:-trtexec}

if ! command -v "${TRTEXEC}" >/dev/null 2>&1; then
    echo "trtexec was not found. Set TRTEXEC=/absolute/path/to/trtexec."
    exit 1
fi

case "${MODEL}" in
    pi05)
        ENGINES=(vision prefix denoise)
        ;;
    groot-n15)
        ENGINES=(vit_fp16 llm_fp16 vlln_vl_self_attention state_encoder \
                 action_encoder dit_fp16 action_decoder)
        ;;
    *)
        echo "Unsupported model: ${MODEL}; expected pi05 or groot-n15"
        exit 1
        ;;
esac

mkdir -p "${OUTPUT_DIR}"
if [[ ! -f "${ENGINE_DIR}/engine_manifest.json" ]]; then
    echo "Missing engine manifest: ${ENGINE_DIR}/engine_manifest.json"
    exit 1
fi
cp "${ENGINE_DIR}/engine_manifest.json" \
   "${OUTPUT_DIR}/engine_manifest.json"
python3 - "${MODEL}" "${ENGINE_DIR}" "${OUTPUT_DIR}/engine_sha256.txt" <<'PY'
import hashlib
import json
import pathlib
import sys

model = sys.argv[1]
engine_dir = pathlib.Path(sys.argv[2])
output = pathlib.Path(sys.argv[3])
expected = {
    'pi05': ('vision', 'prefix', 'denoise'),
    'groot-n15': ('vit_fp16', 'llm_fp16', 'vlln_vl_self_attention',
                  'state_encoder', 'action_encoder', 'dit_fp16',
                  'action_decoder'),
}[model]
manifest = json.loads(
    (engine_dir / 'engine_manifest.json').read_text(encoding='utf-8'))
if (manifest.get('model') != model or manifest.get('engine_model') != model):
    raise ValueError('Engine manifest model does not match the benchmark')
identities = manifest.get('engines')
if not isinstance(identities, dict) or set(identities) != set(expected):
    raise ValueError('Engine manifest has an unexpected plan set')
lines = []
for name in expected:
    identity = identities[name]
    path = engine_dir / f'{name}.engine'
    if identity.get('file') != path.name or not path.is_file():
        raise ValueError(f'Engine file identity mismatch: {name}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    actual = digest.hexdigest()
    if (identity.get('bytes') != path.stat().st_size
            or identity.get('sha256') != actual):
        raise ValueError(f'Engine hash/size does not match manifest: {name}')
    lines.append(f'{actual}  {path}')
output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
PY
{
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
        --format=csv,noheader
} >"${OUTPUT_DIR}/gpu_environment.txt"
"${TRTEXEC}" --version >"${OUTPUT_DIR}/trtexec_version.txt" 2>&1

for name in "${ENGINES[@]}"; do
    engine="${ENGINE_DIR}/${name}.engine"
    if [[ ! -f "${engine}" ]]; then
        echo "Missing TensorRT engine: ${engine}"
        exit 1
    fi
    "${TRTEXEC}" \
        --loadEngine="${engine}" \
        --warmUp=1000 \
        --duration=10 \
        --iterations=100 \
        --useSpinWait \
        --noDataTransfers \
        --noCudaGraph \
        --separateProfileRun \
        --dumpProfile \
        --exportTimes="${OUTPUT_DIR}/${name}_times.json" \
        --exportProfile="${OUTPUT_DIR}/${name}_profile.json" \
        >"${OUTPUT_DIR}/${name}.log" 2>&1
done

echo "trtexec benchmark logs written to ${OUTPUT_DIR}"
