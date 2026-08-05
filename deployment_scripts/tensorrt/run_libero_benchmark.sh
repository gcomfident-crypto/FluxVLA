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

# Run the five single-GPU LIBERO-10 benchmark groups and validate their pairing.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "${script_dir}/../.." && pwd)
project_pythonpath="${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${project_root}"

benchmark_gpu=${1:-3}
benchmark_root=${2:-benchmark_results/tensorrt_gpu3_$(date +%Y%m%d_%H%M%S)}

pi05_checkpoint=${PI05_CHECKPOINT:-checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/step-038064-epoch-24-loss=0.0170.safetensors}
groot_checkpoint=${GROOT_CHECKPOINT:-checkpoints/gr00t_eagle_3b_libero_10_full_finetune_bs64/checkpoints/step-028548-epoch-18-loss=0.0546.safetensors}
pi05_engine_dir=deployment_artifacts/engines/pi05
groot_engine_dir=deployment_artifacts/engines/groot_n15

for checkpoint in "${pi05_checkpoint}" "${groot_checkpoint}"; do
    if [[ ! -f "${checkpoint}" ]]; then
        echo "Missing checkpoint: ${checkpoint}"
        exit 1
    fi
done

mkdir -p "${benchmark_root}/environment"
PYTHONPATH="${project_pythonpath}" python - "${project_root}" \
    >"${benchmark_root}/environment/import_identity.txt" <<'PY'
import importlib
import pathlib
import sys

project_root = pathlib.Path(sys.argv[1]).resolve()
for module_name in (
        'fluxvla',
        'fluxvla.engines.runners.libero_eval_runner',
        'fluxvla.models.vlas.llava_tensorrt',
):
    module = importlib.import_module(module_name)
    module_path = pathlib.Path(module.__file__).resolve()
    try:
        module_path.relative_to(project_root)
    except ValueError as exc:
        raise RuntimeError(
            f'{module_name} resolved outside this checkout: {module_path}') \
            from exc
    print(f'{module_name}={module_path}')
PY
for bundle in pi05 groot_n15; do
    if [[ "${bundle}" == "pi05" ]]; then
        engine_dir=${pi05_engine_dir}
    else
        engine_dir=${groot_engine_dir}
    fi
    if [[ ! -f "${engine_dir}/engine_manifest.json" ]]; then
        echo "Missing engine manifest: ${engine_dir}/engine_manifest.json"
        exit 1
    fi
    cp "${engine_dir}/engine_manifest.json" \
       "${benchmark_root}/environment/${bundle}_engine_manifest.json"
    sha256sum "${engine_dir}"/*.engine \
        >"${benchmark_root}/environment/${bundle}_engine_sha256.txt"
done
git rev-parse HEAD >"${benchmark_root}/environment/git_commit.txt"
git status --short >"${benchmark_root}/environment/git_status.txt"
python -m pip list --format=freeze \
    >"${benchmark_root}/environment/pip_freeze.txt"
nvidia-smi -i "${benchmark_gpu}" -q \
    >"${benchmark_root}/environment/nvidia_smi_gpu${benchmark_gpu}.txt"

declare -A benchmark_runs

run_group() {
    local key=$1
    local config=$2
    local checkpoint=$3
    local backend=$4
    local model=$5
    local output="${benchmark_root}/${key}"

    mkdir -p "${output}"
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="${benchmark_gpu}" \
    MUJOCO_GL=egl \
    WANDB_MODE=disabled \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${project_pythonpath}" \
    torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/eval.py \
        --config "${config}" \
        --ckpt-path "${checkpoint}" \
        --cfg-options \
        eval.seed=7 \
        eval.num_trials_per_task=50 \
        eval.task_ids='[0,1,2,3,4,5,6,7,8,9]' \
        eval.inference_seed=7 \
        eval.preprocess_every_step=False \
        eval.save_rollout_videos=False \
        eval.save_failed_rollout_videos=False \
        eval.enable_time_profile=True \
        eval.enable_module_profile=True \
        eval.time_profile_warmup_calls=5 \
        eval.time_profile_backend="${backend}" \
        eval.time_profile_model="${model}" \
        eval.result_output_dir="${output}" \
        eval.result_gpu_id="${benchmark_gpu}" \
        eval.run_id_suffix="${key}" \
        2>&1 | tee "${output}/console.log"

    local summary
    summary=$(find "${output}" -type f -name summary.json -print -quit)
    if [[ -z "${summary}" ]]; then
        echo "No summary.json produced for ${key}"
        exit 1
    fi
    benchmark_runs["${key}"]=$(dirname "${summary}")
    echo "${key}=${benchmark_runs[${key}]}" \
        >>"${benchmark_root}/run_directories.txt"
}

run_group pi05_pytorch_bf16 \
    configs/pi05/pi05_paligemma_libero_10_pytorch.py \
    "${pi05_checkpoint}" pytorch pi05
run_group pi05_tensorrt_bf16 \
    configs/pi05/pi05_paligemma_libero_10_tensorrt.py \
    "${pi05_checkpoint}" tensorrt pi05
run_group groot_pytorch_bf16 \
    configs/gr00t/gr00t_eagle_3b_libero_10_pytorch.py \
    "${groot_checkpoint}" pytorch groot_n15
run_group groot_pytorch_fp16 \
    configs/gr00t/gr00t_eagle_3b_libero_10_pytorch_fp16.py \
    "${groot_checkpoint}" pytorch groot_n15
run_group groot_tensorrt_fp16 \
    configs/gr00t/gr00t_eagle_3b_libero_10_tensorrt.py \
    "${groot_checkpoint}" tensorrt groot_n15

python -m deployment_scripts.tensorrt.summarize_benchmark \
    --pi05-pytorch-bf16 "${benchmark_runs[pi05_pytorch_bf16]}" \
    --pi05-tensorrt-bf16 "${benchmark_runs[pi05_tensorrt_bf16]}" \
    --groot-pytorch-bf16 "${benchmark_runs[groot_pytorch_bf16]}" \
    --groot-pytorch-fp16 "${benchmark_runs[groot_pytorch_fp16]}" \
    --groot-tensorrt-fp16 "${benchmark_runs[groot_tensorrt_fp16]}" \
    --expected-trials-per-task 50 \
    --expected-cuda-visible-device "${benchmark_gpu}" \
    --output-dir "${benchmark_root}/summary"

echo "Validated benchmark written to ${benchmark_root}"
