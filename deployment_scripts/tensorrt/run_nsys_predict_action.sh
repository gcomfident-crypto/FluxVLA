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

# Capture one steady-state public predict_action call for each TensorRT model.
# Physical GPU 3 is intentionally mapped to cuda:0 in each child process.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INVOCATION_DIR="${PWD}"
PHYSICAL_GPU=3

PI_ENGINE_DIR=""
PI_CHECKPOINT=""
PI_CONFIG="${PROJECT_ROOT}/configs/pi05/pi05_paligemma_libero_10_tensorrt.py"
GR_ENGINE_DIR=""
GR_CHECKPOINT=""
GR_CONFIG="${PROJECT_ROOT}/configs/gr00t/gr00t_eagle_3b_libero_10_tensorrt.py"
OUTPUT_DIR=""
PYTHON_BIN="${PYTHON:-python}"
NSYS_BIN="${NSYS:-nsys}"
WARMUP_CALLS=1
SEED=7
OUTPUT_CREATED=0

usage() {
  cat <<'EOF'
Usage:
  bash deployment_scripts/tensorrt/run_nsys_predict_action.sh \
    --pi-engine-dir /path/to/engines/pi05 \
    --pi-checkpoint /path/to/pi05.safetensors \
    --gr-engine-dir /path/to/engines/groot_n15 \
    --gr-checkpoint /path/to/groot.safetensors \
    --output-dir /path/to/new-or-empty/nsys-output [options]

Required:
  --pi-engine-dir DIR     PI0.5 TensorRT engine bundle.
  --pi-checkpoint PATH    PI0.5 checkpoint file or checkpoint directory.
  --gr-engine-dir DIR     GR00T-N1.5 TensorRT engine bundle.
  --gr-checkpoint PATH    GR00T checkpoint file or checkpoint directory.
  --output-dir DIR        New or empty output directory. Non-empty directories
                          are rejected and existing reports are never replaced.

Optional:
  --pi-config PATH        PI0.5 TensorRT config.
  --gr-config PATH        GR00T-N1.5 TensorRT config.
  --python COMMAND        Python interpreter (default: $PYTHON or python).
  --nsys COMMAND          Nsight Systems CLI (default: $NSYS or nsys).
  --warmup-calls N        Calls before the captured call (default: 1).
  --seed N                Deterministic smoke input seed (default: 7).
  -h, --help              Show this help.

The GPU selection is fixed deliberately: physical GPU 3 is exposed as the
single logical device and smoke_test.py receives --device cuda:0. Each report
captures only one predict_action call via cudaProfilerStart/Stop; model load,
TensorRT context creation, and warm-up calls are outside the capture range.
EOF
}

die() {
  echo "[nsys] ERROR: $*" >&2
  exit 1
}

require_value() {
  local option="$1"
  local count="$2"
  [[ "${count}" -ge 2 ]] || die "${option} requires a value"
}

absolute_path() {
  local value="$1"
  if [[ "${value}" == /* ]]; then
    realpath -m -- "${value}"
  else
    realpath -m -- "${INVOCATION_DIR}/${value}"
  fi
}

resolve_command() {
  local value="$1"
  local resolved
  if [[ "${value}" == */* ]]; then
    resolved="$(absolute_path "${value}")"
    [[ -x "${resolved}" && -f "${resolved}" ]] || \
      die "command is not executable: ${resolved}"
  else
    resolved="$(command -v -- "${value}" 2>/dev/null)" || \
      die "command was not found: ${value}"
  fi
  printf '%s\n' "${resolved}"
}

shell_join() {
  local separator=""
  local argument
  for argument in "$@"; do
    printf '%s%q' "${separator}" "${argument}"
    separator=' '
  done
  printf '\n'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pi-engine-dir)
      require_value "$1" "$#"
      PI_ENGINE_DIR="$2"
      shift 2
      ;;
    --pi-checkpoint)
      require_value "$1" "$#"
      PI_CHECKPOINT="$2"
      shift 2
      ;;
    --pi-config)
      require_value "$1" "$#"
      PI_CONFIG="$2"
      shift 2
      ;;
    --gr-engine-dir)
      require_value "$1" "$#"
      GR_ENGINE_DIR="$2"
      shift 2
      ;;
    --gr-checkpoint)
      require_value "$1" "$#"
      GR_CHECKPOINT="$2"
      shift 2
      ;;
    --gr-config)
      require_value "$1" "$#"
      GR_CONFIG="$2"
      shift 2
      ;;
    --output-dir)
      require_value "$1" "$#"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --python)
      require_value "$1" "$#"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --nsys)
      require_value "$1" "$#"
      NSYS_BIN="$2"
      shift 2
      ;;
    --warmup-calls)
      require_value "$1" "$#"
      WARMUP_CALLS="$2"
      shift 2
      ;;
    --seed)
      require_value "$1" "$#"
      SEED="$2"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n "${PI_ENGINE_DIR}" ]] || die "--pi-engine-dir is required"
[[ -n "${PI_CHECKPOINT}" ]] || die "--pi-checkpoint is required"
[[ -n "${GR_ENGINE_DIR}" ]] || die "--gr-engine-dir is required"
[[ -n "${GR_CHECKPOINT}" ]] || die "--gr-checkpoint is required"
[[ -n "${OUTPUT_DIR}" ]] || die "--output-dir is required"
[[ "${WARMUP_CALLS}" =~ ^[0-9]+$ ]] || \
  die "--warmup-calls must be a non-negative integer"
[[ "${SEED}" =~ ^[0-9]+$ ]] || die "--seed must be a non-negative integer"

PI_ENGINE_DIR="$(absolute_path "${PI_ENGINE_DIR}")"
PI_CHECKPOINT="$(absolute_path "${PI_CHECKPOINT}")"
PI_CONFIG="$(absolute_path "${PI_CONFIG}")"
GR_ENGINE_DIR="$(absolute_path "${GR_ENGINE_DIR}")"
GR_CHECKPOINT="$(absolute_path "${GR_CHECKPOINT}")"
GR_CONFIG="$(absolute_path "${GR_CONFIG}")"
OUTPUT_DIR="$(absolute_path "${OUTPUT_DIR}")"
PYTHON_BIN="$(resolve_command "${PYTHON_BIN}")"
NSYS_BIN="$(resolve_command "${NSYS_BIN}")"

[[ -f "${PI_ENGINE_DIR}/engine_manifest.json" ]] || \
  die "PI0.5 engine manifest not found: ${PI_ENGINE_DIR}/engine_manifest.json"
[[ -e "${PI_CHECKPOINT}" || -L "${PI_CHECKPOINT}" ]] || \
  die "PI0.5 checkpoint not found: ${PI_CHECKPOINT}"
[[ -f "${PI_CONFIG}" ]] || die "PI0.5 config not found: ${PI_CONFIG}"
[[ -f "${GR_ENGINE_DIR}/engine_manifest.json" ]] || \
  die "GR00T engine manifest not found: ${GR_ENGINE_DIR}/engine_manifest.json"
[[ -e "${GR_CHECKPOINT}" || -L "${GR_CHECKPOINT}" ]] || \
  die "GR00T checkpoint not found: ${GR_CHECKPOINT}"
[[ -f "${GR_CONFIG}" ]] || die "GR00T config not found: ${GR_CONFIG}"

if [[ -L "${OUTPUT_DIR}" ]]; then
  die "--output-dir must not be a symbolic link: ${OUTPUT_DIR}"
fi
if [[ -e "${OUTPUT_DIR}" && ! -d "${OUTPUT_DIR}" ]]; then
  die "--output-dir is not a directory: ${OUTPUT_DIR}"
fi
if [[ -d "${OUTPUT_DIR}" ]] \
  && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  die "--output-dir is not empty: ${OUTPUT_DIR}"
fi

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi was not found"
nvidia-smi "--id=${PHYSICAL_GPU}" --query-gpu=index \
  --format=csv,noheader,nounits >/dev/null || \
  die "physical GPU ${PHYSICAL_GPU} is unavailable"
NSYS_PROFILE_HELP="$("${NSYS_BIN}" profile --help 2>&1)" || \
  die "could not read nsys profile --help"
grep -Eq -- '--capture-range(=|[[:space:]])' <<< "${NSYS_PROFILE_HELP}" || \
  die "this nsys does not advertise --capture-range"
grep -q -- '--capture-range-end' <<< "${NSYS_PROFILE_HELP}" || \
  die "this nsys does not advertise --capture-range-end"

mkdir -p -- "${OUTPUT_DIR}"
OUTPUT_CREATED=1

on_exit() {
  local status=$?
  if [[ "${status}" -ne 0 && "${OUTPUT_CREATED}" -eq 1 ]]; then
    printf 'FAILED exit_code=%s completed_at_utc=%s\n' \
      "${status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > "${OUTPUT_DIR}/status.txt"
  fi
  exit "${status}"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  printf 'cd %q\n' "${PROJECT_ROOT}"
} > "${OUTPUT_DIR}/commands.sh"
chmod +x "${OUTPUT_DIR}/commands.sh"

{
  printf 'captured_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'project_root=%s\n' "${PROJECT_ROOT}"
  printf 'physical_gpu_index=%s\n' "${PHYSICAL_GPU}"
  printf 'CUDA_DEVICE_ORDER=PCI_BUS_ID\n'
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${PHYSICAL_GPU}"
  printf 'visible_torch_device=cuda:0\n'
  printf 'python=%s\n' "${PYTHON_BIN}"
  "${PYTHON_BIN}" --version 2>&1
  "${PYTHON_BIN}" -c \
    'import torch; print(f"torch={torch.__version__}"); print(f"torch_cuda={torch.version.cuda}")'
  printf 'nsys=%s\n' "${NSYS_BIN}"
  "${NSYS_BIN}" --version 2>&1
  nvidia-smi "--id=${PHYSICAL_GPU}" \
    --query-gpu=index,uuid,name,driver_version,pci.bus_id \
    --format=csv,noheader
} > "${OUTPUT_DIR}/environment.txt"

run_profile() {
  local model="$1"
  local engine_dir="$2"
  local checkpoint="$3"
  local config="$4"
  local report_stem="$5"
  local log_path="${OUTPUT_DIR}/${report_stem}.log"
  local report_path="${OUTPUT_DIR}/${report_stem}.nsys-rep"
  local command=(
    "${NSYS_BIN}"
    profile
    --trace=cuda,nvtx,osrt
    --sample=none
    --cpuctxsw=none
    --stats=false
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --force-overwrite=false
    --output="${OUTPUT_DIR}/${report_stem}"
    env
    CUDA_DEVICE_ORDER=PCI_BUS_ID
    "CUDA_VISIBLE_DEVICES=${PHYSICAL_GPU}"
    PYTHONUNBUFFERED=1
    TOKENIZERS_PARALLELISM=false
    "${PYTHON_BIN}"
    -m
    deployment_scripts.tensorrt.smoke_test
    "${model}"
    --engine-dir
    "${engine_dir}"
    --checkpoint
    "${checkpoint}"
    --config
    "${config}"
    --device
    cuda:0
    --seed
    "${SEED}"
    --warmup-calls
    "${WARMUP_CALLS}"
    --cuda-profiler-range
  )

  printf '\n# %s\n' "${model}" >> "${OUTPUT_DIR}/commands.sh"
  shell_join "${command[@]}" >> "${OUTPUT_DIR}/commands.sh"
  echo "[nsys] Capturing ${model} on physical GPU ${PHYSICAL_GPU}"

  local pipeline_status=()
  local profile_status
  local tee_status
  set +e
  (
    cd "${PROJECT_ROOT}"
    "${command[@]}"
  ) 2>&1 | tee "${log_path}"
  pipeline_status=("${PIPESTATUS[@]}")
  set -e
  profile_status="${pipeline_status[0]}"
  tee_status="${pipeline_status[1]}"
  [[ "${profile_status}" -eq 0 ]] || \
    die "nsys ${model} capture failed with exit code ${profile_status}"
  [[ "${tee_status}" -eq 0 ]] || die "could not write ${log_path}"
  [[ -s "${report_path}" ]] || die "expected report was not created: ${report_path}"
}

run_profile pi05 "${PI_ENGINE_DIR}" "${PI_CHECKPOINT}" "${PI_CONFIG}" \
  pi05_predict_action
run_profile groot-n15 "${GR_ENGINE_DIR}" "${GR_CHECKPOINT}" "${GR_CONFIG}" \
  groot_n15_predict_action

printf 'SUCCESS completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${OUTPUT_DIR}/status.txt"
echo "[nsys] Reports written to ${OUTPUT_DIR}"
