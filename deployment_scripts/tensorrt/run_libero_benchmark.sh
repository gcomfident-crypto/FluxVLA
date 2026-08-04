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

# Run four LIBERO-10 latency evaluations serially on one physical GPU:
# PI0.5 eager PyTorch, PI0.5 TensorRT, GR00T eager PyTorch, and GR00T
# TensorRT. The selected PyTorch configs do not use the repository's
# Triton/CUDA-Graph inference implementations.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INVOCATION_DIR="${PWD}"

PI_PYTORCH_CONFIG="${PROJECT_ROOT}/configs/pi05/pi05_paligemma_libero_10_full_finetune.py"
PI_TENSORRT_CONFIG="${PROJECT_ROOT}/configs/pi05/pi05_paligemma_libero_10_tensorrt.py"
GR_PYTORCH_CONFIG="${PROJECT_ROOT}/configs/gr00t/gr00t_eagle_3b_libero_10_pytorch.py"
GR_TENSORRT_CONFIG="${PROJECT_ROOT}/configs/gr00t/gr00t_eagle_3b_libero_10_tensorrt.py"

GPU=""
GPU_UUID=""
PI_CHECKPOINT=""
GR_CHECKPOINT=""
OUTPUT_ROOT=""
PI_ENGINE_DIR="${PROJECT_ROOT}/deployment_artifacts/engines/pi05"
GR_ENGINE_DIR="${PROJECT_ROOT}/deployment_artifacts/engines/groot_n15"
MODE="formal"
TRIALS_PER_TASK=50
ENABLE_MODULE_PROFILE=False
DRY_RUN=0

MONITOR_PID=""
CURRENT_GROUP=""
GIT_COMMIT_SNAPSHOT=""
GIT_STATUS_SNAPSHOT=""

TELEMETRY_QUERY="index,uuid,timestamp,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
TELEMETRY_HEADER="gpu_index,gpu_uuid,timestamp,utilization_gpu_percent,memory_used_mib,memory_total_mib,power_draw_w,temperature_gpu_c"

usage() {
  cat <<'EOF'
Usage:
  bash deployment_scripts/tensorrt/run_libero_benchmark.sh \
    --gpu 3 \
    --pi-checkpoint /path/to/pi05.safetensors \
    --gr-checkpoint /path/to/gr00t.safetensors \
    --output-root /path/to/benchmark-output \
    --mode formal [options]

Required options:
  --gpu INDEX             Physical nvidia-smi GPU index. It is mapped to
                          cuda:0 inside each single-process evaluation.
  --pi-checkpoint PATH    PI0.5 checkpoint used by both PI evaluations.
  --gr-checkpoint PATH    GR00T checkpoint used by both GR evaluations.
  --output-root DIR       New per-group output directories are created here.

Optional:
  --pi-engine-dir DIR     PI0.5 TensorRT engine directory.
  --gr-engine-dir DIR     GR00T TensorRT engine directory.
  --mode MODE             formal: 50 trials/task without module events;
                          module: 1 diagnostic trial/task with CUDA events.
                          Run the two modes into different output roots.
  --dry-run               Print the four serial commands without writing files
                          or starting evaluations/telemetry.
  -h, --help              Show this help.

Formal evaluation uses all 10 LIBERO-10 tasks, 50 trials per task, five global
profiling warm-up calls, preprocess_every_step=False, and no rollout videos.
Module mode is a separate 1-trial/task diagnostic run so CUDA-event collection
cannot perturb the formal headline latency.
TensorRT plans must already exist in their engine directories.
EOF
}

die() {
  echo "[benchmark] ERROR: $*" >&2
  exit 1
}

require_option_value() {
  local option="$1"
  local count="$2"
  if [[ "${count}" -lt 2 ]]; then
    die "${option} requires a value"
  fi
}

absolute_path() {
  local value="$1"
  if [[ "${value}" == /* ]]; then
    realpath -m -- "${value}"
  else
    realpath -m -- "${INVOCATION_DIR}/${value}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      require_option_value "$1" "$#"
      GPU="$2"
      shift 2
      ;;
    --pi-checkpoint | --pi-ckpt)
      require_option_value "$1" "$#"
      PI_CHECKPOINT="$2"
      shift 2
      ;;
    --gr-checkpoint | --gr-ckpt)
      require_option_value "$1" "$#"
      GR_CHECKPOINT="$2"
      shift 2
      ;;
    --output-root)
      require_option_value "$1" "$#"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --pi-engine-dir)
      require_option_value "$1" "$#"
      PI_ENGINE_DIR="$2"
      shift 2
      ;;
    --gr-engine-dir)
      require_option_value "$1" "$#"
      GR_ENGINE_DIR="$2"
      shift 2
      ;;
    --mode)
      require_option_value "$1" "$#"
      MODE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

[[ -n "${GPU}" ]] || die "--gpu is required"
[[ "${GPU}" =~ ^[0-9]+$ ]] || die "--gpu must be a non-negative integer"
[[ -n "${PI_CHECKPOINT}" ]] || die "--pi-checkpoint is required"
[[ -n "${GR_CHECKPOINT}" ]] || die "--gr-checkpoint is required"
[[ -n "${OUTPUT_ROOT}" ]] || die "--output-root is required"
case "${MODE}" in
  formal)
    TRIALS_PER_TASK=50
    ENABLE_MODULE_PROFILE=False
    ;;
  module | module-profile)
    MODE="module"
    TRIALS_PER_TASK=1
    ENABLE_MODULE_PROFILE=True
    ;;
  *)
    die "--mode must be formal or module"
    ;;
esac

PI_CHECKPOINT="$(absolute_path "${PI_CHECKPOINT}")"
GR_CHECKPOINT="$(absolute_path "${GR_CHECKPOINT}")"
OUTPUT_ROOT="$(absolute_path "${OUTPUT_ROOT}")"
PI_ENGINE_DIR="$(absolute_path "${PI_ENGINE_DIR}")"
GR_ENGINE_DIR="$(absolute_path "${GR_ENGINE_DIR}")"

cleanup_monitor() {
  local pid="${MONITOR_PID:-}"
  if [[ -z "${pid}" ]]; then
    return
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
  MONITOR_PID=""
}

on_exit() {
  local status=$?
  cleanup_monitor
  if [[ "${status}" -ne 0 && -n "${CURRENT_GROUP}" ]]; then
    echo "[benchmark] Aborted in ${CURRENT_GROUP}; later groups were not run." >&2
  fi
  exit "${status}"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

shell_join() {
  local arg
  local separator=""
  for arg in "$@"; do
    printf '%s%q' "${separator}" "${arg}"
    separator=' '
  done
  printf '\n'
}

EVAL_COMMAND=()
EVAL_ENV=()
TELEMETRY_COMMAND=()

build_commands() {
  local group_name="$1"
  local model="$2"
  local backend="$3"
  local config="$4"
  local checkpoint="$5"
  local inference_steps="$6"
  local run_dir="$7"
  local results_dir="${run_dir}/results"
  local cfg_options=(
    "eval.task_suite_name=libero_10"
    "eval.task_ids=[0,1,2,3,4,5,6,7,8,9]"
    "eval.num_trials_per_task=${TRIALS_PER_TASK}"
    "eval.eval_shard_strategy=episode"
    "eval.eval_chunk_size=10"
    "eval.num_inference_steps=${inference_steps}"
    "eval.inference_seed=7"
    "eval.preprocess_every_step=False"
    "eval.enable_time_profile=True"
    "eval.enable_module_profile=${ENABLE_MODULE_PROFILE}"
    "eval.time_profile_warmup_calls=5"
    "eval.time_profile_backend=${backend}"
    "eval.time_profile_model=${model}"
    "eval.save_rollout_videos=False"
    "eval.save_failed_rollout_videos=False"
    "eval.save_multi_view_rollout_videos=False"
    "eval.run_id_suffix=${group_name}"
    "eval.result_output_dir=${results_dir}"
    "eval.result_gpu_id=${GPU}"
  )

  if [[ "${model}" == "pi05" && "${backend}" == "tensorrt" ]]; then
    cfg_options+=("inference_model.engine_dir=${PI_ENGINE_DIR}")
  elif [[ "${model}" == "gr00t" && "${backend}" == "tensorrt" ]]; then
    cfg_options+=(
      "inference_model.vlm_backbone.engine_dir=${GR_ENGINE_DIR}"
      "inference_model.vla_head.engine_dir=${GR_ENGINE_DIR}"
    )
  fi

  EVAL_ENV=(
    env
    "CUDA_DEVICE_ORDER=PCI_BUS_ID"
    "CUDA_VISIBLE_DEVICES=${GPU_UUID:-${GPU}}"
    "WANDB_MODE=disabled"
    "WANDB_DISABLED=true"
    "WANDB_SILENT=true"
    "MUJOCO_GL=egl"
    "TOKENIZERS_PARALLELISM=false"
    "OMP_NUM_THREADS=1"
    "PYTHONUNBUFFERED=1"
  )
  EVAL_COMMAND=(
    torchrun
    --standalone
    --nnodes=1
    --nproc-per-node=1
    "${PROJECT_ROOT}/scripts/eval.py"
    --config "${config}"
    --ckpt-path "${checkpoint}"
    --cfg-options "${cfg_options[@]}"
  )
  TELEMETRY_COMMAND=(
    nvidia-smi
    "--id=${GPU}"
    "--query-gpu=${TELEMETRY_QUERY}"
    --format=csv,noheader,nounits
    --loop-ms=200
  )
}

write_metadata() {
  local group_name="$1"
  local model="$2"
  local backend="$3"
  local config="$4"
  local checkpoint="$5"
  local run_dir="$6"
  local engine_dir="not_applicable"

  if [[ "${backend}" == "tensorrt" && "${model}" == "pi05" ]]; then
    engine_dir="${PI_ENGINE_DIR}"
  elif [[ "${backend}" == "tensorrt" ]]; then
    engine_dir="${GR_ENGINE_DIR}"
  fi

  {
    echo "# GPU telemetry"
    shell_join "${TELEMETRY_COMMAND[@]}"
    echo
    echo "# Evaluation (run from ${PROJECT_ROOT})"
    printf 'cd %q && ' "${PROJECT_ROOT}"
    shell_join "${EVAL_ENV[@]}" "${EVAL_COMMAND[@]}"
  } > "${run_dir}/command.txt"

  {
    printf 'captured_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'benchmark_group=%s\n' "${group_name}"
    printf 'benchmark_mode=%s\n' "${MODE}"
    printf 'trials_per_task=%s\n' "${TRIALS_PER_TASK}"
    printf 'module_profile=%s\n' "${ENABLE_MODULE_PROFILE}"
    printf 'model=%s\n' "${model}"
    printf 'backend=%s\n' "${backend}"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'kernel=%s\n' "$(uname -srmo)"
    printf 'project_root=%s\n' "${PROJECT_ROOT}"
    printf 'config=%s\n' "${config}"
    printf 'checkpoint=%s\n' "${checkpoint}"
    printf 'engine_dir=%s\n' "${engine_dir}"
    printf 'physical_gpu_index=%s\n' "${GPU}"
    printf 'physical_gpu_uuid=%s\n' "${GPU_UUID}"
    printf 'CUDA_DEVICE_ORDER=PCI_BUS_ID\n'
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "${GPU_UUID}"
    printf 'visible_torch_device=cuda:0\n'
    printf 'WANDB_MODE=disabled\n'
    printf 'WANDB_DISABLED=true\n'
    printf 'OMP_NUM_THREADS=1\n'
    printf 'timezone=%s\n' "$(date +%Z%z)"
    printf 'python_executable=%s\n' "$(command -v python)"
    python --version 2>&1
    python -c 'import torch; print(f"torch={torch.__version__}"); print(f"torch_cuda={torch.version.cuda}")'
    printf 'torchrun_executable=%s\n' "$(command -v torchrun)"
    nvidia-smi "--id=${GPU}" \
      --query-gpu=index,uuid,name,driver_version,pci.bus_id \
      --format=csv,noheader
  } > "${run_dir}/env.txt"

  printf '%s\n' "${GIT_COMMIT_SNAPSHOT}" > "${run_dir}/git_commit.txt"
  printf '%s\n' "${GIT_STATUS_SNAPSHOT}" > "${run_dir}/git_status.txt"
}

start_telemetry() {
  local run_dir="$1"
  printf '%s\n' "${TELEMETRY_HEADER}" > "${run_dir}/telemetry.csv"
  "${TELEMETRY_COMMAND[@]}" >> "${run_dir}/telemetry.csv" \
    2> "${run_dir}/telemetry.stderr.log" &
  MONITOR_PID=$!

  # Allow one 200 ms sample to be emitted and fail before model startup if the
  # monitor cannot query the selected physical GPU.
  sleep 0.25
  if ! kill -0 "${MONITOR_PID}" 2>/dev/null; then
    local monitor_status
    set +e
    wait "${MONITOR_PID}"
    monitor_status=$?
    set -e
    MONITOR_PID=""
    echo "[benchmark] nvidia-smi telemetry exited early (${monitor_status})." >&2
    return 1
  fi
}

assert_gpu_idle() {
  local sample
  local utilization
  local memory_used
  for sample in 1 2 3 4 5; do
    IFS=',' read -r utilization memory_used < <(
      nvidia-smi "--id=${GPU_UUID}" \
        --query-gpu=utilization.gpu,memory.used \
        --format=csv,noheader,nounits | tr -d ' ')
    [[ "${utilization}" =~ ^[0-9]+$ ]] || \
      die "could not parse GPU ${GPU} utilization: ${utilization}"
    [[ "${memory_used}" =~ ^[0-9]+$ ]] || \
      die "could not parse GPU ${GPU} memory use: ${memory_used}"
    if (( utilization > 0 )); then
      die "physical GPU ${GPU} (${GPU_UUID}) is busy: ${utilization}% utilization"
    fi
    # Drivers or an idle multi-GPU process may retain a small context. More
    # than 1 GiB is treated as an active allocation and fails closed.
    if (( memory_used > 1024 )); then
      die "physical GPU ${GPU} (${GPU_UUID}) is busy: ${memory_used} MiB used"
    fi
    if (( sample < 5 )); then
      sleep 1
    fi
  done
}

run_preflight_smoke() {
  local smoke_env=(
    env
    "CUDA_DEVICE_ORDER=PCI_BUS_ID"
    "CUDA_VISIBLE_DEVICES=${GPU_UUID}"
    "MUJOCO_GL=egl"
    "TOKENIZERS_PARALLELISM=false"
    "PYTHONUNBUFFERED=1"
  )
  local pi_command=(
    python -m deployment_scripts.tensorrt.smoke_test pi05
    --engine-dir "${PI_ENGINE_DIR}"
    --checkpoint "${PI_CHECKPOINT}"
    --device cuda:0
  )
  local gr_command=(
    python -m deployment_scripts.tensorrt.smoke_test groot-n15
    --engine-dir "${GR_ENGINE_DIR}"
    --checkpoint "${GR_CHECKPOINT}"
    --device cuda:0
  )
  {
    printf 'cd %q && ' "${PROJECT_ROOT}"
    shell_join "${smoke_env[@]}" "${pi_command[@]}"
    printf 'cd %q && ' "${PROJECT_ROOT}"
    shell_join "${smoke_env[@]}" "${gr_command[@]}"
  } > "${OUTPUT_ROOT}/preflight_smoke_commands.txt"
  (
    cd "${PROJECT_ROOT}"
    "${smoke_env[@]}" "${pi_command[@]}"
  ) 2>&1 | tee "${OUTPUT_ROOT}/preflight_pi05_smoke.log"
  (
    cd "${PROJECT_ROOT}"
    "${smoke_env[@]}" "${gr_command[@]}"
  ) 2>&1 | tee "${OUTPUT_ROOT}/preflight_groot_smoke.log"
}

run_group() {
  local group_name="$1"
  local model="$2"
  local backend="$3"
  local config="$4"
  local checkpoint="$5"
  local inference_steps="$6"
  local run_dir="${OUTPUT_ROOT}/${group_name}"
  local results_dir="${run_dir}/results"

  CURRENT_GROUP="${group_name}"
  build_commands "${group_name}" "${model}" "${backend}" "${config}" \
    "${checkpoint}" "${inference_steps}" "${run_dir}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] ${group_name}"
    printf '  output: %s\n' "${run_dir}"
    printf '  telemetry: '
    shell_join "${TELEMETRY_COMMAND[@]}"
    printf '  evaluation: cd %q && ' "${PROJECT_ROOT}"
    shell_join "${EVAL_ENV[@]}" "${EVAL_COMMAND[@]}"
    echo
    CURRENT_GROUP=""
    return
  fi

  [[ ! -e "${run_dir}" && ! -L "${run_dir}" ]] || \
    die "output already exists: ${run_dir}"
  mkdir -p "${run_dir}/results"
  write_metadata "${group_name}" "${model}" "${backend}" "${config}" \
    "${checkpoint}" "${run_dir}"

  echo "[benchmark] Starting ${group_name} on physical GPU ${GPU}"
  assert_gpu_idle
  start_telemetry "${run_dir}"

  local pipeline_status=()
  local eval_status
  local tee_status
  set +e
  (
    cd "${PROJECT_ROOT}"
    "${EVAL_ENV[@]}" "${EVAL_COMMAND[@]}"
  ) 2>&1 | tee "${run_dir}/stdout.log"
  pipeline_status=("${PIPESTATUS[@]}")
  set -e
  eval_status="${pipeline_status[0]}"
  tee_status="${pipeline_status[1]}"

  local monitor_failed=0
  if ! kill -0 "${MONITOR_PID}" 2>/dev/null; then
    monitor_failed=1
    wait "${MONITOR_PID}" 2>/dev/null || true
    MONITOR_PID=""
  else
    cleanup_monitor
  fi

  if [[ "${eval_status}" -ne 0 ]]; then
    printf 'FAILED eval_exit_code=%s\n' "${eval_status}" > "${run_dir}/status.txt"
    echo "[benchmark] ${group_name} failed with exit code ${eval_status}." >&2
    return "${eval_status}"
  fi
  if [[ "${tee_status}" -ne 0 ]]; then
    printf 'FAILED tee_exit_code=%s\n' "${tee_status}" > "${run_dir}/status.txt"
    echo "[benchmark] Failed to write ${run_dir}/stdout.log." >&2
    return "${tee_status}"
  fi
  if [[ "${monitor_failed}" -ne 0 ]]; then
    printf 'FAILED telemetry_exited_early\n' > "${run_dir}/status.txt"
    echo "[benchmark] Telemetry exited before ${group_name} completed." >&2
    return 1
  fi

  local profile_candidates=()
  local profile_csv
  while IFS= read -r profile_csv; do
    profile_candidates+=("${profile_csv}")
  done < <(find "${results_dir}" -type f \
    -path '*/time_profile/inference_calls.csv' -print | sort)
  if [[ "${#profile_candidates[@]}" -ne 1 ]]; then
    printf 'FAILED profile_run_count=%s\n' \
      "${#profile_candidates[@]}" > "${run_dir}/status.txt"
    die "${group_name} produced ${#profile_candidates[@]} profile runs; expected 1"
  fi
  local eval_run_dir
  eval_run_dir="${profile_candidates[0]%/time_profile/inference_calls.csv}"
  printf '%s\n' "${eval_run_dir}" > "${run_dir}/eval_run_dir.txt"
  ln -s "$(realpath --relative-to="${run_dir}" \
    "${eval_run_dir}/time_profile")" "${run_dir}/time_profile"

  printf 'SUCCESS completed_at_utc=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${run_dir}/status.txt"
  echo "[benchmark] Completed ${group_name}"
  CURRENT_GROUP=""
}

preflight() {
  local required_file
  local group_name
  for required_file in \
    "${PI_PYTORCH_CONFIG}" \
    "${PI_TENSORRT_CONFIG}" \
    "${GR_PYTORCH_CONFIG}" \
    "${GR_TENSORRT_CONFIG}" \
    "${PI_CHECKPOINT}" \
    "${GR_CHECKPOINT}" \
    "${PI_ENGINE_DIR}/engine_manifest.json" \
    "${GR_ENGINE_DIR}/engine_manifest.json"; do
    [[ -f "${required_file}" ]] || die "required file not found: ${required_file}"
  done

  command -v python >/dev/null 2>&1 || die "python was not found"
  command -v torchrun >/dev/null 2>&1 || die "torchrun was not found"
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi was not found"
  command -v git >/dev/null 2>&1 || die "git was not found"
  nvidia-smi "--id=${GPU}" --query-gpu=index --format=csv,noheader,nounits \
    >/dev/null || die "nvidia-smi GPU index ${GPU} is unavailable"
  GPU_UUID="$(nvidia-smi "--id=${GPU}" --query-gpu=uuid \
    --format=csv,noheader | tr -d '[:space:]')"
  [[ "${GPU_UUID}" == GPU-* ]] || \
    die "could not resolve physical GPU ${GPU} UUID"

  # Check every destination before the first long-running evaluation starts.
  # This prevents, for example, group 01 from completing only for group 02 to
  # discover an old output directory and abort the benchmark midway.
  for group_name in \
    01_pi05_pytorch \
    02_pi05_tensorrt \
    03_gr00t_pytorch \
    04_gr00t_tensorrt; do
    [[ ! -e "${OUTPUT_ROOT}/${group_name}" \
      && ! -L "${OUTPUT_ROOT}/${group_name}" ]] || \
      die "output already exists: ${OUTPUT_ROOT}/${group_name}"
  done

  GIT_COMMIT_SNAPSHOT="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
  GIT_STATUS_SNAPSHOT="$(git -C "${PROJECT_ROOT}" status --short --branch)"
  mkdir -p "${OUTPUT_ROOT}"
  nvidia-smi "--id=${GPU_UUID}" \
    --query-gpu=index,uuid,name,pci.bus_id,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader > "${OUTPUT_ROOT}/gpu_preflight.csv"
  nvidia-smi "--id=${GPU_UUID}" \
    --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits \
    > "${OUTPUT_ROOT}/gpu_preflight_compute_processes.csv" || true
  assert_gpu_idle
  run_preflight_smoke
}

if [[ "${DRY_RUN}" -eq 0 ]]; then
  preflight
fi

# These calls are intentionally foreground and serial. A failure returns from
# run_group under `set -e`, invokes the telemetry cleanup trap, and prevents all
# later groups from starting.
run_group "01_pi05_pytorch" "pi05" "pytorch" "${PI_PYTORCH_CONFIG}" \
  "${PI_CHECKPOINT}" 10
run_group "02_pi05_tensorrt" "pi05" "tensorrt" "${PI_TENSORRT_CONFIG}" \
  "${PI_CHECKPOINT}" 10
run_group "03_gr00t_pytorch" "gr00t" "pytorch" "${GR_PYTORCH_CONFIG}" \
  "${GR_CHECKPOINT}" 4
run_group "04_gr00t_tensorrt" "gr00t" "tensorrt" "${GR_TENSORRT_CONFIG}" \
  "${GR_CHECKPOINT}" 4

CURRENT_GROUP=""
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[dry-run] Four groups would run serially; no files were written."
else
  echo "[benchmark] All four serial groups completed: ${OUTPUT_ROOT}"
fi
