# PI0.5 and GR00T-N1.5 TensorRT Benchmark Protocol

This protocol compares the eager PyTorch and TensorRT inference paths for
PI0.5 and GR00T-N1.5 on LIBERO-10. It also retains the official TensorRT
engine-level benchmark, layer profile, Engine Inspector output, and two short
Nsight Systems traces.

The protocol has no target speedup or task-success acceptance threshold. It
records the observed results. Numerical parity and functional smoke checks are
regression guards, not claims about LIBERO task success.

## 1. Fixed experiment contract

Run every GPU command on physical GPU 3. A child process sees that one device
as `cuda:0`:

```text
physical GPU:         3
CUDA_VISIBLE_DEVICES: 3
PyTorch device:       cuda:0
trtexec device:       0 (inside the one-device namespace)
```

The formal comparison contains four foreground, serial groups:

| Order | Group | Config | Trials |
| ---: | --- | --- | ---: |
| 1 | PI0.5 eager PyTorch | `configs/pi05/pi05_paligemma_libero_10_full_finetune.py` | 10 tasks x 50 |
| 2 | PI0.5 TensorRT | `configs/pi05/pi05_paligemma_libero_10_tensorrt.py` | 10 tasks x 50 |
| 3 | GR00T-N1.5 eager PyTorch | `configs/gr00t/gr00t_eagle_3b_libero_10_pytorch.py` | 10 tasks x 50 |
| 4 | GR00T-N1.5 TensorRT | `configs/gr00t/gr00t_eagle_3b_libero_10_tensorrt.py` | 10 tasks x 50 |

This is 500 formal trials per group and 2,000 formal trials in total. The
dedicated eager GR00T config is required: the older optimized inference config
uses repository Triton and CUDA Graph paths and is not a valid baseline here.

Run a second diagnostic pass with one trial per task and group (40 additional
diagnostic trials). Only that pass enables module-level CUDA events. Keeping it
separate prevents event instrumentation from perturbing the formal headline
latency.

The following are deliberately disabled or out of scope:

- repository Triton inference kernels;
- CUDA Graph capture;
- INT8, FP8, FP4, or other quantization;
- simultaneous execution of two experiment groups;
- cross-GPU comparisons;
- a minimum speedup or success-rate gate.

Use the same checkpoint within each model's PyTorch/TensorRT pair. PI0.5 keeps
10 Euler steps and GR00T-N1.5 keeps four denoising steps. Every public model
call uses deterministic inference seed 7 on both backends, in addition to the
runner's global seed 7, so action-noise generation is paired rather than left
dependent on backend-specific model-loading RNG consumption.

## 2. Environment and paths

Run commands from the FluxVLA source root. Use a new artifact root for this
experiment; do not reuse a directory containing an older run.

```bash
cd /absolute/path/to/FluxVLA
set -euo pipefail

export TRTEXEC=/absolute/path/to/matching/TensorRT/bin/trtexec
export PI_CKPT=/absolute/path/to/pi05/checkpoint.safetensors
export GR_CKPT=/absolute/path/to/gr00t_n15/checkpoint.safetensors
export BENCH_ROOT=/absolute/path/to/new/pi05_gr00t_trt_benchmark

export ONNX_ROOT="${BENCH_ROOT}/onnx"
export ENGINE_ROOT="${BENCH_ROOT}/engines"
export VALIDATION_ROOT="${BENCH_ROOT}/validation"
export ENGINE_BENCH_ROOT="${BENCH_ROOT}/engine_benchmark"
export FORMAL_ROOT="${BENCH_ROOT}/libero_formal"
export MODULE_ROOT="${BENCH_ROOT}/libero_module"
export NSYS_ROOT="${BENCH_ROOT}/nsys"
export SUMMARY_ROOT="${BENCH_ROOT}/summary"
```

Record the software and GPU identities before starting:

```bash
mkdir -p "${BENCH_ROOT}/environment"
git rev-parse HEAD | tee "${BENCH_ROOT}/environment/git_commit.txt"
git status --short --branch | tee "${BENCH_ROOT}/environment/git_status.txt"
python --version 2>&1 | tee "${BENCH_ROOT}/environment/python.txt"
python -c 'import torch, tensorrt as trt; print("torch", torch.__version__); print("torch CUDA", torch.version.cuda); print("TensorRT", trt.__version__)' \
  | tee "${BENCH_ROOT}/environment/python_packages.txt"
"${TRTEXEC}" --version 2>&1 \
  | tee "${BENCH_ROOT}/environment/trtexec.txt"
nvidia-smi --id=3 \
  --query-gpu=index,uuid,name,driver_version,pci.bus_id \
  --format=csv \
  | tee "${BENCH_ROOT}/environment/gpu3.txt"
nsys --version 2>&1 | tee "${BENCH_ROOT}/environment/nsys.txt"
```

The TensorRT Python runtime and `trtexec` must have the same
`major.minor.patch` version. Plans are rebuilt on GPU 3; a plan copied from a
different GPU or TensorRT version is not a benchmark input.

## 3. Export ONNX and rebuild plans on GPU 3

Export the static ONNX bundles:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
python -m deployment_scripts.tensorrt.export_onnx pi05 \
  --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --checkpoint "${PI_CKPT}" \
  --output-dir "${ONNX_ROOT}" \
  --device cuda:0

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
python -m deployment_scripts.tensorrt.export_onnx groot-n15 \
  --config configs/gr00t/gr00t_eagle_3b_libero_10_full_finetune.py \
  --checkpoint "${GR_CKPT}" \
  --output-dir "${ONNX_ROOT}" \
  --device cuda:0 \
  --sequence-length 600
```

Build every engine on the same physical GPU. `build_engine.sh` uses verbose
TensorRT logging (`kINFO` and detailed profiling verbosity), writes each exact
`trtexec` command, and keeps the timing cache and build duration.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
TRTEXEC="${TRTEXEC}" \
bash deployment_scripts/tensorrt/build_engine.sh \
  pi05 "${ONNX_ROOT}/pi05" "${ENGINE_ROOT}/pi05"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
TRTEXEC="${TRTEXEC}" \
bash deployment_scripts/tensorrt/build_engine.sh \
  groot-n15 "${ONNX_ROOT}/groot_n15" "${ENGINE_ROOT}/groot_n15"
```

The build must produce:

- PI0.5: `vision`, `prefix`, and `denoise` engines;
- GR00T-N1.5: `vit_fp16`, `llm_fp16`,
  `vlln_vl_self_attention`, `state_encoder`, `action_encoder`, `dit_fp16`,
  and `action_decoder` engines;
- `engine_manifest.json`, `timing.cache`, and `logs/` for each model;
- one `*.log`, `*.command.txt`, `*.build_time_seconds`, and
  `*.built_at_utc` record for every engine.

The command does not enable CUDA Graph or quantization.

## 4. Numerical parity and public-API smoke checks

Save every validation command's stdout/stderr. The smoke command executes
FluxVLA's public `predict_action` path and checks shape, dtype, and finite
values.

```bash
mkdir -p "${VALIDATION_ROOT}"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
python -m deployment_scripts.tensorrt.parity_check pi05 \
  --engine-dir "${ENGINE_ROOT}/pi05" \
  --checkpoint "${PI_CKPT}" \
  --device cuda:0 \
  --report-only \
  --output-json "${VALIDATION_ROOT}/pi05_parity.json" \
  2>&1 | tee "${VALIDATION_ROOT}/pi05_parity.log"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
python -m deployment_scripts.tensorrt.smoke_test pi05 \
  --engine-dir "${ENGINE_ROOT}/pi05" \
  --checkpoint "${PI_CKPT}" \
  --device cuda:0 \
  2>&1 | tee "${VALIDATION_ROOT}/pi05_smoke.log"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
python -m deployment_scripts.tensorrt.parity_check groot-n15 \
  --engine-dir "${ENGINE_ROOT}/groot_n15" \
  --checkpoint "${GR_CKPT}" \
  --device cuda:0 \
  --report-only \
  --output-json "${VALIDATION_ROOT}/groot_n15_parity.json" \
  2>&1 | tee "${VALIDATION_ROOT}/groot_n15_parity.log"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
python -m deployment_scripts.tensorrt.smoke_test groot-n15 \
  --engine-dir "${ENGINE_ROOT}/groot_n15" \
  --checkpoint "${GR_CKPT}" \
  --device cuda:0 \
  2>&1 | tee "${VALIDATION_ROOT}/groot_n15_smoke.log"
```

Parity reports maximum absolute error, RMSE, and cosine similarity for the
native PyTorch, ONNX-wrapper, and TensorRT outputs. It is not a latency test
and does not replace LIBERO evaluation.

## 5. Official `trtexec` benchmark, layer profile, and Inspector

Run the engine microbenchmark separately for the two model bundles:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
python -m deployment_scripts.tensorrt.benchmark_engines pi05 \
  --engine-dir "${ENGINE_ROOT}/pi05" \
  --onnx-dir "${ONNX_ROOT}/pi05" \
  --output-dir "${ENGINE_BENCH_ROOT}/pi05" \
  --trtexec "${TRTEXEC}" \
  --device 0 \
  --seed 0 \
  --warmup-ms 1000 \
  --duration-seconds 10 \
  --iterations 100 \
  --profile-iterations 20

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
python -m deployment_scripts.tensorrt.benchmark_engines groot-n15 \
  --engine-dir "${ENGINE_ROOT}/groot_n15" \
  --onnx-dir "${ONNX_ROOT}/groot_n15" \
  --output-dir "${ENGINE_BENCH_ROOT}/groot_n15" \
  --trtexec "${TRTEXEC}" \
  --device 0 \
  --seed 0 \
  --warmup-ms 1000 \
  --duration-seconds 10 \
  --iterations 100 \
  --profile-iterations 20
```

For each engine this command runs three independent official-tool passes:

1. `perf`: unprofiled timing samples and headline engine metrics;
2. `profile`: the perturbing TensorRT per-layer profiler;
3. `inspector`: raw Engine Inspector JSON without inference.

`engine_benchmark.csv` is derived only from the unprofiled performance pass.
Do not use the per-layer profile duration as an end-to-end latency. Inputs are
deterministically generated from the static ONNX contract and remain
device-resident by default; pass `--include-data-transfers` only for a separate
explicit transfer experiment.

The PI0.5 action-level engine call weights are 1 x Vision, 1 x Prefix, and
10 x Denoise. The GR00T-N1.5 weights are 1 x each backbone/state stage and
4 x each action encoder/DiT/action decoder stage. Weighted engine sums are
component microbenchmarks, not substitutes for public `predict_action` or
pipeline timings.

Convert every unmodified `--exportProfile` JSON into a provenance-preserving
raw-layer CSV and a separate Top-10 list for each Engine:

```bash
python -m deployment_scripts.tensorrt.summarize_trtexec_profile \
  "${ENGINE_BENCH_ROOT}"/pi05/engines/*/profile/profile.json \
  "${ENGINE_BENCH_ROOT}"/groot_n15/engines/*/profile/profile.json \
  --output-dir "${ENGINE_BENCH_ROOT}/layer_profile_summary"
```

The summary groups only byte-identical layer names within one source JSON. It
retains the source path and SHA256 and does not rename, fuse, or reinterpret
TensorRT layers.

## 6. Formal LIBERO-10 run: 2,000 trials

Preview all four serial commands without creating output:

```bash
bash deployment_scripts/tensorrt/run_libero_benchmark.sh \
  --gpu 3 \
  --pi-checkpoint "${PI_CKPT}" \
  --gr-checkpoint "${GR_CKPT}" \
  --pi-engine-dir "${ENGINE_ROOT}/pi05" \
  --gr-engine-dir "${ENGINE_ROOT}/groot_n15" \
  --output-root "${FORMAL_ROOT}" \
  --mode formal \
  --dry-run
```

Then run the four groups. They execute serially in the fixed table order:

```bash
bash deployment_scripts/tensorrt/run_libero_benchmark.sh \
  --gpu 3 \
  --pi-checkpoint "${PI_CKPT}" \
  --gr-checkpoint "${GR_CKPT}" \
  --pi-engine-dir "${ENGINE_ROOT}/pi05" \
  --gr-engine-dir "${ENGINE_ROOT}/groot_n15" \
  --output-root "${FORMAL_ROOT}" \
  --mode formal
```

The runner uses five warm-up model calls per group, marks them in the raw call
CSV, and excludes them from reported distributions. It disables rollout video
and W&B network logging and records GPU telemetry per group.

Formal `model_inference_ms` is the synchronized public model-call boundary.
Formal `model_pipeline_total_ms` is the arithmetic sum of:

```text
preprocess_ms
+ model_inference_ms
+ output_transfer_ms
+ action_denorm_ms
```

Simulation stepping, rendering, video encoding, logging I/O, and other
environment work are outside that pipeline metric. Task success and episode
wall time remain in the trial CSV as separate outcome/context fields.

## 7. Separate module diagnostic run

Use a different output root and enable module timing only in this pass:

```bash
bash deployment_scripts/tensorrt/run_libero_benchmark.sh \
  --gpu 3 \
  --pi-checkpoint "${PI_CKPT}" \
  --gr-checkpoint "${GR_CKPT}" \
  --pi-engine-dir "${ENGINE_ROOT}/pi05" \
  --gr-engine-dir "${ENGINE_ROOT}/groot_n15" \
  --output-root "${MODULE_ROOT}" \
  --mode module
```

This pass records one diagnostic trial for each of the 10 tasks and four
groups. The module comparison includes:

| Model | Stages |
| --- | --- |
| PI0.5 | Vision, Prefix, Denoise aggregate, Denoise step |
| GR00T-N1.5 | ViT, LLM, backbone processor, state encoder, action expert aggregate, action encoder, DiT, action decoder |

Repeated stages record both aggregate duration and call count. The summary
uses the same semantic stage boundaries for PyTorch and TensorRT.

## 8. Short Nsight Systems traces of public `predict_action`

Create two official `.nsys-rep` files with the safe wrapper:

```bash
bash deployment_scripts/tensorrt/run_nsys_predict_action.sh \
  --pi-engine-dir "${ENGINE_ROOT}/pi05" \
  --pi-checkpoint "${PI_CKPT}" \
  --gr-engine-dir "${ENGINE_ROOT}/groot_n15" \
  --gr-checkpoint "${GR_CKPT}" \
  --output-dir "${NSYS_ROOT}"
```

The script is fixed to physical GPU 3 and passes `cuda:0` to FluxVLA after
`CUDA_VISIBLE_DEVICES=3`. It warms up once, then uses
`cudaProfilerStart/Stop` to capture exactly one checked public
`predict_action` call per TensorRT model. Model loading, engine context
creation, and warm-up are not part of the trace. The script records the exact
commands, environment, and text logs and refuses a non-empty output directory;
it never overwrites an existing report.

Open the two unmodified reports with the NVIDIA Nsight Systems GUI:

```text
pi05_predict_action.nsys-rep
groot_n15_predict_action.nsys-rep
```

Nsight Systems is a diagnostic timeline. Its instrumented durations are not
headline benchmark values.

## 9. Validate, aggregate, and render the report

After both LIBERO passes have completed, generate the traceable CSVs, Markdown
report, SVG, and PNG:

```bash
python -m deployment_scripts.tensorrt.summarize_benchmark \
  --pi05-pytorch "${FORMAL_ROOT}/01_pi05_pytorch" \
  --pi05-tensorrt "${FORMAL_ROOT}/02_pi05_tensorrt" \
  --groot-pytorch "${FORMAL_ROOT}/03_gr00t_pytorch" \
  --groot-tensorrt "${FORMAL_ROOT}/04_gr00t_tensorrt" \
  --pi05-pytorch-module-profile "${MODULE_ROOT}/01_pi05_pytorch" \
  --pi05-tensorrt-module-profile "${MODULE_ROOT}/02_pi05_tensorrt" \
  --groot-pytorch-module-profile "${MODULE_ROOT}/03_gr00t_pytorch" \
  --groot-tensorrt-module-profile "${MODULE_ROOT}/04_gr00t_tensorrt" \
  --output-dir "${SUMMARY_ROOT}" \
  --gpu "physical GPU 3" \
  --precision "PI0.5 BF16; GR00T-N1.5 FP16; no quantization" \
  --expected-trials-per-group 500 \
  --expected-tasks-per-group 10
```

The command rejects missing/duplicate trials, the wrong backend/model labels,
inconsistent pipeline arithmetic, missing required module values, and a figure
whose serialized source no longer matches the summary tables. It computes
PyTorch-versus-TensorRT speedups but imposes no speedup or success threshold.

`REPORT.md` embeds a two-panel latency figure similar in purpose to NVIDIA's
VLA deployment diagram: one PI0.5 panel and one GR00T-N1.5 panel, each showing
PyTorch and TensorRT module timing plus formal model/pipeline markers. The
figure is generated only from `figure_source.csv`, which retains paths back to
the formal and diagnostic raw inputs. No HTML is generated.

## 10. Artifact contract

A complete run has this structure (additional raw files may be present):

```text
${BENCH_ROOT}/
|-- environment/
|   |-- git_commit.txt
|   |-- git_status.txt
|   |-- gpu3.txt
|   |-- nsys.txt
|   |-- python.txt
|   |-- python_packages.txt
|   `-- trtexec.txt
|-- onnx/
|   |-- pi05/export_manifest.json
|   `-- groot_n15/export_manifest.json
|-- engines/
|   |-- pi05/{engine_manifest.json,timing.cache,logs/,*.engine}
|   `-- groot_n15/{engine_manifest.json,timing.cache,logs/,*.engine}
|-- validation/
|   |-- pi05_parity.log
|   |-- pi05_parity.json
|   |-- pi05_smoke.log
|   |-- groot_n15_parity.log
|   |-- groot_n15_parity.json
|   `-- groot_n15_smoke.log
|-- engine_benchmark/
|   |-- pi05/{engine_benchmark.csv,benchmark_manifest.json,engines/}
|   |-- groot_n15/{engine_benchmark.csv,benchmark_manifest.json,engines/}
|   `-- layer_profile_summary/{trtexec_layer_profile_raw.csv,
|                              trtexec_layer_profile_top10.csv}
|-- libero_formal/{01_pi05_pytorch,02_pi05_tensorrt,
|                  03_gr00t_pytorch,04_gr00t_tensorrt}/
|-- libero_module/{01_pi05_pytorch,02_pi05_tensorrt,
|                  03_gr00t_pytorch,04_gr00t_tensorrt}/
|-- nsys/
|   |-- commands.sh
|   |-- environment.txt
|   |-- pi05_predict_action.nsys-rep
|   |-- groot_n15_predict_action.nsys-rep
|   |-- *.log
|   `-- status.txt
`-- summary/
    |-- REPORT.md
    |-- overall_comparison.csv
    |-- module_comparison.csv
    |-- figure_source.csv
    |-- *_trial_results.csv
    |-- *_task_summary.csv
    `-- assets/
        |-- vla_pytorch_vs_tensorrt_latency.svg
        `-- vla_pytorch_vs_tensorrt_latency.png
```

Each LIBERO group directory must contain `command.txt`, `env.txt`,
`git_commit.txt`, `git_status.txt`, `stdout.log`, `telemetry.csv`,
`status.txt`, and a `time_profile/` link with `inference_calls.csv` and
`trials.csv`. A group is complete only when `status.txt` begins with
`SUCCESS`.

Retain raw CSV/JSON/log inputs with the derived report. Do not edit generated
numbers or the SVG/PNG by hand. If a long run fails, preserve the partial
directory for diagnosis and start the replacement in a new output root; the
run and Nsight wrappers intentionally avoid overwriting prior evidence.
