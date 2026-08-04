# FluxVLA TensorRT Deployment

This directory provides TensorRT inference for PI0.5 and GR00T-N1.5 in
FluxVLA.

```text
checkpoint -> ONNX -> TensorRT engine -> FluxVLA inference
```

- PI0.5: 3 engines for Vision, Prefix and Denoise (V/P/D).
- GR00T-N1.5: 7 engines for ViT, LLM and the action head.
- The original PyTorch configs remain available.

This root-level directory contains deployment commands and ONNX export
wrappers. Reusable contracts, checkpoint validation and engine execution live
under `fluxvla/deployment/tensorrt`, so the runtime is included in installed
FluxVLA packages while these scripts remain standalone deployment tools.
The GR00T partitioning reference and implementation provenance are documented
in [ATTRIBUTION.md](ATTRIBUTION.md).

All commands in this guide must be run from the root of a FluxVLA source
checkout. The `deployment_scripts/`, `configs/`, this guide, and
`requirements-tensorrt.txt` are source-tree deployment tools and are not
included in an installed FluxVLA wheel.

## 1. Installation

Install FluxVLA first, then install the optional TensorRT dependencies:

```bash
python -m pip install -r requirements-tensorrt.txt
```

If only an installed FluxVLA wheel is available, install the runtime
dependencies directly:

```bash
python -m pip install \
  --extra-index-url https://pypi.nvidia.com \
  onnx==1.18.0 \
  tensorrt-cu12==11.1.0.106 \
  tensorrt-cu12-bindings==11.1.0.106 \
  tensorrt-cu12-libs==11.1.0.106
```

Exporting ONNX, building engines, and running the supplied smoke command still
require a source checkout because their modules, configs, and shell script are
not installed by the wheel.

`trtexec` and the TensorRT Python package must use the same
`major.minor.patch` version. If `trtexec` is not in `PATH`, set
`TRTEXEC=/absolute/path/to/trtexec` when building engines.

The packages in `requirements-tensorrt.txt` install ONNX and the TensorRT
Python runtime. They do **not** install the `trtexec` command-line tool.
Install the matching full TensorRT SDK separately (for example, from an
NVIDIA tar or Debian package), then verify both components before exporting or
building:

```bash
python -c "import tensorrt as trt; print(trt.__version__)"
export TRTEXEC=/absolute/path/to/trtexec
"${TRTEXEC}" --version
```

### Current support scope

| Environment                                  | Status                                                       | Notes                                                                                                                                                                                                            |
| -------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Linux x86_64, CUDA 12, TensorRT 11.1         | Supported reference environment                              | `requirements-tensorrt.txt` targets this environment. Use an NVIDIA driver compatible with the installed CUDA/TensorRT stack.                                                                                    |
| NVIDIA RTX 3090                              | Validated                                                    | Both PI0.5 and GR00T-N1.5 real-engine smoke and numerical-parity checks have passed.                                                                                                                             |
| Other x86_64 RTX, A-series, or H-series GPUs | Not yet validated                                            | Re-export or reuse the ONNX files, but rebuild every TensorRT engine on the target system and rerun smoke, numerical-parity, and task evaluation.                                                                |
| Linux aarch64, Jetson Orin, or Thor          | Not supported by the current pip requirements; not validated | The pinned TensorRT wheels used here are not the installation path for these platforms. A platform-provided TensorRT/JetPack stack, matching build tools, code validation, and newly built engines are required. |
| Windows                                      | Not supported or validated                                   | The scripts and environment instructions currently target Linux.                                                                                                                                                 |
| CPU-only or non-NVIDIA GPU systems           | Not supported                                                | TensorRT execution requires a supported NVIDIA CUDA GPU.                                                                                                                                                         |

TensorRT plans are not treated as portable artifacts in this implementation.
Do not copy an engine built for one GPU or TensorRT version to another target.
The runtime checks the TensorRT `major.minor.patch` identity, and the build does
not enable TensorRT hardware-compatibility or version-compatible plan modes.
Build the plans again on each target environment.

The currently supported model profiles are also static:

| Model      | Precision                              | Supported input profile                                                                          |
| ---------- | -------------------------------------- | ------------------------------------------------------------------------------------------------ |
| PI0.5      | BF16 neural I/O, FP32 actions          | Batch 1, two 224x224 image views, 180 language tokens, action horizon 10, exactly 10 Euler steps |
| GR00T-N1.5 | FP16 neural I/O, FP32 returned actions | Batch 1, two 224x224 image views, sequence length 600, action horizon 10, four denoising steps   |

All seven GR00T-N1.5 ONNX graphs and TensorRT engines use these static shapes;
the build does not create dynamic TensorRT optimization profiles.

Different image counts (for example, three cameras), batch sizes, sequence
lengths, precisions, action shapes, or denoising schedules are outside the
current runtime contract. Supporting them requires updating the contract,
re-exporting ONNX, rebuilding the engines, and repeating correctness and task
validation.

## 2. Export ONNX

PI0.5:

```bash
python -m deployment_scripts.tensorrt.export_onnx pi05 \
  --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --checkpoint checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/step-038064-epoch-24-loss=0.0170.safetensors
```

GR00T-N1.5:

```bash
python -m deployment_scripts.tensorrt.export_onnx groot-n15 \
  --config configs/gr00t/gr00t_eagle_3b_libero_10_full_finetune.py \
  --checkpoint checkpoints/gr00t_eagle_3b_libero_10_full_finetune_bs64/checkpoints/step-028548-epoch-18-loss=0.0546.safetensors \
  --sequence-length 600
```

The default output directories are:

```text
deployment_artifacts/onnx/pi05
deployment_artifacts/onnx/groot_n15
```

Each directory contains the exported ONNX graphs and
`export_manifest.json`. Run the following command for optional export
arguments:

```bash
python -m deployment_scripts.tensorrt.export_onnx --help
```

## 3. Build TensorRT Engines

```bash
# PI0.5
bash deployment_scripts/tensorrt/build_engine.sh \
  pi05 deployment_artifacts/onnx/pi05 deployment_artifacts/engines/pi05

# GR00T-N1.5
bash deployment_scripts/tensorrt/build_engine.sh \
  groot-n15 deployment_artifacts/onnx/groot_n15 \
  deployment_artifacts/engines/groot_n15
```

The script generates TensorRT engines, build logs, a timing cache and
`engine_manifest.json`. The manifests record the checkpoint, model contract,
TensorRT version and artifact hashes; inference rejects mismatched files.

## 4. Run Inference

Run one functional smoke inference with each real engine bundle:

```bash
python -m deployment_scripts.tensorrt.smoke_test pi05 \
  --engine-dir deployment_artifacts/engines/pi05 \
  --checkpoint checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/step-038064-epoch-24-loss=0.0170.safetensors

python -m deployment_scripts.tensorrt.smoke_test groot-n15 \
  --engine-dir deployment_artifacts/engines/groot_n15 \
  --checkpoint checkpoints/gr00t_eagle_3b_libero_10_full_finetune_bs64/checkpoints/step-028548-epoch-18-loss=0.0546.safetensors
```

The smoke command checks output shape, dtype and finite values. It is not a
latency benchmark or a task-success evaluation.

Run deterministic numerical parity checks against both the native PyTorch
model and the ONNX export wrappers:

```bash
python -m deployment_scripts.tensorrt.parity_check pi05 \
  --engine-dir deployment_artifacts/engines/pi05 \
  --checkpoint checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/step-038064-epoch-24-loss=0.0170.safetensors

python -m deployment_scripts.tensorrt.parity_check groot-n15 \
  --engine-dir deployment_artifacts/engines/groot_n15 \
  --checkpoint checkpoints/gr00t_eagle_3b_libero_10_full_finetune_bs64/checkpoints/step-028548-epoch-18-loss=0.0546.safetensors
```

The parity command uses fixed synthetic inputs and noise, reports maximum
absolute error, RMSE and cosine similarity for all three implementation pairs,
and exits with a non-zero status if a configured tolerance is exceeded. It is
a numerical regression check rather than a LIBERO task-success evaluation.

For LIBERO evaluation, select the TensorRT config instead of the original
PyTorch config:

```text
PI0.5 PyTorch:   configs/pi05/pi05_paligemma_libero_10_full_finetune.py
PI0.5 TensorRT:  configs/pi05/pi05_paligemma_libero_10_tensorrt.py
GR00T PyTorch:   configs/gr00t/gr00t_eagle_3b_libero_10_full_finetune.py
GR00T TensorRT:  configs/gr00t/gr00t_eagle_3b_libero_10_tensorrt.py
```

Use the selected config with the normal FluxVLA `scripts/eval.py` command.
Engine paths can be overridden through `--cfg-options`.

## 5. Engine Partitions

| Model      | Engines                                                                                                           |
| ---------- | ----------------------------------------------------------------------------------------------------------------- |
| PI0.5      | `vision`, `prefix`, `denoise`                                                                                     |
| GR00T-N1.5 | `vit_fp16`, `llm_fp16`, `vlln_vl_self_attention`, `state_encoder`, `action_encoder`, `dit_fp16`, `action_decoder` |

PI0.5 executes Vision once, Prefix once and Denoise ten times per action
prediction. GR00T retains lightweight PyTorch glue operations such as token
embedding, image-token splicing, tensor concatenation and Euler updates; the
neural modules represented by the seven engines run through TensorRT.

## 6. Notes

- Inference currently uses batch size 1 and manifest-defined input shapes.
- PI0.5 keeps action horizon 10 and exactly 10 Euler steps.
- TensorRT configs fail on missing or incompatible engines; they do not
  silently fall back to the PyTorch neural path.
- TensorRT plans should be rebuilt for the target GPU and TensorRT version.
- The reproducible PI0.5/GR00T PyTorch-versus-TensorRT experiment, official
  `trtexec` profiling, Engine Inspector artifacts, and Nsight Systems commands
  are documented in [BENCHMARK.md](BENCHMARK.md).
- Quantization and cross-GPU validation are outside the current implementation
  scope.
