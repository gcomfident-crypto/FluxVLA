# TensorRT Deployment

TensorRT inference for FluxVLA PI0.5 and GR00T-N1.5:

```text
checkpoint -> ONNX partitions -> TensorRT plans -> FluxVLA evaluation
```

- PI0.5: Vision, Prefix, and Denoise plans.
- GR00T-N1.5: ViT, LLM, backbone processor, state encoder, action encoder,
  DiT, and action decoder plans.
- The TensorRT path does not use Triton kernels or CUDA Graph.
- The original eager PyTorch path remains selectable.

The GR00T partition follows the
[Isaac-GR00T N1.5 deployment design](https://github.com/NVIDIA/Isaac-GR00T/blob/n1.5-release/deployment_scripts/README.md).
Run every command below from the FluxVLA source root.

## 1. Environment

Install FluxVLA first, then the optional deployment packages:

```bash
python -m pip install -r requirements-tensorrt.txt
python -c "import onnx, tensorrt; print(onnx.__version__, tensorrt.__version__)"
```

The reusable runtime under `fluxvla/deployment/tensorrt` is included in the
FluxVLA package. The CLI scripts, configs, and this guide are source-tree
tools, so export, build, smoke, parity, and benchmark commands require a
FluxVLA source checkout.

The pip requirements do not install `trtexec`. Install the matching full
TensorRT SDK separately. The Python bindings and `trtexec` must have the same
`major.minor.patch` version.

```bash
trtexec --version
# If it is not in PATH:
export TRTEXEC=/absolute/path/to/trtexec
```

### Supported profiles

| Model      | Precision                                                          | Static profile                                                                    |
| ---------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| PI0.5      | BF16 learned-network tensors; FP32 flow state and returned actions | Batch 1, two 224x224 views, 180 language tokens, horizon 10, 10 Euler steps       |
| GR00T-N1.5 | FP16 neural I/O, FP32 returned actions                             | Batch 1, two 224x224 views, sequence length 600, horizon 10, four denoising steps |

The reference installation is Linux x86_64, CUDA 12, and TensorRT 11.1. RTX
3090 has passed real-plan smoke and numerical-parity checks. Other RTX,
A-series, and H-series GPUs require rebuilding and revalidation. The current
pip requirements do not support aarch64 Jetson Orin or Thor; those platforms
need their platform TensorRT/JetPack stack and have not been validated here.

TensorRT plans are not treated as portable. Rebuild them after changing the
GPU architecture or TensorRT version. Supporting a different camera count,
batch, sequence length, action shape, or solver schedule requires changing the
contract, exporting ONNX again, rebuilding plans, and repeating validation.

## 2. Export ONNX

```bash
# PI0.5
python -m deployment_scripts.tensorrt.export_onnx pi05 \
  --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --checkpoint checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/step-038064-epoch-24-loss=0.0170.safetensors

# GR00T-N1.5
python -m deployment_scripts.tensorrt.export_onnx groot-n15 \
  --config configs/gr00t/gr00t_eagle_3b_libero_10_full_finetune.py \
  --checkpoint checkpoints/gr00t_eagle_3b_libero_10_full_finetune_bs64/checkpoints/step-028548-epoch-18-loss=0.0546.safetensors \
  --sequence-length 600
```

The default outputs are `deployment_artifacts/onnx/pi05` and
`deployment_artifacts/onnx/groot_n15`. Each complete export is published as a
directory and contains `export_manifest.json`, graph hashes, static shapes,
and checkpoint identity.

## 3. Build Plans

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
  bash deployment_scripts/tensorrt/build_engine.sh \
  pi05 deployment_artifacts/onnx/pi05 deployment_artifacts/engines/pi05

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
  bash deployment_scripts/tensorrt/build_engine.sh \
  groot-n15 deployment_artifacts/onnx/groot_n15 \
  deployment_artifacts/engines/groot_n15
```

The build writes verbose TensorRT logs, a timing cache, plan hashes, and
`engine_manifest.json`. Runtime loading fails if the plan bundle, checkpoint,
TensorRT version, or static contract does not match.
Use the physical target GPU when building; the commands above select GPU3 for
the formal experiment in this document.

## 4. Validate

Run one real-plan inference for each model:

```bash
python -m deployment_scripts.tensorrt.smoke_test pi05 \
  --engine-dir deployment_artifacts/engines/pi05 \
  --checkpoint checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/step-038064-epoch-24-loss=0.0170.safetensors

python -m deployment_scripts.tensorrt.smoke_test groot-n15 \
  --engine-dir deployment_artifacts/engines/groot_n15 \
  --checkpoint checkpoints/gr00t_eagle_3b_libero_10_full_finetune_bs64/checkpoints/step-028548-epoch-18-loss=0.0546.safetensors
```

Compare native PyTorch, the ONNX export wrapper, and TensorRT with identical
synthetic inputs and explicit noise:

```bash
python -m deployment_scripts.tensorrt.parity_check pi05 \
  --engine-dir deployment_artifacts/engines/pi05 \
  --checkpoint checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/step-038064-epoch-24-loss=0.0170.safetensors

python -m deployment_scripts.tensorrt.parity_check groot-n15 \
  --engine-dir deployment_artifacts/engines/groot_n15 \
  --checkpoint checkpoints/gr00t_eagle_3b_libero_10_full_finetune_bs64/checkpoints/step-028548-epoch-18-loss=0.0546.safetensors
```

## 5. LIBERO-10 Benchmark

The formal protocol uses one GPU, all 10 LIBERO-10 tasks, 50 trials per task,
five model-only warm-up calls, no rollout videos, and a deterministic changing
noise seed (`7 + episode-local model-call index`). Simulator stepping is not
part of the latency metric.

| Run                      | Config                                                   | Comparison role                      |
| ------------------------ | -------------------------------------------------------- | ------------------------------------ |
| PI0.5 PyTorch BF16       | `configs/pi05/pi05_paligemma_libero_10_pytorch.py`       | PI0.5 speed and quality baseline     |
| PI0.5 TensorRT BF16      | `configs/pi05/pi05_paligemma_libero_10_tensorrt.py`      | PI0.5 TensorRT                       |
| GR00T-N1.5 PyTorch BF16  | `configs/gr00t/gr00t_eagle_3b_libero_10_pytorch.py`      | Original-precision quality reference |
| GR00T-N1.5 PyTorch FP16  | `configs/gr00t/gr00t_eagle_3b_libero_10_pytorch_fp16.py` | Fair GR00T speed denominator         |
| GR00T-N1.5 TensorRT FP16 | `configs/gr00t/gr00t_eagle_3b_libero_10_tensorrt.py`     | GR00T TensorRT                       |

The supplied runner executes all five rows with the exact paired settings,
keeps their raw artifacts in separate directories, and invokes the strict
summarizer only after every run completes:

```bash
bash deployment_scripts/tensorrt/run_libero_benchmark.sh \
  3 benchmark_results/pi05_gr00t_tensorrt_gpu3
```

The first argument is the physical GPU index and the second is a new output
directory. The script records `CUDA_VISIBLE_DEVICES`, `nvidia-smi`, the Git
state, Python packages, all console logs, and the resolved directory of every
run. Set `PI05_CHECKPOINT` or `GROOT_CHECKPOINT` only when the checkpoints are
stored outside the documented default paths.

Every run writes `time_profile/inference_calls.csv`. Its principal fields are:

- `inference_ms`: synchronized `predict_action` time.
- `pipeline_total_ms`: preprocessing + inference + output transfer + action
  denormalization.
- module fields: CUDA-event attribution for PI0.5 V/P/D or GR00T backbone and
  action-head partitions.
- `initial_state_sha256` and `inference_seed`: pairing evidence.

Module attribution records CUDA Events and therefore adds a small, symmetric
profiling overhead to `inference_ms`. The official plan-only numbers below are
measured separately by `trtexec` without these events.

The GR00T speedup is calculated from PyTorch FP16 versus TensorRT FP16. PyTorch
BF16 is kept as a separate quality reference; comparing BF16 directly with
FP16 would mix backend and precision effects.

### Official plan benchmark

Use `trtexec` to measure the serialized plans without H2D/D2H or CUDA Graph:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
  bash deployment_scripts/tensorrt/benchmark_engines.sh \
  pi05 deployment_artifacts/engines/pi05 benchmark_results/trtexec/pi05

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
  bash deployment_scripts/tensorrt/benchmark_engines.sh \
  groot-n15 deployment_artifacts/engines/groot_n15 \
  benchmark_results/trtexec/groot_n15
```

## 6. Performance

The previous GR00T result was removed because PyTorch did not receive the same
seed control and used BF16 while TensorRT used FP16. This section will be
populated only from the corrected five-run GPU3 experiment described above.
