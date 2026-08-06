# TensorRT Deployment

> **Validation scope:** This implementation has currently been validated only on the NVIDIA GeForce RTX 5090. Other GPUs have not been validated and require rebuilding the TensorRT plans on the target device.

TensorRT inference for FluxVLA PI0.5 and GR00T-N1.5:

```text
checkpoint -> ONNX partitions -> TensorRT plans -> FluxVLA evaluation
```

- PI0.5: Vision, Prefix, and Denoise plans.
- GR00T-N1.5: ViT, LLM, backbone processor, state encoder, action encoder, DiT, and action decoder plans.
- The original eager PyTorch path remains selectable through its original evaluation config.

The GR00T partition boundaries follow the [Isaac-GR00T N1.5 deployment design](https://github.com/NVIDIA/Isaac-GR00T/blob/n1.5-release/deployment_scripts/README.md).

## 1. Environment

Install FluxVLA and the TensorRT dependencies:

```bash
python -m pip install -r requirements-tensorrt.txt
```

Install `trtexec` separately from the TensorRT SDK and ensure that it matches the TensorRT Python package version. Follow the [official TensorRT installation guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) for platform-specific installation instructions:

```bash
python -c "import tensorrt; print(tensorrt.__version__)"
trtexec --version
```

Run the following commands from the FluxVLA repository root.

### Supported profiles

| Model      | Floating-point profile                                         | Static profile                                                                    |
| ---------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| PI0.5      | BF16 main neural compute; FP32 flow state and returned actions | Batch 1, two 224x224 views, 180 language tokens, horizon 10, 10 Euler steps       |
| GR00T-N1.5 | FP16 main neural compute; FP32 returned actions                | Batch 1, two 224x224 views, sequence length 600, horizon 10, four denoising steps |

The original GR00T-N1.5 PyTorch configuration uses BF16. The TensorRT profile uses FP16 following the NVIDIA Isaac-GR00T N1.5 deployment reference, and the PyTorch FP16 configuration is retained as the precision-matched speed baseline.

The reference installation is Linux x86_64, CUDA 12, and TensorRT 11.1. The current pip requirements do not support aarch64 Jetson Orin or Thor; those platforms need their platform TensorRT/JetPack stack and have not been validated here.

TensorRT plans are not portable across GPU architectures or TensorRT versions. Rebuild them after changing either. Supporting a different camera count, batch, sequence length, action shape, or solver schedule requires changing the contract, exporting ONNX again, and rebuilding the plans.

Both profiles require exactly two present, valid camera views. The documented LIBERO transform emits `img_masks=[[True, True]]`; image validity is baked into the exported graphs rather than exposed as a TensorRT binding. Missing or variably valid views are unsupported.

## 2. Export ONNX

```bash
# PI0.5
python -m scripts.tensorrt.export_onnx pi05 \
  --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --checkpoint checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/step-038064-epoch-24-loss=0.0170.safetensors

# GR00T-N1.5
python -m scripts.tensorrt.export_onnx groot-n15 \
  --config configs/gr00t/gr00t_eagle_3b_libero_10_full_finetune.py \
  --checkpoint checkpoints/gr00t_eagle_3b_libero_10_full_finetune_bs64/checkpoints/step-028548-epoch-18-loss=0.0546.safetensors \
  --sequence-length 600
```

The default outputs are `deployment_artifacts/onnx/pi05` and `deployment_artifacts/onnx/groot_n15`. Each complete export contains an `export_manifest.json`, graph hashes, static shapes, and checkpoint identity.

## 3. Build Plans

```bash
bash scripts/tensorrt/build_engine.sh \
  pi05 deployment_artifacts/onnx/pi05 deployment_artifacts/engines/pi05

bash scripts/tensorrt/build_engine.sh \
  groot-n15 deployment_artifacts/onnx/groot_n15 \
  deployment_artifacts/engines/groot_n15
```

The build writes verbose TensorRT logs, a timing cache, plan hashes, and an `engine_manifest.json`. Runtime loading fails if the plan bundle, checkpoint, TensorRT version, or static contract does not match. Build on the physical GPU that will run the plans.

## 4. Run Evaluation

PI0.5 TensorRT:

```bash
MUJOCO_GL=egl WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/eval.py \
  --config configs/pi05/pi05_paligemma_libero_10_tensorrt.py \
  --ckpt-path checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/step-038064-epoch-24-loss=0.0170.safetensors
```

GR00T-N1.5 TensorRT:

```bash
MUJOCO_GL=egl WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/eval.py \
  --config configs/gr00t/gr00t_eagle_3b_libero_10_tensorrt.py \
  --ckpt-path checkpoints/gr00t_eagle_3b_libero_10_full_finetune_bs64/checkpoints/step-028548-epoch-18-loss=0.0546.safetensors
```

The TensorRT configs expect plans under `deployment_artifacts/engines/pi05` and `deployment_artifacts/engines/groot_n15`. To run eager PyTorch instead, select the corresponding original full-finetune config.

## 5. Performance

The following results were measured on one NVIDIA GeForce RTX 5090 with batch size 1, all 10 LIBERO-10 tasks, 50 trials per task, and five warm-up calls. Simulator stepping is excluded from policy and pipeline latency.

### Timing methodology

Complete policy inference uses synchronized wall-clock timing. The runner calls `torch.cuda.synchronize()` before and after `predict_action()` and measures the interval with `time.perf_counter_ns()`. This captures the latency observed by the caller, including Python dispatch, TensorRT enqueue, lightweight orchestration, and GPU execution. PyTorch and TensorRT use the same boundary, so this measurement is used for the primary end-to-end speedup.

Module attribution uses `torch.cuda.Event` pairs recorded on the executing CUDA stream. Events measure GPU elapsed time inside Vision, Prefix, Denoise, ViT, LLM, DiT, and other module boundaries without synchronizing after every module; they are resolved only after the complete policy call has been synchronized. This avoids adding a device-wide synchronization between modules and is therefore used for hotspot and time-distribution analysis rather than caller-observed end-to-end latency.

`pipeline_total_ms` is the sum of preprocessing, synchronized policy inference, output transfer, and action denormalization. It is not an independently timed wall-clock interval and does not include MuJoCo `env.step()` or simulator rendering. Nested module rows must not be added to their parent rows.

### End-to-end policy latency

| Model      | Backend / role                 | Success rate | Policy inference (ms) | Speedup | Pipeline (ms)   | Speedup |
| ---------- | ------------------------------ | ------------ | --------------------- | ------- | --------------- | ------- |
| PI0.5      | PyTorch BF16 baseline          | 96.8%        | 169.70 +/- 0.87       | 1.00x   | 174.56 +/- 0.95 | 1.00x   |
| PI0.5      | TensorRT BF16                  | 96.4%        | 42.53 +/- 0.07        | 3.99x   | 47.36 +/- 0.20  | 3.69x   |
| GR00T-N1.5 | PyTorch BF16 quality reference | 90.8%        | 59.67 +/- 0.63        | -       | 65.25 +/- 0.75  | -       |
| GR00T-N1.5 | PyTorch FP16 speed baseline    | 89.8%        | 59.88 +/- 0.63        | 1.00x   | 65.44 +/- 0.76  | 1.00x   |
| GR00T-N1.5 | TensorRT FP16                  | 88.2%        | 12.18 +/- 0.08        | 4.92x   | 17.78 +/- 0.24  | 3.68x   |

The GR00T BF16 row reports the original PyTorch precision and serves as a quality reference. GR00T speedup compares PyTorch FP16 with TensorRT FP16 so that backend and precision effects are not mixed. A policy call returns one horizon-10 action chunk.

### Module latency

Each value is an aggregate per policy call. PI0.5 invokes its Denoise plan 10 times; GR00T invokes Action Encoder, DiT, and Action Decoder four times each.

| PI0.5 module | PyTorch BF16 (ms) | TensorRT BF16 (ms) | Speedup |
| ------------ | ----------------- | ------------------ | ------- |
| Vision       | 16.801 +/- 0.200  | 4.368 +/- 0.033    | 3.85x   |
| Prefix       | 24.126 +/- 0.074  | 15.042 +/- 0.011   | 1.60x   |
| Denoise      | 128.489 +/- 0.708 | 22.859 +/- 0.013   | 5.62x   |

| GR00T-N1.5 module  | PyTorch FP16 (ms) | TensorRT FP16 (ms) | Speedup |
| ------------------ | ----------------- | ------------------ | ------- |
| Backbone total     | 27.242 +/- 0.398  | 5.638 +/- 0.059    | 4.83x   |
| ViT                | 11.244 +/- 0.214  | 2.439 +/- 0.035    | 4.61x   |
| LLM                | 15.641 +/- 0.200  | 3.042 +/- 0.021    | 5.14x   |
| Backbone processor | 1.690 +/- 0.005   | 0.897 +/- 0.001    | 1.88x   |
| State encoder      | 0.048 +/- 0.001   | 0.016 +/- 0.001    | 3.00x   |
| Action expert      | 30.748 +/- 0.270  | 5.460 +/- 0.008    | 5.63x   |
| Action encoder     | 1.284 +/- 0.027   | 0.144 +/- 0.003    | 8.92x   |
| DiT                | 28.123 +/- 0.253  | 5.198 +/- 0.007    | 5.41x   |
| Action decoder     | 0.722 +/- 0.018   | 0.058 +/- 0.003    | 12.53x  |

TensorRT-backed inference executes every exported neural partition through a TensorRT plan. Tensor assembly, random-number generation, and lightweight orchestration remain in PyTorch.
