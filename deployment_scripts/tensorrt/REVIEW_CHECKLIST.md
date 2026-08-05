# TensorRT Review and Reproduction Checklist

This is the release gate for the PI0.5 and GR00T-N1.5 TensorRT change. A box is
checked only when a command or retained artifact supports it.

## Verification record

- Review date: 2026-08-06.
- Branch: `fix/tensorrt-review-rebenchmark`.
- Model/evaluation source: clean commit
  `ae27b27efd5059d96bb1e0c7ba1e803ee4872d3a`.
- Formal result root:
  `benchmark_results/pi05_gr00t_trt_gpu3_corrected_20260805_235000`.
- Raw archive SHA256:
  `2a63f7e69965986d016446d0be122d957cbc9c94587bf7d43eaa04ed4257ea4b`.
- Strict outputs: `summary_verified/benchmark_summary.csv`,
  `module_summary.csv`, `task_summary.csv`, and `benchmark_data.json`.
- Build/runtime identity: `environment/` and
  `preparation/engine_build_logs/`.

## Code and contracts

- [x] Ordinary PyTorch model imports succeed when TensorRT imports are blocked.
- [x] Packaged runtime modules are under `fluxvla.deployment.tensorrt`; the
  source-only `deployment_scripts` directory is not an installed dependency.
- [x] PI0.5 is a static BF16 profile: batch 1, two valid 224x224 views, 180
  language tokens, horizon 10, and 10 Euler steps.
- [x] GR00T-N1.5 is a static FP16 profile: batch 1, two valid 224x224 views,
  sequence length 600, horizon 10, and four denoising steps.
- [x] Engine manifests bind plans to checkpoint SHA256, TensorRT version,
  shapes, dtypes, and plan hashes; real runtime loading is fail-closed.
- [x] Dedicated configs keep eager PyTorch selectable.
- [x] Exported neural partitions have no PyTorch neural fallback. The path does
  not invoke FluxVLA's Triton acceleration kernels or enable CUDA Graph.
- [x] PI0.5 Euler updates are in the denoise plan; GR00T Euler updates remain in
  lightweight PyTorch orchestration.

## Correctness and fairness

- [x] `FlowMatchingHead` forwards explicit noise, seed, and inference-step
  controls.
- [x] Seeded GR00T PyTorch FP16 and TensorRT FP16 start from the same FP32
  random values cast to FP16.
- [x] Each paired trial has the same initial-state SHA256 and deterministic
  per-episode seed schedule; diverged actions are not mislabeled as identical
  trajectories.
- [x] PI0.5 compares BF16 PyTorch with BF16 TensorRT.
- [x] GR00T speedup compares FP16 PyTorch with FP16 TensorRT; BF16 is reported
  only as the original-quality reference.
- [x] Expanded paired configs have identical evaluation and data settings.
- [x] Real-plan smoke tests pass with finite FP32 outputs for both models.
- [x] PyTorch / ONNX-wrapper / TensorRT parity passes all recorded thresholds
  for both models.

## Static and local validation

- [x] `py_compile` passes for every changed Python file.
- [x] `bash -n` passes for all TensorRT deployment shell scripts.
- [x] `git diff --check 539e04b` passes.
- [x] Flake8, isort, YAPF, codespell, Markdown formatting, and all configured
  pre-commit hooks pass over every file changed from `539e04b`.
- [x] Packaging-boundary and TensorRT-blocked ordinary-import checks pass.
- [x] Non-fatal dependency and process-group warnings are retained in logs and
  are not reported as TensorRT correctness failures.

## GPU3 LIBERO-10 experiment

- [x] `environment/nvidia_smi_gpu3.txt` and continuous `nvidia-smi pmon`
  evidence show the formal runs used physical GPU3, an RTX 5090.
- [x] Every run covers tasks 0-9 with 50 trials each (500 trials).
- [x] PI0.5 PyTorch BF16 and TensorRT BF16 runs completed.
- [x] GR00T PyTorch BF16, PyTorch FP16, and TensorRT FP16 runs completed.
- [x] Every per-call CSV records stage latency, pipeline total, seed, initial
  state, returned/executed actions, and module attribution.
- [x] Strict validation confirms complete trials, contiguous call chains,
  failure terminal steps, pipeline component sums, and paired initial states.
- [x] Official `trtexec` times, profiles, and logs were collected for all ten
  plans without data transfers or CUDA Graph.
- [x] Checkpoint SHA256, plan SHA256, action horizon, solver steps, and relevant
  static contracts match within each comparison.

## Documentation and handoff

- [x] README commands cover environment, export, build, smoke, parity, formal
  rollout evaluation, and official plan benchmarking.
- [x] The five-row result table reports trials, successes, success rate, calls,
  inference, pipeline, and fair speedups.
- [x] The 50-row task CSV and README table report all 10 tasks for all 5 runs.
- [x] Module tables disclose parent/nested overlap and per-policy call counts.
- [x] The `trtexec` table reports mean, median, P95, throughput, and plan call
  counts without presenting plan-only values as end-to-end latency.
- [x] The generated figure values match `benchmark_summary.csv`; no value was
  edited into the image manually.
- [x] Parity thresholds and all three comparisons are recorded.
- [x] Raw CSVs, logs, environment metadata, hashes, exact postprocessor, and
  summary artifacts are retained locally.
- [x] A final independent review of the complete diff finds no unresolved P0
  or P1 TensorRT issue.
