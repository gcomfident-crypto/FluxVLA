# TensorRT Review and Reproduction Checklist

This file is the release gate for the PI0.5 and GR00T-N1.5 TensorRT changes.
Every item must cite an actual command or artifact before it is checked.

## Code and contracts

- [ ] TensorRT modules import without requiring TensorRT for ordinary PyTorch
  configs.
- [ ] PI0.5 remains a static BF16 profile: batch 1, two 224x224 views, 180
  language tokens, horizon 10, and 10 Euler steps.
- [ ] GR00T-N1.5 remains a static FP16 profile: batch 1, two 224x224 views,
  sequence length 600, horizon 10, and four denoising steps.
- [ ] Engine manifests bind plans to checkpoint, TensorRT version, shapes, and
  artifact hashes.
- [ ] PyTorch remains selectable through dedicated benchmark configs.
- [ ] No Triton kernel or CUDA Graph is used by the TensorRT path.

## Correctness and fairness

- [ ] `FlowMatchingHead` forwards `noise`, `seed`, and inference-step controls.
- [ ] Seeded GR00T PyTorch FP16 and TensorRT FP16 start from the same FP32
  random values cast to FP16.
- [ ] Each episode uses the same deterministic, changing seed sequence across
  paired backends.
- [ ] PI0.5 PyTorch and TensorRT use the same BF16 precision contract.
- [ ] GR00T speedup uses PyTorch FP16 as its denominator; PyTorch BF16 is
  reported separately as the original-quality reference.
- [ ] Real-engine smoke checks pass for both models.
- [ ] PyTorch / ONNX-wrapper / TensorRT numerical parity checks pass for both
  models.

## Static and local validation

- [ ] Python compilation passes for every changed Python file.
- [ ] `git diff --check` passes.
- [ ] Flake8, isort, YAPF, codespell, and Markdown formatting pass.
- [ ] TensorRT runtime/config unit-level checks pass.
- [ ] Known failures from optional unrelated project dependencies are recorded
  and are not presented as TensorRT failures.

## GPU3 LIBERO-10 experiment

- [ ] All formal runs use only server GPU3.
- [ ] Each run covers 10 tasks x 50 trials (500 trials).
- [ ] PI0.5 PyTorch BF16 run completed.
- [ ] PI0.5 TensorRT BF16 run completed.
- [ ] GR00T PyTorch BF16 quality-reference run completed.
- [ ] GR00T PyTorch FP16 speed-baseline run completed.
- [ ] GR00T TensorRT FP16 run completed.
- [ ] Per-call CSV records inference, preprocessing, output transfer,
  denormalization, pipeline total, seed, and module attribution.
- [ ] Official `trtexec` engine benchmarks and logs were collected.
- [ ] Initial states, checkpoint, action horizon, and solver steps match within
  every comparison.

## Documentation and handoff

- [ ] README commands reproduce export, build, parity, and formal evaluation.
- [ ] README performance tables contain only the new verified results.
- [ ] README contains a generated PyTorch-versus-TensorRT performance figure.
- [ ] Raw CSVs, logs, environment metadata, and summary artifacts are retained.
- [ ] A final independent review found no unresolved critical or high-severity
  TensorRT issue.
