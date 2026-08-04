# Third-Party Attribution for TensorRT Deployment

## NVIDIA Isaac-GR00T reference

The GR00T-N1.5 TensorRT deployment design in this directory was informed by
the seven-engine partition and deployment structure published in NVIDIA
Isaac-GR00T:

- Project: NVIDIA Isaac-GR00T
- Upstream path: `deployment_scripts/`
- Upstream ref: tag `n1.5-release`
- Upstream commit: `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96`
- Source: <https://github.com/NVIDIA/Isaac-GR00T/tree/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96/deployment_scripts>
- License: Apache License 2.0
- Upstream copyright: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES

The FluxVLA files in this directory and under
`fluxvla/deployment/tensorrt/` are FluxVLA-specific reimplementations. They use
FluxVLA model objects, checkpoint handling, manifests, PyTorch stream
integration, and runtime contracts; they do not vendor an upstream NVIDIA
deployment source file. The attribution above records the architectural
reference and does not assign NVIDIA authorship to the FluxVLA-specific code.

If a future change copies or substantially adapts an Isaac-GR00T source file,
that file must retain the applicable upstream copyright and SPDX license header
and identify the FluxVLA modifications.

PI0.5's three-engine Vision/Prefix/Denoise partition is a FluxVLA-specific
design and is not derived from the Isaac-GR00T deployment scripts.

## TensorRT dependency

NVIDIA TensorRT is an optional external dependency and is not redistributed by
this repository. TensorRT Python packages and the separately installed
TensorRT SDK remain subject to NVIDIA's applicable license terms.
