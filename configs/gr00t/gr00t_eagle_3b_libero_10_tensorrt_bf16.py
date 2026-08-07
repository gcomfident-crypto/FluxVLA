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
"""TensorRT BF16 GR00T-N1.5 LIBERO-10 evaluation configuration."""

_base_ = ['./gr00t_eagle_3b_libero_10_tensorrt.py']

_engine_dir = './deployment_artifacts/engines/groot_n15_bf16'

inference_model = dict(
    vlm_backbone=dict(dtype='bf16', engine_dir=_engine_dir),
    vla_head=dict(dtype='bf16', engine_dir=_engine_dir),
)

eval = dict(
    mixed_precision_dtype='bf16',
    enable_mixed_precision_training=True,
)
