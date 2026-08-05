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
"""Eager PyTorch FP16 GR00T-N1.5 TensorRT speed baseline."""

_base_ = ['./gr00tn15_eagle_3b_libero_10_pytorch.py']

# Construct Eagle directly in FP16. Constructing in BF16 and converting later
# would round the checkpoint twice and would not match the TensorRT export.
inference_model = dict(vlm_backbone=dict(dtype='float16'))

eval = dict(
    mixed_precision_dtype='fp16',
    enable_mixed_precision_training=True,
)
