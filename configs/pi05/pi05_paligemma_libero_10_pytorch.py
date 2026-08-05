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
"""Eager PyTorch PI0.5 LIBERO-10 benchmark configuration."""

_base_ = ['./pi05_paligemma_libero_10_full_finetune.py']

eval = dict(
    eval_chunk_size=10,
    num_inference_steps=10,
    mixed_precision_dtype='bf16',
    enable_mixed_precision_training=True,
)
