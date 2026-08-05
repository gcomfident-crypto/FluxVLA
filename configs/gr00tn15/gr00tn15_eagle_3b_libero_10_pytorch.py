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
"""Eager PyTorch BF16 GR00T-N1.5 LIBERO-10 quality reference."""

_base_ = ['./gr00tn15_eagle_3b_libero_10_full_finetune.py']

# The full-finetune config selects a custom inference implementation for eval.
# Replace it with the ordinary eager modules used by the training model.
inference_model = dict(
    _delete_=True,
    type='LlavaVLA',
    pretrained_name_or_path='./checkpoints/GR00T-N1.5-3B',
    vlm_backbone=dict(
        type='EagleBackbone',
        dtype='bf16',
        vlm_path='fluxvla/models/third_party_models/eagle2_hg_model'),
    vla_head=dict(
        type='FlowMatchingHead',
        state_dim=64,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_inference_timesteps=4,
        noise_beta_alpha=1.0,
        noise_beta_beta=1.5,
        action_dim=32,
        ori_action_dim=7),
    freeze_vlm_backbone=False,
    name_mapping={
        'vlm_backbone.vlm': 'backbone.eagle_model',
        'vla_head': 'action_head'
    },
    freeze_projector=False)

eval = dict(
    model_family='groot',
    eval_chunk_size=10,
    num_inference_steps=4,
    mixed_precision_dtype='bf16',
    enable_mixed_precision_training=True,
)
