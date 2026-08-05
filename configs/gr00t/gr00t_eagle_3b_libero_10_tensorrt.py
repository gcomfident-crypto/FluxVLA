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
"""TensorRT-backed GR00T-N1.5 LIBERO-10 evaluation configuration."""

_base_ = ['./gr00t_eagle_3b_libero_10_full_finetune.py']

_engine_dir = './deployment_artifacts/engines/groot_n15'

inference_model = dict(
    type='LlavaTensorRT',
    pretrained_name_or_path='./checkpoints/GR00T-N1.5-3B',
    vlm_backbone=dict(
        type='EagleTensorRTBackbone',
        vlm_path='fluxvla/models/third_party_models/eagle2_hg_model',
        dtype='float16',
        engine_dir=_engine_dir,
        image_views=2,
        sequence_length=600,
        state_dim=64,
        action_horizon=10,
        action_dim=32,
        future_tokens=32,
        input_embedding_dim=1536,
        hidden_size=1024,
        denoise_steps=4,
        strict_manifest=True),
    vla_head=dict(
        type='FlowMatchingTensorRTHead',
        engine_dir=_engine_dir,
        image_views=2,
        sequence_length=600,
        state_dim=64,
        hidden_size=1024,
        input_embedding_dim=1536,
        num_steps=10,
        num_inference_timesteps=4,
        action_dim=32,
        ori_action_dim=7,
        strict_manifest=True,
        diffusion_model_cfg=dict(
            attention_head_dim=48,
            cross_attention_dim=2048,
            dropout=0.2,
            final_dropout=True,
            interleave_self_attention=True,
            norm_type='ada_norm',
            num_attention_heads=32,
            num_layers=16,
            output_dim=1024,
            positional_embeddings=None)),
    freeze_vlm_backbone=True,
    name_mapping={
        'vlm_backbone.vlm': 'backbone.eagle_model',
        'vla_head': 'action_head'
    },
    freeze_projector=True)

eval = dict(
    model_family='groot',
    eval_chunk_size=10,
    num_inference_steps=4,
    mixed_precision_dtype='fp16',
    enable_mixed_precision_training=True,
)
