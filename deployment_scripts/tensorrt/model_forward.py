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
"""ONNX-exportable model partitions for PI0.5 and GR00T-N1.5.

GR00T follows NVIDIA's seven-engine split from
``Isaac-GR00T/deployment_scripts``.  PI0.5 uses an analogous three-engine
split: vision/projector (V), prefix KV cache (P), and one Euler denoising
step (D).  Only ordinary PyTorch tensor operations are used here; this module
does not import TensorRT, Triton, or CUDA Graph utilities.
"""

from __future__ import annotations
import math

import torch
import torch.nn.functional as F
from torch import nn

from fluxvla.deployment.tensorrt.pi05.contract import PI05ExportContract


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    midpoint = value.shape[-1] // 2
    return torch.cat((-value[..., midpoint:], value[..., :midpoint]), dim=-1)


def _apply_rotary_single(value: torch.Tensor, cosine: torch.Tensor,
                         sine: torch.Tensor) -> torch.Tensor:
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    return value * cosine + _rotate_half(value) * sine


def _apply_rotary_pair(
        query: torch.Tensor, key: torch.Tensor, cosine: torch.Tensor,
        sine: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (_apply_rotary_single(query, cosine, sine),
            _apply_rotary_single(key, cosine, sine))


def _repeat_kv(value: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return value
    batch, kv_heads, sequence, head_dim = value.shape
    value = value[:, :, None, :, :].expand(batch, kv_heads, repeats, sequence,
                                           head_dim)
    return value.reshape(batch, kv_heads * repeats, sequence, head_dim)


def _rms_norm(hidden: torch.Tensor, norm: nn.Module) -> torch.Tensor:
    input_dtype = hidden.dtype
    variance = torch.mean(torch.square(hidden.float()), dim=-1, keepdim=True)
    normalized = hidden * torch.rsqrt(variance + float(norm.eps))
    normalized = normalized * (1.0 + norm.weight.float())
    return normalized.to(input_dtype)


def _gated_residual(x: torch.Tensor, y: torch.Tensor,
                    gate: torch.Tensor | None) -> torch.Tensor:
    return x + y if gate is None else x + y * gate


class PI05VisionExportWrapper(nn.Module):
    """PI0.5 SigLIP plus projector, preserving ordered per-view execution."""

    def __init__(
        self,
        model: nn.Module,
        contract: PI05ExportContract = PI05ExportContract()
    ) -> None:
        super().__init__()
        self.vision = model.vision_backbone.vision
        self.projector = model.projector
        self.contract = contract
        # Export the attention math as ordinary MatMul/Softmax/MatMul nodes.
        # PyTorch's legacy ONNX lowering for SDPA can materialize a complex
        # scalar while decomposing the explicit scale, which ONNX/TensorRT
        # cannot represent.  Transformers 5 dispatches both implementations
        # through the same SiglipAttention module, so selecting ``eager`` here
        # changes only the exportable formulation, not the weights or contract.
        self.vision.config._attn_implementation = 'eager'
        vision_config = self.vision.config
        patch_count = (
            (contract.image_height // int(vision_config.patch_size)) *
            (contract.image_width // int(vision_config.patch_size)))
        if patch_count != contract.image_tokens_per_view:
            raise ValueError(
                'PI0.5 image token contract does not match SigLIP')
        if int(vision_config.hidden_size) != 1152:
            raise ValueError(
                'PI0.5 TensorRT v1 expects SigLIP hidden size 1152')
        projection = self.projector.projector
        if (projection.in_features, projection.out_features) != (1152, 2048):
            raise ValueError('PI0.5 projector must be 1152 -> 2048')
        self.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        expected = (self.contract.image_views, 3, self.contract.image_height,
                    self.contract.image_width)
        if tuple(images.shape) != expected:
            raise ValueError(f'images shape must be {expected}')
        projected = []
        # The unaccelerated FluxVLA path invokes SigLIP once for each view.
        # Keep that ordering inside one exported graph to preserve numerics.
        for view in range(self.contract.image_views):
            hidden = self.vision(images[view:view + 1]).last_hidden_state
            projected.append(self.projector(hidden))
        return torch.cat(projected, dim=1)


def _prefix_inputs(
    llm_backbone: nn.Module,
    image_prefix: torch.Tensor,
    lang_tokens: torch.Tensor,
    lang_mask: torch.Tensor,
    image_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    language = llm_backbone.embed_tokens(lang_tokens)
    language = language * math.sqrt(language.shape[-1])
    hidden = torch.cat((image_prefix, language), dim=1)
    valid = torch.cat((image_valid, lang_mask), dim=1)
    positions = torch.cumsum(valid, dim=1, dtype=torch.int32) - 1
    valid_2d = valid[:, :, None] & valid[:, None, :]
    zero = torch.zeros((), dtype=hidden.dtype, device=hidden.device)
    blocked = torch.full((),
                         torch.finfo(hidden.dtype).min,
                         dtype=hidden.dtype,
                         device=hidden.device)
    mask = torch.where(valid_2d[:, None, :, :], zero, blocked)
    return hidden, positions, mask, valid


class PI05PrefixExportWrapper(nn.Module):
    """PI0.5 main Gemma prefix with explicit K/V tensor outputs."""

    def __init__(
        self,
        model: nn.Module,
        contract: PI05ExportContract = PI05ExportContract()
    ) -> None:
        super().__init__()
        self.llm_backbone = model.llm_backbone
        self.contract = contract
        config = self.llm_backbone.config
        actual = (len(self.llm_backbone.layers), int(config.hidden_size),
                  int(config.num_key_value_heads), int(config.head_dim))
        expected = (contract.kv_layers, contract.prefix_hidden,
                    contract.kv_heads, contract.head_dim)
        if actual != expected:
            raise ValueError(f'PI0.5 prefix contract mismatch: '
                             f'{actual} != {expected}')
        rotary = self.llm_backbone.rotary_emb
        self.register_buffer(
            'rope_inv_freq',
            rotary.inv_freq.detach().float(),
            persistent=False)
        self.rope_attention_scaling = float(rotary.attention_scaling)
        self.register_buffer(
            'image_valid',
            torch.ones(
                contract.batch,
                contract.image_tokens,
                dtype=torch.bool,
                device=self.llm_backbone.embed_tokens.weight.device),
            persistent=False)
        self.eval()

    def _rotary(self, hidden: torch.Tensor,
                positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv = self.rope_inv_freq[None, :, None].expand(positions.shape[0], -1,
                                                       1)
        frequencies = torch.matmul(inv, positions[:, None, :].float())
        embedding = torch.cat(
            (frequencies.transpose(1, 2), frequencies.transpose(1, 2)), dim=-1)
        cosine = torch.cos(embedding) * self.rope_attention_scaling
        sine = torch.sin(embedding) * self.rope_attention_scaling
        return cosine.to(hidden.dtype), sine.to(hidden.dtype)

    def _project_cache(self, layer: nn.Module, hidden: torch.Tensor,
                       cosine: torch.Tensor,
                       sine: torch.Tensor) -> tuple[torch.Tensor, ...]:
        normalized = _rms_norm(hidden, layer.input_layernorm)
        batch, sequence = normalized.shape[:2]
        shape = (batch, sequence, -1, layer.self_attn.head_dim)
        key = layer.self_attn.k_proj(normalized).view(shape).transpose(1, 2)
        value = layer.self_attn.v_proj(normalized).view(shape).transpose(1, 2)
        return normalized, _apply_rotary_single(key, cosine, sine), value

    def _layer(self, layer: nn.Module, hidden: torch.Tensor,
               cosine: torch.Tensor, sine: torch.Tensor, mask: torch.Tensor,
               query_valid: torch.Tensor) -> tuple[torch.Tensor, ...]:
        residual = hidden
        normalized, key, value = self._project_cache(layer, hidden, cosine,
                                                     sine)
        batch, sequence = normalized.shape[:2]
        shape = (batch, sequence, -1, layer.self_attn.head_dim)
        query = layer.self_attn.q_proj(normalized).view(shape).transpose(1, 2)
        query = _apply_rotary_single(query, cosine, sine)
        attended = F.scaled_dot_product_attention(
            query,
            _repeat_kv(key, layer.self_attn.num_key_value_groups),
            _repeat_kv(value, layer.self_attn.num_key_value_groups),
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            scale=None)
        zero = torch.zeros((), dtype=attended.dtype, device=attended.device)
        attended = torch.where(query_valid[:, None, :, None], attended, zero)
        attended = attended.transpose(1, 2).contiguous().reshape(
            batch, sequence, -1)
        hidden = residual + layer.self_attn.o_proj(attended)

        residual = hidden
        normalized = _rms_norm(hidden, layer.post_attention_layernorm)
        gate = layer.mlp.gate_proj(normalized)
        up = layer.mlp.up_proj(normalized)
        hidden = layer.mlp.down_proj(F.gelu(gate, approximate='tanh') * up)
        return residual + hidden, key, value

    def forward(self, image_prefix: torch.Tensor, lang_tokens: torch.Tensor,
                lang_mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden, positions, mask, valid = _prefix_inputs(
            self.llm_backbone, image_prefix, lang_tokens, lang_mask,
            self.image_valid)
        cosine, sine = self._rotary(hidden, positions)
        caches = []
        for layer in self.llm_backbone.layers[:-1]:
            hidden, key, value = self._layer(layer, hidden, cosine, sine, mask,
                                             valid)
            caches.extend((key, value))
        _, key, value = self._project_cache(self.llm_backbone.layers[-1],
                                            hidden, cosine, sine)
        caches.extend((key, value))
        return tuple(caches)


class PI05DenoiseStepExportWrapper(nn.Module):
    """One PI0.5 expert step, including the original Euler update."""

    def __init__(
        self,
        model: nn.Module,
        contract: PI05ExportContract = PI05ExportContract()
    ) -> None:
        super().__init__()
        self.contract = contract
        self.action_in_proj = model.action_in_proj
        self.time_mlp_in = model.time_mlp_in
        self.time_mlp_out = model.time_mlp_out
        self.expert = model.llm_expert
        self.action_out_proj = model.action_out_proj
        self.proj_width = int(model.proj_width)
        config = self.expert.config
        actual = (len(self.expert.layers), int(config.hidden_size),
                  int(config.num_key_value_heads), int(config.head_dim),
                  int(model.n_action_steps), int(model.max_action_dim),
                  int(model.num_steps))
        expected = (contract.kv_layers, contract.expert_hidden,
                    contract.kv_heads, contract.head_dim,
                    contract.action_horizon, contract.action_dim,
                    contract.denoise_steps)
        if actual != expected:
            raise ValueError(f'PI0.5 denoise contract mismatch: '
                             f'{actual} != {expected}')
        weight = self.action_in_proj.projector.weight
        fraction = torch.linspace(
            0.0,
            1.0,
            self.proj_width // 2,
            dtype=torch.float32,
            device=weight.device)
        period = 4e-3 * (4.0 / 4e-3)**fraction
        self.register_buffer(
            'time_scale', 2.0 * math.pi / period, persistent=False)
        self.register_buffer(
            'rope_inv_freq',
            self.expert.rotary_emb.inv_freq.detach().float(),
            persistent=False)
        self.rope_attention_scaling = float(
            self.expert.rotary_emb.attention_scaling)
        self.register_buffer(
            'image_valid',
            torch.ones(
                contract.batch,
                contract.image_tokens,
                dtype=torch.bool,
                device=weight.device),
            persistent=False)
        self.register_buffer(
            'suffix_positions',
            torch.arange(
                contract.action_horizon,
                dtype=torch.int32,
                device=weight.device)[None, :],
            persistent=False)
        self.register_buffer(
            'euler_dt',
            torch.tensor(
                -1.0 / contract.denoise_steps,
                dtype=torch.float32,
                device=weight.device),
            persistent=False)
        self.eval()

    def _suffix(self, x_t: torch.Tensor,
                timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = self.action_in_proj.projector.weight.dtype
        hidden = self.action_in_proj(x_t.to(dtype))
        phase = timestep[..., None] * self.time_scale[None, :]
        time = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
        time = F.silu(self.time_mlp_in(time.to(dtype)))
        condition = F.silu(self.time_mlp_out(time))
        return hidden, condition

    def _positions_and_mask(
            self, hidden: torch.Tensor,
            lang_mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        prefix_valid = torch.cat((self.image_valid, lang_mask), dim=1)
        positions = (
            prefix_valid.to(torch.int32).sum(dim=1, keepdim=True) +
            self.suffix_positions)
        suffix_valid = torch.ones(
            self.contract.batch,
            self.contract.action_horizon,
            dtype=torch.bool,
            device=lang_mask.device)
        key_valid = torch.cat((prefix_valid, suffix_valid), dim=1)
        key_valid = key_valid[:, None, None, :].expand(
            self.contract.batch, 1, self.contract.action_horizon,
            self.contract.prefix_tokens + self.contract.action_horizon)
        zero = torch.zeros((), dtype=hidden.dtype, device=hidden.device)
        blocked = torch.full((),
                             torch.finfo(hidden.dtype).min,
                             dtype=hidden.dtype,
                             device=hidden.device)
        return positions, torch.where(key_valid, zero, blocked)

    def _rotary(self, hidden: torch.Tensor,
                positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv = self.rope_inv_freq[None, :, None].expand(positions.shape[0], -1,
                                                       1)
        frequencies = torch.matmul(inv, positions[:, None, :].float())
        frequencies = frequencies.transpose(1, 2)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        cosine = torch.cos(embedding) * self.rope_attention_scaling
        sine = torch.sin(embedding) * self.rope_attention_scaling
        return cosine.to(hidden.dtype), sine.to(hidden.dtype)

    def _layer(self, layer: nn.Module, hidden: torch.Tensor,
               condition: torch.Tensor, cosine: torch.Tensor,
               sine: torch.Tensor, mask: torch.Tensor,
               prefix_key: torch.Tensor,
               prefix_value: torch.Tensor) -> torch.Tensor:
        residual = hidden
        normalized, gate = layer.input_layernorm(hidden, condition)
        attention = layer.self_attn
        batch, sequence = normalized.shape[:2]
        shape = (batch, sequence, -1, attention.head_dim)
        query = attention.q_proj(normalized).view(shape).transpose(1, 2)
        key = attention.k_proj(normalized).view(shape).transpose(1, 2)
        value = attention.v_proj(normalized).view(shape).transpose(1, 2)
        query, key = _apply_rotary_pair(query, key, cosine, sine)
        key = _repeat_kv(
            torch.cat((prefix_key, key), dim=2),
            attention.num_key_value_groups)
        value = _repeat_kv(
            torch.cat((prefix_value, value), dim=2),
            attention.num_key_value_groups)
        weights = torch.matmul(query, key.transpose(2, 3))
        weights = F.softmax(
            weights * attention.scaling + mask, dim=-1,
            dtype=torch.float32).to(query.dtype)
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).contiguous().reshape(
            batch, sequence, -1)
        hidden = _gated_residual(residual, attention.o_proj(attended), gate)
        residual = hidden
        normalized, gate = layer.post_attention_layernorm(hidden, condition)
        return _gated_residual(residual, layer.mlp(normalized), gate)

    def forward(self, x_t: torch.Tensor, timestep: torch.Tensor,
                lang_mask: torch.Tensor,
                *prefix_kv: torch.Tensor) -> torch.Tensor:
        if len(prefix_kv) != self.contract.kv_layers * 2:
            raise ValueError('PI0.5 denoise received wrong K/V cache count')
        hidden, condition = self._suffix(x_t, timestep)
        positions, mask = self._positions_and_mask(hidden, lang_mask)
        cosine, sine = self._rotary(hidden, positions)
        for index, layer in enumerate(self.expert.layers):
            hidden = self._layer(layer, hidden, condition, cosine, sine, mask,
                                 prefix_kv[index * 2],
                                 prefix_kv[index * 2 + 1])
        hidden, _ = self.expert.norm(hidden, condition)
        output_dtype = self.action_out_proj.projector.weight.dtype
        velocity = self.action_out_proj(hidden.float().to(output_dtype))
        increment = velocity * self.euler_dt.to(velocity.dtype)
        return x_t + increment.float()


class GR00TVisionExportWrapper(nn.Module):
    """Eagle SigLIP vision transformer with explicit position IDs.

    The explicit-position-ID export interface follows the design described by
    NVIDIA's N1.5 deployment, but this wrapper is independently implemented
    against FluxVLA's vendored Eagle model object. See ``ATTRIBUTION.md``.
    """

    def __init__(self, eagle_backbone: nn.Module) -> None:
        super().__init__()
        self.vision = eagle_backbone.vlm.vision_model.vision_model
        self.eval()

    def forward(self, pixel_values: torch.Tensor,
                position_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.vision.embeddings
        patch = embeddings.patch_embedding(
            pixel_values.to(embeddings.patch_embedding.weight.dtype))
        hidden = patch.flatten(2).transpose(1, 2)
        hidden = hidden + embeddings.position_embedding(position_ids)
        output = self.vision.encoder(
            inputs_embeds=hidden,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True)
        return self.vision.post_layernorm(output.last_hidden_state)


class GR00TLLMExportWrapper(nn.Module):
    """Selected Eagle language layers and final feature projection."""

    def __init__(self, eagle_backbone: nn.Module) -> None:
        super().__init__()
        self.language_model = eagle_backbone.vlm.language_model
        self.eagle_linear = eagle_backbone.eagle_linear
        self.select_layer = int(eagle_backbone.select_layer)
        self.eval()

    def forward(self, inputs_embeds: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True)
        return self.eagle_linear(output.hidden_states[self.select_layer])


class GR00TBackboneOutputExportWrapper(nn.Module):
    """VLLN and VL self-attention partition from the GR00T action head."""

    def __init__(self, action_head: nn.Module) -> None:
        super().__init__()
        self.vlln = action_head.vlln
        self.vl_self_attention = action_head.vl_self_attention
        self.eval()

    def forward(self, backbone_features: torch.Tensor) -> torch.Tensor:
        return self.vl_self_attention(self.vlln(backbone_features))


def finite_tensor(name: str, value: torch.Tensor) -> None:
    """Raise a useful error when an exported/runtime result is non-finite."""
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f'{name} contains NaN or Inf')
