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
"""Shared utilities for simulation evaluation runners."""

import copy
import os
import weakref
from pathlib import Path
from typing import Any, Dict

import torch

from fluxvla.engines.utils import initialize_overwatch
from fluxvla.engines.utils.name_map import str_to_dtype

overwatch = initialize_overwatch(__name__)


class BaseEvalRunner:
    """Common helper mixin for environment evaluation runners.

    This base class intentionally keeps the task-specific evaluation loops in
    their concrete runners. It only centralizes small, side-effect-free helpers
    that are shared by LIBERO and RoboCasa evaluation code.
    """

    @staticmethod
    def build_eval_vla(cfg: Dict):
        """Build the VLA model used by an evaluation config."""
        from fluxvla.engines import build_vla_from_cfg

        if hasattr(cfg, 'inference_model'):
            return build_vla_from_cfg(cfg.inference_model).eval()
        return build_vla_from_cfg(cfg.model).eval()

    @staticmethod
    def _set_model_cfg_value(model_cfg, key: str, value) -> None:
        if isinstance(model_cfg, dict):
            model_cfg[key] = value
        else:
            setattr(model_cfg, key, value)

    @staticmethod
    def _resolve_model_build_dtype(dtype):
        if dtype is None:
            return None
        if isinstance(dtype, torch.dtype):
            return dtype
        return str_to_dtype(str(dtype))

    @classmethod
    def prepare_eval_model_cfg(cls,
                               cfg,
                               model_build_device: str = None,
                               model_build_dtype=None):
        """Copy the eval model config and apply construction overrides."""
        if hasattr(cfg, 'inference_model'):
            model_cfg = copy.deepcopy(cfg.inference_model)
        else:
            model_cfg = copy.deepcopy(cfg.model)

        if model_build_device is not None:
            cls._set_model_cfg_value(model_cfg, 'device',
                                     str(model_build_device))
        resolved_dtype = cls._resolve_model_build_dtype(model_build_dtype)
        if resolved_dtype is not None:
            cls._set_model_cfg_value(model_cfg, 'torch_dtype', resolved_dtype)
        return model_cfg

    @staticmethod
    def default_stats_path(ckpt_path: str) -> str:
        """Return the checkpoint-relative dataset statistics path."""
        return os.path.join(
            Path(ckpt_path).resolve().parent.parent, 'dataset_statistics.json')

    def load_eval_state_dict(
        self, state_dict: Dict,
        allowed_missing_key_prefixes=()) -> None:  # noqa: E125
        """Load eval weights with optional missing-key prefixes."""
        prepare = getattr(self.vla, 'prepare_eval_state_dict', None)
        if callable(prepare):
            state_dict = prepare(state_dict)
        missing, unexpected = self.vla.load_state_dict(
            state_dict, strict=False)
        unexpected = list(unexpected)
        assert not unexpected, (
            'Unexpected keys while loading eval checkpoint: '
            f'{unexpected[:10]}')
        if isinstance(allowed_missing_key_prefixes, str):
            allowed_missing_key_prefixes = (allowed_missing_key_prefixes, )
        else:
            allowed_missing_key_prefixes = tuple(allowed_missing_key_prefixes)
        offending = [
            k for k in missing if not any(
                k.startswith(prefix)
                for prefix in allowed_missing_key_prefixes)
        ]
        assert not offending, ('Missing keys while loading eval checkpoint: '
                               f'{offending[:10]}')

    def call_model_hook(self,
                        hook_name: str,
                        *args,
                        reverse: bool = False,
                        continue_on_error: bool = False):
        """Call an optional deployment lifecycle hook on model modules.

        TensorRT adapters use these hooks to bind an engine bundle to the
        evaluation checkpoint, deserialize plans after CUDA placement, and
        release execution contexts during runner cleanup.  Ordinary PyTorch
        modules do not implement the hooks and are unaffected.
        """
        # Weak references are important for TensorRT preparation: adapter
        # hooks delete the PyTorch modules covered by an engine before plan
        # deserialization.  A normal list would keep those large modules alive
        # until every hook finished and create an avoidable peak-memory spike.
        modules = [weakref.ref(module) for module in self.vla.modules()]
        if reverse:
            modules.reverse()
        called = []
        errors = []
        for module_ref in modules:
            module = module_ref()
            if module is None:
                continue
            hook = getattr(module, hook_name, None)
            if callable(hook):
                try:
                    hook(*args)
                    called.append(module)
                except Exception as exc:  # noqa: BLE001
                    if not continue_on_error:
                        raise
                    errors.append(exc)
        if errors:
            raise RuntimeError(
                f'{len(errors)} model hook(s) failed during {hook_name}') \
                from errors[0]
        return called

    def set_common_eval_attrs(self, cfg: Dict, seed: int, ckpt_path: str,
                              model_family: str, mixed_precision_dtype: str,
                              enable_mixed_precision_training: bool) -> None:
        """Set attributes shared by evaluation runners."""
        self.cfg = cfg
        self.seed = seed
        self.ckpt_path = ckpt_path
        self.model_family = model_family
        self.mixed_precision_dtype = str_to_dtype(mixed_precision_dtype)
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.device_id = overwatch.local_rank()
        self.distributed_state = overwatch.distributed_state

    def update_model_norm_stats(self, norm_stats: Dict[str, Any]) -> None:
        """Attach normalization statistics to the wrapped VLA model."""
        self.vla.norm_stats = norm_stats
