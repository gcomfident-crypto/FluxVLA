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
"""Small, PyTorch-native TensorRT runtime.

The implementation follows the runtime pattern in NVIDIA Isaac-GR00T N1.5,
but fixes two integration issues that matter inside FluxVLA:

* TensorRT runs on the current PyTorch CUDA stream, so no PyCUDA context is
  created and no per-engine ``stream.synchronize()`` is needed.
* high-level inference retains input and output tensors for a stable concrete
  shape instead of allocating them for every engine invocation.

This module never captures a CUDA graph.  It supports one execution context
and one in-flight stream per :class:`TensorRTRuntime`.  Use a separate
runtime/context for concurrent streams.
"""

from __future__ import annotations
import re
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional

import torch


def _version_triplet(value: str) -> tuple[int, int, int]:
    """Return the numeric TensorRT ``major.minor.patch`` identity."""
    match = re.search(r'(\d+)\.(\d+)\.(\d+)', str(value))
    if match is None:
        raise ValueError(f'Cannot parse TensorRT version: {value!r}')
    return tuple(int(part) for part in match.groups())


def _require_tensorrt():
    try:
        import tensorrt as trt
    except (ImportError, OSError) as exc:  # pragma: no cover - host-specific
        raise RuntimeError(
            'TensorRT Python bindings are required. On the supported Linux '
            'x86_64 reference environment, install tensorrt-cu12==11.1.0.106 '
            'from the NVIDIA Python package index. Source-checkout users can '
            'also follow deployment_scripts/tensorrt/README.md.') from exc
    return trt


def _canonical_mode(mode: Any) -> str:
    value = str(mode).lower()
    if value in {'input', 'tensoriomode.input'} or value.endswith('.input'):
        return 'input'
    if value in {'output', 'tensoriomode.output'} or value.endswith('.output'):
        return 'output'
    raise ValueError(f'Unsupported TensorRT I/O mode: {mode!r}')


def _torch_dtype(trt_dtype: Any, trt_module: Any = None) -> torch.dtype:
    """Translate TensorRT dtype enums without relying on NumPy BF16."""
    value = str(trt_dtype).lower().replace('datatype.', '')
    aliases = {
        'float': torch.float32,
        'float32': torch.float32,
        'fp32': torch.float32,
        'half': torch.float16,
        'float16': torch.float16,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'bfloat16': torch.bfloat16,
        'int8': torch.int8,
        'int32': torch.int32,
        'int64': torch.int64,
        'bool': torch.bool,
        'uint8': torch.uint8,
    }
    if value in aliases:
        return aliases[value]
    # TensorRT enums sometimes stringify as an integer in test doubles or
    # older bindings.  Prefer the official NumPy bridge when available.
    trt = trt_module if trt_module is not None else _require_tensorrt()
    try:
        import numpy as np
        numpy_dtype = np.dtype(trt.nptype(trt_dtype))
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(f'Unsupported TensorRT tensor dtype: {trt_dtype!r}') \
            from exc
    numpy_map = {
        np.dtype('float32'): torch.float32,
        np.dtype('float16'): torch.float16,
        np.dtype('int8'): torch.int8,
        np.dtype('int32'): torch.int32,
        np.dtype('int64'): torch.int64,
        np.dtype('bool'): torch.bool,
        np.dtype('uint8'): torch.uint8,
    }
    if numpy_dtype not in numpy_map:
        raise TypeError(f'Unsupported TensorRT tensor dtype: {trt_dtype!r}')
    return numpy_map[numpy_dtype]


@dataclass(frozen=True)
class TensorSpec:
    """TensorRT binding contract.

    A ``-1`` dimension is a wildcard.  Engine-declared specs may contain
    wildcards, while every tensor submitted for execution must be concrete.
    """

    name: str
    mode: str
    shape: tuple[int, ...]
    dtype: torch.dtype

    def __post_init__(self) -> None:
        object.__setattr__(self, 'mode', _canonical_mode(self.mode))
        object.__setattr__(self, 'shape', tuple(int(v) for v in self.shape))
        if not self.name:
            raise ValueError('TensorSpec.name must not be empty')
        if any(value == 0 or value < -1 for value in self.shape):
            raise ValueError(f'Invalid shape for {self.name}: {self.shape}')
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError(f'{self.name} dtype must be torch.dtype')

    def accepts_shape(self, actual: Iterable[int]) -> bool:
        actual = tuple(int(v) for v in actual)
        return len(actual) == len(self.shape) and all(
            expected == -1 or expected == value
            for expected, value in zip(self.shape, actual))


class TensorRTRuntime:
    """Deserialize and execute one TensorRT engine.

    The runtime is deliberately model-agnostic.  Model code owns orchestration
    (PI0.5 V/P/D or GR00T's seven sub-engines), while this class owns one
    engine, one execution context, named bindings, and reusable output memory.
    """

    def __init__(
        self,
        engine_path: str | Path,
        *,
        expected_specs: Optional[Iterable[TensorSpec]] = None,
        expected_tensorrt_version: Optional[str] = None,
        profile_index: int = 0,
        _trt_module: Any = None,
    ) -> None:
        self.engine_path = Path(engine_path).expanduser()
        self.expected_tensorrt_version = expected_tensorrt_version
        self.profile_index = int(profile_index)
        if self.profile_index < 0:
            raise ValueError('profile_index must be non-negative')
        expected_specs = (
            tuple(expected_specs) if expected_specs is not None else None)
        self._expected_specs = ({spec.name: spec
                                 for spec in expected_specs}
                                if expected_specs is not None else None)
        if (expected_specs is not None
                and len(self._expected_specs) != len(expected_specs)):
            raise ValueError('expected_specs contains duplicate names')

        # The private injection point keeps CPU/fake tests independent of an
        # installed TensorRT distribution.  Production always takes the lazy
        # import path in ``prepare``.
        self._injected_trt = _trt_module
        self._trt = None
        self._logger = None
        self._runtime = None
        self._engine = None
        self._context = None
        self._device: Optional[torch.device] = None
        self._specs: dict[str, TensorSpec] = {}
        self._inputs: MutableMapping[str, torch.Tensor] = {}
        self._outputs: MutableMapping[str, torch.Tensor] = {}
        self._addresses: dict[str, int] = {}
        self._stream: Optional[torch.cuda.Stream] = None
        self._stream_handle: Optional[int] = None
        self._profile_selected = False
        self._closed = False
        # Preparation, first enqueue, execution, and teardown share one
        # re-entrant lock so two concurrent first calls cannot deserialize
        # separate engines into the same runtime object.
        self._lock = threading.RLock()

    @property
    def prepared(self) -> bool:
        return self._context is not None

    @property
    def specs(self) -> Mapping[str, TensorSpec]:
        if not self.prepared:
            raise RuntimeError('TensorRT runtime has not been prepared')
        return dict(self._specs)

    @property
    def outputs(self) -> Mapping[str, torch.Tensor]:
        return dict(self._outputs)

    @property
    def inputs(self) -> Mapping[str, torch.Tensor]:
        return dict(self._inputs)

    def _inspect_specs(self) -> dict[str, TensorSpec]:
        specs = {}
        for index in range(int(self._engine.num_io_tensors)):
            name = str(self._engine.get_tensor_name(index))
            spec = TensorSpec(
                name=name,
                mode=_canonical_mode(self._engine.get_tensor_mode(name)),
                shape=tuple(self._engine.get_tensor_shape(name)),
                dtype=_torch_dtype(
                    self._engine.get_tensor_dtype(name), self._trt),
            )
            if name in specs:
                raise RuntimeError(f'Duplicate TensorRT tensor name: {name}')
            specs[name] = spec
        if not specs:
            raise RuntimeError(
                f'Engine has no I/O tensors: {self.engine_path}')
        return specs

    def _audit_expected_specs(self) -> None:
        if self._expected_specs is None:
            return
        if set(self._expected_specs) != set(self._specs):
            missing = sorted(set(self._expected_specs) - set(self._specs))
            extra = sorted(set(self._specs) - set(self._expected_specs))
            raise RuntimeError(
                f'Engine binding names do not match the expected contract; '
                f'missing={missing}, extra={extra}')
        failures = []
        for name, expected in self._expected_specs.items():
            actual = self._specs[name]
            if expected.mode != actual.mode:
                failures.append(
                    f'{name}: mode {actual.mode} != {expected.mode}')
            if expected.dtype != actual.dtype:
                failures.append(
                    f'{name}: dtype {actual.dtype} != {expected.dtype}')
            # A concrete model contract must not silently accept a dynamic
            # engine binding.  ``-1`` is a wildcard only when the expected
            # contract explicitly asks for one; an engine-side ``-1`` is not
            # equivalent to a concrete expected dimension.
            if len(expected.shape) != len(actual.shape) or any(
                    exp != -1 and exp != got
                    for exp, got in zip(expected.shape, actual.shape)):
                failures.append(
                    f'{name}: shape {actual.shape} != {expected.shape}')
        if failures:
            raise RuntimeError('Engine binding contract mismatch: ' +
                               '; '.join(failures))

    def prepare(self, device: str | torch.device = 'cuda') -> None:
        with self._lock:
            self._prepare_locked(device)

    def _prepare_locked(self, device: str | torch.device = 'cuda') -> None:
        if self._closed:
            raise RuntimeError('TensorRT runtime is closed')
        device = torch.device(device)
        if device.type != 'cuda' and self._injected_trt is None:
            raise ValueError(f'TensorRT requires a CUDA device, got {device}')
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        if self.prepared:
            if device != self._device:
                raise RuntimeError(
                    f'Runtime already prepared on {self._device}, not {device}'
                )
            return
        if not self.engine_path.is_file():
            raise FileNotFoundError(f'TensorRT engine not found: '
                                    f'{self.engine_path}')

        trt = (
            self._injected_trt
            if self._injected_trt is not None else _require_tensorrt())
        if self.expected_tensorrt_version is not None:
            runtime_version = getattr(trt, '__version__', None)
            if runtime_version is None:
                raise RuntimeError(
                    'TensorRT binding does not expose __version__; cannot '
                    'verify engine/runtime compatibility')
            expected = _version_triplet(self.expected_tensorrt_version)
            actual = _version_triplet(runtime_version)
            if actual != expected:
                raise RuntimeError(
                    'TensorRT engine/runtime version mismatch: engine was '
                    f'built with {self.expected_tensorrt_version}, Python '
                    f'runtime is {runtime_version}')
        device_context = (
            torch.cuda.device(device)
            if device.type == 'cuda' else nullcontext())
        with device_context:
            logger = trt.Logger(trt.Logger.ERROR)
            runtime = trt.Runtime(logger)
            engine_bytes = self.engine_path.read_bytes()
            engine = runtime.deserialize_cuda_engine(engine_bytes)
            if engine is None:
                raise RuntimeError(f'Failed to deserialize TensorRT engine: '
                                   f'{self.engine_path}')
            profile_count = int(
                getattr(engine, 'num_optimization_profiles', 1))
            if profile_count < 1:
                profile_count = 1
            if self.profile_index >= profile_count:
                raise ValueError(
                    f'profile_index={self.profile_index} is outside engine '
                    f'range [0, {profile_count - 1}]')
            context = engine.create_execution_context()
            if context is None:
                raise RuntimeError(f'Failed to create execution context: '
                                   f'{self.engine_path}')

        try:
            self._trt = trt
            self._logger = logger
            self._runtime = runtime
            self._engine = engine
            self._context = context
            self._device = device
            self._specs = self._inspect_specs()
            self._audit_expected_specs()
        except Exception:
            self._clear_resources()
            raise

    def _select_stream(
            self, stream: Optional[torch.cuda.Stream]) -> torch.cuda.Stream:
        if stream is None:
            if self._device.type != 'cuda':
                raise ValueError(
                    'CPU/fake execution requires an explicit fake stream')
            stream = torch.cuda.current_stream(self._device)
        stream_device = torch.device(stream.device)
        if stream_device.type == 'cuda' and stream_device.index is None:
            stream_device = torch.device('cuda', torch.cuda.current_device())
        if stream_device != self._device:
            raise ValueError(
                f'CUDA stream is on {stream_device}, runtime is on '
                f'{self._device}')
        handle = int(stream.cuda_stream)
        if self._device.type == 'cuda':
            # Tensor inputs are expected to be ready on PyTorch's current
            # stream.  When a caller selects another stream, establish the
            # producer-to-consumer dependency before binding/copying them.
            current = torch.cuda.current_stream(self._device)
            if int(current.cuda_stream) != handle:
                stream.wait_stream(current)
        if self._stream_handle is None:
            self._stream = stream
            self._stream_handle = handle
        elif handle != self._stream_handle:
            raise RuntimeError(
                'A TensorRTRuntime is stream-affine after its first enqueue; '
                'create one runtime per concurrent CUDA stream')
        return stream

    def _validate_tensor(self, name: str, tensor: torch.Tensor,
                         spec: TensorSpec) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'{name} must be a torch.Tensor')
        if tensor.device != self._device:
            raise ValueError(
                f'{name} is on {tensor.device}, expected {self._device}')
        if tensor.dtype != spec.dtype:
            raise TypeError(
                f'{name} dtype is {tensor.dtype}, expected {spec.dtype}')
        if not tensor.is_contiguous():
            raise ValueError(f'{name} must be contiguous')
        if not spec.accepts_shape(tensor.shape):
            raise ValueError(
                f'{name} shape is {tuple(tensor.shape)}, engine declares '
                f'{spec.shape}')
        if self._expected_specs is not None:
            expected = self._expected_specs[name]
            if not expected.accepts_shape(tensor.shape):
                raise ValueError(
                    f'{name} shape is {tuple(tensor.shape)}, expected '
                    f'contract requires {expected.shape}')

    def _set_input_shapes(self, inputs: Mapping[str, torch.Tensor]) -> None:
        for name, tensor in inputs.items():
            spec = self._specs[name]
            if -1 in spec.shape:
                accepted = self._context.set_input_shape(
                    name, tuple(int(v) for v in tensor.shape))
                if accepted is False:
                    raise RuntimeError(
                        f'TensorRT rejected input shape for {name}: '
                        f'{tuple(tensor.shape)}')

    def _infer_shapes(self) -> None:
        """Run named-tensor shape inference after input addresses are bound.

        TensorRT 10.13 and 11 require shape-inference I/O addresses before
        ``infer_shapes``.  Binding inputs first works for both ordinary
        dynamic dimensions and shape tensors.
        """
        infer_shapes = getattr(self._context, 'infer_shapes', None)
        if callable(infer_shapes):
            missing = list(infer_shapes())
            if missing:
                raise RuntimeError(
                    f'Not all TensorRT shapes are specified: {missing}')

    def _select_profile(self, stream: torch.cuda.Stream) -> None:
        if self._profile_selected:
            return
        if self.profile_index != 0:
            select = getattr(self._context, 'set_optimization_profile_async',
                             None)
            if not callable(select):
                raise RuntimeError(
                    'This TensorRT binding cannot select a non-default '
                    'optimization profile asynchronously')
            accepted = select(self.profile_index, int(stream.cuda_stream))
            if accepted is False:
                raise RuntimeError(f'TensorRT rejected optimization profile '
                                   f'{self.profile_index}')
        self._profile_selected = True

    def _resolved_output(self, name: str) -> TensorSpec:
        declared = self._specs[name]
        shape = tuple(int(v) for v in self._context.get_tensor_shape(name))
        if any(value < 0 for value in shape):
            raise RuntimeError(
                f'TensorRT did not resolve output shape for {name}: {shape}')
        return TensorSpec(name, 'output', shape, declared.dtype)

    def _output_buffer(self, spec: TensorSpec) -> torch.Tensor:
        current = self._outputs.get(spec.name)
        if (current is None or tuple(current.shape) != spec.shape
                or current.dtype != spec.dtype
                or current.device != self._device):
            current = torch.empty(
                spec.shape, dtype=spec.dtype, device=self._device)
            self._outputs[spec.name] = current
        return current

    def _input_buffer(self, name: str, source: torch.Tensor) -> torch.Tensor:
        current = self._inputs.get(name)
        if (current is None or tuple(current.shape) != tuple(source.shape)
                or current.dtype != source.dtype
                or current.device != self._device):
            current = torch.empty_like(source, device=self._device)
            self._inputs[name] = current
        return current

    def _bind(self, name: str, tensor: torch.Tensor) -> None:
        address = int(tensor.data_ptr())
        if self._addresses.get(name) == address:
            return
        accepted = self._context.set_tensor_address(name, address)
        if accepted is False:
            raise RuntimeError(f'TensorRT rejected address for {name}')
        self._addresses[name] = address

    def _record_stream(self, tensor: torch.Tensor,
                       stream: torch.cuda.Stream) -> None:
        # CPU tensors appear only through the private fake-TRT test seam.
        if tensor.device.type == 'cuda':
            tensor.record_stream(stream)

    def _stream_context(self, stream: torch.cuda.Stream):
        if self._device.type == 'cuda':
            return torch.cuda.stream(stream)
        return nullcontext()

    def _enqueue_impl(
        self,
        inputs: Mapping[str, torch.Tensor],
        *,
        outputs: Optional[Mapping[str, torch.Tensor]],
        stream: torch.cuda.Stream,
    ) -> Mapping[str, torch.Tensor]:
        expected_inputs = {
            name
            for name, spec in self._specs.items() if spec.mode == 'input'
        }
        if set(inputs) != expected_inputs:
            missing = sorted(expected_inputs - set(inputs))
            extra = sorted(set(inputs) - expected_inputs)
            raise ValueError(
                f'Input names mismatch; missing={missing}, extra={extra}')
        self._select_profile(stream)
        for name, tensor in inputs.items():
            self._validate_tensor(name, tensor, self._specs[name])
        self._set_input_shapes(inputs)

        # Bind all inputs before infer_shapes.  TensorRT 10.13/11 needs this
        # ordering when an input participates in shape inference.
        for name, tensor in inputs.items():
            self._bind(name, tensor)
            self._record_stream(tensor, stream)
        self._infer_shapes()

        expected_outputs = {
            name
            for name, spec in self._specs.items() if spec.mode == 'output'
        }
        if outputs is not None and not set(outputs).issubset(expected_outputs):
            raise ValueError(f'Unknown output overrides: '
                             f'{sorted(set(outputs) - expected_outputs)}')
        bound_outputs = {}
        for name in expected_outputs:
            resolved = self._resolved_output(name)
            tensor = outputs[name] if outputs and name in outputs \
                else self._output_buffer(resolved)
            self._validate_tensor(name, tensor, resolved)
            bound_outputs[name] = tensor
            self._bind(name, tensor)
            self._record_stream(tensor, stream)

        succeeded = self._context.execute_async_v3(
            stream_handle=int(stream.cuda_stream))
        if succeeded is False:
            raise RuntimeError(
                f'TensorRT execution failed: {self.engine_path}')
        return bound_outputs

    def enqueue(
        self,
        inputs: Mapping[str, torch.Tensor],
        *,
        outputs: Optional[Mapping[str, torch.Tensor]] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> Mapping[str, torch.Tensor]:
        """Enqueue one inference without synchronizing the CUDA stream."""
        with self._lock:
            if not self.prepared:
                device = (
                    next(iter(inputs.values())).device if inputs else 'cuda')
                self._prepare_locked(device)
            selected_stream = self._select_stream(stream)
            return self._enqueue_impl(
                inputs, outputs=outputs, stream=selected_stream)

    def infer(
        self,
        inputs: Mapping[str, torch.Tensor],
        *,
        outputs: Optional[Mapping[str, torch.Tensor]] = None,
        stream: Optional[torch.cuda.Stream] = None,
        copy_outputs: bool = True,
    ) -> Mapping[str, torch.Tensor]:
        """Convenience wrapper around :meth:`enqueue`.

        Cloning is enqueued on the same stream and does not synchronize it.
        Set ``copy_outputs=False`` in an orchestrated hot path where outputs
        are consumed before the same runtime is enqueued again.
        """
        with self._lock:
            if not self.prepared:
                device = (
                    next(iter(inputs.values())).device if inputs else 'cuda')
                self._prepare_locked(device)
            selected_stream = self._select_stream(stream)
            persistent_inputs = {}
            with self._stream_context(selected_stream):
                for name, source in inputs.items():
                    if name not in self._specs:
                        # Let _enqueue_impl produce the complete name diff.
                        persistent_inputs[name] = source
                        continue
                    self._validate_tensor(name, source, self._specs[name])
                    target = self._input_buffer(name, source)
                    target.copy_(source, non_blocking=True)
                    self._record_stream(source, selected_stream)
                    persistent_inputs[name] = target
                result = self._enqueue_impl(
                    persistent_inputs, outputs=outputs, stream=selected_stream)
                if copy_outputs:
                    return {
                        name: tensor.clone()
                        for name, tensor in result.items()
                    }
                return result

    def _clear_resources(self) -> None:
        self._inputs.clear()
        self._outputs.clear()
        self._addresses.clear()
        self._stream = None
        self._stream_handle = None
        self._profile_selected = False
        self._context = None
        self._engine = None
        self._runtime = None
        self._logger = None
        self._trt = None
        self._device = None

    def close(self, *, synchronize: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            if synchronize and self._stream is not None:
                self._stream.synchronize()
            self._clear_resources()
            self._closed = True

    def __enter__(self) -> 'TensorRTRuntime':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
