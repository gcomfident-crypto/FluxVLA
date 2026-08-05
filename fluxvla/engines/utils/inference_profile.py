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
"""Optional CUDA-event timing for inference module boundaries."""

from __future__ import annotations
from collections import defaultdict
from contextlib import contextmanager
from typing import DefaultDict, Dict, Iterator, Optional

import torch


class CudaStageProfiler:
    """Collect repeated CUDA stages without synchronizing each stage."""

    def __init__(self) -> None:
        self.enabled = False
        self._active = False
        self._open = {}
        self._events: DefaultDict[str, list] = defaultdict(list)

    def configure(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            self.reset()

    def reset(self) -> None:
        self._active = False
        self._open.clear()
        self._events.clear()

    def begin_call(self) -> None:
        self._open.clear()
        self._events.clear()
        self._active = self.enabled

    def start(self,
              name: str,
              stream: Optional[torch.cuda.Stream] = None) -> None:
        if not self._active:
            return
        if name in self._open:
            raise RuntimeError(f'CUDA profile stage is already open: {name}')
        event = torch.cuda.Event(enable_timing=True)
        event.record(stream)
        self._open[name] = event

    def stop(self,
             name: str,
             stream: Optional[torch.cuda.Stream] = None) -> None:
        if not self._active:
            return
        start = self._open.pop(name, None)
        if start is None:
            raise RuntimeError(f'CUDA profile stage is not open: {name}')
        end = torch.cuda.Event(enable_timing=True)
        end.record(stream)
        self._events[name].append((start, end))

    @contextmanager
    def stage(
        self,
        name: str,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> Iterator[None]:
        self.start(name, stream)
        try:
            yield
        finally:
            self.stop(name, stream)

    def resolve(self) -> Dict[str, float | int]:
        """Resolve events after the caller synchronizes the model call."""
        if not self.enabled:
            return {}
        if self._open:
            raise RuntimeError('Cannot resolve CUDA profile with open stages: '
                               f'{sorted(self._open)}')
        result: Dict[str, float | int] = {}
        for name, pairs in self._events.items():
            values = [float(start.elapsed_time(end)) for start, end in pairs]
            result[f'{name}_ms'] = sum(values)
            result[f'{name}_calls'] = len(values)
            result[f'{name}_mean_ms'] = sum(values) / len(values)
        self._active = False
        return result


def _profiler(module) -> CudaStageProfiler:
    profiler = getattr(module, '_cuda_stage_profiler', None)
    if profiler is None:
        profiler = CudaStageProfiler()
        setattr(module, '_cuda_stage_profiler', profiler)
    return profiler


def configure_module_profile(module, enabled: bool) -> None:
    _profiler(module).configure(enabled)


def begin_module_profile(module) -> None:
    _profiler(module).begin_call()


def start_module_stage(module,
                       name: str,
                       stream: Optional[torch.cuda.Stream] = None) -> None:
    _profiler(module).start(name, stream)


def stop_module_stage(module,
                      name: str,
                      stream: Optional[torch.cuda.Stream] = None) -> None:
    _profiler(module).stop(name, stream)


def module_stage(module,
                 name: str,
                 stream: Optional[torch.cuda.Stream] = None):
    return _profiler(module).stage(name, stream)


def resolve_module_profile(module) -> Dict[str, float | int]:
    return _profiler(module).resolve()
