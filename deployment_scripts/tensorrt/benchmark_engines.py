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
"""Benchmark the static PI0.5 or GR00T-N1.5 TensorRT engine bundle.

This command is deliberately limited to engine-level measurements.  For every
engine it creates deterministic raw inputs from the static ONNX graph and runs
three independent ``trtexec`` processes:

* ``perf`` exports per-inference timings without CUDA Graph capture;
* ``profile`` exports the perturbing per-layer profiler measurements; and
* ``inspector`` exports layer information without running inference.

Commands, logs, raw NVIDIA JSON, input identities and the derived
``engine_benchmark.csv`` are kept together beneath ``--output-dir``.  The CSV
is derived only from the unprofiled performance pass.  It is a component
microbenchmark, not end-to-end policy latency.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_SCHEMA_VERSION = 1
_STAT_NAMES = ('mean', 'std', 'min', 'p50', 'p90', 'p95', 'p99', 'max')
_METRIC_COLUMNS = {
    'h2d': 'h2d',
    'gpu': 'gpu_compute',
    'd2h': 'd2h',
    'host': 'host_latency',
    'enqueue': 'enqueue',
}
_NUMBER_PATTERN = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'


class BenchmarkError(RuntimeError):
    """An actionable benchmark configuration or artifact error."""


@dataclass(frozen=True)
class EngineSpec:
    """Map one serialized engine to its ONNX export-manifest key."""

    name: str
    onnx_key: str
    calls_per_action: int


@dataclass(frozen=True)
class StaticInput:
    """One concrete, non-initializer ONNX graph input."""

    name: str
    elem_type: int
    dtype_name: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class RuntimeCliPolicy:
    """Version-specific trtexec arguments that preserve the protocol."""

    arguments: tuple[str, ...]
    cuda_graph_control: str
    data_transfer_control: str


_ENGINE_SPECS = {
    'pi05': (
        EngineSpec('vision', 'vision', 1),
        EngineSpec('prefix', 'prefix', 1),
        EngineSpec('denoise', 'denoise', 10),
    ),
    'groot-n15': (
        EngineSpec('vit_fp16', 'vit', 1),
        EngineSpec('llm_fp16', 'llm', 1),
        EngineSpec('vlln_vl_self_attention', 'backbone_output', 1),
        EngineSpec('state_encoder', 'state_encoder', 1),
        EngineSpec('action_encoder', 'action_encoder', 4),
        EngineSpec('dit_fp16', 'dit', 4),
        EngineSpec('action_decoder', 'action_decoder', 4),
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8')


def _read_json(path: Path, description: str) -> Any:
    if not path.is_file():
        raise BenchmarkError(f'{description} not found: {path}')
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f'invalid {description}: {path}: {exc}') from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _safe_child(directory: Path, filename: str, description: str) -> Path:
    relative = Path(filename)
    candidate = (directory / relative).resolve()
    if (relative.is_absolute() or '..' in relative.parts
            or candidate.parent != directory.resolve()):
        raise BenchmarkError(
            f'unsafe {description} filename in manifest: {filename!r}')
    return candidate


def _resolve_executable(value: str) -> Path:
    expanded = os.path.expanduser(value)
    if os.sep in expanded:
        candidate = Path(expanded)
    else:
        found = shutil.which(expanded)
        if found is None:
            raise BenchmarkError(
                f'trtexec executable not found: {value!r}; pass '
                '--trtexec=/absolute/path/to/trtexec')
        candidate = Path(found)
    candidate = candidate.resolve()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise BenchmarkError(f'trtexec is not executable: {candidate}')
    return candidate


def _prepare_output_dir(path: Path, force: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise BenchmarkError(f'output path is not a directory: {path}')
    if path.is_dir() and any(path.iterdir()) and not force:
        raise BenchmarkError(
            f'output directory is not empty: {path}; choose a new directory '
            'or pass --force to overwrite known output files')
    path.mkdir(parents=True, exist_ok=True)
    return path


def _remove_stale_file(path: Path) -> None:
    """Remove only a known generated file before producing its replacement."""
    if not path.exists():
        return
    if not path.is_file():
        raise BenchmarkError(f'expected output path is not a file: {path}')
    path.unlink()


def _command_record(argv: Sequence[str], pass_name: str, **extra: Any) -> dict:
    value = {
        'schema_version': _SCHEMA_VERSION,
        'pass': pass_name,
        'argv': list(argv),
        'shell_command': shlex.join(argv),
        'cwd': str(Path.cwd().resolve()),
    }
    value.update(extra)
    return value


def _probe_trtexec(executable: Path, output_dir: Path) -> tuple[set[str], str]:
    argv = [str(executable), '--help']
    log_path = output_dir / 'trtexec_help.log'
    command_path = output_dir / 'trtexec_help_command.json'
    started_at = _utc_now()
    start = time.perf_counter()
    try:
        process = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False)
    except OSError as exc:
        raise BenchmarkError(
            f'could not execute trtexec --help: {exc}') from exc
    elapsed = time.perf_counter() - start
    log_path.write_bytes(process.stdout)
    help_text = process.stdout.decode('utf-8', errors='replace')
    flags = set(re.findall(r'--([A-Za-z][A-Za-z0-9]*)', help_text))
    version = _parse_tensorrt_version(help_text)
    _write_json(
        command_path,
        _command_record(
            argv,
            'help',
            started_at_utc=started_at,
            finished_at_utc=_utc_now(),
            elapsed_seconds=elapsed,
            returncode=process.returncode,
            accepted_nonzero_returncode=True,
            detected_tensorrt_version=version,
            detected_flag_count=len(flags),
            log=str(log_path.relative_to(output_dir))))
    # Some trtexec releases return a non-zero status after printing help when
    # no model was supplied.  Capability detection is therefore based on the
    # actual help body rather than that status alone.
    if not flags or 'loadEngine' not in flags:
        raise BenchmarkError(
            f'trtexec --help did not expose expected options; return code '
            f'{process.returncode}, see {log_path}')
    major = int(version.split('.', 1)[0])
    if major not in (10, 11):
        raise BenchmarkError(
            f'unsupported TensorRT major version {version}; this command '
            'supports trtexec from TensorRT 10 or 11')
    return flags, version


def _parse_tensorrt_version(text: str) -> str:
    dotted = re.search(
        r'TensorRT\s+(?:version\s*)?v?(\d+)\.(\d+)\.(\d+)',
        text,
        flags=re.IGNORECASE)
    if dotted:
        return '.'.join(dotted.groups())
    encoded = re.search(r'TensorRT\s+v(\d{5,})', text, flags=re.IGNORECASE)
    if encoded:
        number = int(encoded.group(1))
        return f'{number // 10000}.{number // 100 % 100}.{number % 100}'
    raise BenchmarkError(
        'could not determine the TensorRT version from trtexec --help')


def _require_flags(supported: set[str], required: Iterable[str],
                   pass_name: str, version: str) -> None:
    missing = sorted(set(required) - supported)
    if missing:
        rendered = ', '.join(f'--{name}' for name in missing)
        raise BenchmarkError(
            f'TensorRT {version} trtexec does not advertise required '
            f'{pass_name} flag(s): {rendered}')


def _validate_cli_capabilities(supported: set[str], version: str,
                               include_transfers: bool) -> RuntimeCliPolicy:
    common = {
        'loadEngine', 'device', 'loadInputs', 'iterations', 'warmUp',
        'duration', 'infStreams'
    }
    _require_flags(supported,
                   common | {'avgRuns', 'percentile', 'exportTimes'},
                   'performance', version)
    _require_flags(
        supported,
        common | {'dumpProfile', 'exportProfile', 'separateProfileRun'},
        'profile', version)
    _require_flags(
        supported,
        {'loadEngine', 'device', 'skipInference', 'exportLayerInfo'},
        'inspector', version)

    major = int(version.split('.', 1)[0])
    runtime_arguments = []
    if 'noCudaGraph' in supported:
        runtime_arguments.append('--noCudaGraph')
        cuda_graph_control = 'explicit --noCudaGraph'
    elif major == 10 and 'useCudaGraph' in supported:
        # TensorRT 10 disables CUDA Graph unless --useCudaGraph is supplied.
        cuda_graph_control = 'TensorRT 10 default; omitted --useCudaGraph'
    else:
        raise BenchmarkError(
            f'TensorRT {version} trtexec exposes neither --noCudaGraph nor '
            'the TensorRT 10 --useCudaGraph opt-in needed to verify that '
            'CUDA Graph is disabled')

    if include_transfers:
        if 'includeDataTransfers' in supported:
            runtime_arguments.append('--includeDataTransfers')
            data_transfer_control = 'explicit --includeDataTransfers'
        elif major == 10 and 'noDataTransfers' in supported:
            # TensorRT 10 includes transfers unless --noDataTransfers is used.
            data_transfer_control = (
                'TensorRT 10 default; omitted --noDataTransfers')
        else:
            raise BenchmarkError(
                f'TensorRT {version} trtexec does not advertise a supported '
                'way to include data transfers')
    elif 'noDataTransfers' in supported:
        runtime_arguments.append('--noDataTransfers')
        data_transfer_control = 'explicit --noDataTransfers'
    elif major == 11 and 'includeDataTransfers' in supported:
        # TensorRT 11 disables transfers unless explicitly included.
        data_transfer_control = (
            'TensorRT 11 default; omitted --includeDataTransfers')
    else:
        raise BenchmarkError(
            f'TensorRT {version} trtexec does not advertise a supported way '
            'to disable data transfers')

    return RuntimeCliPolicy(
        arguments=tuple(runtime_arguments),
        cuda_graph_control=cuda_graph_control,
        data_transfer_control=data_transfer_control)


def _load_manifests(model: str, engine_dir: Path,
                    onnx_dir: Path) -> tuple[dict, dict]:
    export_manifest = _read_json(onnx_dir / 'export_manifest.json',
                                 'ONNX export manifest')
    engine_manifest = _read_json(engine_dir / 'engine_manifest.json',
                                 'engine manifest')
    for description, value in (('ONNX export manifest', export_manifest),
                               ('engine manifest', engine_manifest)):
        if not isinstance(value, dict):
            raise BenchmarkError(f'{description} must be a JSON object')
        if value.get('schema_version') != 1:
            raise BenchmarkError(
                f'{description} has unsupported schema_version '
                f'{value.get("schema_version")!r}')
        if value.get('shape_mode') != 'static':
            raise BenchmarkError(f'{description} must declare static shapes')
        if value.get('model') != model:
            raise BenchmarkError(
                f'{description} model {value.get("model")!r} does not match '
                f'{model!r}')
    if engine_manifest.get('engine_model') != model:
        raise BenchmarkError(
            'engine manifest engine_model does not match model')
    if export_manifest.get('checkpoint_sha256') != engine_manifest.get(
            'checkpoint_sha256'):
        raise BenchmarkError(
            'ONNX and engine manifests identify different checkpoints')
    if export_manifest.get('contract') != engine_manifest.get('contract'):
        raise BenchmarkError(
            'ONNX and engine manifests declare different static contracts')
    if export_manifest.get('onnx') != engine_manifest.get('onnx'):
        raise BenchmarkError(
            'ONNX and engine manifests declare different ONNX file maps')
    if export_manifest.get('onnx_artifacts') != engine_manifest.get(
            'onnx_artifacts'):
        raise BenchmarkError(
            'ONNX and engine manifests declare different ONNX artifact '
            'identities')

    onnx_files = export_manifest.get('onnx')
    onnx_artifacts = export_manifest.get('onnx_artifacts')
    engine_files = engine_manifest.get('engines')
    if not isinstance(onnx_files, dict):
        raise BenchmarkError('ONNX export manifest has no ONNX file map')
    if not isinstance(onnx_artifacts, dict):
        raise BenchmarkError(
            'ONNX export manifest has no ONNX artifact identities')
    if not isinstance(engine_files, dict):
        raise BenchmarkError('engine manifest has no engine identity map')
    specs = _ENGINE_SPECS[model]
    expected_engine_names = {spec.name for spec in specs}
    if set(engine_files) != expected_engine_names:
        raise BenchmarkError(
            'engine manifest engine set mismatch; expected '
            f'{sorted(expected_engine_names)}, got {sorted(engine_files)}')
    expected_onnx_keys = {spec.onnx_key for spec in specs}
    if set(onnx_files) != expected_onnx_keys:
        raise BenchmarkError(
            'ONNX manifest graph set mismatch; expected '
            f'{sorted(expected_onnx_keys)}, got {sorted(onnx_files)}')
    if set(onnx_artifacts) != expected_onnx_keys:
        raise BenchmarkError(
            'ONNX artifact identity set mismatch; expected '
            f'{sorted(expected_onnx_keys)}, got {sorted(onnx_artifacts)}')
    return export_manifest, engine_manifest


def _validate_tool_version(engine_manifest: Mapping[str, Any],
                           tool_version: str) -> None:
    engine_version = engine_manifest.get('tensorrt_version')
    if not isinstance(engine_version, str) or not engine_version:
        raise BenchmarkError(
            'engine manifest has no TensorRT version identity')
    engine_match = re.match(r'^(\d+)\.(\d+)\.(\d+)', engine_version)
    tool_match = re.match(r'^(\d+)\.(\d+)\.(\d+)', tool_version)
    if not engine_match or not tool_match:
        raise BenchmarkError(
            f'cannot compare TensorRT versions {engine_version!r} and '
            f'{tool_version!r}')
    if engine_match.groups() != tool_match.groups():
        raise BenchmarkError(
            f'engine was built with TensorRT {engine_version}, but selected '
            f'trtexec is TensorRT {tool_version}')


def _engine_and_onnx_paths(
        spec: EngineSpec, engine_dir: Path, onnx_dir: Path,
        export_manifest: Mapping[str, Any],
        engine_manifest: Mapping[str, Any]) -> tuple[Path, Path]:
    onnx_name = export_manifest['onnx'][spec.onnx_key]
    identity = engine_manifest['engines'][spec.name]
    if not isinstance(onnx_name, str):
        raise BenchmarkError(f'invalid ONNX filename for {spec.onnx_key}')
    if not isinstance(identity, dict) or not isinstance(
            identity.get('file'), str):
        raise BenchmarkError(f'invalid engine identity for {spec.name}')
    engine_path = _safe_child(engine_dir, identity['file'], 'engine')
    onnx_path = _safe_child(onnx_dir, onnx_name, 'ONNX')
    if identity['file'] != f'{spec.name}.engine':
        raise BenchmarkError(
            f'engine manifest filename mismatch for {spec.name}: '
            f'{identity["file"]!r}')
    if not engine_path.is_file():
        raise BenchmarkError(f'engine file not found: {engine_path}')
    if not onnx_path.is_file():
        raise BenchmarkError(f'ONNX file not found: {onnx_path}')
    _validate_file_identity(engine_path, identity, f'engine {spec.name!r}')
    artifact_identity = export_manifest['onnx_artifacts'][spec.onnx_key]
    _validate_onnx_artifact_identity(onnx_dir, onnx_path, artifact_identity,
                                     spec.onnx_key)
    return engine_path, onnx_path


def _validate_file_identity(path: Path, identity: Mapping[str, Any],
                            description: str) -> None:
    expected_bytes = identity.get('bytes')
    expected_sha256 = identity.get('sha256')
    if (type(expected_bytes) is not int or expected_bytes < 0
            or not isinstance(expected_sha256, str)
            or re.fullmatch(r'[0-9a-fA-F]{64}', expected_sha256) is None):
        raise BenchmarkError(
            f'{description} has an invalid size/SHA256 identity in its '
            'manifest')
    if path.stat().st_size != expected_bytes:
        raise BenchmarkError(
            f'{description} size does not match its manifest: {path}')
    actual_sha256 = _sha256(path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise BenchmarkError(
            f'{description} SHA256 does not match its manifest: {path}')


def _validate_onnx_artifact_identity(onnx_dir: Path, onnx_path: Path,
                                     identity: Any, graph_name: str) -> None:
    if not isinstance(identity,
                      dict) or identity.get('graph') != onnx_path.name:
        raise BenchmarkError(
            f'invalid ONNX artifact identity for {graph_name!r}')
    files = identity.get('files')
    if (not isinstance(files, list) or not files
            or not all(isinstance(item, dict) for item in files)):
        raise BenchmarkError(
            f'ONNX artifact identity for {graph_name!r} has no file list')
    validated_paths = set()
    for item in files:
        filename = item.get('file')
        if not isinstance(filename, str):
            raise BenchmarkError(
                f'ONNX artifact for {graph_name!r} has an invalid filename')
        path = _safe_child(onnx_dir, filename, 'ONNX artifact')
        if path in validated_paths:
            raise BenchmarkError(
                f'ONNX artifact for {graph_name!r} lists a duplicate file: '
                f'{filename!r}')
        if not path.is_file():
            raise BenchmarkError(f'ONNX artifact file not found: {path}')
        _validate_file_identity(path, item,
                                f'ONNX artifact {graph_name!r}/{filename!r}')
        validated_paths.add(path)
    if onnx_path not in validated_paths:
        raise BenchmarkError(
            f'ONNX artifact identity for {graph_name!r} does not include '
            f'the graph file {onnx_path.name!r}')


def _require_onnx_and_numpy():
    try:
        import numpy as np
        import onnx
    except ImportError as exc:
        raise BenchmarkError(
            'ONNX and NumPy are required to generate deterministic raw '
            'inputs; install requirements-tensorrt.txt') from exc
    return onnx, np


def _read_static_inputs(onnx_module, onnx_path: Path) -> list[StaticInput]:
    try:
        model = onnx_module.load(str(onnx_path), load_external_data=False)
    except Exception as exc:
        raise BenchmarkError(f'could not load ONNX graph {onnx_path}: {exc}') \
            from exc
    initializer_names = {value.name for value in model.graph.initializer}
    inputs = []
    for value in model.graph.input:
        if value.name in initializer_names:
            continue
        if not value.type.HasField('tensor_type'):
            raise BenchmarkError(
                f'ONNX input {value.name!r} in {onnx_path.name} is not a '
                'tensor')
        tensor_type = value.type.tensor_type
        if not tensor_type.elem_type:
            raise BenchmarkError(
                f'ONNX input {value.name!r} has no element type')
        shape = []
        for index, dimension in enumerate(tensor_type.shape.dim):
            if not dimension.HasField('dim_value') or dimension.dim_value <= 0:
                symbolic = dimension.dim_param or str(dimension.dim_value)
                raise BenchmarkError(
                    f'ONNX input {value.name!r} dimension {index} is not a '
                    f'positive static value: {symbolic!r}')
            shape.append(int(dimension.dim_value))
        try:
            dtype_name = onnx_module.TensorProto.DataType.Name(
                tensor_type.elem_type)
        except ValueError:
            dtype_name = f'UNKNOWN_{tensor_type.elem_type}'
        inputs.append(
            StaticInput(
                name=value.name,
                elem_type=int(tensor_type.elem_type),
                dtype_name=dtype_name,
                shape=tuple(shape)))
    if not inputs:
        raise BenchmarkError(f'ONNX graph has no runtime inputs: {onnx_path}')
    names = [value.name for value in inputs]
    if len(names) != len(set(names)):
        raise BenchmarkError(
            f'ONNX graph has duplicate input names: {onnx_path}')
    return inputs


def _input_seed(seed: int, model: str, engine: str, name: str) -> int:
    encoded = f'{seed}:{model}:{engine}:{name}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'little')


def _semantic_integer_values(np, shape: tuple[int, ...], name: str,
                             dtype) -> Any:
    lowered = name.lower()
    if 'mask' in lowered:
        return np.ones(shape, dtype=dtype)
    if 'position' in lowered and shape:
        values = np.arange(shape[-1], dtype=np.int64)
        values = np.broadcast_to(values, shape)
        return np.ascontiguousarray(values, dtype=dtype)
    # Token IDs, timesteps and embodiment IDs are deliberately kept at zero.
    # Zero is in range for their embedding/gather operations and avoids
    # semantically invalid random integer indices.
    return np.zeros(shape, dtype=dtype)


def _float_values(np, count: int, shape: tuple[int, ...], seed: int):
    rng = np.random.Generator(np.random.PCG64(seed))
    values = rng.standard_normal(count, dtype=np.float32)
    np.clip(values, -3.0, 3.0, out=values)
    return values.reshape(shape)


def _raw_input_bytes(onnx_module, np, value: StaticInput, seed: int) \
        -> tuple[bytes, str, str]:
    tensor_proto = onnx_module.TensorProto
    count = math.prod(value.shape) if value.shape else 1
    if value.elem_type == tensor_proto.FLOAT:
        array = _float_values(np, count, value.shape, seed).astype(
            np.dtype('<f4'), copy=False)
        generator = 'pcg64-standard-normal-clipped[-3,3]'
    elif value.elem_type == tensor_proto.FLOAT16:
        array = _float_values(np, count, value.shape,
                              seed).astype(np.dtype('<f2'))
        generator = 'pcg64-standard-normal-clipped[-3,3]'
    elif value.elem_type == tensor_proto.DOUBLE:
        array = _float_values(np, count, value.shape,
                              seed).astype(np.dtype('<f8'))
        generator = 'pcg64-standard-normal-clipped[-3,3]'
    elif value.elem_type == tensor_proto.BFLOAT16:
        float_values = _float_values(np, count, value.shape, seed)
        bits = np.ascontiguousarray(
            float_values, dtype=np.float32).view(np.uint32)
        # Round-to-nearest-even before retaining the high 16 BF16 bits.
        rounding_bias = np.uint32(0x7FFF) + ((bits >> 16) & 1)
        array = ((bits + rounding_bias) >> 16).astype(np.dtype('<u2'))
        generator = 'pcg64-standard-normal-clipped[-3,3]-bf16-rne'
    elif value.elem_type == tensor_proto.BOOL:
        array = np.ones(value.shape, dtype=np.bool_)
        generator = 'semantic-ones'
    else:
        integer_types = {
            tensor_proto.INT8: np.dtype('i1'),
            tensor_proto.UINT8: np.dtype('u1'),
            tensor_proto.INT16: np.dtype('<i2'),
            tensor_proto.UINT16: np.dtype('<u2'),
            tensor_proto.INT32: np.dtype('<i4'),
            tensor_proto.UINT32: np.dtype('<u4'),
            tensor_proto.INT64: np.dtype('<i8'),
            tensor_proto.UINT64: np.dtype('<u8'),
        }
        dtype = integer_types.get(value.elem_type)
        if dtype is None:
            raise BenchmarkError(
                f'unsupported ONNX input dtype {value.dtype_name} for '
                f'{value.name!r}')
        array = _semantic_integer_values(np, value.shape, value.name, dtype)
        generator = ('semantic-ones' if 'mask' in value.name.lower() else
                     'semantic-position-range'
                     if 'position' in value.name.lower() else 'semantic-zeros')
    array = np.ascontiguousarray(array)
    return array.tobytes(order='C'), str(array.dtype), generator


def _safe_input_filename(name: str) -> str:
    stem = re.sub(r'[^A-Za-z0-9_.-]+', '_', name).strip('._') or 'input'
    suffix = hashlib.sha256(name.encode('utf-8')).hexdigest()[:8]
    return f'{stem}-{suffix}.bin'


def _generate_engine_inputs(model: str, spec: EngineSpec, onnx_path: Path,
                            output_dir: Path, seed: int, onnx_module,
                            np) -> tuple[list[dict], str]:
    static_inputs = _read_static_inputs(onnx_module, onnx_path)
    input_dir = output_dir / 'inputs' / spec.name
    input_dir.mkdir(parents=True, exist_ok=True)
    identities = []
    load_items = []
    for value in static_inputs:
        if any(delimiter in value.name for delimiter in ",:'"):
            raise BenchmarkError(
                f'ONNX input name cannot be represented safely in trtexec '
                f'--loadInputs: {value.name!r}')
        path = input_dir / _safe_input_filename(value.name)
        raw, storage_dtype, generator = _raw_input_bytes(
            onnx_module, np, value,
            _input_seed(seed, model, spec.name, value.name))
        path.write_bytes(raw)
        absolute = path.resolve()
        if any(delimiter in str(absolute) for delimiter in ",:'"):
            raise BenchmarkError(
                f'input path cannot contain comma, colon, or apostrophe for '
                f'trtexec --loadInputs: {absolute}')
        identity = {
            'name': value.name,
            'shape': list(value.shape),
            'onnx_dtype': value.dtype_name,
            'storage_dtype': storage_dtype,
            'generator': generator,
            'file': str(path.relative_to(output_dir)),
            'bytes': path.stat().st_size,
            'sha256': _sha256(path),
        }
        identities.append(identity)
        load_items.append(f'{value.name}:{absolute}')
    _write_json(
        input_dir / 'inputs.json', {
            'schema_version': _SCHEMA_VERSION,
            'model': model,
            'engine': spec.name,
            'onnx': str(onnx_path),
            'seed': seed,
            'inputs': identities,
        })
    return identities, ','.join(load_items)


def _perf_command(executable: Path, engine_path: Path, load_inputs: str,
                  output_path: Path, args: argparse.Namespace,
                  policy: RuntimeCliPolicy) -> list[str]:
    return [
        str(executable),
        f'--loadEngine={engine_path}',
        f'--loadInputs={load_inputs}',
        f'--device={args.device}',
        '--infStreams=1',
        *policy.arguments,
        f'--warmUp={args.warmup_ms}',
        f'--duration={args.duration_seconds}',
        f'--iterations={args.iterations}',
        '--avgRuns=1',
        '--percentile=50,90,95,99',
        f'--exportTimes={output_path}',
    ]


def _profile_command(executable: Path, engine_path: Path, load_inputs: str,
                     output_path: Path, args: argparse.Namespace,
                     policy: RuntimeCliPolicy) -> list[str]:
    return [
        str(executable),
        f'--loadEngine={engine_path}',
        f'--loadInputs={load_inputs}',
        f'--device={args.device}',
        '--infStreams=1',
        *policy.arguments,
        f'--warmUp={args.warmup_ms}',
        '--duration=0',
        f'--iterations={args.profile_iterations}',
        '--dumpProfile',
        '--separateProfileRun',
        f'--exportProfile={output_path}',
    ]


def _inspector_command(executable: Path, engine_path: Path, output_path: Path,
                       args: argparse.Namespace) -> list[str]:
    return [
        str(executable),
        f'--loadEngine={engine_path}',
        f'--device={args.device}',
        '--skipInference',
        f'--exportLayerInfo={output_path}',
    ]


def _run_pass(argv: Sequence[str], engine: str, pass_name: str, pass_dir: Path,
              output_dir: Path) -> Path:
    pass_dir.mkdir(parents=True, exist_ok=True)
    log_path = pass_dir / f'{pass_name}.log'
    command_path = pass_dir / 'command.json'
    started_at = _utc_now()
    _write_json(
        command_path,
        _command_record(
            argv,
            pass_name,
            engine=engine,
            status='running',
            started_at_utc=started_at,
            log=str(log_path.relative_to(output_dir))))
    start = time.perf_counter()
    returncode = None
    error = None
    try:
        with log_path.open('wb') as stream:
            process = subprocess.run(
                argv, stdout=stream, stderr=subprocess.STDOUT, check=False)
        returncode = process.returncode
    except OSError as exc:
        error = str(exc)
    elapsed = time.perf_counter() - start
    record = _command_record(
        argv,
        pass_name,
        engine=engine,
        status=('completed' if returncode == 0 else 'failed'),
        started_at_utc=started_at,
        finished_at_utc=_utc_now(),
        elapsed_seconds=elapsed,
        returncode=returncode,
        log=str(log_path.relative_to(output_dir)))
    if error is not None:
        record['error'] = error
    _write_json(command_path, record)
    if error is not None:
        raise BenchmarkError(
            f'could not execute {pass_name} pass for {engine}: {error}; see '
            f'{command_path}')
    if returncode != 0:
        raise BenchmarkError(
            f'{pass_name} pass failed for {engine} with exit code '
            f'{returncode}; see {log_path}')
    if not log_path.is_file() or log_path.stat().st_size == 0:
        raise BenchmarkError(
            f'{pass_name} pass produced no log for {engine}: {log_path}')
    return log_path


def _validate_export_json(path: Path, pass_name: str, engine: str) -> Any:
    value = _read_json(path, f'{pass_name} JSON for {engine}')
    if pass_name == 'profile':
        if (not isinstance(value, list) or not value
                or not all(isinstance(item, dict) for item in value)):
            raise BenchmarkError(
                f'profile JSON for {engine} is not a non-empty layer list: '
                f'{path}')
    elif pass_name == 'inspector':
        if (not isinstance(value, dict)
                or not isinstance(value.get('Layers'), list)
                or not value['Layers']):
            raise BenchmarkError(
                f'inspector JSON for {engine} has no non-empty Layers list: '
                f'{path}; ensure the engine was built with detailed profiling '
                'verbosity')
    return value


def _timing_records(value: Any, path: Path) -> list[dict]:
    if isinstance(value, list):
        records = value
    elif isinstance(value, dict):
        records = None
        for key in ('timings', 'iterations', 'records', 'data'):
            candidate = value.get(key)
            if isinstance(candidate, list):
                records = candidate
                break
        if records is None:
            raise BenchmarkError(
                f'unrecognized exportTimes object schema in {path}; keys are '
                f'{sorted(value)}')
    else:
        raise BenchmarkError(
            f'exportTimes JSON must be a list or object: {path}')
    if not records or not all(isinstance(item, dict) for item in records):
        raise BenchmarkError(
            f'exportTimes JSON contains no timing records: {path}')
    return records


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _record_value(record: Mapping[str, Any], *keys: str) -> float | None:
    lowered = {str(key).lower(): value for key, value in record.items()}
    for key in keys:
        value = _finite_number(lowered.get(key.lower()))
        if value is not None:
            return value
    return None


def _record_delta(record: Mapping[str, Any], starts: Sequence[str],
                  ends: Sequence[str]) -> float | None:
    start = _record_value(record, *starts)
    end = _record_value(record, *ends)
    if start is None or end is None:
        return None
    difference = end - start
    if difference < -1e-9:
        return None
    return max(0.0, difference)


def _extract_record_metrics(
        record: Mapping[str, Any]) -> dict[str, float | None]:
    h2d = _record_value(record, 'h2dMs', 'h2dLatencyMs')
    if h2d is None:
        h2d = _record_delta(record, ('startH2dMs', ), ('endH2dMs', ))
    gpu = _record_value(record, 'computeMs', 'gpuComputeMs',
                        'gpuComputeTimeMs')
    if gpu is None:
        gpu = _record_delta(record, ('startComputeMs', ), ('endComputeMs', ))
    d2h = _record_value(record, 'd2hMs', 'd2hLatencyMs')
    if d2h is None:
        d2h = _record_delta(record, ('startD2hMs', ), ('endD2hMs', ))
    host = _record_value(record, 'latencyMs', 'hostLatencyMs')
    if host is None and all(value is not None for value in (h2d, gpu, d2h)):
        host = float(h2d) + float(gpu) + float(d2h)
    enqueue = _record_value(record, 'enqueueMs', 'enqMs', 'enqueueTimeMs')
    if enqueue is None:
        enqueue = _record_delta(record, ('startEnqMs', 'startEnqueueMs'),
                                ('endEnqMs', 'endEnqueueMs'))
    return {
        'h2d': h2d,
        'gpu': gpu,
        'd2h': d2h,
        'host': host,
        'enqueue': enqueue,
    }


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError('percentile requires at least one value')
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    fraction = rank - lower
    return (sorted_values[lower] * (1.0 - fraction) +
            sorted_values[upper] * fraction)


def _summarize(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    checked = []
    for value in values:
        parsed = _finite_number(value)
        if parsed is None or parsed < 0:
            raise BenchmarkError(
                f'timing series contains invalid value: {value!r}')
        checked.append(parsed)
    ordered = sorted(checked)
    return {
        'mean': statistics.fmean(checked),
        'std': statistics.pstdev(checked),
        'min': ordered[0],
        'p50': _percentile(ordered, 50),
        'p90': _percentile(ordered, 90),
        'p95': _percentile(ordered, 95),
        'p99': _percentile(ordered, 99),
        'max': ordered[-1],
    }


def _parse_log(log_path: Path) -> dict:
    text = log_path.read_text(encoding='utf-8', errors='replace')
    result: dict[str, Any] = {'summaries': {}}
    throughput_matches = re.findall(
        rf'Throughput:\s*({_NUMBER_PATTERN})\s*qps', text, flags=re.IGNORECASE)
    if throughput_matches:
        result['throughput_qps'] = float(throughput_matches[-1])
    scalar_patterns = {
        'gpu_name': r'Selected Device:\s*(.+?)\s*$',
        'gpu_uuid': r'Selected Device UUID:\s*(\S+)',
        'compute_capability': r'Compute Capability:\s*(\S+)',
    }
    for name, pattern in scalar_patterns.items():
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            result[name] = matches[-1].strip()
    labels = {
        'H2D Latency': 'h2d',
        'GPU Compute Time': 'gpu',
        'D2H Latency': 'd2h',
        'Host Latency': 'host',
        'Latency': 'host',
        'Enqueue Time': 'enqueue',
    }
    label_pattern = '|'.join(re.escape(label) for label in labels)
    for line in text.splitlines():
        match = re.search(
            rf'\b({label_pattern}):\s*(.*)$', line, flags=re.IGNORECASE)
        if not match or 'mean' not in match.group(2).lower():
            continue
        canonical_label = next(label for label in labels
                               if label.lower() == match.group(1).lower())
        metric = labels[canonical_label]
        values = {}
        for value_match in re.finditer(
                rf'(min|max|mean|median|percentile\(\s*'
                rf'({_NUMBER_PATTERN})%\s*\))\s*=\s*'
                rf'({_NUMBER_PATTERN})\s*ms',
                match.group(2),
                flags=re.IGNORECASE):
            key = value_match.group(1).lower()
            if key == 'median':
                key = 'p50'
            elif key.startswith('percentile'):
                percentile = float(value_match.group(2))
                key = f'p{percentile:g}'
            values[key] = float(value_match.group(3))
        if values:
            result['summaries'][metric] = values
    return result


def _fallback_throughput(records: Sequence[Mapping[str, Any]],
                         host_values: Sequence[float]) -> tuple[float, str]:
    starts = []
    ends = []
    for record in records:
        start = _record_value(record, 'startH2dMs', 'startComputeMs')
        end = _record_value(record, 'endD2hMs', 'endComputeMs')
        if start is not None and end is not None:
            starts.append(start)
            ends.append(end)
    if starts and ends:
        span_ms = max(ends) - min(starts)
        if span_ms > 0:
            return len(records) * 1000.0 / span_ms, 'exportTimes-span'
    if host_values:
        mean_ms = statistics.fmean(host_values)
        if mean_ms > 0:
            return 1000.0 / mean_ms, 'inverse-host-mean'
    raise BenchmarkError('could not derive throughput from log or exportTimes')


def _parse_performance(times_path: Path, log_path: Path) -> dict:
    raw = _read_json(times_path, 'trtexec exportTimes JSON')
    records = _timing_records(raw, times_path)
    series = {name: [] for name in _METRIC_COLUMNS}
    for record in records:
        values = _extract_record_metrics(record)
        for name, value in values.items():
            if value is not None:
                series[name].append(value)
    sample_count = len(records)
    incomplete = {
        name: len(values)
        for name, values in series.items() if len(values) != sample_count
    }
    if incomplete:
        keys = sorted({str(key) for record in records for key in record})
        raise BenchmarkError(
            f'exportTimes records do not provide a complete H2D/GPU/D2H/'
            f'host/enqueue timing matrix in {times_path}; expected '
            f'{sample_count} values per metric, got {incomplete}; keys are '
            f'{keys}')
    log_values = _parse_log(log_path)
    summaries = {name: _summarize(values) for name, values in series.items()}
    if 'throughput_qps' in log_values:
        throughput = float(log_values['throughput_qps'])
        throughput_source = 'trtexec-log'
    else:
        throughput, throughput_source = _fallback_throughput(
            records, series['host'])
    if not math.isfinite(throughput) or throughput <= 0:
        raise BenchmarkError(f'invalid throughput value: {throughput!r}')
    return {
        'sample_count': sample_count,
        'series_count': {name: len(values)
                         for name, values in series.items()},
        'metrics': summaries,
        'throughput_qps': throughput,
        'throughput_source': throughput_source,
        'log': log_values,
    }


def _relative(path: Path, output_dir: Path) -> str:
    return str(path.resolve().relative_to(output_dir.resolve()))


def _build_csv_row(model: str, spec: EngineSpec, engine_path: Path,
                   onnx_path: Path, input_identities: Sequence[Mapping[str,
                                                                       Any]],
                   engine_manifest: Mapping[str, Any], tool_version: str,
                   metrics: Mapping[str, Any], paths: Mapping[str, Path],
                   output_dir: Path, args: argparse.Namespace) -> dict:
    identity = engine_manifest['engines'][spec.name]
    build_elapsed_seconds = _finite_number(
        identity.get('build_elapsed_seconds'))
    if build_elapsed_seconds is None or build_elapsed_seconds < 0:
        raise BenchmarkError(
            f'engine manifest has no valid build time for {spec.name}; '
            'rebuild it with the current build_engine.sh')
    row = {
        'schema_version':
        _SCHEMA_VERSION,
        'model':
        model,
        'engine':
        spec.name,
        'calls_per_action':
        spec.calls_per_action,
        'engine_file':
        engine_path.name,
        'engine_sha256':
        identity.get('sha256', ''),
        'engine_bytes':
        engine_path.stat().st_size,
        'build_elapsed_seconds':
        build_elapsed_seconds,
        'built_at_utc':
        identity.get('built_at_utc', ''),
        'build_log_file':
        identity.get('build_log_file', ''),
        'onnx_file':
        onnx_path.name,
        'checkpoint_sha256':
        engine_manifest.get('checkpoint_sha256', ''),
        'tensorrt_version':
        tool_version,
        'gpu_name':
        metrics.get('log', {}).get('gpu_name', ''),
        'gpu_uuid':
        metrics.get('log', {}).get('gpu_uuid', ''),
        'compute_capability':
        metrics.get('log', {}).get('compute_capability', ''),
        'device':
        args.device,
        'input_count':
        len(input_identities),
        'input_dtypes':
        ';'.join(str(item['onnx_dtype']) for item in input_identities),
        'seed':
        args.seed,
        'warmup_ms':
        args.warmup_ms,
        'duration_seconds':
        args.duration_seconds,
        'minimum_iterations':
        args.iterations,
        'profile_iterations':
        args.profile_iterations,
        'sample_count':
        metrics['sample_count'],
        'inf_streams':
        1,
        'cuda_graph':
        False,
        'data_transfers':
        ('included' if args.include_data_transfers else 'disabled'),
        'throughput_qps':
        metrics['throughput_qps'],
        'throughput_source':
        metrics['throughput_source'],
        'times_json':
        _relative(paths['times'], output_dir),
        'profile_json':
        _relative(paths['profile'], output_dir),
        'inspector_json':
        _relative(paths['inspector'], output_dir),
        'perf_log':
        _relative(paths['perf_log'], output_dir),
        'profile_log':
        _relative(paths['profile_log'], output_dir),
        'inspector_log':
        _relative(paths['inspector_log'], output_dir),
    }
    for metric, prefix in _METRIC_COLUMNS.items():
        summary = metrics['metrics'].get(metric, {})
        for statistic in _STAT_NAMES:
            row[f'{prefix}_{statistic}_ms'] = summary.get(statistic, '')
    gpu_mean = row.get('gpu_compute_mean_ms')
    row['weighted_gpu_compute_mean_ms'] = (
        float(gpu_mean) *
        spec.calls_per_action if gpu_mean not in ('', None) else '')
    return row


def _csv_fieldnames() -> list[str]:
    fields = [
        'schema_version', 'model', 'engine', 'calls_per_action', 'engine_file',
        'engine_sha256', 'engine_bytes', 'build_elapsed_seconds',
        'built_at_utc', 'build_log_file', 'onnx_file', 'checkpoint_sha256',
        'tensorrt_version', 'gpu_name', 'gpu_uuid', 'compute_capability',
        'device', 'input_count', 'input_dtypes', 'seed', 'warmup_ms',
        'duration_seconds', 'minimum_iterations', 'profile_iterations',
        'sample_count', 'inf_streams', 'cuda_graph', 'data_transfers',
        'throughput_qps', 'throughput_source'
    ]
    for prefix in _METRIC_COLUMNS.values():
        fields.extend(f'{prefix}_{statistic}_ms' for statistic in _STAT_NAMES)
    fields.extend([
        'weighted_gpu_compute_mean_ms', 'times_json', 'profile_json',
        'inspector_json', 'perf_log', 'profile_log', 'inspector_log'
    ])
    return fields


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BenchmarkError(f'cannot write non-finite CSV value: {value}')
        return format(value, '.12g')
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = _csv_fieldnames()
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            extrasaction='raise',
            lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {name: _csv_value(row.get(name, ''))
                 for name in fields})


def _validate_args(args: argparse.Namespace) -> None:
    if args.device < 0:
        raise BenchmarkError('--device must be non-negative')
    if args.seed < 0:
        raise BenchmarkError('--seed must be non-negative')
    if args.warmup_ms < 0:
        raise BenchmarkError('--warmup-ms must be non-negative')
    if args.duration_seconds < 0:
        raise BenchmarkError('--duration-seconds must be non-negative')
    if args.iterations <= 0:
        raise BenchmarkError('--iterations must be positive')
    if args.profile_iterations <= 0:
        raise BenchmarkError('--profile-iterations must be positive')


def _benchmark(args: argparse.Namespace) -> Path:
    _validate_args(args)
    if sys.byteorder != 'little':
        raise BenchmarkError(
            'raw input serialization currently supports little-endian hosts '
            'only')
    engine_dir = args.engine_dir.expanduser().resolve()
    onnx_dir = args.onnx_dir.expanduser().resolve()
    if not engine_dir.is_dir():
        raise BenchmarkError(f'engine directory not found: {engine_dir}')
    if not onnx_dir.is_dir():
        raise BenchmarkError(f'ONNX directory not found: {onnx_dir}')
    requested_output_dir = args.output_dir.expanduser().resolve()
    if any(delimiter in str(requested_output_dir) for delimiter in ",:'"):
        raise BenchmarkError(
            '--output-dir absolute path cannot contain comma, colon, or '
            f'apostrophe because trtexec --loadInputs cannot represent it: '
            f'{requested_output_dir}')
    output_dir = _prepare_output_dir(requested_output_dir, args.force)
    if output_dir in (engine_dir, onnx_dir):
        raise BenchmarkError(
            '--output-dir must differ from --engine-dir and --onnx-dir')
    _remove_stale_file(output_dir / 'engine_benchmark.csv')
    _remove_stale_file(output_dir / 'benchmark_manifest.json')
    executable = _resolve_executable(args.trtexec)
    supported_flags, tool_version = _probe_trtexec(executable, output_dir)
    runtime_policy = _validate_cli_capabilities(supported_flags, tool_version,
                                                args.include_data_transfers)
    export_manifest, engine_manifest = _load_manifests(args.model, engine_dir,
                                                       onnx_dir)
    _validate_tool_version(engine_manifest, tool_version)
    onnx_module, np = _require_onnx_and_numpy()

    rows = []
    engine_artifacts = []
    for spec in _ENGINE_SPECS[args.model]:
        print(
            f'[{spec.name}] generating deterministic ONNX inputs', flush=True)
        engine_path, onnx_path = _engine_and_onnx_paths(
            spec, engine_dir, onnx_dir, export_manifest, engine_manifest)
        input_identities, load_inputs = _generate_engine_inputs(
            args.model, spec, onnx_path, output_dir, args.seed, onnx_module,
            np)
        engine_output = output_dir / 'engines' / spec.name
        paths = {
            'times': engine_output / 'perf' / 'times.json',
            'profile': engine_output / 'profile' / 'profile.json',
            'inspector': engine_output / 'inspector' / 'inspector.json',
        }
        for path in paths.values():
            _remove_stale_file(path)
        commands = {
            'perf':
            _perf_command(executable, engine_path, load_inputs, paths['times'],
                          args, runtime_policy),
            'profile':
            _profile_command(executable, engine_path, load_inputs,
                             paths['profile'], args, runtime_policy),
            'inspector':
            _inspector_command(executable, engine_path, paths['inspector'],
                               args),
        }
        for pass_name, argv in commands.items():
            # Every used flag was already checked as a capability group; this
            # per-command check prevents future edits from adding an unprobed
            # flag silently.
            used = {
                value[2:].split('=', 1)[0]
                for value in argv[1:] if value.startswith('--')
            }
            _require_flags(supported_flags, used, pass_name, tool_version)
            print(f'[{spec.name}] running {pass_name} pass', flush=True)
            log = _run_pass(argv, spec.name, pass_name,
                            engine_output / pass_name, output_dir)
            paths[f'{pass_name}_log'] = log
        _validate_export_json(paths['profile'], 'profile', spec.name)
        _validate_export_json(paths['inspector'], 'inspector', spec.name)
        metrics = _parse_performance(paths['times'], paths['perf_log'])
        parsed_path = engine_output / 'perf' / 'parsed_metrics.json'
        _write_json(parsed_path, metrics)
        row = _build_csv_row(args.model, spec, engine_path, onnx_path,
                             input_identities, engine_manifest, tool_version,
                             metrics, paths, output_dir, args)
        rows.append(row)
        engine_artifacts.append({
            'engine':
            spec.name,
            'calls_per_action':
            spec.calls_per_action,
            'engine_file':
            str(engine_path),
            'onnx_file':
            str(onnx_path),
            'inputs':
            input_identities,
            'times_json':
            _relative(paths['times'], output_dir),
            'profile_json':
            _relative(paths['profile'], output_dir),
            'inspector_json':
            _relative(paths['inspector'], output_dir),
            'parsed_metrics':
            _relative(parsed_path, output_dir),
        })

    csv_path = output_dir / 'engine_benchmark.csv'
    _write_csv(csv_path, rows)
    _write_json(
        output_dir / 'benchmark_manifest.json', {
            'schema_version': _SCHEMA_VERSION,
            'created_at_utc': _utc_now(),
            'model': args.model,
            'engine_dir': str(engine_dir),
            'onnx_dir': str(onnx_dir),
            'output_dir': str(output_dir),
            'trtexec': str(executable),
            'tensorrt_version': tool_version,
            'protocol': {
                'seed':
                args.seed,
                'device':
                args.device,
                'warmup_ms':
                args.warmup_ms,
                'duration_seconds':
                args.duration_seconds,
                'minimum_iterations':
                args.iterations,
                'profile_iterations':
                args.profile_iterations,
                'inf_streams':
                1,
                'cuda_graph':
                False,
                'cuda_graph_control':
                runtime_policy.cuda_graph_control,
                'data_transfers':
                ('included' if args.include_data_transfers else 'disabled'),
                'data_transfer_control':
                runtime_policy.data_transfer_control,
                'passes': ['perf', 'profile', 'inspector'],
            },
            'engine_manifest': str(engine_dir / 'engine_manifest.json'),
            'export_manifest': str(onnx_dir / 'export_manifest.json'),
            'engines': engine_artifacts,
            'engine_benchmark_csv': str(csv_path.relative_to(output_dir)),
            'engine_benchmark_csv_sha256': _sha256(csv_path),
        })
    return csv_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=('pi05', 'groot-n15'))
    parser.add_argument('--engine-dir', required=True, type=Path)
    parser.add_argument('--onnx-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument(
        '--trtexec',
        default=os.environ.get('TRTEXEC', 'trtexec'),
        help='trtexec executable (default: TRTEXEC or trtexec from PATH)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warmup-ms', type=int, default=1000)
    parser.add_argument('--duration-seconds', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--profile-iterations', type=int, default=20)
    parser.add_argument(
        '--include-data-transfers',
        action='store_true',
        help='measure H2D/D2H transfers (default: device-resident inputs)')
    parser.add_argument(
        '--force',
        action='store_true',
        help='allow overwriting known files in a non-empty output directory')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        csv_path = _benchmark(args)
    except BenchmarkError as exc:
        raise SystemExit(f'benchmark_engines: error: {exc}') from exc
    print(f'Engine benchmark CSV: {csv_path}')


if __name__ == '__main__':
    main()
