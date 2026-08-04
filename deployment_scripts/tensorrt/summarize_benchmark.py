#!/usr/bin/env python3
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
"""Validate and summarize formal and diagnostic LIBERO benchmark runs.

Four formal evaluation inputs provide success rates and end-to-end latency.
Four separate module-profile inputs provide CUDA-event module metrics.  Each
input resolves from either a direct evaluation ``run_dir`` or a
``run_libero_benchmark.sh`` group wrapper; formal inputs must also expose the
wrapper's ``telemetry.csv``.  Warm-up inference rows are excluded from every
latency statistic.  The generated figure is rendered from the emitted
``figure_source.csv`` so every plotted number has a machine-readable source.
"""

from __future__ import annotations
import argparse
import csv
import math
import os
import textwrap
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterable, Mapping, Sequence

SCHEMA_VERSION = 1

PIPELINE_COMPONENTS = (
    'preprocess_ms',
    'model_inference_ms',
    'output_transfer_ms',
    'action_denorm_ms',
)
PIPELINE_FIELD = 'model_pipeline_total_ms'
METRICS = (
    ('preprocess', 'preprocess_ms'),
    ('model_inference', 'model_inference_ms'),
    ('output_transfer', 'output_transfer_ms'),
    ('action_denorm', 'action_denorm_ms'),
    ('model_pipeline', PIPELINE_FIELD),
)
STATISTICS = ('mean', 'std', 'p50', 'p90', 'p95', 'p99')
PERCENTILES = {
    'p50': 50.0,
    'p90': 90.0,
    'p95': 95.0,
    'p99': 99.0,
}
TELEMETRY_REQUIRED_COLUMNS = {
    'gpu_index', 'gpu_uuid', 'timestamp', 'utilization_gpu_percent',
    'memory_used_mib', 'memory_total_mib', 'power_draw_w', 'temperature_gpu_c'
}
TELEMETRY_METRICS = (
    ('gpu_util', 'utilization_gpu_percent', 'pct'),
    ('memory_used', 'memory_used_mib', 'mib'),
    ('power', 'power_draw_w', 'w'),
    ('temperature', 'temperature_gpu_c', 'c'),
)


@dataclass(frozen=True)
class ModuleSpec:
    key: str
    label: str
    short_label: str
    parent: str
    hierarchy_level: int
    figure_included: bool
    color: str
    hatch: str


MODULES_BY_MODEL = {
    'pi05': (
        ModuleSpec('vision', 'Vision', 'Vision', '', 0, True, '#0072B2', '//'),
        ModuleSpec('prefix', 'Prefix', 'Prefix', '', 0, True, '#E69F00',
                   '\\\\'),
        ModuleSpec('denoise', 'Denoise', 'Denoise', '', 0, True, '#009E73',
                   'xx'),
        ModuleSpec('denoise_step', 'Denoise step', 'D-step', 'denoise', 1,
                   False, '#56B4E9', '..'),
    ),
    'groot': (
        ModuleSpec('backbone_total', 'Backbone total', 'Backbone', '', 0,
                   False, '#0072B2', '//'),
        ModuleSpec('vit', 'Vision transformer', 'ViT', 'backbone_total', 1,
                   True, '#56B4E9', '..'),
        ModuleSpec('llm', 'Language model', 'LLM', 'backbone_total', 1, True,
                   '#E69F00', '\\\\'),
        ModuleSpec('backbone_processor', 'Backbone processor', 'Processor', '',
                   0, True, '#56B4E9', '..'),
        ModuleSpec('state_encoder', 'State encoder', 'State', '', 0, True,
                   '#E69F00', '\\\\'),
        ModuleSpec('action_expert', 'Action expert', 'Expert', '', 0, True,
                   '#009E73', 'xx'),
        ModuleSpec('action_encoder', 'Action encoder', 'A-encoder',
                   'action_expert', 1, False, '#CC79A7', 'oo'),
        ModuleSpec('dit', 'Diffusion transformer', 'DiT', 'action_expert', 1,
                   False, '#D55E00', '++'),
        ModuleSpec('action_decoder', 'Action decoder', 'A-decoder',
                   'action_expert', 1, False, '#F0E442', '**'),
    ),
}


@dataclass(frozen=True)
class FigureSegmentSpec:
    key: str
    label: str
    short_label: str
    detail_label: str
    source_role: str
    source_key: str
    repeat_count: int
    detail_source_modules: str
    color: str
    hatch: str


FIGURE_SEGMENTS_BY_MODEL = {
    'pi05': (
        FigureSegmentSpec('preprocess', 'Preprocess', 'Prep', '', 'formal',
                          'preprocess', 1, '', '#A6A6A6', '..'),
        FigureSegmentSpec('vision', 'Vision', 'Vision', '',
                          'module_diagnostic', 'vision', 1, '', '#0072B2',
                          '//'),
        FigureSegmentSpec('prefix', 'Prefix', 'Prefix', '',
                          'module_diagnostic', 'prefix', 1, '', '#E69F00',
                          '\\'),
        FigureSegmentSpec('denoise', 'Denoise ×10', 'Denoise ×10', '',
                          'module_diagnostic', 'denoise', 10, 'denoise_step',
                          '#009E73', 'xx'),
        FigureSegmentSpec('output_transfer', 'Output Transfer', 'Output', '',
                          'formal', 'output_transfer', 1, '', '#CC79A7', 'oo'),
        FigureSegmentSpec('action_denorm', 'Action Denorm', 'Denorm', '',
                          'formal', 'action_denorm', 1, '', '#D55E00', '++'),
    ),
    'groot': (
        FigureSegmentSpec('preprocess', 'Preprocess', 'Prep', '', 'formal',
                          'preprocess', 1, '', '#A6A6A6', '..'),
        FigureSegmentSpec('vit', 'ViT', 'ViT', '', 'module_diagnostic', 'vit',
                          1, '', '#0072B2', '//'),
        FigureSegmentSpec('llm', 'LLM', 'LLM', '', 'module_diagnostic', 'llm',
                          1, '', '#E69F00', '\\'),
        FigureSegmentSpec('backbone_processor', 'Backbone Processor', 'B-Proc',
                          '', 'module_diagnostic', 'backbone_processor', 1, '',
                          '#56B4E9', '..'),
        FigureSegmentSpec('state_encoder', 'State Encoder', 'State', '',
                          'module_diagnostic', 'state_encoder', 1, '',
                          '#F0E442', '++'),
        FigureSegmentSpec('action_expert', 'Action Expert', 'Expert',
                          'Action Encoder → DiT → Decoder ×4',
                          'module_diagnostic', 'action_expert', 4,
                          'action_encoder;dit;action_decoder', '#009E73',
                          'xx'),
        FigureSegmentSpec('output_transfer', 'Output Transfer', 'Output', '',
                          'formal', 'output_transfer', 1, '', '#CC79A7', 'oo'),
        FigureSegmentSpec('action_denorm', 'Action Denorm', 'Denorm', '',
                          'formal', 'action_denorm', 1, '', '#D55E00', '++'),
    ),
}


@dataclass(frozen=True)
class GroupSpec:
    key: str
    model_key: str
    model_label: str
    backend: str
    backend_label: str
    cli_option: str
    input_root: Path
    run_dir: Path
    run_dir_display: str
    telemetry_path: Path | None


@dataclass
class LoadedGroup:
    spec: GroupSpec
    calls: list[dict]
    measured_calls: list[dict]
    trials: list[dict]
    trial_results: list[dict]
    source_model: str
    telemetry_rows: list[dict]
    telemetry_timezone: str


@dataclass
class LoadedModuleGroup:
    spec: GroupSpec
    calls: list[dict]
    measured_calls: list[dict]
    trials: list[dict]
    source_model: str


class BenchmarkValidationError(ValueError):
    """Raised when profiler CSV data violates the benchmark contract."""


def _required_call_columns(module_keys: Sequence[str] = ()) -> set[str]:
    """Return core columns plus model-specific diagnostic module columns."""
    columns = {
        'schema_version', 'rank', 'model', 'backend', 'task_id', 'trial_id',
        'call_index', 'trial_call_index', 'env_step', 'warmup',
        'actions_returned', 'actions_executed', *PIPELINE_COMPONENTS,
        PIPELINE_FIELD
    }
    if module_keys:
        columns.add('module_profile')
    for name in module_keys:
        columns.update((f'{name}_ms', f'{name}_calls', f'{name}_mean_ms'))
    return columns


TRIAL_REQUIRED_COLUMNS = {
    'schema_version', 'rank', 'model', 'backend', 'task_id', 'trial_id',
    'initial_state_index', 'initial_state_sha256', 'seed', 'inference_seed',
    'status', 'error_type', 'error_message', 'success', 'env_steps',
    'model_calls_total', 'model_calls_measured', 'actions_executed',
    'episode_start_unix', 'episode_end_unix', 'episode_wall_s',
    'preprocess_total_ms', 'preprocess_mean_ms', 'model_inference_total_ms',
    'model_inference_mean_ms', 'output_transfer_total_ms',
    'output_transfer_mean_ms', 'action_denorm_total_ms',
    'action_denorm_mean_ms', 'model_pipeline_total_ms',
    'model_pipeline_mean_ms', 'torch_peak_allocated_mib'
}


def _read_csv(path: Path, required: set[str]) -> tuple[list[str], list[dict]]:
    if not path.is_file():
        raise BenchmarkValidationError(f'Required CSV does not exist: {path}')
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise BenchmarkValidationError(f'CSV has no header: {path}')
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise BenchmarkValidationError(
                f'CSV has duplicate columns: {path}')
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise BenchmarkValidationError(
                f'CSV is missing required columns {missing}: {path}')
        rows = list(reader)
    if not rows:
        raise BenchmarkValidationError(f'CSV contains no data rows: {path}')
    return list(reader.fieldnames), rows


def _context(path: Path, row_number: int, field: str) -> str:
    return f'{path}:{row_number} field {field!r}'


def _parse_int(raw: str, path: Path, row_number: int, field: str) -> int:
    value = str(raw).strip()
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkValidationError(
            f'{_context(path, row_number, field)} must be an integer, got '
            f'{raw!r}') from exc


def _parse_optional_int(raw: str, path: Path, row_number: int,
                        field: str) -> int | None:
    if str(raw).strip() == '':
        return None
    return _parse_int(raw, path, row_number, field)


def _parse_float(raw: str,
                 path: Path,
                 row_number: int,
                 field: str,
                 *,
                 allow_blank: bool = False) -> float | None:
    value = str(raw).strip()
    if allow_blank and value == '':
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkValidationError(
            f'{_context(path, row_number, field)} must be numeric, got '
            f'{raw!r}') from exc
    if not math.isfinite(result):
        raise BenchmarkValidationError(
            f'{_context(path, row_number, field)} must be finite, got '
            f'{raw!r}')
    return result


def _parse_warmup(raw: str, path: Path, row_number: int) -> bool:
    value = str(raw).strip().lower()
    if value in {'1', 'true', 'yes'}:
        return True
    if value in {'0', 'false', 'no'}:
        return False
    raise BenchmarkValidationError(
        f'{_context(path, row_number, "warmup")} must be 0 or 1, got '
        f'{raw!r}')


def _nonnegative(value: float | int, path: Path, row_number: int,
                 field: str) -> None:
    if value < 0:
        raise BenchmarkValidationError(
            f'{_context(path, row_number, field)} must be non-negative, got '
            f'{value}')


def _parse_call_rows(
    path: Path, raw_rows: Sequence[dict], module_keys: Sequence[str] = ()
) -> list[dict]:  # noqa: E125
    rows = []
    integer_fields = ('schema_version', 'rank', 'task_id', 'trial_id',
                      'call_index', 'trial_call_index', 'env_step',
                      'actions_returned', 'actions_executed')
    for offset, raw in enumerate(raw_rows, start=2):
        row: dict = {
            field: _parse_int(raw[field], path, offset, field)
            for field in integer_fields
        }
        row['model'] = str(raw['model']).strip()
        row['backend'] = str(raw['backend']).strip().lower()
        row['warmup'] = _parse_warmup(raw['warmup'], path, offset)
        if module_keys:
            row['module_profile'] = _parse_int(raw['module_profile'], path,
                                               offset, 'module_profile')
            if row['module_profile'] != 1:
                raise BenchmarkValidationError(
                    f'{_context(path, offset, "module_profile")} must be 1 '
                    'for a diagnostic module-profile input')
        if not row['model']:
            raise BenchmarkValidationError(
                f'{_context(path, offset, "model")} must not be blank')
        for field in (*PIPELINE_COMPONENTS, PIPELINE_FIELD):
            row[field] = _parse_float(raw[field], path, offset, field)
            _nonnegative(row[field], path, offset, field)
        for module in module_keys:
            ms_field = f'{module}_ms'
            calls_field = f'{module}_calls'
            mean_field = f'{module}_mean_ms'
            ms = _parse_float(
                raw[ms_field], path, offset, ms_field, allow_blank=True)
            calls_raw = str(raw[calls_field]).strip()
            mean = _parse_float(
                raw[mean_field], path, offset, mean_field, allow_blank=True)
            if ms is None:
                if calls_raw != '' or mean is not None:
                    raise BenchmarkValidationError(
                        f'{path}:{offset} has a partial {module} module record'
                    )
                row[ms_field] = None
                row[calls_field] = None
                row[mean_field] = None
                continue
            calls = _parse_int(calls_raw, path, offset, calls_field)
            if calls <= 0 or mean is None:
                raise BenchmarkValidationError(
                    f'{path}:{offset} has invalid {module} calls/mean values')
            _nonnegative(ms, path, offset, ms_field)
            _nonnegative(mean, path, offset, mean_field)
            row[ms_field] = ms
            row[calls_field] = calls
            row[mean_field] = mean
        rows.append(row)
    return rows


def _parse_trial_rows(path: Path, raw_rows: Sequence[dict]) -> list[dict]:
    rows = []
    integer_fields = ('schema_version', 'rank', 'task_id', 'trial_id',
                      'initial_state_index', 'seed', 'success', 'env_steps',
                      'model_calls_total', 'model_calls_measured',
                      'actions_executed')
    float_fields = ('episode_start_unix', 'episode_end_unix', 'episode_wall_s',
                    'preprocess_total_ms', 'model_inference_total_ms',
                    'output_transfer_total_ms', 'action_denorm_total_ms',
                    'model_pipeline_total_ms', 'torch_peak_allocated_mib')
    optional_float_fields = ('preprocess_mean_ms', 'model_inference_mean_ms',
                             'output_transfer_mean_ms',
                             'action_denorm_mean_ms', 'model_pipeline_mean_ms')
    for offset, raw in enumerate(raw_rows, start=2):
        status = str(raw['status']).strip().lower()
        error_type = str(raw['error_type']).strip()
        error_message = str(raw['error_message']).strip()
        if status != 'completed':
            raise BenchmarkValidationError(
                f'{path}:{offset} has trial status={status!r}, '
                f'error_type={error_type!r}, error_message={error_message!r}; '
                'only completed trials may be summarized')
        if error_type or error_message:
            raise BenchmarkValidationError(
                f'{path}:{offset} completed trial must have blank '
                'error_type/error_message')
        row: dict = {
            field: _parse_int(raw[field], path, offset, field)
            for field in integer_fields
        }
        row['model'] = str(raw['model']).strip()
        row['backend'] = str(raw['backend']).strip().lower()
        row['status'] = status
        row['error_type'] = error_type
        row['error_message'] = error_message
        row['initial_state_sha256'] = str(
            raw['initial_state_sha256']).strip().lower()
        row['inference_seed'] = _parse_optional_int(raw['inference_seed'],
                                                    path, offset,
                                                    'inference_seed')
        if (len(row['initial_state_sha256']) != 64
                or any(char not in '0123456789abcdef'
                       for char in row['initial_state_sha256'])):
            raise BenchmarkValidationError(
                f'{_context(path, offset, "initial_state_sha256")} must be '
                'a 64-character hexadecimal SHA-256 digest')
        for field in ('initial_state_index', 'seed'):
            _nonnegative(row[field], path, offset, field)
        if row['inference_seed'] is not None:
            _nonnegative(row['inference_seed'], path, offset, 'inference_seed')
        for field in float_fields:
            row[field] = _parse_float(raw[field], path, offset, field)
            _nonnegative(row[field], path, offset, field)
        for field in optional_float_fields:
            row[field] = _parse_float(
                raw[field], path, offset, field, allow_blank=True)
            if row[field] is not None:
                _nonnegative(row[field], path, offset, field)
        if row['success'] not in (0, 1):
            raise BenchmarkValidationError(
                f'{_context(path, offset, "success")} must be 0 or 1')
        if row['episode_end_unix'] < row['episode_start_unix']:
            raise BenchmarkValidationError(
                f'{path}:{offset} episode_end_unix precedes '
                'episode_start_unix')
        rows.append(row)
    return rows


def _parse_local_telemetry_timestamp(raw: str, path: Path,
                                     row_number: int) -> float:
    value = str(raw).strip()
    formats = (
        '%Y/%m/%d %H:%M:%S.%f',
        '%Y/%m/%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
    )
    for timestamp_format in formats:
        try:
            # NVIDIA emits a timezone-naive server-local timestamp.  Calling
            # timestamp() on a naive datetime deliberately applies the local
            # timezone configured on the server, including its DST rules.
            return datetime.strptime(value, timestamp_format).timestamp()
        except ValueError:
            continue
    raise BenchmarkValidationError(
        f'{_context(path, row_number, "timestamp")} has unsupported '
        f'nvidia-smi local timestamp {raw!r}')


def _local_timezone_label() -> str:
    current = datetime.now().astimezone()
    offset = current.strftime('%z')
    rendered_offset = (f'{offset[:3]}:{offset[3:]}'
                       if len(offset) == 5 else offset)
    return f'{current.tzname() or "local"} (UTC{rendered_offset})'


def _parse_telemetry_rows(path: Path) -> tuple[list[dict], str]:
    _, raw_rows = _read_csv(path, TELEMETRY_REQUIRED_COLUMNS)
    rows = []
    for offset, raw in enumerate(raw_rows, start=2):
        row = {
            'gpu_index':
            _parse_int(raw['gpu_index'], path, offset, 'gpu_index'),
            'gpu_uuid':
            str(raw['gpu_uuid']).strip(),
            'timestamp':
            str(raw['timestamp']).strip(),
            'timestamp_unix':
            _parse_local_telemetry_timestamp(raw['timestamp'], path, offset),
        }
        if not row['gpu_uuid']:
            raise BenchmarkValidationError(
                f'{_context(path, offset, "gpu_uuid")} must not be blank')
        for field in ('utilization_gpu_percent', 'memory_used_mib',
                      'memory_total_mib', 'power_draw_w', 'temperature_gpu_c'):
            row[field] = _parse_float(raw[field], path, offset, field)
            _nonnegative(row[field], path, offset, field)
        if row['utilization_gpu_percent'] > 100.0:
            raise BenchmarkValidationError(
                f'{_context(path, offset, "utilization_gpu_percent")} '
                'must not exceed 100')
        if row['memory_used_mib'] > row['memory_total_mib']:
            raise BenchmarkValidationError(
                f'{path}:{offset} memory_used_mib exceeds memory_total_mib')
        if row['temperature_gpu_c'] > 150.0:
            raise BenchmarkValidationError(
                f'{_context(path, offset, "temperature_gpu_c")} exceeds '
                'the validation limit of 150 C')
        rows.append(row)

    if any(rows[index]['timestamp_unix'] > rows[index + 1]['timestamp_unix']
           for index in range(len(rows) - 1)):
        raise BenchmarkValidationError(
            f'Telemetry timestamps must be non-decreasing: {path}')
    gpu_indices = {row['gpu_index'] for row in rows}
    gpu_uuids = {row['gpu_uuid'] for row in rows}
    if len(gpu_indices) != 1 or len(gpu_uuids) != 1:
        raise BenchmarkValidationError(
            f'Telemetry must describe exactly one GPU, found indices '
            f'{sorted(gpu_indices)} and UUIDs {sorted(gpu_uuids)}: {path}')
    return rows, _local_timezone_label()


def _isclose(left: float, right: float, abs_tol_ms: float,
             rel_tol: float) -> bool:
    return math.isclose(left, right, abs_tol=abs_tol_ms, rel_tol=rel_tol)


def _assert_close(label: str, left: float, right: float, abs_tol_ms: float,
                  rel_tol: float) -> None:
    if not _isclose(left, right, abs_tol_ms, rel_tol):
        raise BenchmarkValidationError(
            f'{label}: {left:.9f} != {right:.9f} '
            f'(abs_tol={abs_tol_ms}, rel_tol={rel_tol})')


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise BenchmarkValidationError(
            'Cannot compute a percentile of no data')
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _stats(values: Iterable[float]) -> dict[str, float | None]:
    data = [float(value) for value in values]
    if not data:
        return {'total': 0.0, **{statistic: None for statistic in STATISTICS}}
    result: dict[str, float | None] = {
        'total': sum(data),
        'mean': fmean(data),
        'std': pstdev(data),
    }
    result.update({
        name: _percentile(data, percentile)
        for name, percentile in PERCENTILES.items()
    })
    return result


def _metric_stats(calls: Sequence[dict],
                  field: str) -> dict[str, float | None]:
    return _stats(row[field] for row in calls)


def _add_telemetry_summary(row: dict, samples: Sequence[dict]) -> None:
    if not samples:
        raise BenchmarkValidationError(
            'Cannot summarize an empty telemetry sample set')
    row['telemetry_samples'] = len(samples)
    for prefix, field, unit in TELEMETRY_METRICS:
        summary = _stats(sample[field] for sample in samples)
        for statistic in STATISTICS:
            row[f'{prefix}_{statistic}_{unit}'] = summary[statistic]
        row[f'{prefix}_max_{unit}'] = max(sample[field] for sample in samples)


def _trial_metric_source(prefix: str) -> tuple[str, str]:
    if prefix == 'model_pipeline':
        return 'model_pipeline_total_ms', 'model_pipeline_mean_ms'
    return f'{prefix}_total_ms', f'{prefix}_mean_ms'


def _validate_trial_aggregate(spec: GroupSpec, trial: dict,
                              all_calls: Sequence[dict],
                              measured_calls: Sequence[dict],
                              abs_tol_ms: float, rel_tol: float) -> None:
    identity = (f'{spec.key} task={trial["task_id"]} '
                f'trial={trial["trial_id"]}')
    if trial['model_calls_total'] != len(all_calls):
        raise BenchmarkValidationError(
            f'{identity}: trials.csv model_calls_total='
            f'{trial["model_calls_total"]}, calls CSV has {len(all_calls)}')
    if trial['model_calls_measured'] != len(measured_calls):
        raise BenchmarkValidationError(
            f'{identity}: trials.csv model_calls_measured='
            f'{trial["model_calls_measured"]}, filtered calls CSV has '
            f'{len(measured_calls)}')
    executed = sum(row['actions_executed'] for row in all_calls)
    if trial['actions_executed'] != executed:
        raise BenchmarkValidationError(
            f'{identity}: trials.csv actions_executed='
            f'{trial["actions_executed"]}, calls CSV has {executed}')
    for prefix, call_field in METRICS:
        values = [row[call_field] for row in measured_calls]
        aggregate = _stats(values)
        total_field, mean_field = _trial_metric_source(prefix)
        _assert_close(f'{identity} {total_field}', trial[total_field],
                      aggregate['total'], abs_tol_ms, rel_tol)
        source_mean = trial[mean_field]
        if aggregate['mean'] is None:
            if source_mean is not None:
                raise BenchmarkValidationError(
                    f'{identity}: {mean_field} must be blank with no measured '
                    'calls')
        elif source_mean is None:
            raise BenchmarkValidationError(
                f'{identity}: {mean_field} is blank despite measured calls')
        else:
            _assert_close(f'{identity} {mean_field}', source_mean,
                          aggregate['mean'], abs_tol_ms, rel_tol)


def _build_trial_result(spec: GroupSpec, source_model: str, trial: dict,
                        all_calls: Sequence[dict],
                        measured_calls: Sequence[dict],
                        telemetry_samples: Sequence[dict],
                        telemetry_timezone: str) -> dict:
    if spec.telemetry_path is None:
        raise AssertionError('Formal group has no telemetry path')
    row = {
        'group_id':
        spec.key,
        'model_family':
        spec.model_label,
        'backend':
        spec.backend,
        'source_model':
        source_model,
        'source_role':
        'formal',
        'source_input':
        spec.cli_option,
        'source_run_dir':
        str(spec.run_dir),
        'source_inference_calls':
        'time_profile/inference_calls.csv',
        'source_trials':
        'time_profile/trials.csv',
        'source_telemetry':
        'telemetry.csv',
        'source_telemetry_path':
        str(spec.telemetry_path),
        'telemetry_timestamp_timezone':
        telemetry_timezone,
        'rank':
        trial['rank'],
        'task_id':
        trial['task_id'],
        'trial_id':
        trial['trial_id'],
        'initial_state_index':
        trial['initial_state_index'],
        'initial_state_sha256':
        trial['initial_state_sha256'],
        'seed':
        trial['seed'],
        'inference_seed':
        trial['inference_seed'],
        'status':
        trial['status'],
        'error_type':
        trial['error_type'],
        'error_message':
        trial['error_message'],
        'success':
        trial['success'],
        'env_steps':
        trial['env_steps'],
        'model_calls_total':
        len(all_calls),
        'warmup_calls_ignored':
        len(all_calls) - len(measured_calls),
        'model_calls_measured':
        len(measured_calls),
        'actions_executed':
        trial['actions_executed'],
        'actions_executed_measured':
        sum(item['actions_executed'] for item in measured_calls),
        'episode_start_unix':
        trial['episode_start_unix'],
        'episode_end_unix':
        trial['episode_end_unix'],
        'episode_wall_s':
        trial['episode_wall_s'],
        'torch_peak_allocated_mib':
        trial['torch_peak_allocated_mib'],
        '_telemetry_samples':
        list(telemetry_samples),
    }
    for prefix, field in METRICS:
        summary = _metric_stats(measured_calls, field)
        for statistic in ('total', *STATISTICS):
            row[f'{prefix}_{statistic}_ms'] = summary[statistic]
    _add_telemetry_summary(row, telemetry_samples)
    return row


def _validate_group(spec: GroupSpec, calls: list[dict], trials: list[dict],
                    telemetry_rows: list[dict], telemetry_timezone: str,
                    expected_trials: int, expected_tasks: int,
                    abs_tol_ms: float, rel_tol: float) -> LoadedGroup:
    if len(trials) != expected_trials:
        raise BenchmarkValidationError(
            f'{spec.key}: expected {expected_trials} trial rows, found '
            f'{len(trials)}')
    task_ids = sorted({row['task_id'] for row in trials})
    if len(task_ids) != expected_tasks:
        raise BenchmarkValidationError(
            f'{spec.key}: expected {expected_tasks} tasks, found '
            f'{len(task_ids)} ({task_ids})')
    if expected_trials % expected_tasks != 0:
        raise BenchmarkValidationError(
            'Expected trials must be divisible by expected tasks')
    expected_per_task = expected_trials // expected_tasks
    per_task = defaultdict(int)
    trial_by_id = {}
    for row in trials:
        key = (row['task_id'], row['trial_id'])
        if key in trial_by_id:
            raise BenchmarkValidationError(
                f'{spec.key}: duplicate trial identity {key}')
        trial_by_id[key] = row
        per_task[row['task_id']] += 1
    bad_counts = {
        task: count
        for task, count in per_task.items() if count != expected_per_task
    }
    if bad_counts:
        raise BenchmarkValidationError(
            f'{spec.key}: expected {expected_per_task} trials per task, got '
            f'{bad_counts}')

    call_ids = set()
    all_calls_by_trial: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in calls:
        call_id = (row['rank'], row['call_index'])
        if call_id in call_ids:
            raise BenchmarkValidationError(
                f'{spec.key}: duplicate call identity {call_id}')
        call_ids.add(call_id)
        trial_key = (row['task_id'], row['trial_id'])
        if trial_key not in trial_by_id:
            raise BenchmarkValidationError(
                f'{spec.key}: inference call references unknown trial '
                f'{trial_key}')
        if row['rank'] != trial_by_id[trial_key]['rank']:
            raise BenchmarkValidationError(
                f'{spec.key}: rank mismatch for trial {trial_key}')
        all_calls_by_trial[trial_key].append(row)

    schema_versions = {row['schema_version'] for row in [*calls, *trials]}
    if schema_versions != {SCHEMA_VERSION}:
        raise BenchmarkValidationError(
            f'{spec.key}: expected schema_version={SCHEMA_VERSION}, found '
            f'{sorted(schema_versions)}')
    backends = {row['backend'] for row in [*calls, *trials]}
    if backends != {spec.backend}:
        raise BenchmarkValidationError(
            f'{spec.key}: expected backend={spec.backend!r}, found '
            f'{sorted(backends)}')
    models = {row['model'] for row in [*calls, *trials]}
    if len(models) != 1:
        raise BenchmarkValidationError(
            f'{spec.key}: model column is inconsistent: {sorted(models)}')
    source_model = next(iter(models))

    measured_calls = [row for row in calls if not row['warmup']]
    if not measured_calls:
        raise BenchmarkValidationError(
            f'{spec.key}: no inference calls remain after warm-up filtering')
    for row in measured_calls:
        expected_pipeline = sum(row[field] for field in PIPELINE_COMPONENTS)
        _assert_close(
            f'{spec.key} call rank={row["rank"]} index={row["call_index"]} '
            f'{PIPELINE_FIELD}', row[PIPELINE_FIELD], expected_pipeline,
            abs_tol_ms, rel_tol)
    trial_results = []
    telemetry_timestamps = [
        sample['timestamp_unix'] for sample in telemetry_rows
    ]
    for trial_key, trial in sorted(trial_by_id.items()):
        trial_calls = sorted(
            all_calls_by_trial.get(trial_key, []),
            key=lambda row: (row['trial_call_index'], row['call_index']))
        measured = [row for row in trial_calls if not row['warmup']]
        _validate_trial_aggregate(spec, trial, trial_calls, measured,
                                  abs_tol_ms, rel_tol)
        telemetry_left = bisect_left(telemetry_timestamps,
                                     trial['episode_start_unix'])
        telemetry_right = bisect_right(telemetry_timestamps,
                                       trial['episode_end_unix'])
        trial_telemetry = telemetry_rows[telemetry_left:telemetry_right]
        if not trial_telemetry:
            raise BenchmarkValidationError(
                f'{spec.key} task={trial["task_id"]} '
                f'trial={trial["trial_id"]}: no telemetry sample falls '
                'within episode_start_unix/episode_end_unix')
        trial_results.append(
            _build_trial_result(spec, source_model, trial, trial_calls,
                                measured, trial_telemetry, telemetry_timezone))
    return LoadedGroup(
        spec=spec,
        calls=calls,
        measured_calls=measured_calls,
        trials=trials,
        trial_results=trial_results,
        source_model=source_model,
        telemetry_rows=telemetry_rows,
        telemetry_timezone=telemetry_timezone)


def _load_group(spec: GroupSpec, expected_trials: int, expected_tasks: int,
                abs_tol_ms: float, rel_tol: float) -> LoadedGroup:
    if spec.telemetry_path is None:
        raise BenchmarkValidationError(
            f'{spec.cli_option}: formal input must be a benchmark group '
            'wrapper containing telemetry.csv; a direct eval run without '
            'telemetry cannot produce trial GPU statistics')
    profile_dir = spec.run_dir / 'time_profile'
    call_path = profile_dir / 'inference_calls.csv'
    trial_path = profile_dir / 'trials.csv'
    _, raw_calls = _read_csv(call_path, _required_call_columns())
    _, raw_trials = _read_csv(trial_path, TRIAL_REQUIRED_COLUMNS)
    calls = _parse_call_rows(call_path, raw_calls)
    trials = _parse_trial_rows(trial_path, raw_trials)
    telemetry_rows, telemetry_timezone = _parse_telemetry_rows(
        spec.telemetry_path)
    return _validate_group(spec, calls, trials, telemetry_rows,
                           telemetry_timezone, expected_trials, expected_tasks,
                           abs_tol_ms, rel_tol)


def _validate_module_group(spec: GroupSpec, calls: list[dict],
                           trials: list[dict], abs_tol_ms: float,
                           rel_tol: float) -> LoadedModuleGroup:
    """Validate a diagnostic run without imposing formal trial counts."""
    trial_by_id = {}
    for trial in trials:
        key = (trial['task_id'], trial['trial_id'])
        if key in trial_by_id:
            raise BenchmarkValidationError(
                f'{spec.key} module profile: duplicate trial identity {key}')
        trial_by_id[key] = trial

    call_ids = set()
    calls_by_trial: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for call in calls:
        call_id = (call['rank'], call['call_index'])
        if call_id in call_ids:
            raise BenchmarkValidationError(
                f'{spec.key} module profile: duplicate call identity '
                f'{call_id}')
        call_ids.add(call_id)
        trial_key = (call['task_id'], call['trial_id'])
        if trial_key not in trial_by_id:
            raise BenchmarkValidationError(
                f'{spec.key} module profile: inference call references '
                f'unknown trial {trial_key}')
        if call['rank'] != trial_by_id[trial_key]['rank']:
            raise BenchmarkValidationError(
                f'{spec.key} module profile: rank mismatch for trial '
                f'{trial_key}')
        calls_by_trial[trial_key].append(call)

    schema_versions = {row['schema_version'] for row in [*calls, *trials]}
    if schema_versions != {SCHEMA_VERSION}:
        raise BenchmarkValidationError(
            f'{spec.key} module profile: expected schema_version='
            f'{SCHEMA_VERSION}, found {sorted(schema_versions)}')
    backends = {row['backend'] for row in [*calls, *trials]}
    if backends != {spec.backend}:
        raise BenchmarkValidationError(
            f'{spec.key} module profile: expected backend={spec.backend!r}, '
            f'found {sorted(backends)}')
    models = {row['model'] for row in [*calls, *trials]}
    if len(models) != 1:
        raise BenchmarkValidationError(
            f'{spec.key} module profile: model column is inconsistent: '
            f'{sorted(models)}')
    source_model = next(iter(models))

    measured_calls = [row for row in calls if not row['warmup']]
    if not measured_calls:
        raise BenchmarkValidationError(
            f'{spec.key} module profile: no inference calls remain after '
            'warm-up filtering')
    for call in measured_calls:
        expected_pipeline = sum(call[field] for field in PIPELINE_COMPONENTS)
        _assert_close(
            f'{spec.key} module profile call rank={call["rank"]} '
            f'index={call["call_index"]} {PIPELINE_FIELD}',
            call[PIPELINE_FIELD], expected_pipeline, abs_tol_ms, rel_tol)

    for trial_key, trial in sorted(trial_by_id.items()):
        trial_calls = sorted(
            calls_by_trial.get(trial_key, []),
            key=lambda row: (row['trial_call_index'], row['call_index']))
        measured = [row for row in trial_calls if not row['warmup']]
        _validate_trial_aggregate(spec, trial, trial_calls, measured,
                                  abs_tol_ms, rel_tol)

    for module in MODULES_BY_MODEL[spec.model_key]:
        present = [
            call for call in measured_calls
            if call[f'{module.key}_ms'] is not None
        ]
        if module.figure_included and len(present) != len(measured_calls):
            raise BenchmarkValidationError(
                f'{spec.key} module profile: required figure module '
                f'{module.key!r} is present in {len(present)}/'
                f'{len(measured_calls)} measured calls')
        if present and len(present) != len(measured_calls):
            raise BenchmarkValidationError(
                f'{spec.key} module profile: module {module.key!r} is only '
                f'present in {len(present)}/{len(measured_calls)} measured '
                'calls')
        for call in present:
            expected_mean = (
                call[f'{module.key}_ms'] / call[f'{module.key}_calls'])
            _assert_close(
                f'{spec.key} module profile call index={call["call_index"]} '
                f'{module.key}_mean_ms', call[f'{module.key}_mean_ms'],
                expected_mean, abs_tol_ms, rel_tol)

    repeated_modules = ((('denoise_step', ),
                         10) if spec.model_key == 'pi05' else
                        (('action_encoder', 'dit', 'action_decoder'), 4))
    module_keys, expected_repeats = repeated_modules
    for module_key in module_keys:
        bad = [
            call['call_index'] for call in measured_calls
            if call[f'{module_key}_calls'] != expected_repeats
        ]
        if bad:
            raise BenchmarkValidationError(
                f'{spec.key} module profile: {module_key} must execute '
                f'exactly {expected_repeats} times per measured inference; '
                f'{len(bad)} calls violate the figure contract')

    return LoadedModuleGroup(
        spec=spec,
        calls=calls,
        measured_calls=measured_calls,
        trials=trials,
        source_model=source_model)


def _load_module_group(spec: GroupSpec, abs_tol_ms: float,
                       rel_tol: float) -> LoadedModuleGroup:
    profile_dir = spec.run_dir / 'time_profile'
    call_path = profile_dir / 'inference_calls.csv'
    trial_path = profile_dir / 'trials.csv'
    module_keys = tuple(module.key
                        for module in MODULES_BY_MODEL[spec.model_key])
    _, raw_calls = _read_csv(call_path, _required_call_columns(module_keys))
    _, raw_trials = _read_csv(trial_path, TRIAL_REQUIRED_COLUMNS)
    calls = _parse_call_rows(call_path, raw_calls, module_keys)
    trials = _parse_trial_rows(trial_path, raw_trials)
    return _validate_module_group(spec, calls, trials, abs_tol_ms, rel_tol)


def _add_distribution(row: dict,
                      prefix: str,
                      values: Iterable[float],
                      suffix: str = '') -> None:
    summary = _stats(values)
    for statistic in STATISTICS:
        row[f'{prefix}_{statistic}{suffix}'] = summary[statistic]


def _build_module_rows(group: LoadedModuleGroup) -> list[dict]:
    result = []
    for order, module in enumerate(MODULES_BY_MODEL[group.spec.model_key]):
        ms_field = f'{module.key}_ms'
        calls_field = f'{module.key}_calls'
        present = [
            row for row in group.measured_calls if row[ms_field] is not None
        ]
        if not present:
            continue
        if len(present) != len(group.measured_calls):
            raise BenchmarkValidationError(
                f'{group.spec.key}: module {module.key!r} is only present in '
                f'{len(present)}/{len(group.measured_calls)} measured calls')
        elapsed = [row[ms_field] for row in present]
        invocations = [row[calls_field] for row in present]
        per_invocation = [row[ms_field] / row[calls_field] for row in present]
        total_invocations = sum(invocations)
        summary = _stats(elapsed)
        row = {
            'group_id': group.spec.key,
            'model_family': group.spec.model_label,
            'backend': group.spec.backend,
            'module_order': order,
            'module_key': module.key,
            'module_label': module.label,
            'parent_module': module.parent,
            'hierarchy_level': module.hierarchy_level,
            'figure_included': int(module.figure_included),
            'measured_inference_calls': len(present),
            'module_event_calls_total': total_invocations,
            'source_model': group.source_model,
            'source_role': 'module_diagnostic',
            'source_input': group.spec.cli_option,
            'source_run_dir': str(group.spec.run_dir),
            'source_inference_calls': 'time_profile/inference_calls.csv',
            'source_trials': 'time_profile/trials.csv',
            'warmup_filter': 'warmup != 1',
        }
        for statistic in STATISTICS:
            row[f'{statistic}_ms'] = summary[statistic]
        _add_distribution(row, 'invocations_per_inference', invocations)
        _add_distribution(row, 'per_invocation', per_invocation, '_ms')
        result.append(row)
    return result


def _build_task_rows(group: LoadedGroup) -> list[dict]:
    result = []
    task_ids = sorted({row['task_id'] for row in group.trials})
    for task_id in task_ids:
        trials = [
            row for row in group.trial_results if row['task_id'] == task_id
        ]
        calls = [
            row for row in group.measured_calls if row['task_id'] == task_id
        ]
        telemetry_samples = [
            sample for trial in trials
            for sample in trial['_telemetry_samples']
        ]
        successes = sum(row['success'] for row in trials)
        row = {
            'group_id':
            group.spec.key,
            'model_family':
            group.spec.model_label,
            'backend':
            group.spec.backend,
            'source_role':
            'formal',
            'source_input':
            group.spec.cli_option,
            'source_run_dir':
            str(group.spec.run_dir),
            'source_inference_calls':
            'time_profile/inference_calls.csv',
            'source_trials':
            'time_profile/trials.csv',
            'source_telemetry':
            'telemetry.csv',
            'source_telemetry_path':
            str(group.spec.telemetry_path),
            'telemetry_timestamp_timezone':
            group.telemetry_timezone,
            'task_id':
            task_id,
            'trials':
            len(trials),
            'successes':
            successes,
            'success_rate_pct':
            successes / len(trials) * 100.0,
            'measured_inference_calls':
            len(calls),
            'warmup_calls_ignored':
            sum(item['warmup_calls_ignored'] for item in trials),
            'actions_executed':
            sum(item['actions_executed'] for item in trials),
        }
        for prefix, field in METRICS:
            _add_distribution(row, prefix, (item[field] for item in calls),
                              '_ms')
        _add_distribution(row, 'episode_wall',
                          (item['episode_wall_s'] for item in trials), '_s')
        _add_distribution(row, 'torch_peak_allocated',
                          (item['torch_peak_allocated_mib']
                           for item in trials), '_mib')
        _add_distribution(row, 'pipeline_total_per_trial',
                          (item['model_pipeline_total_ms'] for item in trials),
                          '_ms')
        _add_telemetry_summary(row, telemetry_samples)
        result.append(row)
    return result


def _build_overall_row(group: LoadedGroup) -> dict:
    successes = sum(row['success'] for row in group.trials)
    telemetry_samples = [
        sample for trial in group.trial_results
        for sample in trial['_telemetry_samples']
    ]
    row = {
        'group_id': group.spec.key,
        'model_family': group.spec.model_label,
        'backend': group.spec.backend,
        'source_model': group.source_model,
        'source_role': 'formal',
        'trial_count': len(group.trials),
        'task_count': len({item['task_id']
                           for item in group.trials}),
        'measured_inference_calls': len(group.measured_calls),
        'warmup_calls_ignored': len(group.calls) - len(group.measured_calls),
        'successes': successes,
        'success_rate_pct': successes / len(group.trials) * 100.0,
        'actions_executed':
        sum(item['actions_executed'] for item in group.trials),
        'source_input': group.spec.cli_option,
        'source_run_dir': str(group.spec.run_dir),
        'source_inference_calls': 'time_profile/inference_calls.csv',
        'source_trials': 'time_profile/trials.csv',
        'source_telemetry': 'telemetry.csv',
        'source_telemetry_path': str(group.spec.telemetry_path),
        'telemetry_timestamp_timezone': group.telemetry_timezone,
    }
    for prefix, field in METRICS:
        _add_distribution(row, prefix,
                          (item[field] for item in group.measured_calls),
                          '_ms')
    _add_distribution(row, 'episode_wall',
                      (item['episode_wall_s'] for item in group.trials), '_s')
    _add_distribution(row, 'torch_peak_allocated',
                      (item['torch_peak_allocated_mib']
                       for item in group.trials), '_mib')
    _add_distribution(row, 'pipeline_total_per_trial',
                      (item['model_pipeline_total_ms']
                       for item in group.trial_results), '_ms')
    _add_telemetry_summary(row, telemetry_samples)
    row['model_speedup_vs_pytorch_x'] = None
    row['pipeline_speedup_vs_pytorch_x'] = None
    return row


def _apply_speedups(overall_rows: list[dict]) -> None:
    by_model: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in overall_rows:
        by_model[row['model_family']][row['backend']] = row
    for model, backends in by_model.items():
        if set(backends) != {'pytorch', 'tensorrt'}:
            raise BenchmarkValidationError(
                f'{model}: expected PyTorch and TensorRT rows, found '
                f'{sorted(backends)}')
        pytorch = backends['pytorch']
        tensorrt = backends['tensorrt']
        for metric, output in (('model_inference_mean_ms',
                                'model_speedup_vs_pytorch_x'),
                               ('model_pipeline_mean_ms',
                                'pipeline_speedup_vs_pytorch_x')):
            baseline = pytorch[metric]
            optimized = tensorrt[metric]
            if baseline is None or optimized is None or optimized <= 0:
                raise BenchmarkValidationError(
                    f'{model}: cannot compute speedup from {metric}')
            speedup = baseline / optimized
            pytorch[output] = 1.0
            tensorrt[output] = speedup


def _apply_module_comparisons(module_rows: list[dict],
                              overall_rows: Sequence[dict]) -> None:
    """Attach paired speedups, shares, and diagnostic contribution values."""
    repeated_children = {
        'denoise_step', 'action_encoder', 'dit', 'action_decoder'
    }
    overall_index = {(row['model_family'], row['backend']): row
                     for row in overall_rows}
    module_index = {(row['model_family'], row['module_key'], row['backend']):
                    row
                    for row in module_rows}
    pair_keys = {(row['model_family'], row['module_key'])
                 for row in module_rows}
    for model_family, module_key in sorted(pair_keys):
        pair = {
            backend: module_index.get((model_family, module_key, backend))
            for backend in ('pytorch', 'tensorrt')
        }
        if pair['pytorch'] is None or pair['tensorrt'] is None:
            raise BenchmarkValidationError(
                f'{model_family}/{module_key}: module comparison requires '
                'both PyTorch and TensorRT rows')
        scope = ('per_invocation'
                 if module_key in repeated_children else 'per_inference')
        comparison_field = ('per_invocation_mean_ms'
                            if scope == 'per_invocation' else 'mean_ms')
        baseline_comparison = pair['pytorch'][comparison_field]
        optimized_comparison = pair['tensorrt'][comparison_field]
        if (baseline_comparison is None or optimized_comparison is None
                or baseline_comparison <= 0 or optimized_comparison <= 0):
            raise BenchmarkValidationError(
                f'{model_family}/{module_key}: cannot compute module '
                f'speedup from {comparison_field}')
        baseline_aggregate = pair['pytorch']['mean_ms']
        optimized_aggregate = pair['tensorrt']['mean_ms']
        pipeline_savings = (
            overall_index[(model_family, 'pytorch')]['model_pipeline_mean_ms']
            - overall_index[(model_family,
                             'tensorrt')]['model_pipeline_mean_ms'])
        module_savings = baseline_aggregate - optimized_aggregate

        for backend, row in pair.items():
            formal = overall_index[(model_family, backend)]
            row['comparison_scope'] = scope
            row['comparison_mean_ms'] = row[comparison_field]
            row['formal_model_inference_share_pct'] = (
                row['mean_ms'] / formal['model_inference_mean_ms'] * 100.0)
            row['formal_pipeline_share_pct'] = (
                row['mean_ms'] / formal['model_pipeline_mean_ms'] * 100.0)
            if backend == 'pytorch':
                row['speedup_vs_pytorch_x'] = 1.0
                row['latency_reduction_vs_pytorch_pct'] = 0.0
                row['module_latency_saved_vs_pytorch_ms'] = 0.0
                row['pipeline_savings_contribution_pct'] = None
            else:
                row['speedup_vs_pytorch_x'] = (
                    baseline_comparison / optimized_comparison)
                row['latency_reduction_vs_pytorch_pct'] = (
                    (baseline_comparison - optimized_comparison) /
                    baseline_comparison * 100.0)
                row['module_latency_saved_vs_pytorch_ms'] = module_savings
                row['pipeline_savings_contribution_pct'] = (
                    module_savings / pipeline_savings *
                    100.0 if not math.isclose(pipeline_savings, 0.0) else None)


TELEMETRY_OUTPUT_FIELDS = [
    'telemetry_samples', *[
        f'{prefix}_{statistic}_{unit}' for prefix, _, unit in TELEMETRY_METRICS
        for statistic in (*STATISTICS, 'max')
    ]
]
TELEMETRY_SOURCE_FIELDS = [
    'source_telemetry', 'source_telemetry_path', 'telemetry_timestamp_timezone'
]

TRIAL_RESULT_FIELDS = [
    'group_id', 'model_family', 'backend', 'source_model', 'source_role',
    'source_input', 'source_run_dir', 'source_inference_calls',
    'source_trials', *TELEMETRY_SOURCE_FIELDS, 'rank', 'task_id', 'trial_id',
    'initial_state_index', 'initial_state_sha256', 'seed', 'inference_seed',
    'status', 'error_type', 'error_message', 'success', 'env_steps',
    'model_calls_total', 'warmup_calls_ignored', 'model_calls_measured',
    'actions_executed', 'actions_executed_measured', 'episode_start_unix',
    'episode_end_unix', 'episode_wall_s', 'torch_peak_allocated_mib', *[
        f'{prefix}_{statistic}_ms' for prefix, _ in METRICS
        for statistic in ('total', *STATISTICS)
    ], *TELEMETRY_OUTPUT_FIELDS
]

TASK_SUMMARY_FIELDS = [
    'group_id', 'model_family', 'backend', 'source_role', 'source_input',
    'source_run_dir', 'source_inference_calls', 'source_trials',
    *TELEMETRY_SOURCE_FIELDS, 'task_id', 'trials', 'successes',
    'success_rate_pct', 'measured_inference_calls', 'warmup_calls_ignored',
    'actions_executed', *[
        f'{prefix}_{statistic}_ms' for prefix, _ in METRICS
        for statistic in STATISTICS
    ], *[f'episode_wall_{statistic}_s' for statistic in STATISTICS],
    *[f'torch_peak_allocated_{statistic}_mib' for statistic in STATISTICS],
    *[f'pipeline_total_per_trial_{statistic}_ms'
      for statistic in STATISTICS], *TELEMETRY_OUTPUT_FIELDS
]

OVERALL_FIELDS = [
    'group_id', 'model_family', 'backend', 'source_model', 'trial_count',
    'task_count', 'measured_inference_calls', 'warmup_calls_ignored',
    'successes', 'success_rate_pct', 'actions_executed', 'source_role',
    'source_input', 'source_run_dir', 'source_inference_calls',
    'source_trials', *TELEMETRY_SOURCE_FIELDS, *[
        f'{prefix}_{statistic}_ms' for prefix, _ in METRICS
        for statistic in STATISTICS
    ], *[f'episode_wall_{statistic}_s' for statistic in STATISTICS],
    *[f'torch_peak_allocated_{statistic}_mib' for statistic in STATISTICS],
    *[f'pipeline_total_per_trial_{statistic}_ms'
      for statistic in STATISTICS], *TELEMETRY_OUTPUT_FIELDS,
    'model_speedup_vs_pytorch_x', 'pipeline_speedup_vs_pytorch_x'
]

MODULE_FIELDS = [
    'group_id', 'model_family', 'backend', 'module_order', 'module_key',
    'module_label', 'parent_module', 'hierarchy_level', 'figure_included',
    'measured_inference_calls', 'module_event_calls_total', 'source_model',
    'source_role', 'source_input', 'source_run_dir', 'source_inference_calls',
    'source_trials', 'warmup_filter',
    *[f'{statistic}_ms' for statistic in STATISTICS],
    *[f'invocations_per_inference_{statistic}' for statistic in STATISTICS],
    *[f'per_invocation_{statistic}_ms'
      for statistic in STATISTICS], 'comparison_scope', 'comparison_mean_ms',
    'speedup_vs_pytorch_x', 'latency_reduction_vs_pytorch_pct',
    'module_latency_saved_vs_pytorch_ms', 'formal_model_inference_share_pct',
    'formal_pipeline_share_pct', 'pipeline_savings_contribution_pct'
]

FIGURE_SOURCE_FIELDS = [
    'figure_id', 'panel_order', 'panel_label', 'group_id', 'model_family',
    'backend_order', 'backend', 'row_label', 'segment_order', 'segment_key',
    'segment_label', 'segment_short_label', 'segment_detail_label',
    'segment_repeat_count', 'detail_source_modules', 'source_role',
    'source_metric_key', 'color_hex', 'hatch',
    *[f'segment_{statistic}_ms' for statistic in STATISTICS],
    *[f'model_inference_{statistic}_ms' for statistic in STATISTICS],
    *[f'model_pipeline_{statistic}_ms'
      for statistic in STATISTICS], 'model_speedup_vs_pytorch_x',
    'pipeline_speedup_vs_pytorch_x', 'formal_measured_inference_calls',
    'formal_trial_count', 'module_measured_inference_calls',
    'module_trial_count', 'formal_source_input', 'formal_source_run_dir',
    'module_source_input', 'module_source_run_dir', 'segment_source_input',
    'segment_source_run_dir', 'segment_source_inference_calls',
    'formal_source_inference_calls', 'formal_source_trials',
    'module_source_inference_calls', 'module_source_trials', 'warmup_filter'
]


def _write_csv(path: Path, fields: Sequence[str],
               rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            extrasaction='raise',
            lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: '' if row.get(field) is None else row.get(field)
                for field in fields
            })
    os.replace(temporary, path)


def _build_figure_source(groups: Sequence[LoadedGroup],
                         module_groups: Sequence[LoadedModuleGroup],
                         module_rows: Sequence[dict],
                         overall_rows: Sequence[dict]) -> list[dict]:
    module_index = {(row['group_id'], row['module_key']): row
                    for row in module_rows}
    overall_index = {row['group_id']: row for row in overall_rows}
    module_group_index = {group.spec.key: group for group in module_groups}
    result = []
    for panel_order, model_key in enumerate(('pi05', 'groot')):
        panel_label = '(a) PI0.5' if model_key == 'pi05' else '(b) GR00T-N1.5'
        model_groups = [
            group for group in groups if group.spec.model_key == model_key
        ]
        model_groups.sort(
            key=lambda item: 0 if item.spec.backend == 'pytorch' else 1)
        for backend_order, group in enumerate(model_groups):
            overall = overall_index[group.spec.key]
            module_group = module_group_index[group.spec.key]
            for segment_order, segment in enumerate(
                    FIGURE_SEGMENTS_BY_MODEL[model_key]):
                if segment.source_role == 'formal':
                    source = overall
                    source_input = group.spec.cli_option
                    source_run_dir = str(group.spec.run_dir)
                else:
                    source = module_index[(group.spec.key, segment.source_key)]
                    source_input = module_group.spec.cli_option
                    source_run_dir = str(module_group.spec.run_dir)
                row = {
                    'figure_id':
                    'vla_pytorch_vs_tensorrt_latency',
                    'panel_order':
                    panel_order,
                    'panel_label':
                    panel_label,
                    'group_id':
                    group.spec.key,
                    'model_family':
                    group.spec.model_label,
                    'backend_order':
                    backend_order,
                    'backend':
                    group.spec.backend,
                    'row_label':
                    group.spec.backend_label,
                    'segment_order':
                    segment_order,
                    'segment_key':
                    segment.key,
                    'segment_label':
                    segment.label,
                    'segment_short_label':
                    segment.short_label,
                    'segment_detail_label':
                    segment.detail_label,
                    'segment_repeat_count':
                    segment.repeat_count,
                    'detail_source_modules':
                    segment.detail_source_modules,
                    'source_role':
                    segment.source_role,
                    'source_metric_key':
                    segment.source_key,
                    'color_hex':
                    segment.color,
                    'hatch':
                    segment.hatch,
                    'model_speedup_vs_pytorch_x':
                    overall['model_speedup_vs_pytorch_x'],
                    'pipeline_speedup_vs_pytorch_x':
                    overall['pipeline_speedup_vs_pytorch_x'],
                    'formal_measured_inference_calls':
                    overall['measured_inference_calls'],
                    'formal_trial_count':
                    overall['trial_count'],
                    'module_measured_inference_calls':
                    len(module_group.measured_calls),
                    'module_trial_count':
                    len(module_group.trials),
                    'formal_source_input':
                    group.spec.cli_option,
                    'formal_source_run_dir':
                    str(group.spec.run_dir),
                    'module_source_input':
                    module_group.spec.cli_option,
                    'module_source_run_dir':
                    str(module_group.spec.run_dir),
                    'segment_source_input':
                    source_input,
                    'segment_source_run_dir':
                    source_run_dir,
                    'segment_source_inference_calls':
                    'time_profile/inference_calls.csv',
                    'formal_source_inference_calls':
                    'time_profile/inference_calls.csv',
                    'formal_source_trials':
                    'time_profile/trials.csv',
                    'module_source_inference_calls':
                    'time_profile/inference_calls.csv',
                    'module_source_trials':
                    'time_profile/trials.csv',
                    'warmup_filter':
                    'warmup != 1',
                }
                for statistic in STATISTICS:
                    source_field = (f'{segment.source_key}_{statistic}_ms'
                                    if segment.source_role == 'formal' else
                                    f'{statistic}_ms')
                    row[f'segment_{statistic}_ms'] = source[source_field]
                    row[f'model_inference_{statistic}_ms'] = overall[
                        f'model_inference_{statistic}_ms']
                    row[f'model_pipeline_{statistic}_ms'] = overall[
                        f'model_pipeline_{statistic}_ms']
                result.append(row)
    expected_rows = sum(
        len(FIGURE_SEGMENTS_BY_MODEL[group.spec.model_key])
        for group in groups)
    if len(result) != expected_rows:
        raise BenchmarkValidationError(
            f'Expected {expected_rows} figure segment rows, built '
            f'{len(result)}')
    return result


def _read_figure_source(path: Path) -> list[dict]:
    _, raw_rows = _read_csv(path, set(FIGURE_SOURCE_FIELDS))
    numeric_ints = ('panel_order', 'backend_order', 'segment_order',
                    'segment_repeat_count', 'formal_measured_inference_calls',
                    'formal_trial_count', 'module_measured_inference_calls',
                    'module_trial_count')
    numeric_floats = (
        *[f'segment_{statistic}_ms' for statistic in STATISTICS],
        *[f'model_inference_{statistic}_ms' for statistic in STATISTICS],
        *[f'model_pipeline_{statistic}_ms' for statistic in STATISTICS],
        'model_speedup_vs_pytorch_x', 'pipeline_speedup_vs_pytorch_x')
    result = []
    for offset, raw in enumerate(raw_rows, start=2):
        row = dict(raw)
        for field in numeric_ints:
            row[field] = _parse_int(raw[field], path, offset, field)
        for field in numeric_floats:
            row[field] = _parse_float(raw[field], path, offset, field)
        result.append(row)
    return result


def _validate_figure_trace(figure_rows: Sequence[dict],
                           source_rows: Sequence[dict],
                           module_rows: Sequence[dict],
                           overall_rows: Sequence[dict], abs_tol_ms: float,
                           rel_tol: float) -> None:
    if len(figure_rows) != len(source_rows):
        raise BenchmarkValidationError(
            'figure_source.csv row count changed after serialization')
    module_index = {(row['group_id'], row['module_key']): row
                    for row in module_rows}
    overall_index = {row['group_id']: row for row in overall_rows}
    module_source_by_group = {}
    for module in module_rows:
        module_source_by_group.setdefault(module['group_id'], module)
    expected_keys = {(row['group_id'], row['segment_key'])
                     for row in source_rows}
    source_index = {(row['group_id'], row['segment_key']): row
                    for row in source_rows}
    actual_keys = {(row['group_id'], row['segment_key'])
                   for row in figure_rows}
    if actual_keys != expected_keys:
        raise BenchmarkValidationError(
            'figure_source.csv segment identities changed after serialization')
    for row in figure_rows:
        key = (row['group_id'], row['segment_key'])
        serialized_from = source_index[key]
        for field in FIGURE_SOURCE_FIELDS:
            expected = serialized_from[field]
            actual = row[field]
            if isinstance(expected, (int, float)):
                _assert_close(f'figure serialization {key} {field}', actual,
                              expected, abs_tol_ms, rel_tol)
            elif actual != expected:
                raise BenchmarkValidationError(
                    f'figure serialization {key} {field}: {actual!r} != '
                    f'{expected!r}')
        overall = overall_index[row['group_id']]
        source_role = row['source_role']
        if source_role == 'formal':
            for statistic in STATISTICS:
                upstream_field = (f'{row["source_metric_key"]}_{statistic}_ms')
                _assert_close(f'figure trace {key} formal segment {statistic}',
                              row[f'segment_{statistic}_ms'],
                              overall[upstream_field], abs_tol_ms, rel_tol)
            expected_source_input = overall['source_input']
            expected_source_run_dir = overall['source_run_dir']
        elif source_role == 'module_diagnostic':
            module_key = (row['group_id'], row['source_metric_key'])
            if module_key not in module_index:
                raise BenchmarkValidationError(
                    f'figure trace {key}: missing module source {module_key}')
            module = module_index[module_key]
            for statistic in STATISTICS:
                _assert_close(
                    f'figure trace {key} diagnostic segment {statistic}',
                    row[f'segment_{statistic}_ms'], module[f'{statistic}_ms'],
                    abs_tol_ms, rel_tol)
            expected_source_input = module['source_input']
            expected_source_run_dir = module['source_run_dir']
            detail_modules = [
                value for value in row['detail_source_modules'].split(';')
                if value
            ]
            for detail_key in detail_modules:
                detail_index = (row['group_id'], detail_key)
                if detail_index not in module_index:
                    raise BenchmarkValidationError(
                        f'figure trace {key}: missing detail module '
                        f'{detail_index}')
                _assert_close(
                    f'figure trace {key} {detail_key} repeat count',
                    module_index[detail_index]
                    ['invocations_per_inference_mean'],
                    row['segment_repeat_count'], abs_tol_ms, rel_tol)
        else:
            raise BenchmarkValidationError(
                f'figure trace {key}: invalid source_role={source_role!r}')
        if row['segment_source_input'] != expected_source_input:
            raise BenchmarkValidationError(
                f'figure trace {key}: segment source input mismatch')
        if row['segment_source_run_dir'] != expected_source_run_dir:
            raise BenchmarkValidationError(
                f'figure trace {key}: segment source run_dir mismatch')
        for prefix in ('model_inference', 'model_pipeline'):
            for statistic in STATISTICS:
                field = f'{prefix}_{statistic}_ms'
                _assert_close(f'figure trace {key} {field}', row[field],
                              overall[field], abs_tol_ms, rel_tol)
        for figure_field, overall_field in (('model_speedup_vs_pytorch_x',
                                             'model_speedup_vs_pytorch_x'),
                                            ('pipeline_speedup_vs_pytorch_x',
                                             'pipeline_speedup_vs_pytorch_x')):
            _assert_close(f'figure trace {key} {figure_field}',
                          row[figure_field], overall[overall_field],
                          abs_tol_ms, rel_tol)
        if row['formal_source_input'] != overall['source_input']:
            raise BenchmarkValidationError(
                f'figure trace {key}: formal source input mismatch')
        if row['formal_source_run_dir'] != overall['source_run_dir']:
            raise BenchmarkValidationError(
                f'figure trace {key}: formal source run_dir mismatch')
        module_source = module_source_by_group[row['group_id']]
        if row['module_source_input'] != module_source['source_input']:
            raise BenchmarkValidationError(
                f'figure trace {key}: module source input mismatch')
        if row['module_source_run_dir'] != module_source['source_run_dir']:
            raise BenchmarkValidationError(
                f'figure trace {key}: module source run_dir mismatch')
        if (row['formal_measured_inference_calls'] !=
                overall['measured_inference_calls']):
            raise BenchmarkValidationError(
                f'figure trace {key}: formal call count mismatch')
        if (row['module_measured_inference_calls'] !=
                module_source['measured_inference_calls']):
            raise BenchmarkValidationError(
                f'figure trace {key}: module call count mismatch')


def _render_figure(figure_source_path: Path, svg_path: Path,
                   png_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise RuntimeError(
            'matplotlib is required to generate benchmark figures') from exc

    rows = _read_figure_source(figure_source_path)
    panels: dict[int,
                 dict[str,
                      list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        panels[row['panel_order']][row['group_id']].append(row)
    if set(panels) != {0, 1}:
        raise BenchmarkValidationError(
            'Figure source must contain panels 0 and 1, found '
            f'{sorted(panels)}')

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.titlesize': 12,
        'axes.labelsize': 9,
        'svg.fonttype': 'none',
        'figure.facecolor': 'white',
        'axes.facecolor': '#FAFAFA',
    })
    figure = plt.figure(figsize=(18.0, 7.0), constrained_layout=False)
    grid = figure.add_gridspec(
        1, 4, width_ratios=(3.7, 2.35, 3.7, 2.35), wspace=0.07)
    bar_axes = []
    info_axes = []
    bar_axes.append(figure.add_subplot(grid[0, 0]))
    info_axes.append(figure.add_subplot(grid[0, 1]))
    bar_axes.append(figure.add_subplot(grid[0, 2]))
    info_axes.append(figure.add_subplot(grid[0, 3]))
    marker_handles = [
        Line2D([0], [0],
               color='#222222',
               marker='|',
               linestyle='None',
               markersize=13,
               markeredgewidth=2.0,
               label='Model mean'),
        Line2D([0], [0],
               color='#CC79A7',
               marker='D',
               markerfacecolor='white',
               linestyle='None',
               markersize=6,
               label='Pipeline mean'),
        Line2D([0], [0],
               color='#444444',
               marker='o',
               markersize=4,
               linewidth=1.2,
               label='Pipeline P50–P95'),
    ]

    for panel_order in (0, 1):
        axis = bar_axes[panel_order]
        info_axis = info_axes[panel_order]
        panel_groups = list(panels[panel_order].values())
        panel_groups.sort(key=lambda group: group[0]['backend_order'])
        if len(panel_groups) != 2:
            raise BenchmarkValidationError(
                f'Panel {panel_order} must contain two backend rows')
        panel_stack_totals = {
            group_rows[0]['group_id']:
            sum(row['segment_mean_ms'] for row in group_rows)
            for group_rows in panel_groups
        }
        panel_max = max(
            max(group_rows[0]['model_pipeline_p95_ms'], group_rows[0]
                ['model_pipeline_mean_ms'], group_rows[0]
                ['model_inference_mean_ms'], panel_stack_totals[
                    group_rows[0]['group_id']]) for group_rows in panel_groups)
        if panel_max <= 0:
            raise BenchmarkValidationError(
                f'Panel {panel_order} latencies must be positive')
        x_limit = panel_max * 1.10
        panel_label = panel_groups[0][0]['panel_label']
        axis.set_title(panel_label, loc='left', fontweight='bold', pad=16)
        y_positions = {0: 1.0, 1: 0.0}
        for group_rows in panel_groups:
            group_rows.sort(key=lambda row: row['segment_order'])
            backend_order = group_rows[0]['backend_order']
            y = y_positions[backend_order]
            left = 0.0
            segment_fragments = []
            for row in group_rows:
                width = row['segment_mean_ms']
                if width <= 0:
                    raise BenchmarkValidationError(
                        f'Figure segment {row["group_id"]}/'
                        f'{row["segment_key"]} must be positive')
                axis.barh(
                    y,
                    width,
                    left=left,
                    height=0.50,
                    color=row['color_hex'],
                    hatch=row['hatch'],
                    edgecolor='#333333',
                    linewidth=0.55,
                    alpha=0.92,
                    zorder=3)
                if row['segment_key'] == 'action_expert':
                    if width >= x_limit * 0.13:
                        label = (f'{row["segment_short_label"]}\n'
                                 'A-Enc→DiT→Dec ×4\n'
                                 f'{width:.1f} ms')
                    else:
                        label = (f'{row["segment_short_label"]}\n'
                                 f'{width:.1f} ms')
                    axis.text(
                        left + width / 2,
                        y,
                        label,
                        ha='center',
                        va='center',
                        fontsize=5.6,
                        fontweight='semibold',
                        color='#111111',
                        zorder=4)
                elif width >= x_limit * 0.075:
                    axis.text(
                        left + width / 2,
                        y,
                        f'{row["segment_short_label"]}\n{width:.1f} ms',
                        ha='center',
                        va='center',
                        fontsize=6.5,
                        fontweight='semibold',
                        color='#111111',
                        zorder=4)
                else:
                    label_y = y + (0.105 if row['segment_order'] %
                                   2 == 0 else -0.105)
                    axis.text(
                        left + width / 2,
                        label_y,
                        f'{width:.1f}\nms',
                        ha='center',
                        va='center',
                        fontsize=5.0,
                        fontweight='semibold',
                        color='#111111',
                        zorder=4)
                fragment = (f'{row["segment_short_label"]} {width:.1f}')
                if row['segment_detail_label']:
                    fragment += f' [{row["segment_detail_label"]}]'
                segment_fragments.append(fragment)
                left += width

            summary = group_rows[0]
            model_mean = summary['model_inference_mean_ms']
            pipeline_mean = summary['model_pipeline_mean_ms']
            pipeline_p50 = summary['model_pipeline_p50_ms']
            pipeline_p95 = summary['model_pipeline_p95_ms']
            axis.plot([pipeline_p50, pipeline_p95], [y - 0.34, y - 0.34],
                      color='#444444',
                      linewidth=1.35,
                      zorder=5)
            axis.scatter([pipeline_p50, pipeline_p95], [y - 0.34, y - 0.34],
                         color='#444444',
                         s=16,
                         zorder=6)
            axis.scatter([model_mean], [y + 0.34],
                         marker='|',
                         color='#222222',
                         s=150,
                         linewidths=2.0,
                         zorder=6)
            axis.scatter([pipeline_mean], [y + 0.34],
                         marker='D',
                         facecolors='white',
                         edgecolors='#CC79A7',
                         s=42,
                         linewidths=1.5,
                         zorder=6)

            info_axis.text(
                0.02,
                y + 0.22,
                f'{summary["row_label"]}: model {model_mean:.1f} ms  |  '
                f'pipeline {pipeline_mean:.1f} ms',
                transform=info_axis.get_yaxis_transform(),
                ha='left',
                va='center',
                fontsize=8.5,
                fontweight='bold',
                color='#222222')
            info_axis.text(
                0.02,
                y - 0.02, f'Pipeline P50 / P95: {pipeline_p50:.1f} / '
                f'{pipeline_p95:.1f} ms',
                transform=info_axis.get_yaxis_transform(),
                ha='left',
                va='center',
                fontsize=8.2,
                color='#444444')
            chain_text = ' → '.join(segment_fragments)
            wrapped = '\n'.join(textwrap.wrap(chain_text, width=48))
            info_axis.text(
                0.02,
                y - 0.31,
                wrapped,
                transform=info_axis.get_yaxis_transform(),
                ha='left',
                va='center',
                fontsize=7.1,
                color='#555555')

        tensorrt_summary = panel_groups[1][0]
        info_axis.text(
            0.02,
            1.58, f'TensorRT speedup\nModel '
            f'{tensorrt_summary["model_speedup_vs_pytorch_x"]:.2f}×  |  '
            f'Pipeline '
            f'{tensorrt_summary["pipeline_speedup_vs_pytorch_x"]:.2f}×',
            transform=info_axis.get_yaxis_transform(),
            ha='left',
            va='top',
            fontsize=9.3,
            fontweight='bold',
            color='#222222',
            bbox={
                'boxstyle': 'round,pad=0.42',
                'facecolor': '#F2F2F2',
                'edgecolor': '#D9D9D9'
            })
        axis.set_yticks([1.0, 0.0], ['PyTorch', 'TensorRT'])
        axis.set_ylim(-0.72, 1.76)
        axis.set_xlim(0.0, x_limit)
        axis.grid(axis='x', color='#D9D9D9', linewidth=0.7, zorder=0)
        axis.set_axisbelow(True)
        axis.spines[['top', 'right', 'left']].set_visible(False)
        axis.spines['bottom'].set_color('#999999')
        axis.tick_params(axis='y', length=0, labelsize=9)
        axis.tick_params(axis='x', labelsize=8)
        panel_model = panel_groups[0][0]['model_family']
        axis.set_xlabel(
            f'Latency / segment mean (ms; shared within {panel_model})')
        info_axis.set_ylim(axis.get_ylim())
        info_axis.set_xlim(0, 1)
        info_axis.axis('off')

    figure.suptitle(
        'LIBERO inference latency: PyTorch vs TensorRT',
        fontsize=16,
        fontweight='bold',
        y=0.99)
    figure.legend(
        handles=marker_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
        fontsize=8.5)
    figure.text(
        0.5,
        0.025, 'Gray/pink/orange edge stages: formal-run means; neural '
        'stages: diagnostic module means. Each panel has its own scale. '
        'Source: figure_source.csv; warmup=1 excluded.',
        ha='center',
        va='bottom',
        fontsize=8.2,
        color='#555555')
    figure.subplots_adjust(left=0.065, right=0.985, top=0.82, bottom=0.14)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        svg_path,
        format='svg',
        bbox_inches='tight',
        metadata={
            'Title': 'LIBERO PyTorch versus TensorRT latency',
            'Description': 'Rendered from figure_source.csv'
        })
    figure.savefig(
        png_path,
        format='png',
        dpi=220,
        bbox_inches='tight',
        facecolor='white',
        metadata={
            'Title': 'LIBERO PyTorch versus TensorRT latency',
            'Description': 'Rendered from figure_source.csv'
        })
    plt.close(figure)
    for path in (svg_path, png_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f'Figure export failed: {path}')


def _markdown_cell(value: object) -> str:
    return str(value).replace('|', '\\|').replace('\n', ' ')


def _fmt_ms(value: float | None) -> str:
    return '—' if value is None else f'{value:.2f}'


def _fmt_speedup(value: float | None) -> str:
    return '—' if value is None else f'{value:.2f}×'


def _build_report(groups: Sequence[LoadedGroup],
                  module_groups: Sequence[LoadedModuleGroup],
                  overall_rows: Sequence[dict], module_rows: Sequence[dict],
                  gpu: str, precision: str, timing_boundaries: str,
                  expected_trials: int, expected_tasks: int, abs_tol_ms: float,
                  rel_tol: float, formal_pipeline_rows_checked: int) -> str:
    lines = [
        '# LIBERO PyTorch vs TensorRT Benchmark',
        '',
        '![PI0.5 and GR00T-N1.5 latency comparison]'
        '(assets/vla_pytorch_vs_tensorrt_latency.png)',
        '',
        '[SVG vector export](assets/vla_pytorch_vs_tensorrt_latency.svg)',
        '',
        '## Overall comparison',
        '',
        '| Model | Backend | Trials | Success | Model Mean±Std (ms) | '
        'Model P50/P90/P95/P99 (ms) | Pipeline Mean±Std (ms) | '
        'Pipeline P50/P90/P95/P99 (ms) | Model speedup | Pipeline speedup |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | '
        '---: |',
    ]
    for row in overall_rows:
        lines.append(
            f'| {_markdown_cell(row["model_family"])} | '
            f'{_markdown_cell(row["backend"])} | {row["trial_count"]} | '
            f'{row["success_rate_pct"]:.1f}% | '
            f'{_fmt_ms(row["model_inference_mean_ms"])} ± '
            f'{_fmt_ms(row["model_inference_std_ms"])} | '
            f'{_fmt_ms(row["model_inference_p50_ms"])} / '
            f'{_fmt_ms(row["model_inference_p90_ms"])} / '
            f'{_fmt_ms(row["model_inference_p95_ms"])} / '
            f'{_fmt_ms(row["model_inference_p99_ms"])} | '
            f'{_fmt_ms(row["model_pipeline_mean_ms"])} ± '
            f'{_fmt_ms(row["model_pipeline_std_ms"])} | '
            f'{_fmt_ms(row["model_pipeline_p50_ms"])} / '
            f'{_fmt_ms(row["model_pipeline_p90_ms"])} / '
            f'{_fmt_ms(row["model_pipeline_p95_ms"])} / '
            f'{_fmt_ms(row["model_pipeline_p99_ms"])} | '
            f'{_fmt_speedup(row["model_speedup_vs_pytorch_x"])} | '
            f'{_fmt_speedup(row["pipeline_speedup_vs_pytorch_x"])} |')

    lines.extend([
        '',
        '## Module comparison (independent diagnostic run)',
        '',
        '| Model | Module | Backend | Scope | Mean (ms) | Speedup | '
        'Reduction | Share of formal model | Share of formal pipeline | '
        'Pipeline saving contribution |',
        '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | '
        '---: |',
    ])
    for row in module_rows:
        contribution = row['pipeline_savings_contribution_pct']
        contribution_text = ('—' if contribution is None else
                             f'{contribution:.1f}%')
        lines.append(f'| {_markdown_cell(row["model_family"])} | '
                     f'{_markdown_cell(row["module_label"])} | '
                     f'{_markdown_cell(row["backend"])} | '
                     f'{_markdown_cell(row["comparison_scope"])} | '
                     f'{row["comparison_mean_ms"]:.2f} | '
                     f'{row["speedup_vs_pytorch_x"]:.2f}× | '
                     f'{row["latency_reduction_vs_pytorch_pct"]:.1f}% | '
                     f'{row["formal_model_inference_share_pct"]:.1f}% | '
                     f'{row["formal_pipeline_share_pct"]:.1f}% | '
                     f'{contribution_text} |')

    lines.extend([
        '',
        '## GPU telemetry',
        '',
        '| Model | Backend | Associated samples | GPU utilization Mean/Max '
        '(%) | Memory used Mean/Max (MiB) | Power Mean/Max (W) | '
        'Temperature Mean/Max (C) |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |',
    ])
    for row in overall_rows:
        lines.append(f'| {_markdown_cell(row["model_family"])} | '
                     f'{_markdown_cell(row["backend"])} | '
                     f'{row["telemetry_samples"]} | '
                     f'{row["gpu_util_mean_pct"]:.2f} / '
                     f'{row["gpu_util_max_pct"]:.2f} | '
                     f'{row["memory_used_mean_mib"]:.2f} / '
                     f'{row["memory_used_max_mib"]:.2f} | '
                     f'{row["power_mean_w"]:.2f} / '
                     f'{row["power_max_w"]:.2f} | '
                     f'{row["temperature_mean_c"]:.2f} / '
                     f'{row["temperature_max_c"]:.2f} |')

    timezone_labels = sorted({group.telemetry_timezone for group in groups})
    telemetry_samples = sum(row['telemetry_samples'] for row in overall_rows)
    figure_segment_rows = sum(
        len(FIGURE_SEGMENTS_BY_MODEL[group.spec.model_key])
        for group in groups)

    total_trials = sum(len(group.trials) for group in groups)
    formal_warmups = sum(
        len(group.calls) - len(group.measured_calls) for group in groups)
    module_warmups = sum(
        len(group.calls) - len(group.measured_calls)
        for group in module_groups)
    module_pipeline_rows_checked = sum(
        len(group.measured_calls) for group in module_groups)
    lines.extend([
        '',
        '## Benchmark metadata',
        '',
        f'- GPU: {_markdown_cell(gpu)}',
        f'- Precision: {_markdown_cell(precision)}',
        f'- Formal evaluation: {expected_trials} trials and {expected_tasks} '
        f'tasks per group; {total_trials} trials total.',
        f'- Warm-up filtering: `warmup=1` excluded from both sources '
        f'({formal_warmups} formal and {module_warmups} diagnostic rows).',
        f'- Timing boundaries: {_markdown_cell(timing_boundaries)}',
        '- Telemetry collection: each formal wrapper samples `nvidia-smi` at '
        'a nominal 200 ms interval. Samples are associated with a trial when '
        '`episode_start_unix <= sample_time <= episode_end_unix`.',
        '- Telemetry timestamp interpretation: `nvidia-smi` timestamps are '
        'timezone-naive server-local time and are parsed with the summarizer '
        f'host local timezone ({_markdown_cell(", ".join(timezone_labels))}).',
        '- Telemetry distributions at trial, task, and overall levels report '
        'Mean, population Std, P50, P90, P95, P99, and Max for GPU '
        'utilization, memory used, power draw, and GPU temperature.',
        '- `torch_peak_allocated_mib` is retained as a PyTorch-allocator '
        'diagnostic and does not include all TensorRT allocations. Use the '
        'episode-window `nvidia-smi` memory statistics for cross-backend GPU '
        'memory comparison.',
        '- Distribution contract: Mean, population Std (`ddof=0`), P50, P90, '
        'P95, and P99 use linear-interpolated percentiles. Latency/module '
        'fields use warm-up-filtered per-inference observations; episode, '
        'memory, and per-trial totals use trial observations.',
        '- Formal runs are the sole source of success, trial/task, model, and '
        'pipeline results. Independent diagnostic runs are the sole source '
        'of module results.',
        '- Figure segment widths are means in milliseconds. Preprocess, '
        'output transfer, and action denormalization use formal-run data; '
        'neural segments use independent diagnostic module-profile data. '
        'Model/pipeline means, speedups, and pipeline P50/P95 markers use '
        'formal runs.',
        '- PI0.5 chain: Preprocess → Vision → Prefix → Denoise ×10 → '
        'Output Transfer → Action Denorm. GR00T-N1.5 chain: Preprocess → '
        'ViT → LLM → Backbone Processor → State Encoder → Action '
        'Expert (Action Encoder → DiT → Decoder ×4) → Output Transfer '
        '→ Action Denorm. PyTorch and TensorRT share a scale within each '
        'model; the two model panels use independent scales.',
        '- Module comparison scope is `per_invocation` for PI0.5 Denoise '
        'step and GR00T Action Encoder/DiT/Action Decoder; other rows are '
        '`per_inference`. Share and saving-contribution fields always use '
        'the aggregate module time per model inference. Parent and child '
        'module ranges overlap, so their shares/contributions must not be '
        'summed. Diagnostic-module savings are divided by the observed '
        'formal pipeline saving as attribution context, not as an additive '
        'causal decomposition.',
        '',
        '## Validation and traceability',
        '',
        f'- PASS: four formal groups each contain exactly {expected_trials} '
        f'unique trials and {expected_tasks} tasks; each group-specific trial '
        f'CSV has {expected_trials} rows and each task summary has '
        f'{expected_tasks} rows.',
        '- PASS: every formal and diagnostic trial has `status=completed` '
        'with blank `error_type` and `error_message`; interrupted, running, '
        'or failed trials are rejected before summarization.',
        f'- PASS: all {total_trials} formal trials have at least one '
        f'episode-window telemetry sample ({telemetry_samples} associated '
        'samples across the four overall summaries).',
        '- PASS: four distinct diagnostic runs have `module_profile=1`; all '
        'plotted modules are complete after warm-up filtering; PI0.5 '
        '`denoise_step` executes 10 times and each GR00T action-expert child '
        'executes 4 times per measured inference.',
        f'- PASS: {formal_pipeline_rows_checked} formal and '
        f'{module_pipeline_rows_checked} diagnostic measured inference rows '
        'satisfy '
        '`pipeline = preprocess + model + output transfer + action denorm` '
        f'within `abs_tol={abs_tol_ms} ms`, `rel_tol={rel_tol}`.',
        '- PASS: formal and diagnostic trial aggregates match their '
        'call-level CSV data; diagnostic module mean values match '
        '`module_ms / module_calls`.',
        f'- PASS: both figure formats were rendered by re-reading all '
        f'{figure_segment_rows} segment rows in `figure_source.csv`; every '
        'segment role, value, repeat count, and formal/diagnostic source path '
        'is checked against its upstream summary.',
        '',
        '## Input mapping',
        '',
        '| Source role | Input | Model | Backend | Supplied path | Resolved '
        'eval run | Relative inference CSV | Resolved inference CSV | '
        'Telemetry CSV |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ])
    for group in groups:
        inference_csv = (
            group.spec.run_dir / 'time_profile' / 'inference_calls.csv')
        lines.append(f'| Formal | `{group.spec.cli_option}` | '
                     f'{_markdown_cell(group.spec.model_label)} | '
                     f'{group.spec.backend} | '
                     f'`{_markdown_cell(group.spec.run_dir_display)}` | '
                     f'`{_markdown_cell(group.spec.run_dir)}` | '
                     '`time_profile/inference_calls.csv` | '
                     f'`{_markdown_cell(inference_csv)}` | '
                     f'`{_markdown_cell(group.spec.telemetry_path)}` |')
    for group in module_groups:
        inference_csv = (
            group.spec.run_dir / 'time_profile' / 'inference_calls.csv')
        lines.append(f'| Module diagnostic | `{group.spec.cli_option}` | '
                     f'{_markdown_cell(group.spec.model_label)} | '
                     f'{group.spec.backend} | '
                     f'`{_markdown_cell(group.spec.run_dir_display)}` | '
                     f'`{_markdown_cell(group.spec.run_dir)}` | '
                     '`time_profile/inference_calls.csv` | '
                     f'`{_markdown_cell(inference_csv)}` | '
                     'Not used |')
    lines.extend([
        '',
        'Inputs may resolve directly or through a group wrapper. Formal '
        'inputs must expose the wrapper-produced `telemetry.csv`; a direct '
        'formal eval without telemetry is rejected. Wrapper resolution is '
        'restricted to a unique qualifying directory below '
        '`results/eval_runs`.',
        '',
        '## Generated artifacts',
        '',
        '- `pi05_pytorch_trial_results.csv`, '
        '`pi05_tensorrt_trial_results.csv`, '
        '`groot_pytorch_trial_results.csv`, and '
        f'`groot_tensorrt_trial_results.csv` — {expected_trials} completed '
        'formal trial rows each, including per-trial telemetry statistics.',
        f'- Matching `*_task_summary.csv` files — {expected_tasks} formal '
        'task rows each.',
        '- `overall_comparison.csv` — four-group comparison and paired '
        'speedups, using formal runs only.',
        '- `module_comparison.csv` — top-level and nested CUDA-event module '
        'statistics, using diagnostic runs only.',
        f'- `figure_source.csv` — {figure_segment_rows} traceable segment '
        'rows consumed by the figure.',
        '- `assets/vla_pytorch_vs_tensorrt_latency.png` and `.svg` — '
        'color-vision-safe raster and vector exports; the report embeds PNG '
        'by default and links the SVG.',
        '',
    ])
    return '\n'.join(lines)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    temporary.write_text(value, encoding='utf-8')
    os.replace(temporary, path)


def _resolve_eval_run_dir(raw: Path,
                          cli_option: str) -> tuple[Path, Path, Path | None]:
    """Resolve a direct eval run or a benchmark group wrapper strictly."""
    requested = Path(raw).expanduser().resolve()
    if not requested.is_dir():
        raise BenchmarkValidationError(
            f'{cli_option}: input directory does not exist: {requested}')

    direct_profile = requested / 'time_profile'
    required_names = ('inference_calls.csv', 'trials.csv')
    telemetry_candidate = requested / 'telemetry.csv'
    telemetry_path = (
        telemetry_candidate.resolve()
        if telemetry_candidate.is_file() else None)
    if direct_profile.exists():
        missing = [
            name for name in required_names
            if not (direct_profile / name).is_file()
        ]
        if missing:
            raise BenchmarkValidationError(
                f'{cli_option}: direct run has an incomplete time_profile; '
                f'missing {missing}: {requested}')
        return direct_profile.resolve().parent, requested, telemetry_path

    eval_runs = requested / 'results' / 'eval_runs'
    if not eval_runs.is_dir():
        raise BenchmarkValidationError(
            f'{cli_option}: neither a direct eval run nor a benchmark group '
            f'wrapper with results/eval_runs: {requested}')
    candidates = set()
    for call_path in eval_runs.rglob('time_profile/inference_calls.csv'):
        run_dir = call_path.parent.parent
        if all((run_dir / 'time_profile' / name).is_file()
               for name in required_names):
            candidates.add(run_dir.resolve())
    ordered = sorted(candidates, key=str)
    if len(ordered) != 1:
        rendered = ', '.join(str(path) for path in ordered) or 'none'
        raise BenchmarkValidationError(
            f'{cli_option}: expected exactly one eval run below '
            f'{eval_runs} containing both profiler CSVs, found '
            f'{len(ordered)} ({rendered})')
    return ordered[0], requested, telemetry_path


def _make_group_specs(definitions: Sequence[tuple]) -> list[GroupSpec]:
    result = []
    for definition in definitions:
        (key, model_key, model_label, backend, backend_label, option,
         raw) = definition
        run_dir, input_root, telemetry_path = _resolve_eval_run_dir(
            raw, option)
        result.append(
            GroupSpec(
                key=key,
                model_key=model_key,
                model_label=model_label,
                backend=backend,
                backend_label=backend_label,
                cli_option=option,
                input_root=input_root,
                run_dir=run_dir,
                run_dir_display=str(raw),
                telemetry_path=telemetry_path))
    return result


def _group_specs(args: argparse.Namespace) -> list[GroupSpec]:
    definitions = (
        ('pi05_pytorch', 'pi05', 'PI0.5', 'pytorch', 'PyTorch',
         '--pi05-pytorch', args.pi05_pytorch),
        ('pi05_tensorrt', 'pi05', 'PI0.5', 'tensorrt', 'TensorRT',
         '--pi05-tensorrt', args.pi05_tensorrt),
        ('groot_pytorch', 'groot', 'GR00T-N1.5', 'pytorch', 'PyTorch',
         '--groot-pytorch', args.groot_pytorch),
        ('groot_tensorrt', 'groot', 'GR00T-N1.5', 'tensorrt', 'TensorRT',
         '--groot-tensorrt', args.groot_tensorrt),
    )
    return _make_group_specs(definitions)


def _module_group_specs(args: argparse.Namespace) -> list[GroupSpec]:
    definitions = (
        ('pi05_pytorch', 'pi05', 'PI0.5', 'pytorch', 'PyTorch',
         '--pi05-pytorch-module-profile', args.pi05_pytorch_module_profile),
        ('pi05_tensorrt', 'pi05', 'PI0.5', 'tensorrt', 'TensorRT',
         '--pi05-tensorrt-module-profile', args.pi05_tensorrt_module_profile),
        ('groot_pytorch', 'groot', 'GR00T-N1.5', 'pytorch', 'PyTorch',
         '--groot-pytorch-module-profile', args.groot_pytorch_module_profile),
        ('groot_tensorrt', 'groot', 'GR00T-N1.5', 'tensorrt', 'TensorRT',
         '--groot-tensorrt-module-profile',
         args.groot_tensorrt_module_profile),
    )
    return _make_group_specs(definitions)


def summarize(args: argparse.Namespace) -> list[Path]:
    if args.expected_trials_per_group <= 0:
        raise BenchmarkValidationError(
            '--expected-trials-per-group must be positive')
    if args.expected_tasks_per_group <= 0:
        raise BenchmarkValidationError(
            '--expected-tasks-per-group must be positive')
    if args.pipeline_abs_tol_ms < 0 or args.pipeline_rel_tol < 0:
        raise BenchmarkValidationError(
            'Pipeline tolerances must be non-negative')

    specs = _group_specs(args)
    module_specs = _module_group_specs(args)
    formal_dirs = [spec.run_dir for spec in specs]
    diagnostic_dirs = [spec.run_dir for spec in module_specs]
    if len(set(formal_dirs)) != len(formal_dirs):
        raise BenchmarkValidationError(
            'The four formal inputs must resolve to four distinct eval runs')
    if len(set(diagnostic_dirs)) != len(diagnostic_dirs):
        raise BenchmarkValidationError(
            'The four module-profile inputs must resolve to four distinct '
            'eval runs')
    overlap = set(formal_dirs) & set(diagnostic_dirs)
    if overlap:
        raise BenchmarkValidationError(
            'Formal and module-profile inputs must resolve to disjoint eval '
            f'runs; overlap: {sorted(str(path) for path in overlap)}')
    for formal, diagnostic in zip(specs, module_specs):
        if formal.key != diagnostic.key:
            raise AssertionError('Formal/diagnostic group ordering changed')

    groups = [
        _load_group(spec, args.expected_trials_per_group,
                    args.expected_tasks_per_group, args.pipeline_abs_tol_ms,
                    args.pipeline_rel_tol) for spec in specs
    ]
    module_groups = [
        _load_module_group(spec, args.pipeline_abs_tol_ms,
                           args.pipeline_rel_tol) for spec in module_specs
    ]

    module_rows = [
        row for group in module_groups for row in _build_module_rows(group)
    ]
    task_rows_by_group = {
        group.spec.key: _build_task_rows(group)
        for group in groups
    }
    for key, rows in task_rows_by_group.items():
        if len(rows) != args.expected_tasks_per_group:
            raise BenchmarkValidationError(
                f'{key}: expected {args.expected_tasks_per_group} task '
                f'summary rows, built {len(rows)}')
    overall_rows = [_build_overall_row(group) for group in groups]
    _apply_speedups(overall_rows)
    _apply_module_comparisons(module_rows, overall_rows)
    figure_source_rows = _build_figure_source(groups, module_groups,
                                              module_rows, overall_rows)

    output_dir = Path(args.output_dir).expanduser().resolve()
    assets_dir = output_dir / 'assets'
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {
        'overall': output_dir / 'overall_comparison.csv',
        'module': output_dir / 'module_comparison.csv',
        'figure_source': output_dir / 'figure_source.csv',
        'report': output_dir / 'REPORT.md',
        'svg': assets_dir / 'vla_pytorch_vs_tensorrt_latency.svg',
        'png': assets_dir / 'vla_pytorch_vs_tensorrt_latency.png',
    }
    for group in groups:
        trial_key = f'{group.spec.key}_trial'
        task_key = f'{group.spec.key}_task'
        output_paths[trial_key] = (
            output_dir / f'{group.spec.key}_trial_results.csv')
        output_paths[task_key] = (
            output_dir / f'{group.spec.key}_task_summary.csv')
        _write_csv(output_paths[trial_key], TRIAL_RESULT_FIELDS,
                   group.trial_results)
        _write_csv(output_paths[task_key], TASK_SUMMARY_FIELDS,
                   task_rows_by_group[group.spec.key])
    _write_csv(output_paths['overall'], OVERALL_FIELDS, overall_rows)
    _write_csv(output_paths['module'], MODULE_FIELDS, module_rows)
    _write_csv(output_paths['figure_source'], FIGURE_SOURCE_FIELDS,
               figure_source_rows)

    serialized_figure_rows = _read_figure_source(output_paths['figure_source'])
    _validate_figure_trace(serialized_figure_rows, figure_source_rows,
                           module_rows, overall_rows, args.pipeline_abs_tol_ms,
                           args.pipeline_rel_tol)
    _render_figure(output_paths['figure_source'], output_paths['svg'],
                   output_paths['png'])
    report = _build_report(groups, module_groups, overall_rows, module_rows,
                           args.gpu, args.precision, args.timing_boundaries,
                           args.expected_trials_per_group,
                           args.expected_tasks_per_group,
                           args.pipeline_abs_tol_ms, args.pipeline_rel_tol,
                           sum(len(group.measured_calls) for group in groups))
    _write_text(output_paths['report'], report)

    for path in output_paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f'Expected output was not created: {path}')
    return list(output_paths.values())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--pi05-pytorch',
        required=True,
        type=Path,
        help='PI0.5 PyTorch formal wrapper with telemetry.csv.')
    parser.add_argument(
        '--pi05-tensorrt',
        required=True,
        type=Path,
        help='PI0.5 TensorRT formal wrapper with telemetry.csv.')
    parser.add_argument(
        '--groot-pytorch',
        required=True,
        type=Path,
        help='GR00T-N1.5 PyTorch formal wrapper with telemetry.csv.')
    parser.add_argument(
        '--groot-tensorrt',
        required=True,
        type=Path,
        help='GR00T-N1.5 TensorRT formal wrapper with telemetry.csv.')
    parser.add_argument(
        '--pi05-pytorch-module-profile',
        '--pi05-pytorch-module-run-dir',
        '--pi05-pytorch-module',
        dest='pi05_pytorch_module_profile',
        required=True,
        type=Path,
        help='Independent PI0.5 PyTorch module-profile run_dir.')
    parser.add_argument(
        '--pi05-tensorrt-module-profile',
        '--pi05-tensorrt-module-run-dir',
        '--pi05-tensorrt-module',
        dest='pi05_tensorrt_module_profile',
        required=True,
        type=Path,
        help='Independent PI0.5 TensorRT module-profile run_dir.')
    parser.add_argument(
        '--groot-pytorch-module-profile',
        '--groot-pytorch-module-run-dir',
        '--groot-pytorch-module',
        dest='groot_pytorch_module_profile',
        required=True,
        type=Path,
        help='Independent GR00T-N1.5 PyTorch module-profile run_dir.')
    parser.add_argument(
        '--groot-tensorrt-module-profile',
        '--groot-tensorrt-module-run-dir',
        '--groot-tensorrt-module',
        dest='groot_tensorrt_module_profile',
        required=True,
        type=Path,
        help='Independent GR00T-N1.5 TensorRT module-profile run_dir.')
    parser.add_argument(
        '--output-dir',
        required=True,
        type=Path,
        help='Directory for CSV, Markdown, SVG, and PNG outputs.')
    parser.add_argument(
        '--gpu',
        default='not provided',
        help='GPU metadata written to REPORT.md.')
    parser.add_argument(
        '--precision',
        default='not provided',
        help='Precision metadata written to REPORT.md.')
    parser.add_argument(
        '--timing-boundaries',
        default=('preprocess: dataset transforms + H2D; model: synchronized '
                 'predict_action; output: slice/cast + D2H + NumPy; denorm: '
                 'executed actions; pipeline: sum of the four stages'),
        help='Timing-boundary metadata written to REPORT.md.')
    parser.add_argument(
        '--expected-trials-per-group',
        type=int,
        default=500,
        help='Required unique trial rows in each formal input group.')
    parser.add_argument(
        '--expected-tasks-per-group',
        type=int,
        default=10,
        help='Required unique task IDs in each formal input group.')
    parser.add_argument(
        '--pipeline-abs-tol-ms',
        type=float,
        default=1e-3,
        help='Absolute tolerance for pipeline and aggregate checks.')
    parser.add_argument(
        '--pipeline-rel-tol',
        type=float,
        default=1e-6,
        help='Relative tolerance for pipeline and aggregate checks.')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        outputs = summarize(args)
    except (BenchmarkValidationError, OSError, RuntimeError) as exc:
        raise SystemExit(f'error: {exc}') from exc
    print('Benchmark summary written:')
    for path in outputs:
        print(f'  {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
