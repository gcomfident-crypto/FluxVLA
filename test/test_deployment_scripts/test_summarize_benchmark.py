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

from pathlib import Path

import pytest

from deployment_scripts.tensorrt import summarize_benchmark as summary


def _group_spec(tmp_path: Path,
                model_key: str = 'pi05',
                backend: str = 'pytorch') -> summary.GroupSpec:
    model_label = 'PI0.5' if model_key == 'pi05' else 'GR00T-N1.5'
    run_dir = tmp_path / f'{model_key}_{backend}'
    return summary.GroupSpec(
        key=f'{model_key}_{backend}',
        model_key=model_key,
        model_label=model_label,
        backend=backend,
        backend_label=backend.title(),
        cli_option=f'--{model_key}-{backend}',
        input_root=run_dir,
        run_dir=run_dir,
        run_dir_display=str(run_dir),
        telemetry_path=run_dir / 'telemetry.csv')


def _figure_rows():
    rows = []
    numeric_ints = {
        'panel_order', 'backend_order', 'segment_order',
        'segment_repeat_count', 'formal_measured_inference_calls',
        'formal_trial_count', 'module_measured_inference_calls',
        'module_trial_count'
    }
    numeric_floats = {
        *[f'segment_{statistic}_ms' for statistic in summary.STATISTICS],
        *[
            f'model_inference_{statistic}_ms'
            for statistic in summary.STATISTICS
        ],
        *[
            f'model_pipeline_{statistic}_ms'
            for statistic in summary.STATISTICS
        ],
        'model_speedup_vs_pytorch_x',
        'pipeline_speedup_vs_pytorch_x',
    }
    model_specs = (
        ('pi05', 0, 'PI0.5', 'pi05_pytorch_vs_tensorrt_latency',
         ('vision', 'prefix', 'denoise')),
        ('groot', 1, 'GR00T-N1.5', 'groot_n15_pytorch_vs_tensorrt_latency',
         ('vit', 'llm', 'action_expert')),
    )
    for model_key, panel_order, model_label, figure_id, segments in model_specs:
        for backend_order, backend in enumerate(('pytorch', 'tensorrt')):
            group_id = f'{model_key}_{backend}'
            for segment_order, segment_key in enumerate(segments):
                row = {field: '' for field in summary.FIGURE_SOURCE_FIELDS}
                for field in numeric_ints:
                    row[field] = 1
                for field in numeric_floats:
                    row[field] = 1.0
                row.update({
                    'figure_id':
                    figure_id,
                    'panel_order':
                    panel_order,
                    'panel_label':
                    model_label,
                    'group_id':
                    group_id,
                    'model_family':
                    model_label,
                    'backend_order':
                    backend_order,
                    'backend':
                    backend,
                    'row_label':
                    'PyTorch' if backend == 'pytorch' else 'TensorRT',
                    'segment_order':
                    segment_order,
                    'segment_key':
                    segment_key,
                    'segment_label':
                    segment_key.replace('_', ' ').title(),
                    'segment_short_label':
                    segment_key[:8],
                    'segment_repeat_count':
                    4 if segment_key == 'action_expert' else 1,
                    'color_hex':
                    '#0072B2',
                    'hatch':
                    '//',
                    'segment_mean_ms':
                    3.0 - backend_order,
                    'model_inference_mean_ms':
                    10.0 - 4.0 * backend_order,
                    'model_pipeline_mean_ms':
                    12.0 - 4.0 * backend_order,
                    'model_pipeline_p50_ms':
                    11.0 - 4.0 * backend_order,
                    'model_pipeline_p95_ms':
                    13.0 - 4.0 * backend_order,
                    'model_speedup_vs_pytorch_x':
                    1.0 if backend_order == 0 else 10.0 / 6.0,
                    'pipeline_speedup_vs_pytorch_x':
                    1.0 if backend_order == 0 else 12.0 / 8.0,
                    'formal_measured_inference_calls':
                    10,
                    'formal_trial_count':
                    1,
                    'module_measured_inference_calls':
                    10,
                    'module_trial_count':
                    1,
                })
                rows.append(row)
    return rows


def _loaded_groups(tmp_path: Path):
    groups = []
    module_groups = []
    overall_rows = []
    for model_key, _, model_label, _ in summary.FIGURE_EXPORTS:
        for backend in ('pytorch', 'tensorrt'):
            group_id = f'{model_key}_{backend}'
            run_dir = tmp_path / group_id
            spec = summary.GroupSpec(
                key=group_id,
                model_key=model_key,
                model_label=model_label,
                backend=backend,
                backend_label=backend.title(),
                cli_option=f'--{group_id}',
                input_root=run_dir,
                run_dir=run_dir,
                run_dir_display=str(run_dir),
                telemetry_path=run_dir / 'telemetry.csv')
            groups.append(
                summary.LoadedGroup(
                    spec=spec,
                    calls=[],
                    measured_calls=[],
                    trials=[],
                    trial_results=[],
                    source_model=model_key,
                    telemetry_rows=[],
                    telemetry_timezone='UTC'))
            module_groups.append(
                summary.LoadedModuleGroup(
                    spec=spec,
                    calls=[],
                    measured_calls=[],
                    trials=[],
                    source_model=model_key))
            overall_rows.append({
                'model_family': model_label,
                'backend': backend,
                'trial_count': 1,
                'success_rate_pct': 100.0,
                'model_inference_mean_ms': 10.0,
                'model_inference_std_ms': 1.0,
                'model_inference_p50_ms': 10.0,
                'model_inference_p90_ms': 11.0,
                'model_inference_p95_ms': 11.5,
                'model_inference_p99_ms': 12.0,
                'model_pipeline_mean_ms': 12.0,
                'model_pipeline_std_ms': 1.0,
                'model_pipeline_p50_ms': 12.0,
                'model_pipeline_p90_ms': 13.0,
                'model_pipeline_p95_ms': 13.5,
                'model_pipeline_p99_ms': 14.0,
                'model_speedup_vs_pytorch_x': 1.0,
                'pipeline_speedup_vs_pytorch_x': 1.0,
                'telemetry_samples': 1,
                'gpu_util_mean_pct': 50.0,
                'gpu_util_max_pct': 60.0,
                'memory_used_mean_mib': 1000.0,
                'memory_used_max_mib': 1100.0,
                'power_mean_w': 100.0,
                'power_max_w': 110.0,
                'temperature_mean_c': 50.0,
                'temperature_max_c': 55.0,
            })
    return groups, module_groups, overall_rows


def test_each_model_figure_is_rendered_from_shared_source_csv(tmp_path):
    source_path = tmp_path / 'figure_source.csv'
    summary._write_csv(source_path, summary.FIGURE_SOURCE_FIELDS,
                       _figure_rows())

    outputs = {}
    for model_key, _, model_label, stem in summary.FIGURE_EXPORTS:
        svg_path = tmp_path / f'{stem}.svg'
        png_path = tmp_path / f'{stem}.png'
        summary._render_model_figure(source_path, model_key, svg_path,
                                     png_path)
        outputs[model_key] = svg_path.read_text(encoding='utf-8')
        assert png_path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert model_label in outputs[model_key]

    assert 'GR00T-N1.5' not in outputs['pi05']
    assert 'PI0.5' not in outputs['groot']


def test_report_embeds_two_pngs_and_links_two_svgs(tmp_path):
    groups, module_groups, overall_rows = _loaded_groups(tmp_path)
    report = summary._build_report(
        groups=groups,
        module_groups=module_groups,
        overall_rows=overall_rows,
        module_rows=[],
        gpu='physical GPU 3',
        precision='BF16/FP16',
        timing_boundaries='test boundaries',
        expected_trials=1,
        expected_tasks=1,
        abs_tol_ms=0.1,
        rel_tol=0.01,
        formal_pipeline_rows_checked=0)

    for _, _, _, stem in summary.FIGURE_EXPORTS:
        assert f'![{stem}' not in report
        assert f'(assets/{stem}.png)' in report
        assert f'(assets/{stem}.svg)' in report
    assert report.count('](assets/') == 4
    assert 'vla_pytorch_vs_tensorrt_latency' not in report
    assert '.html' not in report.lower()


def test_wrapper_status_requires_exact_success_first_token(tmp_path):
    wrapper = tmp_path / 'wrapper'
    profile = wrapper / 'time_profile'
    profile.mkdir(parents=True)
    for name in ('inference_calls.csv', 'trials.csv'):
        (profile / name).write_text('header\n', encoding='utf-8')

    with pytest.raises(
            summary.BenchmarkValidationError, match='missing status.txt'):
        summary._resolve_eval_run_dir(wrapper, '--test-input')

    status_path = wrapper / 'status.txt'
    for invalid in ('', 'FAILED eval_exit_code=1\n',
                    'SUCCESSFUL completed_at_utc=now\n', '\nSUCCESS\n'):
        status_path.write_text(invalid, encoding='utf-8')
        with pytest.raises(
                summary.BenchmarkValidationError,
                match='must start with the token SUCCESS'):
            summary._resolve_eval_run_dir(wrapper, '--test-input')

    status_path.write_text(
        'SUCCESS completed_at_utc=2026-08-05T00:00:00Z\n', encoding='utf-8')
    run_dir, input_root, _ = summary._resolve_eval_run_dir(
        wrapper, '--test-input')
    assert run_dir == wrapper.resolve()
    assert input_root == wrapper.resolve()


@pytest.mark.parametrize(
    ('model_key', 'expected_model', 'wrong_model'),
    (('pi05', 'pi05', 'pi0.5'), ('groot', 'gr00t', 'groot-n1.5')))
def test_profiler_model_label_must_match_runner_exactly(
        tmp_path, model_key, expected_model, wrong_model):
    spec = _group_spec(tmp_path, model_key=model_key)
    assert summary._validate_source_model(spec, [{
        'model': expected_model
    }], '') == expected_model
    with pytest.raises(
            summary.BenchmarkValidationError,
            match=f"expected model='{expected_model}'"):
        summary._validate_source_model(spec, [{'model': wrong_model}], '')
    with pytest.raises(
            summary.BenchmarkValidationError,
            match=f"expected model='{expected_model}'"):
        summary._validate_source_model(spec, [{
            'model': expected_model
        }, {
            'model': wrong_model
        }], '')


def test_module_diagnostic_requires_ten_completed_tasks_one_trial_each(
        tmp_path):
    spec = _group_spec(tmp_path)
    valid = [{
        'task_id': task_id,
        'status': 'completed'
    } for task_id in range(10)]
    summary._validate_module_trial_contract(spec, valid)

    with pytest.raises(
            summary.BenchmarkValidationError,
            match='expected 10 completed trials'):
        summary._validate_module_trial_contract(spec, valid[:-1])

    duplicate_task = [dict(trial) for trial in valid]
    duplicate_task[-1]['task_id'] = 8
    with pytest.raises(
            summary.BenchmarkValidationError,
            match='expected 10 distinct tasks'):
        summary._validate_module_trial_contract(spec, duplicate_task)

    interrupted = [dict(trial) for trial in valid]
    interrupted[-1]['status'] = 'interrupted'
    with pytest.raises(
            summary.BenchmarkValidationError, match='status=completed'):
        summary._validate_module_trial_contract(spec, interrupted)
