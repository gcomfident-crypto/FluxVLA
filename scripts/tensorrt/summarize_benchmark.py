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
"""Validate and summarize five paired LIBERO TensorRT benchmark runs."""

from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

RUN_KEYS = (
    'pi05_pytorch_bf16',
    'pi05_tensorrt_bf16',
    'groot_pytorch_bf16',
    'groot_pytorch_fp16',
    'groot_tensorrt_fp16',
)
EXPECTED = {
    'pi05_pytorch_bf16': {
        'label': ('PI0.5', 'PyTorch BF16'),
        'config': 'pi05_paligemma_libero_10_pytorch',
        'model': 'pi05',
        'backend': 'pytorch',
        'dtype': 'torch.bfloat16',
        'steps': 10,
        'module_calls': {
            'vision': 1,
            'prefix': 1,
            'denoise': 1,
            'denoise_step': 10,
        },
    },
    'pi05_tensorrt_bf16': {
        'label': ('PI0.5', 'TensorRT BF16'),
        'config': 'pi05_paligemma_libero_10_tensorrt',
        'model': 'pi05',
        'backend': 'tensorrt',
        'dtype': 'torch.bfloat16',
        'steps': 10,
        'module_calls': {
            'vision': 1,
            'prefix': 1,
            'denoise': 1,
            'denoise_step': 10,
        },
    },
    'groot_pytorch_bf16': {
        'label': ('GR00T-N1.5', 'PyTorch BF16'),
        'config': 'gr00t_eagle_3b_libero_10_pytorch',
        'model': 'groot_n15',
        'backend': 'pytorch',
        'dtype': 'torch.bfloat16',
        'steps': 4,
        'module_calls': {
            'backbone_total': 1,
            'vit': 1,
            'llm': 1,
            'backbone_processor': 1,
            'state_encoder': 1,
            'action_expert': 1,
            'action_encoder': 4,
            'dit': 4,
            'action_decoder': 4,
        },
    },
    'groot_pytorch_fp16': {
        'label': ('GR00T-N1.5', 'PyTorch FP16'),
        'config': 'gr00t_eagle_3b_libero_10_pytorch_fp16',
        'model': 'groot_n15',
        'backend': 'pytorch',
        'dtype': 'torch.float16',
        'steps': 4,
        'module_calls': {
            'backbone_total': 1,
            'vit': 1,
            'llm': 1,
            'backbone_processor': 1,
            'state_encoder': 1,
            'action_expert': 1,
            'action_encoder': 4,
            'dit': 4,
            'action_decoder': 4,
        },
    },
    'groot_tensorrt_fp16': {
        'label': ('GR00T-N1.5', 'TensorRT FP16'),
        'config': 'gr00t_eagle_3b_libero_10_tensorrt',
        'model': 'groot_n15',
        'backend': 'tensorrt',
        'dtype': 'torch.float16',
        'steps': 4,
        'module_calls': {
            'backbone_total': 1,
            'vit': 1,
            'llm': 1,
            'backbone_processor': 1,
            'state_encoder': 1,
            'action_expert': 1,
            'action_encoder': 4,
            'dit': 4,
            'action_decoder': 4,
        },
    },
}
METRICS = ('preprocess_ms', 'inference_ms', 'output_transfer_ms', 'denorm_ms',
           'pipeline_total_ms')
LIBERO_10_MAX_STEPS = 520
LIBERO_WAIT_STEPS = 10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in RUN_KEYS:
        parser.add_argument(
            f'--{key.replace("_", "-")}', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--expected-trials-per-task', type=int, default=50)
    parser.add_argument('--expected-cuda-visible-device', default='3')
    return parser.parse_args()


def _config_stem(value: str) -> str:
    return Path(str(value)).stem


def _load_inputs(path: Path) -> tuple[dict, dict, list[dict]]:
    summary_path = path / 'summary.json'
    profile_dir = path / 'time_profile'
    calls_path = profile_dir / 'inference_calls.csv'
    environment_path = profile_dir / 'environment.json'
    for required in (summary_path, calls_path, environment_path):
        if not required.is_file():
            raise FileNotFoundError(f'Missing benchmark artifact: {required}')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    environment = json.loads(environment_path.read_text(encoding='utf-8'))
    with calls_path.open(newline='', encoding='utf-8') as stream:
        calls = list(csv.DictReader(stream))
    if not calls:
        raise ValueError(f'No measured calls in {calls_path}')
    return summary, environment, calls


def _load_trial_outcomes(path: Path, trials_per_task: int) -> dict:
    """Load the per-trial terminal outcome files written after evaluation."""
    outcomes = {}
    expected_trial_ids = set(range(trials_per_task))
    for task_id in range(10):
        result_path = (path / 'libero_10' / f'task{task_id}_results.json')
        if not result_path.is_file():
            raise FileNotFoundError(
                f'Missing per-task benchmark artifact: {result_path}')
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if (result.get('task_suite') != 'libero_10'
                or int(result.get('task_id', -1)) != task_id
                or int(result.get('total_episodes', -1)) != trials_per_task):
            raise ValueError(f'Invalid per-task result: {result_path}')
        successes = {
            int(value)
            for value in result.get('success_episodes', [])
        }
        failures = {int(value) for value in result.get('failure_episodes', [])}
        if (successes & failures or successes | failures != expected_trial_ids
                or len(successes) != int(result.get('successes', -1))):
            raise ValueError(f'Incomplete trial outcomes in {result_path}')
        for trial_id in successes:
            outcomes[(task_id, trial_id)] = True
        for trial_id in failures:
            outcomes[(task_id, trial_id)] = False
    return outcomes


def _validate_protocol(key: str, path: Path, summary: dict, environment: dict,
                       calls: list[dict], trials_per_task: int,
                       visible_device: str) -> tuple[dict, dict, dict]:
    expected = EXPECTED[key]
    expected_tasks = set(range(10))
    expected_trials = {(task_id, trial_id)
                       for task_id in expected_tasks
                       for trial_id in range(trials_per_task)}
    trial_outcomes = _load_trial_outcomes(path, trials_per_task)
    suites = summary.get('suite_stats', {})
    if set(suites) != {'libero_10'}:
        raise ValueError(f'{key}: expected only the libero_10 suite')
    suite = suites['libero_10']
    if int(suite.get('total_tasks', -1)) != 10:
        raise ValueError(f'{key}: expected 10 tasks')
    if int(suite.get('total_trials', -1)) != len(expected_trials):
        raise ValueError(f'{key}: expected {len(expected_trials)} trials')
    task_results = summary.get('task_results', {})
    expected_task_keys = {f'libero_10_{task_id}' for task_id in expected_tasks}
    if set(task_results) != expected_task_keys or any(
            int(value.get('total_episodes', -1)) != trials_per_task
            for value in task_results.values()):
        raise ValueError(f'{key}: every task must contain {trials_per_task} '
                         'episodes')
    task_successes = [
        int(task_results[name].get('successes', -1))
        for name in expected_task_keys
    ]
    suite_successes = int(suite.get('total_successes', -1))
    if (any(value < 0 or value > trials_per_task for value in task_successes)
            or sum(task_successes) != suite_successes):
        raise ValueError(f'{key}: task and suite success counts disagree')
    for task_id in expected_tasks:
        detailed_successes = sum(trial_outcomes[(task_id, trial_id)]
                                 for trial_id in range(trials_per_task))
        summary_successes = int(
            task_results[f'libero_10_{task_id}']['successes'])
        if detailed_successes != summary_successes:
            raise ValueError(f'{key}: detailed and summary outcomes disagree '
                             f'for task {task_id}')

    if _config_stem(summary.get('config', '')) != expected['config']:
        raise ValueError(f'{key}: summary config does not match protocol')
    if _config_stem(environment.get('config', '')) != expected['config']:
        raise ValueError(f'{key}: environment config does not match protocol')
    if str(environment.get('cuda_visible_devices')) != visible_device:
        raise ValueError(f'{key}: expected CUDA_VISIBLE_DEVICES='
                         f'{visible_device}')
    required_environment = {
        'cuda_device_order': 'PCI_BUS_ID',
        'rank': 0,
        'world_size': 1,
        'logical_cuda_device': 0,
        'mixed_precision_dtype': expected['dtype'],
        'seed': 7,
        'num_trials_per_task': trials_per_task,
        'task_ids': list(range(10)),
        'eval_chunk_size': 10,
        'num_inference_steps': expected['steps'],
        'inference_seed': 7,
        'preprocess_every_step': False,
        'enable_module_profile': True,
        'time_profile_warmup_calls': 5,
        'time_profile_backend': expected['backend'],
        'time_profile_model': expected['model'],
    }
    for name, value in required_environment.items():
        if environment.get(name) != value:
            raise ValueError(f'{key}: environment {name}='
                             f'{environment.get(name)!r}, expected {value!r}')
    checkpoint = str(environment.get('checkpoint', ''))
    summary_checkpoint = str(summary.get('ckpt', ''))
    if (not checkpoint or not summary_checkpoint
            or Path(summary_checkpoint).expanduser().resolve() !=
            Path(checkpoint).expanduser().resolve()):
        raise ValueError(f'{key}: checkpoint identity is inconsistent')

    trial_states = {}
    call_seeds = {}
    trial_call_ids = {}
    trial_rows = {}
    for row in calls:
        if int(row.get('schema_version', -1)) != 1 or int(row.get('rank',
                                                                  -1)) != 0:
            raise ValueError(f'{key}: invalid profile schema or rank')
        if (row.get('model') != expected['model']
                or row.get('backend') != expected['backend']):
            raise ValueError(f'{key}: profile model/backend label mismatch')
        trial_key = (int(row['task_id']), int(row['trial_id']))
        if trial_key not in expected_trials:
            raise ValueError(f'{key}: unexpected trial key {trial_key}')
        state_hash = row.get('initial_state_sha256', '')
        if len(state_hash) != 64:
            raise ValueError(f'{key}: missing initial-state SHA256')
        previous = trial_states.setdefault(trial_key, state_hash)
        if previous != state_hash:
            raise ValueError(f'{key}: state changed within {trial_key}')
        call_id = int(row['trial_call_index'])
        seed_text = row.get('inference_seed', '')
        if not seed_text or int(seed_text) != 7 + call_id:
            raise ValueError(f'{key}: invalid seed for {trial_key}:{call_id}')
        call_key = (*trial_key, call_id)
        if call_key in call_seeds:
            raise ValueError(f'{key}: duplicate call key {call_key}')
        call_seeds[call_key] = int(seed_text)
        trial_call_ids.setdefault(trial_key, []).append(call_id)
        trial_rows.setdefault(trial_key, []).append(row)
        actions_returned = int(row.get('actions_returned', -1))
        actions_executed = int(row.get('actions_executed', -1))
        if actions_returned != 10 or not 1 <= actions_executed <= 10:
            raise ValueError(f'{key}: invalid returned/executed action count')
        for name in METRICS:
            value = float(row[name])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{key}: invalid {name} value')
        component_total = sum(float(row[name]) for name in METRICS[:-1])
        if not math.isclose(
                float(row['pipeline_total_ms']),
                component_total,
                rel_tol=1e-9,
                abs_tol=1e-6):
            raise ValueError(f'{key}: pipeline total does not equal its '
                             'four measured components')
        for module, expected_calls in expected['module_calls'].items():
            elapsed = row.get(f'{module}_ms', '')
            calls_value = row.get(f'{module}_calls', '')
            if (elapsed == '' or not math.isfinite(float(elapsed))
                    or float(elapsed) < 0
                    or int(calls_value) != expected_calls):
                raise ValueError(f'{key}: invalid {module} attribution')
    if set(trial_states) != expected_trials:
        raise ValueError(f'{key}: profile does not cover every trial')
    for trial_key, call_ids in trial_call_ids.items():
        if sorted(call_ids) != list(range(len(call_ids))):
            raise ValueError(f'{key}: non-contiguous calls in {trial_key}')
        rows = sorted(
            trial_rows[trial_key],
            key=lambda row: int(row['trial_call_index']))
        if int(rows[0]['env_step']) != LIBERO_WAIT_STEPS:
            raise ValueError(f'{key}: invalid first env_step in {trial_key}')
        for previous, current in zip(rows, rows[1:]):
            expected_step = (
                int(previous['env_step']) + int(previous['actions_executed']))
            if int(current['env_step']) != expected_step:
                raise ValueError(f'{key}: truncated call chain in {trial_key}')
        if not trial_outcomes[trial_key]:
            terminal_step = (
                int(rows[-1]['env_step']) + int(rows[-1]['actions_executed']))
            expected_terminal_step = LIBERO_WAIT_STEPS + LIBERO_10_MAX_STEPS
            if terminal_step != expected_terminal_step:
                raise ValueError(f'{key}: failed trial {trial_key} ends at '
                                 f'env_step {terminal_step}, expected '
                                 f'{expected_terminal_step}')
    return trial_states, call_seeds, trial_outcomes


def _read_run(key: str, path: Path, trials_per_task: int,
              visible_device: str) -> dict:
    summary, environment, calls = _load_inputs(path)
    trial_states, call_seeds, trial_outcomes = _validate_protocol(
        key, path, summary, environment, calls, trials_per_task,
        visible_device)
    suite = summary['suite_stats']['libero_10']
    trials = int(suite['total_trials'])
    successes = int(suite['total_successes'])
    values = {}
    for name in METRICS:
        series = [float(row[name]) for row in calls]
        values[f'{name}_mean'] = statistics.fmean(series)
        values[f'{name}_std'] = statistics.pstdev(series)
    modules = {}
    for name in EXPECTED[key]['module_calls']:
        series = [float(row[f'{name}_ms']) for row in calls]
        modules[name] = {
            'mean_ms': statistics.fmean(series),
            'std_ms': statistics.pstdev(series),
        }
    return {
        'run_dir': str(path.resolve()),
        'config': EXPECTED[key]['config'],
        'checkpoint': str(environment['checkpoint']),
        'environment': environment,
        'trials': trials,
        'successes': successes,
        'success_rate': successes / trials * 100.0,
        'calls': len(calls),
        'metrics': values,
        'modules': modules,
        'trial_states': {
            f'{task_id}:{trial_id}': value
            for (task_id, trial_id), value in trial_states.items()
        },
        'call_seeds': {
            f'{task_id}:{trial_id}:{call_id}': value
            for (task_id, trial_id, call_id), value in call_seeds.items()
        },
        'trial_outcomes': {
            f'{task_id}:{trial_id}': value
            for (task_id, trial_id), value in trial_outcomes.items()
        },
        'task_results': summary['task_results'],
    }


def _validate_group(runs: dict, keys: tuple[str, ...]) -> None:
    checkpoints = {runs[key]['checkpoint'] for key in keys}
    states = {
        json.dumps(runs[key]['trial_states'], sort_keys=True)
        for key in keys
    }
    if len(checkpoints) != 1:
        raise ValueError(f'Checkpoint paths differ in group: {keys}')
    if len(states) != 1:
        raise ValueError(f'Initial states differ in group: {keys}')
    reference = runs[keys[0]]['call_seeds']
    for key in keys[1:]:
        candidate = runs[key]['call_seeds']
        shared = set(reference).intersection(candidate)
        if not shared or any(reference[name] != candidate[name]
                             for name in shared):
            raise ValueError(f'Seed sequence differs in group: {keys}')


def _write_summary(output: Path, runs: dict) -> None:
    fields = ('key', 'model', 'backend', 'trials', 'successes', 'success_rate',
              'calls', 'inference_mean_ms', 'inference_std_ms',
              'pipeline_mean_ms', 'pipeline_std_ms')
    with (output / 'benchmark_summary.csv').open(
            'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in RUN_KEYS:
            run = runs[key]
            model, backend = EXPECTED[key]['label']
            writer.writerow({
                'key':
                key,
                'model':
                model,
                'backend':
                backend,
                'trials':
                run['trials'],
                'successes':
                run['successes'],
                'success_rate':
                f'{run["success_rate"]:.6f}',
                'calls':
                run['calls'],
                'inference_mean_ms':
                f'{run["metrics"]["inference_ms_mean"]:.6f}',
                'inference_std_ms':
                f'{run["metrics"]["inference_ms_std"]:.6f}',
                'pipeline_mean_ms':
                f'{run["metrics"]["pipeline_total_ms_mean"]:.6f}',
                'pipeline_std_ms':
                f'{run["metrics"]["pipeline_total_ms_std"]:.6f}',
            })
    with (output / 'module_summary.csv').open(
            'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=('key', 'model', 'backend', 'module', 'mean_ms',
                        'std_ms'))
        writer.writeheader()
        for key in RUN_KEYS:
            model, backend = EXPECTED[key]['label']
            for module, values in runs[key]['modules'].items():
                writer.writerow({
                    'key': key,
                    'model': model,
                    'backend': backend,
                    'module': module,
                    'mean_ms': f'{values["mean_ms"]:.6f}',
                    'std_ms': f'{values["std_ms"]:.6f}',
                })
    with (output / 'task_summary.csv').open(
            'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=('key', 'model', 'backend', 'task_id', 'successes',
                        'trials', 'success_rate', 'task_description'))
        writer.writeheader()
        for key in RUN_KEYS:
            model, backend = EXPECTED[key]['label']
            for task_id in range(10):
                result = runs[key]['task_results'][f'libero_10_{task_id}']
                trials = int(result['total_episodes'])
                successes = int(result['successes'])
                writer.writerow({
                    'key': key,
                    'model': model,
                    'backend': backend,
                    'task_id': task_id,
                    'successes': successes,
                    'trials': trials,
                    'success_rate': f'{successes / trials * 100.0:.6f}',
                    'task_description': result['task_description'],
                })
    (output / 'benchmark_data.json').write_text(
        json.dumps(runs, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _plot(output: Path, runs: dict, visible_device: str) -> None:
    import matplotlib.pyplot as plt

    comparisons = (
        ('PI0.5', 'pi05_pytorch_bf16', 'pi05_tensorrt_bf16'),
        ('GR00T-N1.5', 'groot_pytorch_fp16', 'groot_tensorrt_fp16'),
    )
    figure, axes = plt.subplots(2, 1, figsize=(12, 6.5))
    colors = ('#4C78A8', '#F58518')
    for axis, (model, pytorch_key, trt_key) in zip(axes, comparisons):
        keys = (pytorch_key, trt_key)
        labels = [EXPECTED[key]['label'][1] for key in keys]
        inference = [runs[key]['metrics']['inference_ms_mean'] for key in keys]
        pipeline = [
            runs[key]['metrics']['pipeline_total_ms_mean'] for key in keys
        ]
        positions = range(len(keys))
        axis.barh([value + 0.18 for value in positions],
                  pipeline,
                  height=0.34,
                  color=colors[0],
                  label='Pipeline')
        axis.barh([value - 0.18 for value in positions],
                  inference,
                  height=0.34,
                  color=colors[1],
                  label='Policy inference')
        for position, (pipeline_ms,
                       inference_ms) in enumerate(zip(pipeline, inference)):
            axis.text(
                pipeline_ms,
                position + 0.18,
                f' {pipeline_ms:.2f} ms',
                va='center')
            axis.text(
                inference_ms,
                position - 0.18,
                f' {inference_ms:.2f} ms',
                va='center')
        axis.set_yticks(list(positions), labels)
        axis.set_title(model)
        axis.set_xlabel('Mean latency per policy call (ms)')
        axis.grid(axis='x', alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.94),
        ncol=2)
    device_name = runs[RUN_KEYS[0]]['environment']['device_name']
    figure.suptitle(
        f'FluxVLA PyTorch vs TensorRT on LIBERO-10 - {device_name} '
        f'(physical GPU {visible_device})',
        y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    figure.savefig(output / 'pytorch_vs_tensorrt.png', dpi=180)
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    if args.expected_trials_per_task <= 0:
        raise ValueError('--expected-trials-per-task must be positive')
    runs = {
        key: _read_run(key, getattr(args, key), args.expected_trials_per_task,
                       args.expected_cuda_visible_device)
        for key in RUN_KEYS
    }
    device_names = {
        runs[key]['environment']['device_name']
        for key in RUN_KEYS
    }
    if len(device_names) != 1:
        raise ValueError('Runs were produced on different GPU models')
    _validate_group(runs, ('pi05_pytorch_bf16', 'pi05_tensorrt_bf16'))
    _validate_group(
        runs,
        ('groot_pytorch_bf16', 'groot_pytorch_fp16', 'groot_tensorrt_fp16'))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_summary(args.output_dir, runs)
    _plot(args.output_dir, runs, args.expected_cuda_visible_device)
    print(f'Benchmark summary written to {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
