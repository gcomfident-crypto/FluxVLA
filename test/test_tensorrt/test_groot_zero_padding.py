# Copyright 2026 Limx Dynamics
"""Keep TensorRT GR00T padding semantics aligned with eager inference."""

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_tensorrt_head_masks_padding_before_and_during_denoising():
    path = (
        PROJECT_ROOT / 'fluxvla/models/heads/flow_matching_tensorrt_head.py')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    predict = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        and node.name == '_predict_action_locked')
    calls = [
        node for node in ast.walk(predict)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == '_zero_padded_dims_'
    ]
    assert len(calls) == 2

    config = (
        PROJECT_ROOT / 'configs/gr00tn15/'
        'gr00tn15_eagle_3b_libero_10_tensorrt.py')
    assert 'zero_padded_action_dims=True' in config.read_text(encoding='utf-8')
