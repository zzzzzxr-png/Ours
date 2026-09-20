"""Minimal reproducibility and bounds check for per-epoch random crops."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from likelihood.dataset import _random_coordinate


stack = np.zeros((1000, 245, 245), dtype=np.float32)
fixed = {
    'init_s': 0, 'end_s': 128,
    'init_h': 0, 'end_h': 128,
    'init_w': 0, 'end_w': 128,
}
first = _random_coordinate(stack, fixed, 1024)
assert first == _random_coordinate(stack, fixed, 1024)
assert first != _random_coordinate(stack, fixed, 1024 + 6992)
assert first['end_s'] - first['init_s'] == 128
assert first['end_h'] - first['init_h'] == 128
assert first['end_w'] - first['init_w'] == 128
assert 0 <= first['init_s'] <= first['end_s'] <= stack.shape[0]
assert 0 <= first['init_h'] <= first['end_h'] <= stack.shape[1]
assert 0 <= first['init_w'] <= first['end_w'] <= stack.shape[2]
print('Random patch-coordinate checks passed:', first)
