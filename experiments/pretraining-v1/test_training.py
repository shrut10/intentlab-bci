"""Optional PyTorch tests: python -m pytest experiments/pretraining-v1/test_training.py -q."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.run_pretraining import Classifier, fit_classifier, masked_loss, pretrain

CONFIG = json.loads((Path(__file__).parent / "protocol.json").read_text())


def test_masked_loss_ignores_visible_channels_and_rejects_empty_mask():
    target = torch.ones(2, 9, 20)
    mask = torch.zeros(2, 9, dtype=torch.bool)
    mask[:, :2] = True
    prediction = target.clone()
    prediction[:, 2:] += 100
    assert masked_loss(prediction, target, mask) == 0
    prediction[:, :2] += 2
    assert masked_loss(prediction, target, mask) == 4
    with pytest.raises(ValueError, match="At least"):
        masked_loss(prediction, target, torch.zeros_like(mask))


def test_training_is_reproducible_and_source_inputs_are_unchanged():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config = dict(CONFIG, pretraining_epochs=1, supervised_epochs=1, batch_size=4)
    x = np.random.default_rng(1).normal(size=(8, 9, 48)).astype(np.float32)
    y = np.tile([0, 1], 4)
    original_x, original_y = x.copy(), y.copy()
    torch.manual_seed(42)
    initial = Classifier(config)
    initial_state = copy.deepcopy(initial.state_dict())
    state_a, _ = pretrain(initial.encoder, x, config, 42)
    state_b, _ = pretrain(initial.encoder, x, config, 42)
    for name in state_a:
        torch.testing.assert_close(state_a[name], state_b[name], rtol=0, atol=0)
    for name in initial_state:
        torch.testing.assert_close(initial.state_dict()[name], initial_state[name], rtol=0, atol=0)
    a, _ = fit_classifier(initial, state_a, x, y, np.arange(8), config, 42)
    b, _ = fit_classifier(initial, state_b, x, y, np.arange(8), config, 42)
    with torch.no_grad():
        torch.testing.assert_close(a(torch.from_numpy(x)), b(torch.from_numpy(x)), rtol=0, atol=0)
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(y, original_y)
