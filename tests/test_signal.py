import numpy as np
import pytest

from intentlab.signal import bandpower, perturb, preprocess


def test_filter_is_independent_of_other_epochs_and_scale():
    rng = np.random.default_rng(91)
    epochs = rng.normal(size=(3, 9, 480))
    batched = preprocess(epochs)
    np.testing.assert_allclose(batched[1], preprocess(epochs[1:2])[0], atol=1e-6)
    np.testing.assert_allclose(batched, preprocess(epochs * 100 + 50), atol=1e-5)
    np.testing.assert_allclose(np.sqrt(np.mean(batched**2, axis=(1, 2))), 1, atol=1e-6)


@pytest.mark.parametrize("bad", [np.zeros((1, 8, 480)), np.zeros((1, 9, 479)), np.full((1, 9, 480), np.nan)])
def test_invalid_signal_rejected(bad):
    with pytest.raises(ValueError):
        preprocess(bad)


def test_bandpower_distinguishes_known_alpha_and_beta_signals():
    t = np.arange(480) / 160
    alpha = np.tile(np.sin(2 * np.pi * 10 * t), (1, 9, 1))
    beta = np.tile(np.sin(2 * np.pi * 25 * t), (1, 9, 1))
    a, b = bandpower(alpha).reshape(9, 3), bandpower(beta).reshape(9, 3)
    assert np.all(a[:, 0] > a[:, 2] + 10)
    assert np.all(b[:, 2] > b[:, 0] + 10)


def test_perturbations_are_repeatable_and_do_not_mutate_original():
    x = np.ones((2, 9, 480), dtype=np.float32)
    a = perturb(x, noise=0.5, drop="C3")
    b = perturb(x, noise=0.5, drop="C3")
    np.testing.assert_array_equal(a, b)
    assert not a[:, 3].any()
    assert np.all(x == 1)
    assert not np.array_equal(a[:, 0], perturb(x, noise=0.5, seed=1)[:, 0])
