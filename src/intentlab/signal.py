"""Shared training/serving transforms. Signal units are microvolts before scaling."""

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

CHANNELS = ["FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4"]
SFREQ = 160
SAMPLES = 480
BANDS = [(8, 13), (13, 20), (20, 30)]


def preprocess(epochs: np.ndarray) -> np.ndarray:
    """Bandpass completed epochs and apply per-epoch, not per-channel, RMS scaling."""
    x = np.asarray(epochs, dtype=np.float64)
    if x.shape[-2:] != (len(CHANNELS), SAMPLES) or not np.isfinite(x).all():
        raise ValueError("Expected finite EEG with 9 channels and 480 samples")
    x = x - x.mean(axis=-1, keepdims=True)
    sos = butter(4, [8, 30], btype="bandpass", fs=SFREQ, output="sos")
    x = sosfiltfilt(sos, x, axis=-1)
    rms = np.sqrt(np.mean(x**2, axis=(-2, -1), keepdims=True))
    return (x / np.maximum(rms, 1e-8)).astype(np.float32)


def bandpower(epochs: np.ndarray) -> np.ndarray:
    """27 log-power features: channels × alpha, low-beta and high-beta bands."""
    f, power = welch(epochs, fs=SFREQ, nperseg=160, axis=-1)
    features = [np.log(np.maximum(power[..., (f >= lo) & (f < hi)].sum(-1), 1e-10)) for lo, hi in BANDS]
    return np.stack(features, axis=-1).reshape(len(epochs), -1)


def csp_features(epochs: np.ndarray, filters: np.ndarray) -> np.ndarray:
    projected = np.einsum("kc,nct->nkt", filters, epochs)
    power = np.mean(projected**2, axis=-1)
    return np.log(np.maximum(power, 1e-10))


def perturb(epochs: np.ndarray, noise: float = 0, drop: str | None = None, seed: int = 2027) -> np.ndarray:
    """Controlled post-preprocessing disturbances; keep scale fixed after occlusion."""
    x = np.array(epochs, dtype=np.float32, copy=True)
    if noise:
        rng = np.random.default_rng(seed)
        rms = np.sqrt(np.mean(x**2, axis=(-2, -1), keepdims=True))
        x += rng.normal(size=x.shape).astype(np.float32) * rms * noise
    if drop:
        x[:, CHANNELS.index(drop), :] = 0
    return x
