"""Training-time audio augmentation (waveform domain, numpy-only).

Applied before feature extraction. Each function takes a float32 waveform and
returns an augmented copy. `augment_waveform` chains a random subset with
probabilistic control so every training clip gets mild, varied perturbation.

Recipe follows the reference VCM projects (vcm-me2, tiny-alexa):
speed 0.9-1.1, time shift, gain, white/pink noise at 5-30 dB SNR,
lowpass for bad-mic bandwidth, occasional pitch shift.
"""
from __future__ import annotations

import numpy as np

SR = 16000


def add_noise(w: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    pwr = np.mean(w ** 2) + 1e-12
    n = noise.astype(np.float32).copy()
    npwr = np.mean(n ** 2) + 1e-12
    scale = np.sqrt(pwr / (10 ** (snr_db / 20.0)) / npwr)
    return w + n * scale


def speed_perturb(w: np.ndarray, factor: float) -> np.ndarray:
    """Resample by linear interpolation (factor>1 speeds up)."""
    n_out = max(1, int(len(w) / factor))
    pos = np.arange(n_out) * factor
    idx = np.floor(pos).astype(int)
    frac = pos - idx
    idx = np.clip(idx, 0, len(w) - 1)
    idx1 = np.clip(idx + 1, 0, len(w) - 1)
    return (w[idx] * (1 - frac) + w[idx1] * frac).astype(np.float32)


def pitch_shift(w: np.ndarray, semitones: float) -> np.ndarray:
    """PSOLA-lite pitch shift: resample by 2^(st/12), restore length."""
    factor = 2.0 ** (semitones / 12.0)
    stretched = speed_perturb(w, factor)  # factor>1 = higher pitch
    # Restore original length (slight timing distortion, acceptable for aug).
    if len(stretched) < len(w):
        stretched = np.pad(stretched, (0, len(w) - len(stretched)))
    elif len(stretched) > len(w):
        stretched = stretched[: len(w)]
    return stretched


def time_shift(w: np.ndarray, max_shift_ms: int = 120) -> np.ndarray:
    max_s = int(SR * max_shift_ms / 1000)
    shift = np.random.randint(-max_s, max_s + 1)
    return np.roll(w, shift)


def gain_perturb(w: np.ndarray, max_db: float = 8.0) -> np.ndarray:
    db = np.random.uniform(-max_db, max_db)
    return (w * (10 ** (db / 20.0))).astype(np.float32)


def lowpass(w: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """FFT brick-wall lowpass (zero-phase, vectorized). Good enough for
    simulating telephone/bad-mic bandwidth during training."""
    n = len(w)
    if n < 8:
        return w
    spec = np.fft.rfft(w)
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)
    spec[freqs > cutoff_hz] = 0.0
    return np.fft.irfft(spec, n).astype(np.float32)


def pink_noise(n: int) -> np.ndarray:
    """Paul Kellet's pink-noise approximation."""
    white = np.random.randn(n)
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = 0.0
    out = np.empty(n)
    for i in range(n):
        w = white[i]
        b0 = 0.99886 * b0 + w * 0.0555179
        b1 = 0.99332 * b1 + w * 0.0750759
        b2 = 0.96900 * b2 + w * 0.1538520
        b3 = 0.86650 * b3 + w * 0.3104856
        b4 = 0.55000 * b4 + w * 0.5329522
        b5 = -0.7616 * b5 - w * 0.0168980
        out[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362) * 0.11
        b6 = w * 0.115926
    return out.astype(np.float32)


def make_noise_pool(n_clips: int = 32, duration_s: float = 3.0,
                    rng: np.random.Generator | None = None) -> list[np.ndarray]:
    """Mix of white and pink noise clips for augmentation."""
    rng = rng or np.random.default_rng(0)
    n = int(duration_s * SR)
    pool = []
    for i in range(n_clips):
        if i % 2 == 0:
            pool.append(rng.standard_normal(n).astype(np.float32) * 0.1)
        else:
            pool.append(pink_noise(n) * 0.5)
    return pool


def augment_waveform(w: np.ndarray, noise_pool: list | None = None,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply a random, mild augmentation chain.

    noise_pool: list of float32 noise waveforms (same SR). If None, only
    deterministic perturbations are used.
    """
    rng = rng or np.random.default_rng()
    out = w
    if rng.random() < 0.8:
        out = speed_perturb(out, float(rng.uniform(0.9, 1.1)))
    if rng.random() < 0.8:
        out = time_shift(out, max_shift_ms=120)
    if rng.random() < 0.5:
        out = pitch_shift(out, float(rng.choice([-2, -1, 1, 2])))
    if rng.random() < 0.7:
        out = gain_perturb(out, max_db=8.0)
    if noise_pool and rng.random() < 0.7:
        n = noise_pool[rng.integers(len(noise_pool))]
        seg = n[rng.integers(0, max(1, len(n) - len(out))) :][: len(out)]
        out = add_noise(out, seg, float(rng.uniform(5, 30)))
    if rng.random() < 0.35:
        out = lowpass(out, float(rng.uniform(2500, 6000)))
    # Re-truncate/zero-pad to original length (speed perturbation changes it).
    if len(out) < len(w):
        out = np.pad(out, (0, len(w) - len(out)))
    elif len(out) > len(w):
        out = out[: len(w)]
    return out
