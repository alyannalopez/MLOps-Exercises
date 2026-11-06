"""Audio feature extraction for the VCM.

All audio is 16 kHz mono float32 in [-1, 1]. The classifier consumes a
fixed-size (n_mels, n_frames) log-mel spectrogram. Clips are center-cropped
or zero-padded to the target length so the model always sees a constant shape.
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000


def _hann(n: int) -> np.ndarray:
    return np.hanning(n)


def log_mel_spectrogram(
    waveform: np.ndarray,
    *,
    sr: int = SAMPLE_RATE,
    n_mels: int = 40,
    frame_length_ms: float = 25.0,
    hop_length_ms: float = 10.0,
    fmin: float = 20.0,
    fmax: float = 8000.0,
) -> np.ndarray:
    """Compute a log-mel spectrogram of shape (n_mels, n_frames).

    Uses a hand-rolled STFT + triangular mel filterbank so we have no
    dependency beyond numpy (keeps the ONNX export path simple and the
    runtime tiny).
    """
    waveform = np.asarray(waveform, dtype=np.float32)
    n_fft = int(sr * frame_length_ms / 1000.0)
    hop = max(1, int(sr * hop_length_ms / 1000.0))

    # Reflect-pad symmetrically, then trim to an exact multiple of the hop so
    # the frame count is deterministic: (len - n_fft) // hop + 1 frames.
    pad = n_fft // 2
    x = np.pad(waveform, (pad, pad), mode="reflect") if len(waveform) > 0 else np.zeros(n_fft, dtype=np.float32)
    rem = (n_fft - ((len(x) - n_fft) % hop)) % hop
    x = x[: len(x) - rem]
    n_frames = 1 + (len(x) - n_fft) // hop
    if n_frames <= 0:
        n_frames = 1
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    idx = np.clip(idx, 0, len(x) - 1)
    frames = x[idx]  # (n_frames, n_fft)
    frames *= _hann(n_fft)[None, :]

    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2 / n_fft  # (n_frames, n_bins)
    power = np.maximum(power, 1e-10)

    mel = _mel_filterbank(sr, n_mels, n_fft // 2 + 1, fmin, fmax)
    feats = power @ mel.T  # (n_frames, n_mels)
    feats = feats.T  # (n_mels, n_frames)
    return np.log(feats + 1e-10).astype(np.float32)


def _mel_filterbank(sr: int, n_mels: int, n_bins: int, fmin: float, fmax: float) -> np.ndarray:
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    m_min, m_max = hz2mel(fmin), hz2mel(fmax)
    mel_pts = np.linspace(m_min, m_max, n_mels + 2)
    hz_pts = mel2hz(mel_pts)
    bins = np.floor((n_bins + 1) * hz_pts / (sr / 2.0)).astype(int)
    bins = np.clip(bins, 0, n_bins)

    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        for b in range(left, center):
            if center > left:
                fb[i - 1, b] = (b - left) / (center - left)
        for b in range(center, right):
            if right > center:
                fb[i - 1, b] = (right - b) / (right - center)
    return fb


def loudest_crop(waveform: np.ndarray, target_samples: int, win: int | None = None) -> np.ndarray:
    """Crop the loudest contiguous `target_samples` window.

    Uses a Hann-windowed RMS profile with a coarse-to-fine search so long
    clips (up to ~30 s) are handled quickly. Falls back to center crop for
    silent/empty input.
    """
    waveform = np.asarray(waveform, dtype=np.float32)
    n = len(waveform)
    if n <= target_samples:
        return waveform
    win = win or target_samples // 2
    if win <= 0:
        win = target_samples
    # Coarse scan with stride = win, then refine around the best coarse hit.
    stride = win
    prof = np.array([
        float(np.sqrt(np.mean(waveform[i : i + win] ** 2)))
        for i in range(0, n - target_samples + 1, stride)
    ])
    best_coarse = int(np.argmax(prof)) * stride
    lo = max(0, best_coarse - stride)
    hi = min(n - target_samples, best_coarse + stride)
    best_start, best_rms = lo, -1.0
    for i in range(lo, hi + 1, max(1, win // 4)):
        r = float(np.sqrt(np.mean(waveform[i : i + target_samples] ** 2)))
        if r > best_rms:
            best_rms, best_start = r, i
    if best_rms <= 1e-6:  # silent -> center crop
        best_start = (n - target_samples) // 2
    return waveform[best_start : best_start + target_samples]


def prepare_clip(
    waveform: np.ndarray,
    *,
    target_seconds: float = 1.0,
    n_mels: int = 40,
    hop_length_ms: float = 10.0,
    crop: str = "loudest",
) -> np.ndarray:
    """Normalize, crop/pad to target length, and extract log-mel.

    crop: "loudest" (loudest contiguous window) or "center".
    Returns (n_mels, n_frames) where n_frames = target_seconds*sr/hop.
    """
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0:
        waveform = np.zeros(SAMPLE_RATE, dtype=np.float32)
    # Peak normalize to a consistent level (robust to mic gain differences).
    peak = float(np.abs(waveform).max())
    if peak > 1e-6:
        waveform = waveform / peak * 0.95

    target_samples = int(target_seconds * SAMPLE_RATE)
    if len(waveform) > target_samples:
        if crop == "loudest":
            waveform = loudest_crop(waveform, target_samples)
        else:
            start = (len(waveform) - target_samples) // 2
            waveform = waveform[start : start + target_samples]
    elif len(waveform) < target_samples:
        waveform = np.pad(waveform, (0, target_samples - len(waveform)))

    return log_mel_spectrogram(waveform, n_mels=n_mels, hop_length_ms=hop_length_ms)


def feature_shape(target_seconds: float = 1.0, n_mels: int = 40, hop_length_ms: float = 10.0):
    # Matches prepare_clip: waveform is cropped/padded to exactly
    # target_seconds*sr samples before the STFT. With symmetric reflect
    # padding of n_fft//2 and hop trimming, the frame count is L // hop.
    hop = int(hop_length_ms * SAMPLE_RATE / 1000.0)
    n_frames = int(target_seconds * SAMPLE_RATE) // hop
    return (n_mels, n_frames)
