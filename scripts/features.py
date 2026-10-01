"""Shared audio feature extraction for the VCM.

All architectures consume the SAME fixed-size feature tensor:

    INPUT  : wav path (any sample rate, any length)
    OUTPUT : np.float32 array of shape (T, F)   [time-major]

Feature families:
  - MFCC      : (T, 40)  -> logistic / DNN baselines
  - log-mel   : (T, 80)  -> 1-D CNN / CRNN

librosa 1.x returns (F, T); we transpose to (T, F) so the time axis is first,
matching the CNN/CRNN conv layout and the ONNX input contract.
"""
from __future__ import annotations
import numpy as np
import librosa

TARGET_SR = 16000
WINDOW_S  = 1.0
N_MFCC    = 40
N_MEL     = 80
HOP_LEN   = 160            # 10 ms hop @ 16 kHz
WIN_LEN   = 400            # 25 ms window @ 16 kHz

# Warm the JIT/numba kernels once so the first real extraction is fast.
def _warm():
    try:
        _y = np.zeros(TARGET_SR, dtype=np.float32)
        librosa.feature.mfcc(y=_y, sr=TARGET_SR, n_mfcc=N_MFCC,
                             hop_length=HOP_LEN, win_length=WIN_LEN)
        librosa.feature.melspectrogram(y=_y, sr=TARGET_SR, n_mels=N_MEL,
                                       hop_length=HOP_LEN, win_length=WIN_LEN)
    except Exception:
        pass
_warm()

def _load(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return y

def _window(y: np.ndarray, n_samples: int) -> np.ndarray:
    """Crop to the loudest n_samples window, or pad short clips."""
    if len(y) >= n_samples:
        frame = HOP_LEN
        if len(y) > n_samples:
            power = np.abs(y).reshape(-1, frame).sum(axis=1)
            win_frames = n_samples // frame
            best = int(np.argmax(power[:max(1, len(power)-win_frames+1)]))
            y = y[best*frame : best*frame+n_samples]
        return y[:n_samples]
    return np.pad(y, (0, n_samples-len(y)), mode="constant")

def mfcc_features(path: str, n_mfcc: int = N_MFCC, window_s: float = WINDOW_S) -> np.ndarray:
    y = _window(_load(path), int(window_s*TARGET_SR))
    feat = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=n_mfcc,
                                hop_length=HOP_LEN, win_length=WIN_LEN)
    return feat.T.astype(np.float32)          # (F,T) -> (T,F)

def logmel_features(path: str, n_mels: int = N_MEL, window_s: float = WINDOW_S) -> np.ndarray:
    y = _window(_load(path), int(window_s*TARGET_SR))
    S = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=n_mels,
                                       hop_length=HOP_LEN, win_length=WIN_LEN)
    return librosa.power_to_db(S, ref=np.max).T.astype(np.float32)  # (T,F)

def shape_of(kind: str) -> tuple[int, int]:
    T = int(WINDOW_S*TARGET_SR // HOP_LEN) + 1
    return (T, N_MFCC) if kind == "mfcc" else (T, N_MEL)

if __name__ == "__main__":
    import sys
    p = sys.argv[1]
    for kind in ("mfcc","logmel"):
        fn = mfcc_features if kind=="mfcc" else logmel_features
        print(kind, fn(p).shape)
