"""Pure-numpy log-mel feature extractor that EXACTLY reproduces librosa's
melspectrogram (n_fft=2048, win_length=400, hop=160, center=True,
norm='slaney', top_db=80, ref=max). No ONNX ops, no torch, no librosa at
inference time — runs on the Pi with just numpy.
"""
from __future__ import annotations
import numpy as np

SR = 16000
WIN = 400
HOP = 160
NFFT = 2048
N_MELS = 80
N = 16000          # 1.0 s @ 16 kHz
T = 1 + N // HOP   # 101 frames (center=True)


def _mel_filterbank() -> np.ndarray:
    import librosa.filters as lf
    return lf.mel(sr=SR, n_fft=NFFT, n_mels=N_MELS, fmin=0.0, fmax=SR/2,
                  htk=False, norm="slaney").astype(np.float32)


def _window() -> np.ndarray:
    # librosa stft: get_window('hann', win_length, fftbins=True) then pad_center to n_fft
    import scipy.signal as sp
    import librosa.util as lu
    w = sp.get_window("hann", WIN, fftbins=True).astype(np.float32)
    w = lu.pad_center(w, size=NFFT)          # pad window out to n_fft
    return w


class LogMelNP:
    def __init__(self):
        self.fb = _mel_filterbank()
        self.win = _window()
        # FFT matrix F (NFFT, NFFT//2+1) complex
        k = np.arange(NFFT)
        j = np.arange(NFFT//2 + 1)
        self.F = np.exp(-2j * np.pi * np.outer(k, j) / NFFT).astype(np.complex64)

    def transform(self, y: np.ndarray) -> np.ndarray:
        """y: 1-D float32 mono @16k (any length). Returns (T, N_MELS) log-mel."""
        y = np.asarray(y, dtype=np.float32)
        # center=True: pad WIN//2 each side
        pad = WIN // 2
        yc = np.pad(y, (pad, pad), mode="constant")
        n_frames = 1 + (len(yc) - WIN) // HOP
        # frame indices
        idx = np.arange(n_frames)[:, None] * HOP + np.arange(WIN)[None, :]
        frames = yc[idx] * self.win                       # (T, WIN)
        # zero-pad each frame to NFFT for the FFT
        frames = np.hstack([frames, np.zeros((frames.shape[0], NFFT - WIN), np.float32)])
        # STFT magnitude via matmul (real/imag split)
        rr = frames @ self.F.real                          # (T, NFFT//2+1)
        ri = frames @ self.F.imag
        mag2 = rr * rr + ri * ri                           # power
        mel = mag2 @ self.fb.T                             # (T, N_MELS)
        logmel = 20.0 * np.log10(mel + 1e-10)
        logmel = logmel - logmel.max()
        logmel = np.clip(logmel, -80.0, None)              # top_db=80
        # align to T
        if logmel.shape[0] < T:
            logmel = np.vstack([logmel, np.repeat(logmel[-1:], T-logmel.shape[0], axis=0)])
        return logmel[:T].astype(np.float32)


if __name__ == "__main__":
    import librosa
    lm = LogMelNP()
    worst = 0
    for seed in range(8):
        y = (np.random.default_rng(seed).standard_normal(N) * 0.05).astype(np.float32)
        mine = lm.transform(y)
        ref = librosa.power_to_db(
            librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS,
                                           hop_length=HOP, win_length=WIN),
            ref=np.max).T.astype(np.float32)
        d = np.abs(mine - ref)
        worst = max(worst, float(d.max()))
        print(f"seed{seed}: maxdiff={d.max():.4f} (middle={d[10:91].max():.5f})")
    print("WORST", round(worst, 4))
