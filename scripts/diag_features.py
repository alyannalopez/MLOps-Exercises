#!/usr/bin/env python3
"""Diagnose selftest failures: feature parity + zero accuracy."""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcm_infer as V
import librosa

MODEL = "models/crnn_1790586409_int8.onnx"
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "optionB")
wavs = sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True))[:6]

vcm = V.VCM(MODEL)

for p in wavs:
    import wave
    with wave.open(p, "rb") as w:
        sr = w.getframerate(); n = w.getnframes()
        raw = np.frombuffer(w.readframes(n), dtype=np.int16)
    print("SR", sr, "LEN", len(raw), "path:", p.split("optionB/")[1][:80])
    y = raw.astype(np.float32) / 32768.0
    mine = vcm.logmel(y)
    # what train's features.py does: check its windowing
    n1 = int(V.WINDOW_S * 16000)
    yy = y[:n1] if len(y) >= n1 else np.pad(y, (0, n1 - len(y)))
    S = librosa.feature.melspectrogram(y=yy, sr=16000, n_mels=V.N_MEL,
                                       hop_length=160, win_length=400)
    ref = librosa.power_to_db(S, ref=np.max).T.astype(np.float32)
    m = min(mine.shape[0], ref.shape[0])
    d = np.abs(mine[:m] - ref[:m])
    print(f"  shapes mine={mine.shape} ref={ref.shape}  maxdiff={d.max():.2f} meandiff={d.mean():.2f}")
    print(f"  mine[0,:5]={mine[0,:5]}  ref[0,:5]={ref[0,:5]}")
    print(f"  mine[-1,:5]={mine[-1,:5]}  ref[-1,:5]={ref[-1,:5]}")
    r = vcm.classify_pcm(raw)
    print(f"  predict: {r['intent']} conf={r['confidence']:.3f} top3={sorted(r['probabilities'].items(), key=lambda kv:-kv[1])[:3]}")
