"""Synthesize a REJECTION / silence class from the SpeechCommands background-noise
files (white noise, pink noise, tap, etc.).

On a real device the "no command" case is ambient silence/noise, so we build the
reject class from random 1 s windows of those loops rather than from unrelated
speech (which would teach the model to reject *words*).

Output: data/processed/reject/*.wav  (16 kHz mono, 1.0 s each)
        data/processed/reject_manifest.csv

Usage:
  python make_reject.py [--n 1500] [--seed 0]
"""
from __future__ import annotations
import argparse, os, glob
import numpy as np
import soundfile as sf
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NOISE_DIR = os.path.join(ROOT, "data", "raw", "extracted", "_background_noise_")
OUTDIR    = os.path.join(ROOT, "data", "processed", "reject")
SR = 16000
WIN = 1.0

def _resample(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x
    n = int(len(x) * sr_out / sr_in)
    idx = np.linspace(0, len(x)-1, n)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)

def build(n: int, seed: int):
    rng = np.random.default_rng(seed)
    sources = [p for p in glob.glob(os.path.join(NOISE_DIR, "*.wav"))]
    if not sources:
        raise SystemExit(f"No noise files in {NOISE_DIR}")
    # preload
    bank = []
    for p in sources:
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = _resample(x, sr, SR)
        bank.append((os.path.basename(p), x))
    total_len = sum(len(x) for _, x in bank)
    n_win = int(WIN * SR)
    os.makedirs(OUTDIR, exist_ok=True)
    rows = []
    for i in range(n):
        name, x = bank[rng.integers(len(bank))]
        if len(x) < n_win:
            # tile short clips
            reps = n_win // len(x) + 2
            x = np.tile(x, reps)[:n_win]
        start = int(rng.integers(0, len(x)-n_win+1))
        seg = x[start:start+n_win]
        # small random gain so it isn't always full-scale
        gain = 0.3 + 0.7*rng.random()
        seg = (seg * gain).astype(np.float32)
        outp = os.path.join(OUTDIR, f"reject_{i:05d}.wav")
        sf.write(outp, seg, SR)
        rows.append({"source":"synthetic_reject","abs_path":outp,
                     "intent_10":-1,"intent_name":"REJECT","condition":"noise",
                     "duration_s":WIN,"weight":1.0})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "reject_manifest.csv"), index=False)
    print(f"Wrote {len(df)} rejection clips -> {OUTDIR}")
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.n, a.seed)
