#!/usr/bin/env python3
"""Step 2: synthesize the REJECT class (ambient noise, no speech).

On a real device the "no command" case is ambient noise/silence. We synthesize
1-second clips from three procedurally generated noise types (white, pink, and
low-level babble-like filtered noise) so the model learns to reject
non-command audio. No external dataset is required.

Output: ME2/data/reject/reject_NNNNN.wav  (16 kHz mono, 1.0 s)
        ME2/data/reject/reject_manifest.csv
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUTDIR = os.path.join(ROOT, "data", "reject")
SR = 16000
WIN = 1.0


def white(rng, n):
    return rng.standard_normal(n) * 0.02


def pink(rng, n):
    # Paul Kellet's pink noise approximation
    out = np.zeros(n)
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = 0.0
    for i in range(n):
        w = rng.standard_normal()
        b0 = 0.99886 * b0 + w * 0.0555179
        b1 = 0.99332 * b1 + w * 0.0750759
        b2 = 0.96900 * b2 + w * 0.1538520
        b3 = 0.86650 * b3 + w * 0.3104856
        b4 = 0.55000 * b4 + w * 0.5329522
        b5 = -0.7616 * b5 - w * 0.0168980
        out[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362) * 0.11
        b6 = w * 0.115926
    return out * 0.05


def babble(rng, n):
    # band-limited noise bursts approximating distant speech
    x = rng.standard_normal(n) * 0.01
    # simple moving-average smoothing ~ lowpass
    k = 64
    kernel = np.ones(k) / k
    x = np.convolve(x, kernel, mode="same")
    # amplitude modulation at ~3-5 Hz (syllable rate)
    t = np.arange(n) / SR
    mod = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(3, 5) * t + rng.uniform(0, 6.28))
    return x * mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=900, help="total reject clips")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    os.makedirs(OUTDIR, exist_ok=True)
    n = int(WIN * SR)
    gens = [white, pink, babble]
    rows = []
    for i in range(a.n):
        g = gens[i % len(gens)]
        x = g(rng, n)
        # normalize to a consistent RMS
        rms = np.sqrt(np.mean(x ** 2)) + 1e-9
        x = x / rms * 0.01
        p = os.path.join(OUTDIR, f"reject_{i:05d}.wav")
        sf.write(p, x.astype(np.float32), SR)
        rows.append({"abs_path": p, "intent_10": -1, "intent_name": "REJECT",
                     "condition": "noise", "duration_s": WIN})
    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "reject_manifest.csv"), index=False)
    print(f"Reject class: {a.n} clips (1 s, 16 kHz) -> {OUTDIR}")


if __name__ == "__main__":
    main()
