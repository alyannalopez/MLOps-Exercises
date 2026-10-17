#!/usr/bin/env python3
"""Stage 3: Pack the dataset into fast-loading numpy bundles.

Reads data/me2-dataset/manifest.csv + audio wavs, peak-normalizes, crops the
loudest 1 s window, and writes per-split .npz bundles:
  data/packs/<split>.npz  ->  waveforms (N, 16000) float32, labels (N,) int64,
                              is_synthetic (N,) int8, speaker (N,) str

Raw (cropped, normalized) waveforms are stored instead of precomputed
features so training can apply waveform-domain augmentation in the data
loader and extract log-mels on the fly.

Also packs a noise pool (white + pink) for augmentation.

Usage:
  python scripts/pack_data.py [--splits train,test] [--noise]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vcm.augment import make_noise_pool  # noqa: E402
from vcm.features import loudest_crop  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "me2-dataset" / "manifest.csv"
AUDIO = ROOT / "data" / "me2-dataset" / "audio"
OUT = ROOT / "data" / "packs"

TARGET_SAMPLES = 16000  # 1 s @ 16 kHz


def _load_norm_crop(path: Path) -> np.ndarray:
    wav, _ = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    peak = float(np.abs(wav).max())
    if peak > 1e-6:
        wav = wav / peak * 0.95
    if len(wav) < TARGET_SAMPLES:
        wav = np.pad(wav, (0, TARGET_SAMPLES - len(wav)))
    return loudest_crop(wav, TARGET_SAMPLES)


def pack_split(split: str, mf: pd.DataFrame) -> None:
    rows = mf[mf["split"] == split].reset_index(drop=True)
    wavs = np.zeros((len(rows), TARGET_SAMPLES), dtype=np.float32)
    labels = np.zeros(len(rows), dtype=np.int64)
    synth = np.zeros(len(rows), dtype=np.int8)
    spk = np.array([""] * len(rows), dtype=object)

    for i, row in tqdm(rows.iterrows(), total=len(rows), desc=split):
        p = AUDIO / split / (Path(row["file"]).stem + ".wav")
        wavs[i] = _load_norm_crop(p)
        labels[i] = int(row["label_id"])
        synth[i] = int(bool(row["is_synthetic"]))
        spk[i] = str(row["speaker_id"])
        # incremental save: a crash loses at most the last 500 clips
        if (i + 1) % 500 == 0:
            OUT.mkdir(parents=True, exist_ok=True)
            tmp = OUT / f"{split}.part.npz"
            np.savez_compressed(tmp, waveforms=wavs[:i + 1], labels=labels[:i + 1],
                                is_synthetic=synth[:i + 1], speaker=spk[:i + 1])
            tmp.replace(OUT / f"{split}.partial.npz")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{split}.npz", waveforms=wavs, labels=labels,
                        is_synthetic=synth, speaker=spk)
    print(f"{split}: {len(rows)} clips -> {OUT / (split + '.npz')}")


def pack_noise() -> None:
    pool = make_noise_pool(n_clips=32, duration_s=3.0,
                           rng=np.random.default_rng(42))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "noise.npz", noise=np.array(pool, dtype=object))
    print(f"noise pool: {len(pool)} clips -> {OUT / 'noise.npz'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,test,holdout,neg_train,neg_test")
    ap.add_argument("--noise", action="store_true")
    args = ap.parse_args()

    mf = pd.read_csv(MANIFEST)
    for s in [x.strip() for x in args.splits.split(",")]:
        pack_split(s, mf)
    if args.noise:
        pack_noise()
    return 0


if __name__ == "__main__":
    sys.exit(main())
