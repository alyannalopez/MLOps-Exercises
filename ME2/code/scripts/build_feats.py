#!/usr/bin/env python3
"""Step 4: extract features for train/val/test.

Uses the shared feature extractor (features.py):
  - MFCC    (T,40)  -> logistic / dnn
  - log-mel (T,80)  -> cnn1d / crnn
Both are cached to .npz so training/eval never re-reads audio.

Output: ME2/data/feats/{mfcc,logmel}_{train,val,test}.npz
Each .npz: X (N,T,F) float32, y (N,) int64 (class index 0..10, REJECT=10),
           paths, intent_name, intent_raw, condition, duration_s
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import features as F          # noqa: E402
from label_map import CLASSES, REJECT  # noqa: E402

FEATS = os.path.join(ROOT, "data", "feats")
SPLITS = os.path.join(ROOT, "data", "splits")
NAME2IDX = {n: i for i, n in enumerate(CLASSES)} | {"REJECT": REJECT}


def extract(split: str):
    df = pd.read_csv(os.path.join(SPLITS, f"{split}.csv")).reset_index(drop=True)
    t0 = time.time()
    mfcc, logmel = [], []
    for i, row in df.iterrows():
        p = row["abs_path"]
        mfcc.append(F.mfcc_features(p))
        logmel.append(F.logmel_features(p))
        if (i + 1) % 2000 == 0:
            print(f"  {split}: {i+1}/{len(df)} ({time.time()-t0:.0f}s)", flush=True)
    mfcc = np.stack(mfcc).astype(np.float32)
    logmel = np.stack(logmel).astype(np.float32)
    y = df["intent_name"].map(NAME2IDX).to_numpy(dtype=np.int64)
    payload = dict(
        paths=df["abs_path"].to_numpy(),
        intent_name=df["intent_name"].to_numpy(),
        intent_raw=df["intent_raw"].to_numpy(),
        condition=df["condition"].fillna("clean").to_numpy(),
        duration_s=df["duration_s"].to_numpy(dtype=np.float32),
    )
    os.makedirs(FEATS, exist_ok=True)
    np.savez_compressed(os.path.join(FEATS, f"mfcc_{split}.npz"),
                        X=mfcc, y=y, **payload)
    np.savez_compressed(os.path.join(FEATS, f"logmel_{split}.npz"),
                        X=logmel, y=y, **payload)
    print(f"{split}: mfcc {mfcc.shape}  logmel {logmel.shape}  ({time.time()-t0:.0f}s)",
          flush=True)


def main():
    for split in ("train", "val", "test"):
        extract(split)
    print("All features written ->", FEATS)


if __name__ == "__main__":
    main()
