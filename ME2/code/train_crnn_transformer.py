#!/usr/bin/env python3
"""Train only CRNN and Transformer (the other 3 are already done)."""
import sys, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / "code"))
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "data" / "features"
MODEL_DIR = ROOT / "models"

from train_models import CRNN, TinyTransformer, train_model, LABEL2IDX, N_CLASSES
import pandas as pd

torch.manual_seed(42)
np.random.seed(42)

# Load features
mfcc_tr = np.load(FEAT_DIR / "train.npz")["mfcc"].copy()
mfcc_ho = np.load(FEAT_DIR / "holdout.npz")["mfcc"].copy()
mfcc_te = np.load(FEAT_DIR / "test.npz")["mfcc"].copy()

man = pd.read_csv(ROOT / "data" / "manifest.csv")
y_all = {}
for split_name in ["train", "test", "holdout"]:
    sub = man[man["_split"] == split_name]
    y_all[split_name] = sub["command"].map(LABEL2IDX).fillna(N_CLASSES-1).astype(int).values

print(f"Train: {mfcc_tr.shape}, Val: {mfcc_ho.shape}, Test: {mfcc_te.shape}")

results = {}
for name, cls in [("crnn", CRNN), ("transformer", TinyTransformer)]:
    # Monkey-patch the MODELS dict
    import train_models
    train_models.MODELS[name] = lambda c=cls: c()
    r = train_model(name, mfcc_tr, y_all["train"], mfcc_ho, y_all["holdout"], mfcc_te, y_all["test"])
    results[name] = r

# Update benchmark summary
with open(MODEL_DIR / "benchmark_summary.json") as f:
    bench = json.load(f)
for name, r in results.items():
    bench[name] = {k: v for k, v in r.items() if k not in ('test_preds', 'test_y')}

with open(MODEL_DIR / "benchmark_summary.json", "w") as f:
    json.dump(bench, f, indent=2)

print(f"\n{'='*60}\nFINAL SUMMARY\n{'='*60}")
for name, r in bench.items():
    print(f"  {name:15s} | {r['params']:>10,} params | val {r['best_val_acc']:.4f} | test {r['test_acc']:.4f}")
