#!/usr/bin/env python3
"""Stage 5: Evaluate a trained VCM checkpoint on the held-out test set.

Computes the metrics required by the ME2 spec:
  - overall accuracy, balanced accuracy, macro P/R/F1 (+95% CI via bootstrap)
  - per-intent precision/recall/F1
  - confusion matrix + top-10 confusion pairs
  - real-voice vs synthetic-voice accuracy
  - OUT_OF_SCOPE: false-accept rate (OOS predicted as in-scope) and
    false-reject rate (in-scope predicted OOS)
  - inference latency (p50/p95/p99) on CPU

The packs store raw 1 s waveforms. Features are computed identically to
training (static_features -> log_mel_spectrogram) so the eval is
apples-to-apples with the training-time validation loop.

By default the evaluator combines test.npz + neg_test.npz, mirroring the
training validation set (which included negatives). Pass --no-neg to
evaluate on test.npz alone.

Usage:
  python scripts/evaluate_vcm.py --checkpoint models/crnn_seed0/best.pt
  python scripts/evaluate_vcm.py --checkpoint models/tcn_seed0/best.pt --tag tcn_test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vcm.models import build, count_params  # noqa: E402
from vcm.features import log_mel_spectrogram  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PACKS = ROOT / "data" / "packs"
RESULTS = ROOT / "results"
N_CLASSES = 20
N_MELS, N_FRAMES = 40, 100
INTENTS = [
    "PLAY_MUSIC", "WEATHER", "TIME", "LIGHT_ON", "LIGHT_OFF",
    "PAUSE", "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN",
    "CALL", "MESSAGE", "LIST_REMINDERS",
    "TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS", "COLOR", "CREATE_REMINDER",
    "OUT_OF_SCOPE",
]


def bootstrap_ci(y_true, y_pred, stat, n_boot=1000, ci=0.95):
    rng = np.random.default_rng(0)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(stat(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(vals, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


def static_features(waveforms: np.ndarray) -> np.ndarray:
    """Clean (unaugmented) log-mel features, identical to training."""
    out = np.empty((len(waveforms), 1, N_MELS, N_FRAMES), dtype=np.float32)
    for i, w in enumerate(waveforms):
        out[i] = log_mel_spectrogram(w, n_mels=N_MELS)[None]
    return out


def load_split(names: list[str]):
    """Load one or more packs and concatenate waveforms/labels/is_synthetic."""
    ws, ys, ss = [], [], []
    for name in names:
        z = np.load(PACKS / f"{name}.npz", allow_pickle=True)
        ws.append(z["waveforms"])
        ys.append(z["labels"])
        ss.append(z["is_synthetic"])
    return np.concatenate(ws), np.concatenate(ys), np.concatenate(ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", help="base split name (test, holdout)")
    ap.add_argument("--no-neg", action="store_true",
                    help="do NOT combine neg_<split>.npz (default: combine, matching training)")
    ap.add_argument("--latency-n", type=int, default=200)
    ap.add_argument("--tag", default=None, help="suffix for the output file name")
    args = ap.parse_args()

    t_start = time.time()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, weights_only=True)
    cfg = ckpt["config"]
    arch = cfg["arch"]
    model = build(arch, n_classes=N_CLASSES, n_mels=N_MELS, n_frames=N_FRAMES)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    n_params = count_params(model)

    # Load data (combine with negatives by default, matching training)
    split_names = [args.split]
    if not args.no_neg:
        neg_name = f"neg_{args.split}"
        if (PACKS / f"{neg_name}.npz").exists():
            split_names.append(neg_name)
    print(f"loading split(s): {split_names}")
    waveforms, y, synth = load_split(split_names)
    print(f"  n_clips={len(y)}  is_synthetic={int(synth.sum())}")

    # Compute features (identical to training)
    t_feat = time.time()
    X_np = static_features(waveforms)
    print(f"  features in {time.time()-t_feat:.1f}s  shape={X_np.shape}")
    X = torch.from_numpy(X_np)

    # Inference
    with torch.no_grad():
        logits = model(X)
    pred = logits.argmax(-1).numpy()
    probs = torch.softmax(logits, -1).numpy()

    # Core metrics
    acc = accuracy_score(y, pred)
    ba = balanced_accuracy_score(y, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    _, _, _, sup = precision_recall_fscore_support(y, pred, average=None, zero_division=0)
    acc_lo, acc_hi = bootstrap_ci(y, pred, accuracy_score)
    ba_lo, ba_hi = bootstrap_ci(y, pred, balanced_accuracy_score)

    # Per-intent
    cm = confusion_matrix(y, pred, labels=list(range(N_CLASSES)))
    per_intent = {}
    for c, name in enumerate(INTENTS):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per_intent[name] = dict(support=int(sup[c]), precision=round(p, 4),
                                recall=round(r, 4), f1=round(f, 4))

    # Top-10 confusion pairs (off-diagonal)
    pairs = []
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if i != j and cm[i, j] > 0:
                pairs.append((INTENTS[i], INTENTS[j], int(cm[i, j])))
    pairs.sort(key=lambda t: -t[2])

    # Real vs synthetic
    real_mask = synth == 0
    syn_mask = synth == 1
    real_acc = float((pred[real_mask] == y[real_mask]).mean()) if real_mask.any() else None
    syn_acc = float((pred[syn_mask] == y[syn_mask]).mean()) if syn_mask.any() else None

    # OOS behavior
    oos_idx = N_CLASSES - 1
    oos_true = y == oos_idx
    inscope_true = ~oos_true
    fa_rate = float(((pred != oos_idx) & oos_true).sum() / max(1, oos_true.sum()))
    fr_rate = float(((pred == oos_idx) & inscope_true).sum() / max(1, inscope_true.sum()))

    # Latency on CPU (single clip, batch=1)
    model.cpu()
    times = []
    with torch.no_grad():
        for i in range(args.latency_n):
            xb = X[i % len(X)][None]
            t0 = time.perf_counter()
            model(xb)
            times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)

    report = {
        "checkpoint": str(args.checkpoint),
        "arch": arch,
        "split": "+".join(split_names),
        "n_clips": int(len(y)),
        "n_params": n_params,
        "accuracy": round(acc, 4), "accuracy_ci95": [round(acc_lo, 4), round(acc_hi, 4)],
        "balanced_accuracy": round(ba, 4), "balanced_accuracy_ci95": [round(ba_lo, 4), round(ba_hi, 4)],
        "macro_precision": round(prec, 4), "macro_recall": round(rec, 4), "macro_f1": round(f1, 4),
        "real_voice_accuracy": round(real_acc, 4) if real_acc is not None else None,
        "synthetic_voice_accuracy": round(syn_acc, 4) if syn_acc is not None else None,
        "oos_false_accept_rate": round(fa_rate, 4),
        "oos_false_reject_rate": round(fr_rate, 4),
        "latency_ms": {"p50": round(float(np.percentile(times, 50)), 2),
                       "p95": round(float(np.percentile(times, 95)), 2),
                       "p99": round(float(np.percentile(times, 99)), 2)},
        "per_intent": per_intent,
        "top_confusion_pairs": [[a, b, n] for a, b, n in pairs[:10]],
        "confusion_matrix": cm.tolist(),
        "eval_seconds": round(time.time() - t_start, 1),
    }

    RESULTS.mkdir(exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out = RESULTS / f"eval_{arch}_{args.split}{tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    print(json.dumps({k: report[k] for k in [
        "arch", "n_clips", "accuracy", "balanced_accuracy", "macro_f1",
        "real_voice_accuracy", "synthetic_voice_accuracy",
        "oos_false_accept_rate", "oos_false_reject_rate",
        "latency_ms", "eval_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
