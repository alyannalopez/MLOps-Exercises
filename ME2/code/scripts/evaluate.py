#!/usr/bin/env python3
"""Step 6: comprehensive evaluation of the best model.

Computes the full metric battery required by the exercise:
  - intent recognition: accuracy, Cohen kappa, P/R/F1 (macro/micro/weighted)
  - per-intent P/R/F1/support + full-command (raw-intent) accuracy
  - confusion matrix (11x11 incl. REJECT)
  - command recall & rejection: reject recall, false-reject rate,
    command success rate (task completion)
  - calibration: ECE (10 bins), Brier score, reliability diagram data
  - efficiency: params, MACs per clip, model size (fp32/int8)
  - latency: inference time p50/p95/p99/max (CPU, batch=1) + RTF
  - slot/value accuracy: per slotted raw-intent accuracy (within correct
    10-class) + "all" row
  - WER proxy: token-F1 between gold command and canonical phrase of the
    predicted intent (ASR-free, constraint #8)
  - robustness: by condition (clean/noisy), by duration bucket,
    clean->noisy delta, per-value accuracy

Output: ME2/models/eval_results.json, ME2/models/eval_detail.csv
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             precision_recall_fscore_support, confusion_matrix)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from models import build as build_model, count_params, N_CLASSES, REJECT  # noqa: E402
from label_map import CLASSES, CANON_PHRASE, SLOTTED_RAW  # noqa: E402

FEATS = os.path.join(ROOT, "data", "feats")
MODELS = os.path.join(ROOT, "models")
IDX2NAME = {i: n for i, n in enumerate(CLASSES)} | {REJECT: "REJECT"}


def load(split, kind):
    z = np.load(os.path.join(FEATS, f"{kind}_{split}.npz"), allow_pickle=True)
    return (torch.from_numpy(z["X"]), torch.from_numpy(z["y"]),
            {k: z[k] for k in ("paths", "intent_name", "intent_raw",
                               "condition", "duration_s")})


def tok_f1(a: str, b: str) -> float:
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return 0.0
    from collections import Counter
    ca, cb = Counter(ta), Counter(tb)
    inter = sum((ca & cb).values())
    p = inter / len(ta)
    r = inter / len(tb)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def estimate_macs(model, x):
    """Rough MAC estimate via forward hooks (conv + linear)."""
    macs = 0
    def count_conv(m, xin, xout):
        nonlocal macs
        xin = xin[0] if isinstance(xin, tuple) else xin
        kh = m.kernel_size
        kh = kh if isinstance(kh, int) else kh[0]
        tin = xin.shape[-1]
        tout = xout.shape[-1]
        macs += m.in_channels * kh * tin * m.out_channels * tout
    def count_lin(m, xin, xout):
        nonlocal macs
        xin = xin[0] if isinstance(xin, tuple) else xin
        macs += m.in_features * m.out_features * xin.shape[0]
    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(count_conv))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(count_lin))
    model.eval()
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return macs // x.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=None, help="arch name (default: best)")
    ap.add_argument("--latency-n", type=int, default=800)
    a = ap.parse_args()
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

    bench = json.load(open(os.path.join(MODELS, "benchmark.json")))
    name = a.model or bench["best"]
    ckpt = torch.load(os.path.join(MODELS, f"{name}_state.pt"), map_location="cpu")
    kind = ckpt["kind"]
    model = build_model(name).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    X, y, meta = load("test", kind)
    Xd, yd = X.to(DEVICE), y.to(DEVICE)
    with torch.no_grad():
        logits = model(Xd)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred = probs.argmax(1)
    gold = y.numpy()
    names = meta["intent_name"]
    raws = meta["intent_raw"]
    cond = meta["condition"]
    dur = meta["duration_s"]
    confs = probs.max(1)

    out = {"model": name, "kind": kind, "device": DEVICE,
           "n_test": int(len(gold)), "classes": CLASSES}

    # ---- intent recognition ----
    acc = accuracy_score(gold, pred)
    prec, rec, f1, sup = precision_recall_fscore_support(
        gold, pred, labels=list(range(N_CLASSES)), zero_division=0)
    out["accuracy"] = round(float(acc), 4)
    out["cohen_kappa"] = round(float(cohen_kappa_score(gold, pred)), 4)
    out["macro"] = {"precision": round(float(prec.mean()), 4),
                    "recall": round(float(rec.mean()), 4),
                    "f1": round(float(f1.mean()), 4)}
    out["micro"] = {"precision": round(float(np.average(prec, weights=sup)), 4),
                    "recall": round(float(np.average(rec, weights=sup)), 4),
                    "f1": round(float(np.average(f1, weights=sup)), 4)}
    w = sup.astype(float)
    w /= w.sum()
    out["weighted"] = {"precision": round(float(np.average(prec, weights=w)), 4),
                       "recall": round(float(np.average(rec, weights=w)), 4),
                       "f1": round(float(np.average(f1, weights=w)), 4)}

    # ---- per-intent ----
    per = []
    for i, cn in enumerate(CLASSES):
        gi = (gold == i)
        ri = (pred == i)
        tp = int((gi & ri).sum())
        p = tp / max(1, int(ri.sum()))
        r = tp / max(1, int(gi.sum()))
        f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        per.append({"intent": cn, "idx": i, "precision": round(p, 4),
                    "recall": round(r, 4), "f1": round(f, 4),
                    "support": int(gi.sum())})
    out["per_intent"] = per

    # ---- confusion matrix ----
    cm = confusion_matrix(gold, pred, labels=list(range(N_CLASSES)))
    out["confusion"] = cm.tolist()

    # ---- command recall & rejection (task completion) ----
    cmd = gold != REJECT
    rej = gold == REJECT
    out["command_recall"] = round(float(((gold == pred) & cmd).sum() / max(1, cmd.sum())), 4)
    out["reject_recall"] = round(float(((gold == pred) & rej).sum() / max(1, rej.sum())), 4)
    out["false_reject_rate"] = round(float(((pred == REJECT) & cmd).sum() / max(1, cmd.sum())), 4)
    out["false_accept_rate"] = round(float(((pred != REJECT) & rej).sum() / max(1, rej.sum())), 4)
    out["task_completion"] = round(float((gold == pred).mean()), 4)

    # ---- calibration ----
    bins = np.linspace(0, 1, 11)
    rel = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confs > lo) & (confs <= hi)
        if m.sum() > 0:
            rel.append({"bin": [round(float(lo), 2), round(float(hi), 2)],
                        "n": int(m.sum()),
                        "mean_conf": round(float(confs[m].mean()), 4),
                        "acc": round(float((gold[m] == pred[m]).mean()), 4)})
    out["reliability"] = rel
    ece = sum(r["n"] / len(gold) * abs(r["acc"] - r["mean_conf"]) for r in rel)
    out["ece"] = round(float(ece), 4)
    out["brier"] = round(float(np.mean(((probs - np.eye(N_CLASSES)[gold]) ** 2).sum(axis=1))), 4)

    # ---- efficiency ----
    params = count_params(model)
    sd = model.state_dict()
    fp32_mb = sum(t.numel() for t in sd.values()) * 4 / 1e6
    macs_model = build_model(name)
    macs_model.load_state_dict(ckpt["state_dict"])
    out["efficiency"] = {
        "params": int(params),
        "macs_per_clip": int(estimate_macs(macs_model, X[:64].cpu())),
        "size_fp32_mb": round(fp32_mb, 3),
        "size_int8_kb": round(fp32_mb * 1024 / 4, 1),
    }

    # ---- latency (CPU, batch=1, realistic on-device condition) ----
    model_cpu = build_model(name)
    model_cpu.load_state_dict({k: v for k, v in ckpt["state_dict"].items()})
    model_cpu.eval()
    n_lat = min(a.latency_n, len(X))
    xs = X[:n_lat].to(torch.float32)
    times = []
    with torch.no_grad():
        for i in range(n_lat):
            t0 = time.perf_counter()
            model_cpu(xs[i:i + 1])
            times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    out["latency_ms"] = {
        "device": "cpu", "n": int(n_lat),
        "mean": round(float(times.mean()), 2),
        "p50": round(float(np.percentile(times, 50)), 2),
        "p95": round(float(np.percentile(times, 95)), 2),
        "p99": round(float(np.percentile(times, 99)), 2),
        "max": round(float(times.max()), 2),
    }
    out["rtf"] = round(out["latency_ms"]["mean"] / 1000.0, 5)  # 1 s audio window

    # ---- slot / value accuracy (within correct 10-class predictions) ----
    pv = {}
    for rv in sorted(set(raws)):
        m = (raws == rv) & cmd
        if m.sum() >= 5:
            pv[rv] = {"acc": round(float((gold[m] == pred[m]).mean()), 4),
                      "n": int(m.sum())}
    out["slot_value_acc"] = pv
    allm = cmd
    out["slot_value_acc"]["ALL"] = {
        "acc": round(float((gold[allm] == pred[allm]).mean()), 4),
        "n": int(allm.sum()),
    }

    # ---- WER proxy (token-F1 vs canonical phrase) ----
    tf1s = []
    for i in range(len(gold)):
        if gold[i] == REJECT:
            continue
        ref = CANON_PHRASE[gold[i]]
        hyp = CANON_PHRASE.get(int(pred[i]), "")
        tf1s.append(tok_f1(ref, hyp))
    out["wer_proxy_token_f1"] = round(float(np.mean(tf1s)), 4) if tf1s else None
    out["wer_note"] = ("ASR-free token-F1 between the gold command and the canonical "
                       "phrase of the predicted intent (higher = closer to the true "
                       "command). No speech recognition used (constraint #8). True WER "
                       "is not measurable without ASR.")

    # ---- robustness ----
    def acc_of(mask):
        return round(float((gold[mask] == pred[mask]).mean()), 4) if mask.sum() > 0 else None
    bc = {c: acc_of(cond == c) for c in np.unique(cond)}
    out["robustness"] = {
        "by_condition": bc,
        "by_duration_bucket": {
            "<1.0s": acc_of(dur < 1.0),
            "1.0-1.5s": acc_of((dur >= 1.0) & (dur < 1.5)),
            ">=1.5s": acc_of(dur >= 1.5),
        },
        "clean_to_noisy_delta": (round(bc["noisy"] - bc["clean"], 4)
                                 if "clean" in bc and "noisy" in bc else None),
        "per_value_acc": {k: v["acc"] for k, v in sorted(pv.items(), key=lambda x: x[1]["acc"])},
    }

    # detail csv
    det = pd.DataFrame({"gold": names, "pred": [IDX2NAME[p] for p in pred],
                        "raw": raws, "condition": cond, "duration_s": dur,
                        "confidence": confs, "correct": gold == pred})
    det.to_csv(os.path.join(MODELS, "eval_detail.csv"), index=False)

    with open(os.path.join(MODELS, "eval_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("confusion", "per_intent", "robustness",
                                   "reliability", "slot_value_acc")}, indent=2))
    print("\nPer-intent:")
    for p in out["per_intent"]:
        print(f"  {p['intent']:18s} P={p['precision']:.3f} R={p['recall']:.3f} "
              f"F1={p['f1']:.3f} n={p['support']}")
    print("\nRobustness:", json.dumps(out["robustness"], indent=2))


if __name__ == "__main__":
    main()
