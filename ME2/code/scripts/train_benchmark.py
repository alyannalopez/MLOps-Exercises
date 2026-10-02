#!/usr/bin/env python3
"""Step 5: train + benchmark all 4 trainable architectures.

Architectures (from models.py, shared contract: (B,T,F)->(B,11)):
  A. logistic  (MFCC mean-pool)     ~450 params
  B. dnn       (MFCC mean-pool)     ~14.6k params
  C. cnn1d     (log-mel)            ~246k params
  D. crnn      (log-mel)            ~332k params
(E. PocketSphinx-style grammar = architecture E, no training — reported
   separately as the classical baseline; see pocketsphinx_note.md.)

Each is trained up to 100 epochs (Adam, cosine LR, early-stop patience on
val acc), seed 42, MPS when available. Weights are NOT rebalanced so the
benchmark reflects the honest per-architecture comparison; the natural class
imbalance is reported transparently.

Output per arch: ME2/models/<arch>_state.pt, <arch>_history.json,
                 <arch>_bench.json
Plus ME2/models/benchmark.json summarizing all four.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from models import build as build_model, count_params  # noqa: E402
from label_map import CLASSES  # noqa: E402

FEATS = os.path.join(ROOT, "data", "feats")
MODELS = os.path.join(ROOT, "models")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load(split, kind):
    z = np.load(os.path.join(FEATS, f"{kind}_{split}.npz"), allow_pickle=True)
    return torch.from_numpy(z["X"]), torch.from_numpy(z["y"]), z


def train_one(name, epochs, lr, bs, patience, seed):
    kind = "mfcc" if name in ("logistic", "dnn") else "logmel"
    Xt, yt, _ = load("train", kind)
    Xv, yv, _ = load("val", kind)
    Xte, yte, _ = load("test", kind)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(name).to(DEVICE)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    dl = DataLoader(TensorDataset(Xt, yt), batch_size=bs, shuffle=True)
    Xvb, yvb = Xv.to(DEVICE), yv.to(DEVICE)
    Xtb, ytb = Xte.to(DEVICE), yte.to(DEVICE)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_acc": []}
    best_val, best_state, bad = -1.0, None, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot, corr, n = 0.0, 0, 0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
            corr += (out.argmax(1) == yb).sum().item()
            n += xb.size(0)
        sched.step()
        model.eval()
        with torch.no_grad():
            vo = model(Xvb)
            vacc = (vo.argmax(1) == yvb).float().mean().item()
        tracc = corr / n
        history["epoch"].append(ep)
        history["train_loss"].append(round(tot / n, 4))
        history["train_acc"].append(round(tracc, 4))
        history["val_acc"].append(round(vacc, 4))
        if vacc > best_val:
            best_val = vacc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
        if ep % 10 == 0 or ep == 1:
            print(f"  [{name}] ep{ep:3d} loss={tot/n:.3f} tr={tracc:.3f} va={vacc:.3f} "
                  f"(best {best_val:.3f})", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        to = model(Xtb)
        tacc = (to.argmax(1) == ytb).float().mean().item()
    train_time = time.time() - t0
    params = count_params(model)

    os.makedirs(MODELS, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "name": name, "kind": kind,
                "classes": CLASSES, "best_val": round(best_val, 4),
                "test_acc": round(float(tacc), 4), "params": params},
               os.path.join(MODELS, f"{name}_state.pt"))
    with open(os.path.join(MODELS, f"{name}_history.json"), "w") as f:
        json.dump(history, f)
    bench = {"name": name, "params": params, "test_acc": round(float(tacc), 4),
             "best_val": round(best_val, 4), "train_time_s": round(train_time, 1),
             "epochs_run": len(history["epoch"])}
    with open(os.path.join(MODELS, f"{name}_bench.json"), "w") as f:
        json.dump(bench, f, indent=2)
    return bench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--archs", type=str, default="logistic,dnn,cnn1d,crnn")
    a = ap.parse_args()
    print(f"Device: {DEVICE}", flush=True)
    results = []
    for name in a.archs.split(","):
        name = name.strip()
        print(f"\n===== {name} =====", flush=True)
        results.append(train_one(name, a.epochs, a.lr, a.bs, a.patience, a.seed))
    results.sort(key=lambda r: r["test_acc"], reverse=True)
    with open(os.path.join(MODELS, "benchmark.json"), "w") as f:
        json.dump({"device": DEVICE, "seed": a.seed, "epochs": a.epochs,
                   "results": results, "best": results[0]["name"]}, f, indent=2)
    print("\n===== BENCHMARK =====", flush=True)
    for r in results:
        print(f"  {r['name']:9s} test={r['test_acc']:.4f}  params={r['params']:>8d}  "
              f"time={r['train_time_s']}s", flush=True)
    print(f"BEST: {results[0]['name']}", flush=True)


if __name__ == "__main__":
    main()
