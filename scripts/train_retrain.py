"""Re-train the 4 VCM models to 50-100 epochs with full history logging.

Unlike train.py this script:
  * records train/val accuracy + loss EVERY epoch
  * saves best-state weights + history JSON (no ONNX export here)
  * evaluates the best state on the full held-out test set at the end

Usage:
  python train_retrain.py --model crnn --epochs 100
  python train_retrain.py --model all --epochs 100
"""
from __future__ import annotations
import argparse, os, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import features as F
from models import build as build_model, count_params, N_CLASSES

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPLITS = os.path.join(ROOT, "data", "processed", "splits")
MODELDIR = os.path.join(ROOT, "models_retrain")
os.makedirs(MODELDIR, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
INTENT_NAME = {0:"REJECT",1:"PLAY_MUSIC",2:"QUESTION_SEARCH",3:"LIGHTS_ON_OFF",
               4:"DIM_COLOR_LIGHTS",5:"SET_TIMER",6:"SET_ALARM",7:"THERMOSTAT",
               8:"MEDIA_CONTROL",9:"REMINDERS_LISTS",10:"CALLS_MESSAGING"}

def to_class(i10: int) -> int:
    return 0 if i10 == -1 else i10

class VCMDataset(Dataset):
    def __init__(self, X, Y, augment=False):
        self.X = X; self.Y = Y; self.aug = augment
    def __len__(self): return len(self.Y)
    def __getitem__(self, i):
        x = self.X[i]
        if self.aug:
            if np.random.random() < 0.5:
                x = np.roll(x, np.random.randint(-3, 4), axis=0)
            x = x * (0.8 + 0.4 * np.random.random())
        return torch.from_numpy(x), torch.tensor(self.Y[i], dtype=torch.long)

def load_split(csv_path, kind):
    df = pd.read_csv(csv_path, low_memory=False).dropna(subset=["abs_path"]).reset_index(drop=True)
    fn = F.mfcc_features if kind == "mfcc" else F.logmel_features
    T, Fd = F.shape_of(kind)
    X = np.zeros((len(df), T, Fd), dtype=np.float32)
    for i, p in enumerate(df.abs_path.values):
        feat = fn(str(p))
        if feat.shape[0] < T:
            feat = np.pad(feat, ((0, T - feat.shape[0]), (0, 0)), mode="edge")
        else:
            feat = feat[:T]
        X[i] = feat
        if (i + 1) % 2000 == 0:
            print(f"  features {i+1}/{len(df)}", flush=True)
    Y = df.intent_10.map(to_class).values
    return X, Y

def class_weights(Y):
    counts = np.bincount(Y, minlength=N_CLASSES).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (N_CLASSES * counts)
    w = w / w.max()
    return torch.tensor(w, dtype=torch.float32).to(DEVICE)

@torch.no_grad()
def evaluate(model, X, Y, bs=256):
    model.eval()
    dl = DataLoader(VCMDataset(X, Y, augment=False), batch_size=bs, shuffle=False)
    correct = 0; total = 0
    per_correct = np.zeros(N_CLASSES); per_seen = np.zeros(N_CLASSES)
    for x, y in dl:
        x = x.to(DEVICE); y = y.to(DEVICE)
        pred = model(x).argmax(1)
        m = (pred == y)
        correct += m.sum().item(); total += y.numel()
        for yi, pi in zip(y.tolist(), pred.tolist()):
            per_seen[yi] += 1
            if pi == yi:
                per_correct[yi] += 1
    per_acc = np.where(per_seen > 0, per_correct / np.maximum(per_seen, 1), np.nan)
    return correct / total, per_acc

def train_one(name: str, epochs: int, bs: int, lr: float, wd: float, seed: int):
    kind = "mfcc" if name in ("logistic", "dnn") else "logmel"
    tag = f"{name}_retrain_{int(time.time())}"
    print(f"\n===== [{name}] kind={kind} epochs={epochs} bs={bs} lr={lr} =====", flush=True)

    # reuse features across models that share a kind (mfcc: logistic+dnn, logmel: cnn1d+crnn)
    cache_tag = f"{kind}_{int(time.time() // 86400)}"
    cache_npz = os.path.join(ROOT, "data", "processed", f"_retrain_feats_{cache_tag}.npz")
    if os.path.exists(cache_npz):
        print(f"loading cached features {cache_npz}", flush=True)
        d = np.load(cache_npz)
        Xtr, Ytr, Xva, Yva, Xte, Yte = d["Xtr"], d["Ytr"], d["Xva"], d["Yva"], d["Xte"], d["Yte"]
    else:
        Xtr, Ytr = load_split(os.path.join(SPLITS, "train.csv"), kind)
        Xva, Yva = load_split(os.path.join(SPLITS, "val.csv"), kind)
        Xte, Yte = load_split(os.path.join(SPLITS, "test.csv"), kind)
        np.savez_compressed(cache_npz, Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva, Xte=Xte, Yte=Yte)
        print(f"cached features -> {cache_npz}", flush=True)
    print(f"train={len(Xtr)} val={len(Xva)} test={len(Xte)}", flush=True)

    torch.manual_seed(seed); np.random.seed(seed)
    tl = DataLoader(VCMDataset(Xtr, Ytr, augment=True), batch_size=bs, shuffle=True)
    vl = DataLoader(VCMDataset(Xva, Yva, augment=False), batch_size=bs * 2, shuffle=False)

    model = build_model(name).to(DEVICE)
    print(f"params={count_params(model):,}", flush=True)
    crit = nn.CrossEntropyLoss(weight=class_weights(Ytr))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = []
    best_acc = 0.0; best_state = None; best_epoch = 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot_loss = 0.0; nb = 0
        for x, y in tl:
            x = x.to(DEVICE); y = y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            tot_loss += loss.item(); nb += 1
        sched.step()
        train_acc, _ = evaluate(model, Xtr, Ytr, bs=512)
        val_acc, _ = evaluate(model, Xva, Yva, bs=512)
        row = {"epoch": ep + 1, "train_loss": tot_loss / nb,
               "train_acc": round(train_acc, 5), "val_acc": round(val_acc, 5)}
        history.append(row)
        if val_acc > best_acc:
            best_acc = val_acc; best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{name}] ep {ep+1:3d}/{epochs}  loss={row['train_loss']:.4f}  "
              f"train={train_acc:.4f}  val={val_acc:.4f}  best={best_acc:.4f}", flush=True)

    # restore best and evaluate on the held-out test set
    model.load_state_dict(best_state)
    test_acc, per_acc = evaluate(model, Xte, Yte, bs=256)
    per_intent = {INTENT_NAME[c]: (None if np.isnan(a) else round(float(a), 4))
                  for c, a in enumerate(per_acc)}
    result = {
        "model": name, "kind": kind, "epochs": epochs, "bs": bs, "lr": lr,
        "params": count_params(build_model(name)),
        "best_val_epoch": best_epoch,
        "best_val_acc": round(float(best_acc), 5),
        "final_train_acc": history[-1]["train_acc"],
        "test_acc": round(float(test_acc), 5),
        "per_intent_test_acc": per_intent,
        "wall_seconds": round(time.time() - t0, 1),
    }
    torch.save(best_state, os.path.join(MODELDIR, f"{tag}_state.pt"))
    with open(os.path.join(MODELDIR, f"{tag}_history.json"), "w") as f:
        json.dump({"result": result, "history": history}, f, indent=1)
    print(f"[{name}] DONE  best_val={best_acc:.4f} (ep {result['best_val_epoch']})  "
          f"TEST={test_acc:.4f}  ({result['wall_seconds']}s)", flush=True)
    return result, history

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["logistic", "dnn", "cnn1d", "crnn", "all"], default="all")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    names = ["logistic", "dnn", "cnn1d", "crnn"] if a.model == "all" else [a.model]
    results = []
    for n in names:
        r, _ = train_one(n, a.epochs, a.bs, a.lr, a.wd, a.seed)
        results.append(r)
    with open(os.path.join(MODELDIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("\n===== SUMMARY =====")
    for r in results:
        print(f"{r['model']:9s} params={r['params']:>8,}  best_val={r['best_val_acc']:.4f}  TEST={r['test_acc']:.4f}")

if __name__ == "__main__":
    main()
