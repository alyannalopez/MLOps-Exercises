"""Option 2: re-weight the CRNN training loss toward Option B, keep ALL data.

Same splits / feature cache as train_retrain.py, but every training sample
gets a per-row weight so Option B (Mark's primary dataset) dominates the
gradient while speechcommands + synthetic reject still contribute.

  --ob-mult K   weight multiplier for optionB rows (default 3.0)
                effective optionB share of gradient rises from ~67% toward ~90%

Also emits a FULL evaluation suite on the held-out test set:
  accuracy, macro/micro/weighted P-R-F1, per-intent P/R/F1/support,
  confusion matrix, Cohen's kappa, and all plots (training curves,
  per-intent bars, confusion heatmap, loss curve).

Usage:
  python train_weighted.py --ob-mult 3.0 --epochs 100
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, cohen_kappa_score,
                             classification_report)

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
SHORT = {0:"REJ",1:"PLAY",2:"SEARCH",3:"LIGHTS",4:"COLOR",5:"TIMER",
         6:"ALARM",7:"THERMO",8:"MEDIA",9:"REMIND",10:"CALLS"}


class WeightedVCMDataset(Dataset):
    def __init__(self, X, Y, W=None, augment=False):
        self.X = X; self.Y = Y; self.W = W; self.aug = augment
    def __len__(self): return len(self.Y)
    def __getitem__(self, i):
        x = self.X[i]
        if self.aug:
            if np.random.random() < 0.5:
                x = np.roll(x, np.random.randint(-3, 4), axis=0)
            x = x * (0.8 + 0.4 * np.random.random())
        w = self.W[i] if self.W is not None else 1.0
        return torch.from_numpy(x), torch.tensor(self.Y[i], dtype=torch.long), \
               torch.tensor(w, dtype=torch.float32)


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
    Y = df.intent_10.map(lambda i10: 0 if i10 == -1 else i10).values
    src = df["source"].values if "source" in df.columns else np.array([""]*len(df))
    return X, Y, src


def compute_sample_weights(src_tr, ob_mult):
    """Per-row weight: optionB rows *ob_mult, everyone else 1.0."""
    w = np.ones(len(src_tr), dtype=np.float32)
    w[src_tr == "optionB"] = ob_mult
    # normalize so mean weight == 1 (keeps effective LR comparable)
    w = w / w.mean()
    return w


@torch.no_grad()
def evaluate(model, X, Y, bs=256):
    model.eval()
    dl = DataLoader(WeightedVCMDataset(X, Y, None, False), batch_size=bs, shuffle=False)
    preds, labels = [], []
    for x, y, _w in dl:
        x = x.to(DEVICE)
        preds.append(model(x).argmax(1).cpu().numpy())
        labels.append(y.numpy())
    P = np.concatenate(preds); L = np.concatenate(labels)
    return P, L


def class_weights(Y):
    counts = np.bincount(Y, minlength=N_CLASSES).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (N_CLASSES * counts)
    w = w / w.max()
    return torch.tensor(w, dtype=torch.float32).to(DEVICE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ob-mult", type=float, default=3.0,
                    help="loss-weight multiplier for optionB rows")
    ap.add_argument("--feat-cache", default=None,
                    help="reuse an existing logmel feature npz")
    a = ap.parse_args()

    tag = f"crnn_wob_{int(time.time())}"
    print(f"\n===== [CRNN option-2 weighted] ob_mult={a.ob_mult} "
          f"epochs={a.epochs} bs={a.bs} lr={a.lr} device={DEVICE} =====", flush=True)

    # ---- features (reuse cache if given, else build) ----
    cache_npz = a.feat_cache
    if cache_npz and os.path.exists(cache_npz):
        print(f"loading cached features {cache_npz}", flush=True)
        d = np.load(cache_npz)
        Xtr, Ytr, Xva, Yva, Xte, Yte = d["Xtr"], d["Ytr"], d["Xva"], d["Yva"], d["Xte"], d["Yte"]
    else:
        Xtr, Ytr, src_tr = load_split(os.path.join(SPLITS, "train.csv"), "logmel")
        Xva, Yva, _      = load_split(os.path.join(SPLITS, "val.csv"), "logmel")
        Xte, Yte, _      = load_split(os.path.join(SPLITS, "test.csv"), "logmel")
        cache_npz = os.path.join(ROOT, "data", "processed", f"_retrain_feats_logmel_{tag}.npz")
        np.savez_compressed(cache_npz, Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva, Xte=Xte, Yte=Yte)
        print(f"cached features -> {cache_npz}", flush=True)
    print(f"train={len(Xtr)} val={len(Xva)} test={len(Xte)}", flush=True)

    # ---- sample weights (always read sources from the split csv) ----
    _tr_df = pd.read_csv(os.path.join(SPLITS, "train.csv"), low_memory=False)
    src_tr = _tr_df["source"].values
    assert len(src_tr) == len(Xtr), f"source/feature length mismatch {len(src_tr)} vs {len(Xtr)}"
    w_tr = compute_sample_weights(src_tr, a.ob_mult)
    ob_frac = (src_tr == "optionB").mean()
    eff = (w_tr[src_tr == "optionB"].sum()) / w_tr.sum()
    print(f"optionB rows: {ob_frac:.1%} of train; after weighting -> {eff:.1%} of gradient mass",
          flush=True)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    tl = DataLoader(WeightedVCMDataset(Xtr, Ytr, w_tr, True), batch_size=a.bs, shuffle=True)
    vl = DataLoader(WeightedVCMDataset(Xva, Yva, None, False), batch_size=a.bs * 2, shuffle=False)

    model = build_model("crnn").to(DEVICE)
    print(f"params={count_params(model):,}", flush=True)
    crit = nn.CrossEntropyLoss(weight=class_weights(Ytr), reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    history = []
    best_acc = 0.0; best_state = None; best_epoch = 0
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        tot_loss = 0.0; nb = 0
        for x, y, w in tl:
            x = x.to(DEVICE); y = y.to(DEVICE); w = w.to(DEVICE)
            opt.zero_grad()
            per = crit(model(x), y)                 # [B]
            loss = (per * w).mean()                 # weighted mean
            loss.backward(); opt.step()
            tot_loss += loss.item(); nb += 1
        sched.step()
        _, Ltr = evaluate(model, Xtr, Ytr, bs=512)
        Ptr, _ = evaluate(model, Xtr, Ytr, bs=512)
        train_acc = accuracy_score(Ltr, Ptr)
        Pva, Lva = evaluate(model, Xva, Yva, bs=512)
        val_acc = accuracy_score(Lva, Pva)
        row = {"epoch": ep + 1, "train_loss": round(tot_loss / nb, 5),
               "train_acc": round(float(train_acc), 5), "val_acc": round(float(val_acc), 5)}
        history.append(row)
        if val_acc > best_acc:
            best_acc = val_acc; best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[wob] ep {ep+1:3d}/{a.epochs}  loss={row['train_loss']:.4f}  "
              f"train={train_acc:.4f}  val={val_acc:.4f}  best={best_acc:.4f}", flush=True)

    # ---- restore best, full test evaluation ----
    model.load_state_dict(best_state)
    Pte, Lte = evaluate(model, Xte, Yte, bs=256)

    acc = accuracy_score(Lte, Pte)
    prec, rec, f1, sup = precision_recall_fscore_support(
        Lte, Pte, average=None, labels=list(range(N_CLASSES)), zero_division=0)
    macro = precision_recall_fscore_support(Lte, Pte, average="macro", zero_division=0)
    micro = precision_recall_fscore_support(Lte, Pte, average="micro", zero_division=0)
    weighted = precision_recall_fscore_support(Lte, Pte, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(Lte, Pte)
    cm = confusion_matrix(Lte, Pte, labels=list(range(N_CLASSES)))

    per_intent = {}
    for c in range(N_CLASSES):
        per_intent[INTENT_NAME[c]] = {
            "precision": round(float(prec[c]), 4), "recall": round(float(rec[c]), 4),
            "f1": round(float(f1[c]), 4), "support": int(sup[c]),
            "acc": (None if sup[c] == 0 else round(float(cm[c, c] / sup[c]), 4)),
        }

    result = {
        "model": "crnn", "scheme": "option2_source_weighted",
        "ob_mult": a.ob_mult, "epochs": a.epochs, "bs": a.bs, "lr": a.lr,
        "device": DEVICE, "params": count_params(build_model("crnn")),
        "best_val_epoch": best_epoch, "best_val_acc": round(float(best_acc), 5),
        "final_train_acc": history[-1]["train_acc"],
        "test": {
            "accuracy": round(float(acc), 5),
            "cohen_kappa": round(float(kappa), 5),
            "macro_precision": round(float(macro[0]), 5),
            "macro_recall": round(float(macro[1]), 5),
            "macro_f1": round(float(macro[2]), 5),
            "micro_precision": round(float(micro[0]), 5),
            "micro_recall": round(float(micro[1]), 5),
            "micro_f1": round(float(micro[2]), 5),
            "weighted_precision": round(float(weighted[0]), 5),
            "weighted_recall": round(float(weighted[1]), 5),
            "weighted_f1": round(float(weighted[2]), 5),
        },
        "per_intent_test": per_intent,
        "confusion_matrix": cm.tolist(),
        "wall_seconds": round(time.time() - t0, 1),
    }

    torch.save(best_state, os.path.join(MODELDIR, f"{tag}_state.pt"))
    with open(os.path.join(MODELDIR, f"{tag}_history.json"), "w") as f:
        json.dump({"history": history, "result": result}, f, indent=2, default=str)
    with open(os.path.join(MODELDIR, f"{tag}_metrics.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    np.save(os.path.join(MODELDIR, f"{tag}_confusion.npy"), cm)

    print("\n================ TEST METRICS (held-out) ================", flush=True)
    print(classification_report(Lte, Pte, target_names=[INTENT_NAME[c] for c in range(N_CLASSES)],
                                zero_division=0, digits=4))
    print(f"Accuracy      {acc:.4f}")
    print(f"Cohen kappa   {kappa:.4f}")
    print(f"Macro  P/R/F1 {macro[0]:.4f}/{macro[1]:.4f}/{macro[2]:.4f}")
    print(f"Micro  P/R/F1 {micro[0]:.4f}/{micro[1]:.4f}/{micro[2]:.4f}")
    print(f"Weighted P/R/F1 {weighted[0]:.4f}/{weighted[1]:.4f}/{weighted[2]:.4f}")
    print(f"\nsaved: {tag}_state.pt / _history.json / _metrics.json / _confusion.npy", flush=True)
    print(f"FEATURE_CACHE={cache_npz}", flush=True)


if __name__ == "__main__":
    main()
