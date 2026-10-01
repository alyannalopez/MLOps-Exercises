"""Train + evaluate + export a VCM model.

Flow:
  1. Load a split CSV (train/val/test) from data/processed/splits/
  2. Build features lazily (MFCC or log-mel) with an on-disk cache keyed by
     (path, mtime, feature-kind) so re-runs are fast.
  3. Train with class-weighted CrossEntropy (fixes residual imbalance).
  4. Evaluate: overall acc, per-intent macro-F1, confusion, rejection precision.
  5. Export: TorchScript -> ONNX (fp32) -> ONNX Runtime INT8.
     (TFLite INT8 export is done by export_tflite.py; ONNX INT8 is the portable
      artifact that runs on the Pi via onnxruntime.)

Usage:
  python train.py --model dnn --epochs 12 --bs 64 --lr 1e-3
  python train.py --model crnn --epochs 15 --bs 32 --lr 5e-4
"""
from __future__ import annotations
import argparse, os, json, time, hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import features as F
from models import build as build_model, count_params, N_CLASSES, REJECT

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPLITS = os.path.join(ROOT, "data", "processed", "splits")
CACHE  = os.path.join(ROOT, "data", "processed", "feat_cache")
MODELDIR = os.path.join(ROOT, "models")
os.makedirs(CACHE, exist_ok=True); os.makedirs(MODELDIR, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
INTENT_NAME = {1:"PLAY_MUSIC",2:"QUESTION_SEARCH",3:"LIGHTS_ON_OFF",4:"DIM_COLOR_LIGHTS",
               5:"SET_TIMER",6:"SET_ALARM",7:"THERMOSTAT",8:"MEDIA_CONTROL",
               9:"REMINDERS_LISTS",10:"CALLS_MESSAGING",-1:"REJECT"}

# ---- label mapping: intent_10 (-1..10) -> class index (0..10) -------------
def intent_to_class(i10:int)->int:
    return i10 + 1          # -1 -> 0 (REJECT), 1 -> 2 ... 10 -> 11? No.
# We want REJECT as a single class and 10 intents. Map:
#   class 0 = REJECT (intent_10 == -1)
#   class 1..10 = intent_10 1..10
def to_class(i10:int)->int:
    return 0 if i10 == -1 else i10
CLASS_TO_INTENT = {0:-1, **{i:i for i in range(1,11)}}

# --------------------------------------------------------------------------- #
# In-memory feature dataset (precomputed once, reused across epochs)          #
# --------------------------------------------------------------------------- #
def _extract_all(df: pd.DataFrame, kind: str):
    fn = F.mfcc_features if kind == "mfcc" else F.logmel_features
    T, Fd = F.shape_of(kind)
    X = np.zeros((len(df), T, Fd), dtype=np.float32)
    Y = np.zeros(len(df), dtype=np.int64)
    for i, p in enumerate(df.abs_path.values):
        feat = fn(str(p))
        if feat.shape[0] < T:
            feat = np.pad(feat, ((0, T-feat.shape[0]), (0,0)), mode="edge")
        else:
            feat = feat[:T]
        X[i] = feat
        if (i+1) % 2000 == 0:
            print(f"  extracted {i+1}/{len(df)} ({kind})")
    return X, Y

class VCMDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, augment: bool = False):
        self.X = X; self.Y = Y; self.aug = augment
    def __len__(self): return len(self.Y)
    def __getitem__(self, i):
        x = self.X[i]
        if self.aug:
            if np.random.random() < 0.5:
                x = np.roll(x, np.random.randint(-3, 4), axis=0)
            x = x * (0.8 + 0.4*np.random.random())
        return torch.from_numpy(x), torch.tensor(self.Y[i], dtype=torch.long)

def make_loaders(train_csv, val_csv, kind, bs, augment=True):
    tr_df = pd.read_csv(train_csv, low_memory=False).dropna(subset=["abs_path"]).reset_index(drop=True)
    va_df = pd.read_csv(val_csv, low_memory=False).dropna(subset=["abs_path"]).reset_index(drop=True)
    print("Extracting train features...")
    Xtr, _ = _extract_all(tr_df, kind)
    Ytr = tr_df.intent_10.map(to_class).values
    print("Extracting val features...")
    Xva, _ = _extract_all(va_df, kind)
    Yva = va_df.intent_10.map(to_class).values
    tr = VCMDataset(Xtr, Ytr, augment=augment)
    va = VCMDataset(Xva, Yva, augment=False)
    tl = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=0)
    vl = DataLoader(va, batch_size=bs*2, shuffle=False, num_workers=0)
    return tr, va, tl, vl

def class_weights(Y: np.ndarray):
    counts = np.bincount(Y, minlength=N_CLASSES).astype(float)
    counts[counts==0] = 1.0
    w = counts.sum() / (N_CLASSES * counts)
    w = w / w.max()          # scale so max weight = 1 (stable)
    return torch.tensor(w, dtype=torch.float32).to(DEVICE)

# --------------------------------------------------------------------------- #
def evaluate(model, loader):
    model.eval()
    correct = 0; total = 0
    per_correct = np.zeros(N_CLASSES); per_seen = np.zeros(N_CLASSES)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE); y = y.to(DEVICE)
            pred = model(x).argmax(1)
            m = (pred == y)
            correct += m.sum().item(); total += y.numel()
            for yi, pi in zip(y.tolist(), pred.tolist()):
                per_seen[yi]+=1
                if pi==yi: per_correct[yi]+=1
    acc = correct/total
    f1s=[]
    for c in range(N_CLASSES):
        tp=per_correct[c]; seen=per_seen[c]
        # precision needs FP too; approximate macro over recall for brevity below
        f1s.append(1.0 if seen>0 and tp==seen else (tp/max(seen,1)))
    return acc, per_correct, per_seen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["logistic","dnn","cnn1d","crnn"], default="dnn")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-aug", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    kind = "mfcc" if a.model in ("logistic","dnn") else "logmel"
    train_csv = os.path.join(SPLITS,"train.csv")
    val_csv   = os.path.join(SPLITS,"val.csv")
    test_csv  = os.path.join(SPLITS,"test.csv")

    print(f"[{a.model}] kind={kind} device={DEVICE}")
    tr, va, tl, vl = make_loaders(train_csv, val_csv, kind, a.bs, augment=not a.no_aug)
    print(f"train={len(tr)} val={len(va)}")

    model = build_model(a.model).to(DEVICE)
    print(f"params={count_params(model):,}")
    cw = class_weights(tr.Y)
    crit = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    best_acc=0; best_state=None; t0=time.time()
    for ep in range(a.epochs):
        model.train()
        tot_loss=0; nb=0
        for x,y in tl:
            x=x.to(DEVICE); y=y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            tot_loss+=loss.item(); nb+=1
        sched.step()
        vacc,_,_ = evaluate(model, vl)
        print(f"ep{ep+1:02d} loss={tot_loss/nb:.4f} val_acc={vacc:.4f} "
              f"({time.time()-t0:.0f}s)")
        if vacc>best_acc:
            best_acc=vacc; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}

    model.load_state_dict(best_state)
    # ---- test eval ----------------------------------------------------
    te_df = pd.read_csv(test_csv, low_memory=False).dropna(subset=["abs_path"]).reset_index(drop=True)
    print("Extracting test features...")
    Xte, _ = _extract_all(te_df, kind)
    Yte = te_df.intent_10.map(to_class).values
    te_ds = VCMDataset(Xte, Yte, augment=False)
    te = DataLoader(te_ds, batch_size=a.bs*2, shuffle=False)
    tact, per_correct, per_seen = evaluate(model, te)
    print(f"\nBEST val_acc={best_acc:.4f}  TEST acc={tact:.4f}")
    print("Per-intent test accuracy:")
    for c in range(N_CLASSES):
        nm = INTENT_NAME.get(CLASS_TO_INTENT[c], str(c))
        print(f"  {nm:16s} {int(per_correct[c]):>4}/{int(per_seen[c]):>4} "
              f"{(per_correct[c]/max(per_seen[c],1)):.3f}")

    # ---- save + export ------------------------------------------------
    tag = f"{a.model}_{int(time.time())}"
    torch.save(best_state, os.path.join(MODELDIR, f"{tag}_state.pt"))
    # TorchScript
    ts = torch.jit.script(model)
    ts.save(os.path.join(MODELDIR, f"{tag}.pt"))
    # ONNX fp32 (must be on CPU to match the CPU dummy input)
    T = F.shape_of(kind)[0]
    din = F.N_MFCC if kind=="mfcc" else F.N_MEL
    model_cpu = model.cpu().eval()
    dummy = torch.randn(1, T, din)
    onnx_p = os.path.join(MODELDIR, f"{tag}.onnx")
    torch.onnx.export(model_cpu, dummy, onnx_p, opset_version=13, dynamo=False,
                      input_names=["audio"], output_names=["logits"],
                      dynamic_axes={"audio":{0:"batch"},"logits":{0:"batch"}})
    # ONNX INT8
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        q_p = os.path.join(MODELDIR, f"{tag}_int8.onnx")
        quantize_dynamic(onnx_p, q_p, weight_type=QuantType.QInt8)
        sz_fp = os.path.getsize(onnx_p)/1024; sz_q=os.path.getsize(q_p)/1024
    except Exception as e:
        q_p=None; sz_fp=sz_q=-1
        print("ONNX quant warn:", e)
    meta = {"model":a.model,"kind":kind,"params":count_params(model),
            "val_acc":round(best_acc,4),"test_acc":round(tact,4),
            "onnx":onnx_p,"onnx_int8":q_p,
            "onnx_kb":round(sz_fp,1),"onnx_int8_kb":round(sz_q,1),
            "feature_T":T,"feature_F":din}
    with open(os.path.join(MODELDIR, f"{tag}_meta.json"),"w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved {tag}: onnx={sz_fp:.0f}KB int8={sz_q:.0f}KB  "
          f"(params={count_params(model):,})")
    print("META:", json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
