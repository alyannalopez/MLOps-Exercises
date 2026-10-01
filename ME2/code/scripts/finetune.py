#!/usr/bin/env python3
"""Fine-tune the VCM CRNN on YOUR OWN voice recordings.

Why this exists
---------------
The deployed CRNN scored ~95% on its synthetic TTS test set but only ~20% on
your live voice. The gap is speaker/acoustic/phrasing domain shift, not model
capacity. Fine-tuning on a few dozen clips of YOU (same room, same mic, your
phrasings) closes that gap while keeping the 10-intent head intact.

Pipeline
--------
1. Load the base CRNN state (the 95% retrain checkpoint).
2. Build a small dataset from <data>/manifest.csv  (your recordings).
   - log-mel features, identical to training (features.logmel_features).
   - optional mix-back of the original synthetic set (--synthetic-frac) so the
     model doesn't forget the base distribution (catastrophic-forgetting guard).
3. Train a few epochs at a LOW lr (full fine-tune by default; --freeze-cnn
   trains only the LSTM+head for a lighter touch).
4. Evaluate on a held-out slice of YOUR clips (per-intent accuracy + confusion).
5. Export a new INT8 ONNX + meta.json into --out (default pi_bundle/) so you
   can drop it straight into the live demo.

Run
----
    python finetune.py --data ../data/finetune \
        --base ../models_retrain/crnn_retrain_1790670993_state.pt \
        --epochs 15 --lr 2e-4

Lighter-touch (only LSTM + head, faster, less overfit risk):
    python finetune.py --data ../data/finetune --base ... --freeze-cnn --epochs 25
"""
from __future__ import annotations
import argparse, os, sys, csv, json, time, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
from models import build as build_model, count_params, N_CLASSES, REJECT

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

INTENT_NAME = {0:"PLAY_MUSIC",1:"QUESTION_SEARCH",2:"LIGHTS_ON_OFF",
               3:"DIM_COLOR_LIGHTS",4:"SET_TIMER",5:"SET_ALARM",6:"THERMOSTAT",
               7:"MEDIA_CONTROL",8:"REMINDERS_LISTS",9:"CALLS_MESSAGING"}
CLASSES = [INTENT_NAME[i] for i in range(REJECT)]  # 0..9 only (10 = REJECT, excluded)


# ---------------------------------------------------------------------------- #
# Data                                                                          #
# ---------------------------------------------------------------------------- #
def load_manifest(data_dir: str):
    mpath = os.path.join(data_dir, "manifest.csv")
    if not os.path.exists(mpath):
        raise SystemExit(f"no manifest at {mpath}\n"
                         f"run record_finetune.py first (or point --data at the right folder).")
    rows = []
    with open(mpath) as f:
        for r in csv.DictReader(f):
            p = r["path"]
            if not os.path.isabs(p):
                p = os.path.join(data_dir, os.path.relpath(p, data_dir))
            if not os.path.exists(p):
                continue
            intent = r.get("intent", "").strip().upper()
            if intent not in INTENT_NAME.values():
                continue
            rows.append((p, list(INTENT_NAME.values()).index(intent)))
    return rows


class FeatDataset(Dataset):
    """Pre-extracts log-mel features once (small N) and caches in memory."""
    def __init__(self, items, augment=False):
        self.X = np.stack([F.logmel_features(p) for p, _ in items]).astype(np.float32)
        self.Y = np.array([y for _, y in items], dtype=np.int64)
        self.aug = augment
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, i):
        x = self.X[i]
        if self.aug:
            if np.random.random() < 0.5:
                x = np.roll(x, np.random.randint(-3, 4), axis=0)
            x = x * (0.8 + 0.4 * np.random.random())
            # light spec noise
            x = x + 0.01 * np.random.randn(*x.shape).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(self.Y[i], dtype=torch.long)


def load_synthetic_backmix(frac: float, seed: int):
    """Sample a fraction of the original synthetic log-mel cache for back-mixing.

    Uses data/processed/_retrain_feats_logmel_*.npz + manifest.csv if present.
    Returns list[(feat_array, class)] or [].
    """
    if frac <= 0:
        return []
    npz = sorted(glob.glob(os.path.join(ROOT, "data", "processed",
                                        "_retrain_feats_logmel_*.npz")))
    mpath = os.path.join(ROOT, "data", "processed", "manifest.csv")
    if not npz or not os.path.exists(mpath):
        print("  [backmix] synthetic cache not found — skipping (fine for small sets)")
        return []
    print(f"  [backmix] loading synthetic cache {os.path.basename(npz[-1])} ...")
    z = np.load(npz[-1], allow_pickle=True)
    X = z[list(z.keys())[0]]  # (N, T, F)
    with open(mpath) as f:
        man = list(csv.DictReader(f))
    # align lengths
    n = min(len(X), len(man))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)[:max(1, int(n * frac))]
    out = []
    for i in perm:
        intent_10 = int(man[i].get("intent_10", -1))
        if intent_10 < 1 or intent_10 > 10:
            continue  # skip OOV/reject
        out.append((X[i].astype(np.float32), intent_10 - 1))
    print(f"  [backmix] using {len(out)} synthetic clips")
    return out


# ---------------------------------------------------------------------------- #
# Train                                                                         #
# ---------------------------------------------------------------------------- #
def freeze_cnn(model: nn.Module):
    for p in model.cnn.parameters():
        p.requires_grad_(False)
    n_freeze = sum(p.numel() for p in model.cnn.parameters())
    print(f"  [freeze-cnn] froze {n_freeze:,} CNN params; training LSTM+head only")


def evaluate(model, ds: FeatDataset):
    model.eval()
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    correct = 0
    per = {c: [0, 0] for c in CLASSES}
    with torch.no_grad():
        for x, y in dl:
            x = x.to(DEVICE)
            pred = model(x).argmax(1).cpu().numpy()
            y = y.numpy()
            correct += int((pred == y).sum())
            for p, t in zip(pred, y):
                per[CLASSES[t]][1] += 1
                if p == t:
                    per[CLASSES[t]][0] += 1
    acc = correct / max(1, len(ds))
    per_intent = {c: (per[c][0] / per[c][1] if per[c][1] else float("nan"))
                  for c in CLASSES}
    return acc, per_intent


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with manifest.csv (your recordings)")
    ap.add_argument("--base", required=True, help="path to base CRNN *_state.pt")
    ap.add_argument("--out", default=os.path.join(ROOT, "pi_bundle"),
                    help="where to write the new ONNX + meta (default pi_bundle)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-4, help="low lr for fine-tuning")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--freeze-cnn", action="store_true",
                    help="train only LSTM+head (lighter, less overfit)")
    ap.add_argument("--synthetic-frac", type=float, default=0.0,
                    help="fraction of original synthetic set to back-mix (0-1, default 0)")
    ap.add_argument("--holdout", type=float, default=0.3,
                    help="fraction of YOUR clips held out for eval (default 0.3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None, help="output model tag")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    t_start = time.time()

    # 1. base model
    model = build_model("crnn").to(DEVICE)
    ckpt = torch.load(a.base, map_location=DEVICE)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    print(f"[base] loaded {a.base}  ({count_params(model):,} params)")
    if a.freeze_cnn:
        freeze_cnn(model)

    # 2. your data
    items = load_manifest(a.data)
    if not items:
        raise SystemExit("no valid clips found in manifest — record some first.")
    from collections import Counter
    cnt = Counter(CLASSES[y] for _, y in items)
    print(f"[data] {len(items)} of your clips: " +
          ", ".join(f"{c}={cnt[c]}" for c in CLASSES if cnt[c]))

    # split your clips train/holdout (stratified-ish by shuffle)
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(items))
    n_hold = max(1, int(len(items) * a.holdout))
    hold_idx, tr_idx = idx[:n_hold], idx[n_hold:]
    train_items = [items[i] for i in tr_idx]
    hold_items = [items[i] for i in hold_idx]

    # back-mix synthetic into TRAIN only
    synth = load_synthetic_backmix(a.synthetic_frac, a.seed)
    train_items = train_items + synth
    print(f"[split] your: train={len(tr_idx)} holdout={len(hold_idx)}  "
          f"+ synthetic back-mix={len(synth)}")

    train_ds = FeatDataset(train_items, augment=True)
    hold_ds = FeatDataset(hold_items, augment=False)
    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True)

    # 3. optimizer (AdamW, low lr)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = nn.CrossEntropyLoss()

    print(f"\n[train] {a.epochs} epochs @ lr={a.lr}  device={DEVICE}")
    best_acc, best_state, best_epoch = -1.0, None, 0
    for ep in range(1, a.epochs + 1):
        model.train()
        tot, corr, loss_sum = 0, 0, 0.0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            corr += int((logits.argmax(1) == y).sum())
            tot += x.size(0)
        sched.step()
        val_acc, _ = evaluate(model, hold_ds)
        tr_acc = corr / max(1, tot)
        print(f"  ep {ep:2d}/{a.epochs}  train={tr_acc:.3f}  "
              f"your-holdout={val_acc:.3f}  loss={loss_sum/max(1,tot):.3f}")
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # 4. restore best + final eval
    if best_state:
        model.load_state_dict(best_state)
    final_acc, per_intent = evaluate(model, hold_ds)
    print(f"\n[result] best epoch {best_epoch}  your-voice holdout acc = {final_acc:.3f}")
    print("  per-intent (your voice):")
    for c in CLASSES:
        v = per_intent[c]
        print(f"    {c:18s} {v if v==v else float('nan'):.3f}" if v == v
              else f"    {c:18s}   (no holdout clips)")

    # 5. export INT8 ONNX
    os.makedirs(a.out, exist_ok=True)
    tag = a.name or f"crnn_ft_{int(time.time())}"
    pt_path = os.path.join(a.out, f"{tag}_state.pt")
    torch.save(best_state or model.state_dict(), pt_path)

    onnx_path = os.path.join(a.out, f"{tag}_int8.onnx")
    float_onnx = onnx_path.replace("_int8.onnx", "_float.onnx")
    onnx_ok = False
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        # Export a FEATURE-input ONNX: input (batch, T, 80) log-mel -> 11 logits.
        # This matches vcm_infer.VCM, which feeds F.logmel_features output (width
        # 80 => kind="logmel"). Use opset 17 + dynamo=False (legacy exporter) —
        # torch 2.14's default dynamo exporter trips on the CRNN's LSTM.
        T, Fl = F.shape_of("logmel")
        dummy = torch.randn(1, T, Fl, device=DEVICE)
        torch.onnx.export(model, (dummy,), float_onnx,
                          input_names=["x"], output_names=["logits"],
                          opset_version=17, dynamo=False,
                          dynamic_axes={"x": {0: "batch"}})
        quantize_dynamic(float_onnx, onnx_path, weight_type=QuantType.QInt8)
        # clean up the float model + any external-data sidecars
        for extra in (float_onnx, float_onnx + ".data"):
            if os.path.exists(extra):
                os.remove(extra)
        onnx_ok = True
        print(f"  [export] {os.path.basename(onnx_path)} "
              f"({os.path.getsize(onnx_path)/1e6:.2f} MB int8)")
    except Exception as e:
        print(f"  [export] ONNX export failed ({e}); wrote .pt only — export manually.")
        for extra in (float_onnx, float_onnx + ".data"):
            if os.path.exists(extra):
                os.remove(extra)

    meta = {
        "model": "crnn", "tag": tag, "fine_tuned": True,
        "base": os.path.basename(a.base),
        "n_user_clips": len(items), "n_synthetic_backmix": len(synth),
        "epochs": a.epochs, "lr": a.lr, "freeze_cnn": a.freeze_cnn,
        "best_epoch": best_epoch, "your_voice_holdout_acc": round(final_acc, 4),
        "per_intent_your_voice": {c: (round(per_intent[c], 4) if per_intent[c] == per_intent[c] else None)
                                  for c in CLASSES},
        "train_wall_seconds": round(time.time() - t_start, 1),
        "onnx": os.path.basename(onnx_path) if onnx_ok else None,
        "pt": os.path.basename(pt_path),
    }
    with open(os.path.join(a.out, f"{tag}_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    print(f"\n[out] {pt_path}")
    if onnx_ok:
        sz = os.path.getsize(onnx_path) / 1e6
        print(f"[out] {onnx_path}  ({sz:.2f} MB int8)")
    print(f"[out] {os.path.join(a.out, tag+'_meta.json')}")
    print(f"\nDone in {time.time()-t_start:.1f}s.")
    if onnx_ok:
        print("\nTo use it in the live demo:")
        print(f"  python live_demo.py --model {tag}_int8.onnx --simulate")


if __name__ == "__main__":
    main()
