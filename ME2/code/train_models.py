#!/usr/bin/env python3
"""Train 5 VCM architectures on the AI231 ME2 dataset.

Architectures:
  A. Logistic Regression (baseline)
  B. DNN (2 hidden layers)
  C. 1-D CNN (3 conv blocks)
  D. CRNN (CNN + BiLSTM)
  E. Small Transformer (2-layer encoder)

Each trained 50-100 epochs with early stopping on val accuracy.
"""
from __future__ import annotations
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "data" / "features"
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"
# DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

SEED = 42
N_CLASSES = 20  # 19 intents + OUT_OF_SCOPE
N_FRAMES = 100
MAX_EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 1e-4))

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Label mapping ──────────────────────────────────────────────────────────
LABELS = [
    "PLAY_MUSIC","WEATHER","TIME","LIGHT_ON","LIGHT_OFF","PAUSE","STOP","NEXT",
    "VOLUME_UP","VOLUME_DOWN","CALL","MESSAGE","LIST_REMINDERS","TIMER","ALARM",
    "TEMPERATURE","BRIGHTNESS","COLOR","CREATE_REMINDER","OUT_OF_SCOPE",
]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}

# ── Models ─────────────────────────────────────────────────────────────────
class Logistic(nn.Module):
    def __init__(self, n_feat=40*N_FRAMES):
        super().__init__()
        self.fc = nn.Linear(n_feat, N_CLASSES)
    def forward(self, x):
        return self.fc(x.flatten(1))

class DNN(nn.Module):
    def __init__(self, n_feat=40*N_FRAMES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, N_CLASSES),
        )
    def forward(self, x):
        return self.net(x.flatten(1))

class CNN1D(nn.Module):
    def __init__(self, n_mfcc=40):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mfcc, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 5, padding=2), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(256, N_CLASSES))
    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        return self.head(self.conv(x.transpose(1, 2)))

class CRNN(nn.Module):
    def __init__(self, n_mfcc=40):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mfcc, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(128, 128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)
        # BiLSTM output: 2*128 = 256
        self.head = nn.Sequential(nn.Linear(256, N_CLASSES))
    def forward(self, x):
        c = self.conv(x.transpose(1, 2))  # (B, 128, T/4)
        c = c.transpose(1, 2)            # (B, T/4, 128)
        l, _ = self.lstm(c)              # (B, T/4, 256)
        l = l.mean(dim=1)                # (B, 256) — mean over time
        return self.head(l)

class TinyTransformer(nn.Module):
    def __init__(self, n_mfcc=40, d_model=128, nhead=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(n_mfcc, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.Linear(d_model, N_CLASSES))
    def forward(self, x):
        h = self.proj(x)       # (B, T, d_model)
        h = self.encoder(h)    # (B, T, d_model)
        h = h.mean(dim=1)      # (B, d_model) — mean over time
        return self.head(h)

MODELS = {
    "logistic":  lambda: Logistic(),
    "dnn":       lambda: DNN(),
    "cnn1d":     lambda: CNN1D(),
    "crnn":      lambda: CRNN(),
    "transformer": lambda: TinyTransformer(),
}

# ── Training loop ──────────────────────────────────────────────────────────
def train_model(name: str, mfcc_tr, y_tr, mfcc_va, y_va, mfcc_te, y_te):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    model = MODELS[name]().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    
    ds_tr = TensorDataset(torch.from_numpy(mfcc_tr), torch.from_numpy(y_tr))
    ds_va = TensorDataset(torch.from_numpy(mfcc_va), torch.from_numpy(y_va))
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE)
    
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_acc": [], "val_loss": []}
    best_val, best_epoch, patience_counter = 0.0, 0, 0
    best_state = None
    
    print(f"\n{'='*60}\nTraining {name} ({n_params:,} params) on {DEVICE}\n{'='*60}")
    
    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(xb)
        sched.step()
        
        # Validate
        model.eval()
        va_correct, va_total, va_loss = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                va_loss += F.cross_entropy(out, yb, reduction='sum').item()
                va_correct += (out.argmax(1) == yb).sum().item()
                va_total += len(xb)
        
        tr_acc = correct / total
        va_acc = va_correct / va_total
        tr_loss = total_loss / total
        va_loss /= va_total
        
        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["val_loss"].append(va_loss)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"  ep {epoch:3d} | tr_loss {tr_loss:.4f} tr_acc {tr_acc:.4f} | va_acc {va_acc:.4f} va_loss {va_loss:.4f}")
        
        if va_acc > best_val:
            best_val, best_epoch = va_acc, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stop at epoch {epoch} (best: ep {best_epoch}, val_acc {best_val:.4f})")
                break
    
    # Restore best
    model.load_state_dict(best_state)
    model.to(DEVICE)
    
    # Test
    model.eval()
    te_correct, te_total = 0, 0
    te_preds, te_y = [], []
    with torch.no_grad():
        xt = torch.from_numpy(mfcc_te).to(DEVICE)
        yt = torch.from_numpy(y_te).to(DEVICE)
        # batch test
        for i in range(0, len(xt), BATCH_SIZE):
            xb, yb = xt[i:i+BATCH_SIZE], yt[i:i+BATCH_SIZE]
            out = model(xb)
            te_correct += (out.argmax(1) == yb).sum().item()
            te_total += len(xb)
            te_preds.extend(out.argmax(1).cpu().numpy().tolist())
            te_y.extend(yb.cpu().numpy().tolist())
    te_acc = te_correct / te_total
    
    result = {
        "model": name,
        "params": n_params,
        "weight_decay": WEIGHT_DECAY,
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val, 4),
        "test_acc": round(te_acc, 4),
        "history": history,
        "test_preds": te_preds,
        "test_y": te_y,
    }
    
    # Save
    torch.save(model.state_dict(), MODEL_DIR / f"{name}_best.pt")
    with open(MODEL_DIR / f"{name}_result.json", "w") as f:
        json.dump({k: v for k, v in result.items() if k not in ("test_preds", "test_y")}, f, indent=2)
    
    print(f"  TEST ACC: {te_acc:.4f} ({te_correct}/{te_total})")
    return result

def main():
    print(f"Device: {DEVICE}")
    # Load features
    for split, prefix in [("train","tr"), ("test","te"), ("holdout","ho")]:
        d = np.load(FEAT_DIR / f"{split}.npz")
        globals()[f"mfcc_{prefix}"] = d["mfcc"]
        globals()[f"mel_{prefix}"] = d["mel"]
    
    # Build label arrays from manifest
    import pandas as pd
    man = pd.read_csv(ROOT / "data" / "manifest.csv")
    
    # Map manifest rows to feature indices (order within each split)
    y_all = {}
    for split_name in ["train", "test", "holdout"]:
        sub = man[man["_split"] == split_name]
        y_all[split_name] = sub["command"].map(LABEL2IDX).fillna(N_CLASSES-1).astype(int).values
    
    # Use holdout as validation, test as final evaluation
    print(f"Train: {mfcc_tr.shape}, Val(holdout): {mfcc_ho.shape}, Test: {mfcc_te.shape}")
    print(f"Label dist (train): {np.bincount(y_all['train'], minlength=N_CLASSES)}")
    
    results = {}
    # for name in MODELS:
    only = os.environ.get("ONLY_MODEL")
    for name in ([only] if only else MODELS):
        r = train_model(name, mfcc_tr, y_all["train"], mfcc_ho, y_all["holdout"], mfcc_te, y_all["test"])
        results[name] = r
    
    # Summary
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for name, r in results.items():
        print(f"  {name:15s} | {r['params']:>10,} params | val {r['best_val_acc']:.4f} | test {r['test_acc']:.4f}")
    
    with open(MODEL_DIR / "benchmark_summary.json", "w") as f:
        json.dump({n: {k: v for k, v in r.items() if k not in ('test_preds','test_y')} for n, r in results.items()}, f, indent=2)

if __name__ == "__main__":
    main()
