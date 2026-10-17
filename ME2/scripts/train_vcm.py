#!/usr/bin/env python3
"""Stage 4: Train a VCM architecture with early stopping.

Trains one architecture (chosen with --arch) on the packed train bundle
(raw 1 s waveforms), applies waveform-domain augmentation in the data
loader, validates on the packed test bundle every epoch, stops when
validation balanced accuracy plateaus, and saves the best checkpoint +
full training log.

With --slots (conformer_s only): trains a two-head model (intent + slot)
using the shared slot vocabulary from vcm/slots.py.  Slot labels come
from the manifest's slot_value column.  Loss = CE(intent) + CE(slot).

Outputs (under models/<arch>_seed<seed>/):
  best.pt          - best model state dict + config
  config.json      - hyperparameters + dataset info + timings
  train_log.json   - per-epoch metrics
  history.csv      - same, tabular

Usage:
  python scripts/train_vcm.py --arch conformer_s --slots --epochs 100 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vcm.augment import augment_waveform  # noqa: E402
from vcm.features import log_mel_spectrogram  # noqa: E402
from vcm.losses import class_weights_from_counts  # noqa: E402
from vcm.models import build, count_params  # noqa: E402
from vcm.slots import N_SLOTS, NO_SLOT_IDX, slot_index_for, INTENTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PACKS = ROOT / "data" / "packs"
MANIFEST = ROOT / "data" / "me2-dataset" / "manifest.csv"
N_CLASSES = 20
N_MELS, N_FRAMES = 40, 100
OOS_ID = N_CLASSES - 1


def load_pack(name: str):
    z = np.load(PACKS / f"{name}.npz", allow_pickle=True)
    return z


def load_slot_labels() -> dict[str, np.ndarray]:
    """Load slot labels from the manifest, keyed by split name.

    Returns {split: ndarray of slot class indices (N,)}.
    The manifest row order matches the pack row order (both sorted by
    split, then by original manifest order).
    """
    mf = pd.read_csv(MANIFEST)
    out = {}
    for split in ["train", "neg_train", "test", "neg_test"]:
        rows = mf[mf["split"] == split].reset_index(drop=True)
        slots = np.array([
            slot_index_for(r["command"], r.get("slot_value", ""))
            for _, r in rows.iterrows()
        ], dtype=np.int64)
        out[split] = slots
    return out


def balanced_accuracy(y_true, y_pred, n_classes=N_CLASSES):
    per_class = []
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() == 0:
            continue
        per_class.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(per_class)) if per_class else 0.0


class AugDataset(Dataset):
    """Yields (logmel, label, slot_label, weight) with per-epoch waveform augmentation."""

    def __init__(self, waveforms: np.ndarray, labels: np.ndarray,
                 slot_labels: np.ndarray | None, weights: np.ndarray,
                 noise_pool: list, rng: np.random.Generator):
        self.w = waveforms
        self.y = labels
        self.s = slot_labels
        self.wt = weights
        self.noise_pool = noise_pool
        self.rng = rng

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        w = self.w[i]
        w = augment_waveform(w, noise_pool=self.noise_pool, rng=self.rng)
        feat = log_mel_spectrogram(w, n_mels=N_MELS)  # (40, 100)
        if self.s is not None:
            return feat[None], int(self.y[i]), int(self.s[i]), float(self.wt[i])
        return feat[None], int(self.y[i]), float(self.wt[i])


def _collate(batch):
    if len(batch[0]) == 4:
        return (torch.from_numpy(np.stack([x[0] for x in batch])),
                torch.tensor([x[1] for x in batch], dtype=torch.long),
                torch.tensor([x[2] for x in batch], dtype=torch.long),
                torch.tensor([x[3] for x in batch], dtype=torch.float32))
    return (torch.from_numpy(np.stack([x[0] for x in batch])),
            torch.tensor([x[1] for x in batch], dtype=torch.long),
            torch.tensor([x[2] for x in batch], dtype=torch.float32))


def static_features(waveforms: np.ndarray) -> np.ndarray:
    """Clean (unaugmented) log-mel features for validation."""
    out = np.empty((len(waveforms), 1, N_MELS, N_FRAMES), dtype=np.float32)
    for i, w in enumerate(waveforms):
        out[i] = log_mel_spectrogram(w, n_mels=N_MELS)[None]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["cnn", "crnn", "tcn", "conformer_s", "mlp"])
    ap.add_argument("--epochs", type=int, default=100, help="max epochs (50-100 per spec; early stopping ends it sooner)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--real-weight", type=float, default=5.0,
                    help="sampling weight boost for real-voice clips")
    ap.add_argument("--oos-weight", type=float, default=3.0,
                    help="sampling weight boost for OUT_OF_SCOPE clips")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--slot-weight", type=float, default=1.0,
                    help="weight for the slot CE loss (two-head mode)")
    ap.add_argument("--slots", action="store_true",
                    help="train two-head model (intent + slot), conformer_s only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None, help="cpu|mps|cuda (default: auto)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device={device}")

    outdir = ROOT / "models" / f"{args.arch}_seed{args.seed}"
    outdir.mkdir(parents=True, exist_ok=True)

    t_load = time.time()
    ztr = load_pack("train")
    zneg = load_pack("neg_train")
    zte = load_pack("test")
    znegv = load_pack("neg_test")
    znoise = load_pack("noise")
    noise_pool = [np.asarray(n, dtype=np.float32) for n in znoise["noise"]]

    wtr = np.concatenate([ztr["waveforms"], zneg["waveforms"]])
    ytr = torch.from_numpy(np.concatenate([ztr["labels"], zneg["labels"]]))
    synth = np.concatenate([ztr["is_synthetic"], zneg["is_synthetic"]])
    wva = np.concatenate([zte["waveforms"], znegv["waveforms"]])
    yva = torch.from_numpy(np.concatenate([zte["labels"], znegv["labels"]]))
    print(f"loaded in {time.time()-t_load:.1f}s | train {wtr.shape} val {wva.shape} (incl. negatives)")

    # Slot labels (two-head mode)
    slot_labels_tr = slot_labels_va = None
    if args.slots:
        all_slots = load_slot_labels()
        # Negatives are all NO_SLOT
        n_neg_tr = len(zneg["labels"])
        n_neg_va = len(znegv["labels"])
        slot_labels_tr = np.concatenate([all_slots["train"], np.full(n_neg_tr, NO_SLOT_IDX, dtype=np.int64)])
        slot_labels_va = np.concatenate([all_slots["test"], np.full(n_neg_va, NO_SLOT_IDX, dtype=np.int64)])
        assert len(slot_labels_tr) == len(ytr), f"slot label mismatch: {len(slot_labels_tr)} vs {len(ytr)}"
        assert len(slot_labels_va) == len(yva), f"slot label mismatch: {len(slot_labels_va)} vs {len(yva)}"
        print(f"slot labels loaded: train={len(slot_labels_tr)}, val={len(slot_labels_va)}")
        print(f"  slot distribution (train): {np.bincount(slot_labels_tr, minlength=N_SLOTS)}")

    t_feat = time.time()
    Xva_t = torch.from_numpy(static_features(wva))
    print(f"val features in {time.time()-t_feat:.1f}s")

    if args.slots and args.arch != "conformer_s":
        print("--slots is only supported with --arch conformer_s", file=sys.stderr)
        sys.exit(1)

    model = build(args.arch, n_classes=N_CLASSES, n_mels=N_MELS, n_frames=N_FRAMES,
                  n_slots=N_SLOTS if args.slots else None).to(device)
    n_params = count_params(model)
    print(f"arch={args.arch} params={n_params:,}")

    # Per-sample weights: inverse-frequency class weight x real-voice boost x OOS boost.
    counts = np.bincount(ytr.numpy(), minlength=N_CLASSES).astype(np.float64)
    cw = class_weights_from_counts(counts).numpy()
    sample_w = cw[ytr.numpy()]
    sample_w[synth == 0] *= args.real_weight
    sample_w[ytr.numpy() == OOS_ID] *= args.oos_weight
    sample_w = sample_w.astype(np.float32)

    ds = AugDataset(wtr, ytr.numpy(), slot_labels_tr, sample_w, noise_pool, rng)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    collate_fn=_collate)
    if args.slots:
        vadl = DataLoader(TensorDataset(Xva_t, yva, torch.from_numpy(slot_labels_va)), batch_size=256)
    else:
        vadl = DataLoader(TensorDataset(Xva_t, yva), batch_size=256)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_ba, best_epoch, bad = -1.0, -1, 0
    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        model.train()
        tot_loss, n = 0.0, 0
        for Xb, yb, *rest in dl:
            Xb = Xb.to(device)
            yb = yb.to(device)
            if args.slots:
                sb = rest[0].to(device)
                wb = rest[1].to(device)
            else:
                sb = None
                wb = rest[0].to(device)
            opt.zero_grad()
            if args.slots:
                intent_logits, slot_logits = model(Xb)
                # Intent loss (weighted CE with label smoothing)
                logp = torch.log_softmax(intent_logits, dim=-1)
                if args.label_smoothing > 0:
                    t = torch.full_like(logp, args.label_smoothing / (N_CLASSES - 1))
                    t.scatter_(1, yb.unsqueeze(1), 1.0 - args.label_smoothing)
                else:
                    t = torch.zeros_like(logp)
                    t.scatter_(1, yb.unsqueeze(1), 1.0)
                intent_loss = (-(t * logp).sum(dim=-1) * wb).mean()
                # Slot loss (plain CE)
                slot_loss = nn.functional.cross_entropy(slot_logits, sb)
                loss = intent_loss + args.slot_weight * slot_loss
            else:
                logits = model(Xb)
                logp = torch.log_softmax(logits, dim=-1)
                if args.label_smoothing > 0:
                    t = torch.full_like(logp, args.label_smoothing / (N_CLASSES - 1))
                    t.scatter_(1, yb.unsqueeze(1), 1.0 - args.label_smoothing)
                else:
                    t = torch.zeros_like(logp)
                    t.scatter_(1, yb.unsqueeze(1), 1.0)
                loss = (-(t * logp).sum(dim=-1) * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_loss += loss.item() * len(Xb)
            n += len(Xb)
        sched.step()

        # Validate (clean features, no augmentation)
        model.eval()
        preds, truths, slot_preds, slot_truths = [], [], [], []
        with torch.no_grad():
            for batch in vadl:
                if args.slots:
                    Xb, yb, sb = batch
                    Xb = Xb.to(device)
                    intent_logits, slot_logits = model(Xb)
                    preds.append(intent_logits.argmax(-1).cpu().numpy())
                    truths.append(yb.cpu().numpy())
                    slot_preds.append(slot_logits.argmax(-1).cpu().numpy())
                    slot_truths.append(sb.cpu().numpy())
                else:
                    Xb, yb = batch
                    Xb = Xb.to(device)
                    logits = model(Xb)
                    preds.append(logits.argmax(-1).cpu().numpy())
                    truths.append(yb.cpu().numpy())
        preds = np.concatenate(preds)
        truths = np.concatenate(truths)
        ba = balanced_accuracy(truths, preds)
        acc = float((preds == truths).mean())
        oos_mask = truths == OOS_ID
        oos_acc = float((preds[oos_mask] == OOS_ID).mean()) if oos_mask.any() else float("nan")

        # Slot accuracy (two-head mode)
        slot_acc_str = ""
        if args.slots:
            slot_preds_arr = np.concatenate(slot_preds)
            slot_truths_arr = np.concatenate(slot_truths)
            slot_acc = float((slot_preds_arr == slot_truths_arr).mean())
            # Slot accuracy for slotted intents only
            slotted_mask = slot_truths_arr != NO_SLOT_IDX
            if slotted_mask.any():
                slotted_slot_acc = float((slot_preds_arr[slotted_mask] == slot_truths_arr[slotted_mask]).mean())
            else:
                slotted_slot_acc = float("nan")
            slot_acc_str = f" slot_acc={slot_acc:.4f} slotted={slotted_slot_acc:.4f}"

        row = dict(epoch=epoch, train_loss=round(tot_loss / n, 6), val_acc=round(acc, 6),
                   val_balanced_acc=round(ba, 6), val_oos_acc=round(oos_acc, 6),
                   epoch_seconds=round(time.time() - t_ep, 1))
        if args.slots:
            row["val_slot_acc"] = round(slot_acc, 6)
        history.append(row)
        print(f"ep {epoch:3d} loss={tot_loss/n:.4f} val_acc={acc:.4f} val_ba={ba:.4f} "
              f"oos={oos_acc:.3f}{slot_acc_str} ({row['epoch_seconds']}s)", flush=True)

        if ba > best_ba:
            best_ba, best_epoch, bad = ba, epoch, 0
            torch.save({"state_dict": model.state_dict(), "config": vars(args),
                        "n_params": n_params, "best_ba": best_ba}, outdir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch} (best ep {best_epoch}, ba {best_ba:.4f})")
                break

    # Final report with best checkpoint
    model.load_state_dict(torch.load(outdir / "best.pt", weights_only=True)["state_dict"])
    model.eval()
    preds, truths, slot_preds, slot_truths = [], [], [], []
    with torch.no_grad():
        for batch in vadl:
            if args.slots:
                Xb, yb, sb = batch
                Xb = Xb.to(device)
                intent_logits, slot_logits = model(Xb)
                preds.append(intent_logits.argmax(-1).cpu().numpy())
                truths.append(yb.cpu().numpy())
                slot_preds.append(slot_logits.argmax(-1).cpu().numpy())
                slot_truths.append(sb.cpu().numpy())
            else:
                Xb, yb = batch
                Xb = Xb.to(device)
                logits = model(Xb)
                preds.append(logits.argmax(-1).cpu().numpy())
                truths.append(yb.cpu().numpy())
    preds, truths = np.concatenate(preds), np.concatenate(truths)
    report = {
        "arch": args.arch, "seed": args.seed, "n_params": n_params,
        "two_head": args.slots,
        "best_epoch": best_epoch, "best_val_balanced_acc": best_ba,
        "final_val_acc": float((preds == truths).mean()),
        "final_val_balanced_acc": balanced_accuracy(truths, preds),
        "final_val_oos_acc": float((preds[truths == OOS_ID] == OOS_ID).mean()),
        "train_seconds": round(time.time() - t0, 1),
        "epochs_run": len(history),
        "device": str(device),
        "n_train": int(len(wtr)), "n_val": int(len(wva)),
        "config": vars(args),
    }
    if args.slots:
        sp, st = np.concatenate(slot_preds), np.concatenate(slot_truths)
        report["final_val_slot_acc"] = float((sp == st).mean())
        slotted_mask = st != NO_SLOT_IDX
        if slotted_mask.any():
            report["final_val_slotted_slot_acc"] = float((sp[slotted_mask] == st[slotted_mask]).mean())
    (outdir / "config.json").write_text(json.dumps(report, indent=2))
    (outdir / "train_log.json").write_text(json.dumps(history, indent=2))
    import csv as _csv
    with open(outdir / "history.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    print(json.dumps({k: report[k] for k in ["arch", "n_params", "best_epoch",
                                              "best_val_balanced_acc",
                                              "final_val_acc", "train_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
