#!/usr/bin/env python3
"""V3: retrain all 4 VCM models (100 ep) + full multi-metric evaluation.

Metrics per model:
  - accuracy / val / test (overall + per-intent)
  - per-intent recall (command recall)
  - confusion matrix
  - inference latency: per-clip (single, warm), p50/p95/p99/max, batch throughput
  - efficiency: params, fp32/int8 size, MACs (FLOPs), training wall time
  - task completion / success rate (incl. reject handling)
  - WER: whisper.cpp transcription of slotted intents vs canonical phrases
Outputs:
  - vcm/models_v3/<model>_state.pt, <model>_history.json
  - vcm/models_v3/eval_results.json  (everything, for the Excel builder)
  - vcm/models_v3/wer_details.csv
"""
from __future__ import annotations
import argparse, os, glob, json, time, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F
from models import build as build_model, count_params, N_CLASSES, REJECT

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPLITS = os.path.join(ROOT, "data", "processed", "splits")
OUTDIR = os.path.join(ROOT, "models_v3")
os.makedirs(OUTDIR, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
INTENT_NAME = {0:"REJECT",1:"PLAY_MUSIC",2:"QUESTION_SEARCH",3:"LIGHTS_ON_OFF",
               4:"DIM_COLOR_LIGHTS",5:"SET_TIMER",6:"SET_ALARM",7:"THERMOSTAT",
               8:"MEDIA_CONTROL",9:"REMINDERS_LISTS",10:"CALLS_MESSAGING"}
CLASSES = [INTENT_NAME[i] for i in range(N_CLASSES)]
SLOTTED = {"SET_TIMER","SET_ALARM","DIM_COLOR_LIGHTS","THERMOSTAT","REMINDERS_LISTS"}

# Canonical reference phrases per (intent, value-group) for WER.
# Value groups are derived from the ACTUAL Option-B test-set folder names
# (verified against the dataset; note the real alarm values are 4:00 AM /
# 9:00 PM and colors include yellow, per the generated corpus).
CANONICAL = {
    "SET_TIMER": {
        "TIMER_10s": "set a timer for ten seconds",
        "TIMER_30s": "set a timer for thirty seconds",
        "TIMER_1m":  "set a timer for one minute",
    },
    "SET_ALARM": {
        "ALARM_4_00AM": "set an alarm for four in the morning",
        "ALARM_9_00PM": "set an alarm for nine in the evening",
    },
    "DIM_COLOR_LIGHTS": {
        "BRIGHTNESS_20":  "set the lights to twenty percent",
        "BRIGHTNESS_60":  "set the lights to sixty percent",
        "BRIGHTNESS_100": "set the lights to one hundred percent",
        "COLOR_RED":   "change the lights to red",
        "COLOR_BLUE":  "change the lights to blue",
        "COLOR_GREEN": "change the lights to green",
        "COLOR_YELLOW":"change the lights to yellow",
    },
    "THERMOSTAT": {
        "TEMPERATURE_18": "set the thermostat to eighteen degrees",
        "TEMPERATURE_22": "set the thermostat to twenty two degrees",
        "TEMPERATURE_26": "set the thermostat to twenty six degrees",
    },
    "REMINDERS_LISTS": {
        "CREATE_REMINDER_DRINK_WATER": "remind me to drink water",
        "CREATE_REMINDER_STUDY":       "remind me to study",
        "CREATE_REMINDER_EXERCISE":    "remind me to exercise",
    },
}

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

def class_weights(Y):
    counts = np.bincount(Y, minlength=N_CLASSES).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (N_CLASSES * counts)
    return torch.tensor(w / w.max(), dtype=torch.float32).to(DEVICE)

@torch.no_grad()
def predict_all(model, X, bs=512):
    """Return (preds, probs) arrays."""
    model.eval()
    dl = DataLoader(VCMDataset(X, np.arange(len(X)), augment=False), batch_size=bs, shuffle=False)
    preds, probs = [], []
    for x, _ in dl:
        x = x.to(DEVICE)
        logits = model(x)
        p = torch.softmax(logits, dim=1)
        preds.append(p.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)

def macs_estimate(model, kind):
    """Count multiply-accumulate ops for one forward pass (batch=1)."""
    total = 0
    T, Fd = F.shape_of(kind)
    dummy = torch.randn(1, T, Fd).to(DEVICE)
    hooks = []
    def hook_fn(module, inp, out):
        nonlocal total
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            if isinstance(module, nn.Conv1d):
                # out_channels * in_channels/groups * kernel * out_time
                k = module.kernel_size[0]
                out_t = out.shape[-1]
                total += module.out_channels * (module.in_channels // module.groups) * k * out_t
            else:
                total += module.in_features * module.out_features
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook_fn))
    model.eval()
    with torch.no_grad():
        model(dummy)
    for h in hooks:
        h.remove()
    return total

def benchmark_latency(model, X, n_clips=200, bs=32, reps=3):
    """Per-clip single inference latency (warm) + batch throughput.

    Uses a representative subset (strided) of the test set.
    """
    idx = np.linspace(0, len(X) - 1, n_clips).astype(int)
    xs = X[idx]
    xt = torch.from_numpy(xs).to(DEVICE)

    # --- single-clip latency (the real-time path) ---
    model.eval()
    # warmup
    with torch.no_grad():
        for i in range(10):
            model(xt[i:i+1])
    lat = []
    with torch.no_grad():
        for i in range(len(xt)):
            t0 = time.perf_counter()
            model(xt[i:i+1])
            lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)

    # --- batch throughput ---
    tp = []
    with torch.no_grad():
        for r in range(reps):
            t0 = time.perf_counter()
            for i in range(0, len(xt), bs):
                model(xt[i:i+bs])
            tp.append((time.perf_counter() - t0) * 1000)
    tp_s = min(tp) / 1000
    return {
        "single_p50_ms": round(float(np.percentile(lat, 50)), 3),
        "single_p95_ms": round(float(np.percentile(lat, 95)), 3),
        "single_p99_ms": round(float(np.percentile(lat, 99)), 3),
        "single_max_ms": round(float(lat.max()), 3),
        "batch_clips_per_sec": round(len(xt) / tp_s, 1),
        "n_measured": int(len(lat)),
    }

def wer(ref: str, hyp: str) -> float:
    """Word error rate (Levenshtein on word sequences)."""
    r = ref.lower().split(); h = hyp.lower().split()
    if not r:
        return 0.0 if not h else 1.0
    d = np.zeros((len(r)+1, len(h)+1), dtype=int)
    for i in range(len(r)+1): d[i,0] = i
    for j in range(len(h)+1): d[0,j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            cost = 0 if r[i-1] == h[j-1] else 1
            d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+cost)
    return d[len(r), len(h)] / len(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-wer", action="store_true")
    ap.add_argument("--only", type=str, default="", help="comma-separated model subset")
    a = ap.parse_args()

    # ---- load cached features ----
    feat_files = glob.glob(os.path.join(ROOT, "data", "processed", "_retrain_feats_*.npz"))
    mfcc_npz = [f for f in feat_files if "mfcc" in f][0]
    logmel_npz = [f for f in feat_files if "logmel" in f][0]
    dm = np.load(mfcc_npz); dl = np.load(logmel_npz)
    Xtr, Ytr = dm["Xtr"], dm["Ytr"]
    Xva, Yva = dm["Xva"], dm["Yva"]
    Xte, Yte = dm["Xte"], dm["Yte"]
    # logmel npz should carry the same Y; sanity check
    assert np.array_equal(Ytr, dl["Ytr"])
    print(f"features: train={len(Xtr)} val={len(Xva)} test={len(Xte)}", flush=True)

    # ---- load test metadata ----
    tdf = pd.read_csv(os.path.join(SPLITS, "test.csv"), low_memory=False).dropna(subset=["abs_path"]).reset_index(drop=True)
    assert len(tdf) == len(Xte), f"test csv {len(tdf)} != feats {len(Xte)}"
    intent_names = tdf["intent_name"].values
    conditions = tdf["condition"].values
    paths = tdf["abs_path"].values

    results = {}
    # resume: load any prior v3 results so partial runs merge cleanly
    prior = os.path.join(OUTDIR, "eval_results.json")
    if os.path.exists(prior):
        with open(prior) as f:
            results = json.load(f)
    names = [n.strip() for n in a.only.split(",") if n.strip()] or ["logistic", "dnn", "cnn1d", "crnn"]
    for name in names:
        kind = "mfcc" if name in ("logistic", "dnn") else "logmel"
        tag = f"{name}_v3_{int(time.time())}"
        print(f"\n========== TRAINING {name} ({kind}, {a.epochs} ep) ==========", flush=True)

        torch.manual_seed(a.seed); np.random.seed(a.seed)
        tl = DataLoader(VCMDataset(Xtr, Ytr, augment=True), batch_size=a.bs, shuffle=True)
        model = build_model(name).to(DEVICE)
        nparams = count_params(model)
        print(f"params={nparams:,}", flush=True)
        crit = nn.CrossEntropyLoss(weight=class_weights(Ytr))
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

        history = []
        best_acc, best_state, best_epoch = 0.0, None, 0
        t0 = time.time()
        for ep in range(a.epochs):
            model.train()
            tot_loss, nb = 0.0, 0
            for x, y in tl:
                x = x.to(DEVICE); y = y.to(DEVICE)
                opt.zero_grad()
                loss = crit(model(x), y)
                loss.backward(); opt.step()
                tot_loss += loss.item(); nb += 1
            sched.step()
            pa = predict_all(model, Xtr, bs=512)[0]
            tr_acc = float((pa == Ytr).mean())
            pv = predict_all(model, Xva, bs=512)[0]
            va_acc = float((pv == Yva).mean())
            history.append({"epoch": ep+1, "train_loss": round(tot_loss/nb,5),
                            "train_acc": round(tr_acc,5), "val_acc": round(va_acc,5)})
            if va_acc > best_acc:
                best_acc, best_epoch = va_acc, ep+1
                best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if (ep+1) % 10 == 0 or ep == 0:
                print(f"[{name}] ep {ep+1:3d}  loss={tot_loss/nb:.4f}  train={tr_acc:.4f}  val={va_acc:.4f}  best={best_acc:.4f}", flush=True)
        train_wall = time.time() - t0
        print(f"[{name}] training done in {train_wall:.0f}s, best val {best_acc:.4f} @ ep {best_epoch}", flush=True)

        # ---- restore best, full test evaluation ----
        model.load_state_dict(best_state)
        model.eval()
        preds, probs = predict_all(model, Xte, bs=256)
        test_acc = float((preds == Yte).mean())

        # per-intent recall
        rec = {}
        for c in range(N_CLASSES):
            mask = Yte == c
            n = int(mask.sum())
            if n == 0:
                rec[CLASSES[c]] = None; continue
            correct = int((preds[mask] == c).sum())
            rec[CLASSES[c]] = {"recall": round(correct/n, 4), "n": n, "correct": correct}

        # confusion matrix
        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        for t, p in zip(Yte, preds):
            cm[t, p] += 1

        # task completion / success rate
        # "task completed" = model outputs the correct actionable intent OR
        # correctly rejects (REJECT class). A command is FAILED if an actionable
        # clip is rejected, or a reject clip triggers an action, or wrong intent.
        is_cmd = Yte != REJECT
        is_rej = Yte == REJECT
        cmd_ok = int(((preds == Yte) & is_cmd).sum())
        rej_ok = int(((preds == REJECT) & is_rej).sum())
        false_trigger = int(((preds != REJECT) & is_rej).sum())   # reject clip -> action
        missed_cmd    = int(((preds == REJECT) & is_cmd).sum())   # command -> reject
        task_success = (cmd_ok + rej_ok) / len(Yte)
        cmd_recall_overall = cmd_ok / int(is_cmd.sum())

        # latency + efficiency
        lat = benchmark_latency(model, Xte)
        macs = macs_estimate(model, kind)
        fp32_mb = nparams * 4 / 1e6
        int8_mb = nparams * 1 / 1e6

        # ---- WER on slotted intents (whisper.cpp) ----
        # Group test clips by the value-folder in their filename, transcribe up
        # to 6 per group, compare against the canonical phrase for that group.
        # NOTE: WER measures the ASR (whisper-tiny) leg of the pipeline, not the
        # intent model -- it is identical across models, so we compute it once.
        wer_rows = []
        if not a.skip_wer and name == "logistic":
            sys.path.insert(0, os.path.join(ROOT, "pi_bundle"))
            from asr import Transcriber
            tr = Transcriber(os.path.join(ROOT, "asr", "ggml-tiny.en.bin"))
            if tr.available:
                for intent, vals in CANONICAL.items():
                    cidx = CLASSES.index(intent)
                    for val, canon in vals.items():
                        matched = [i for i in range(len(Yte))
                                   if Yte[i] == cidx and val in os.path.basename(paths[i])]
                        for i in matched[:6]:
                            txt = tr.transcribe_wav(paths[i])
                            w = wer(canon, txt)
                            wer_rows.append({"intent": intent, "value": val,
                                             "reference": canon, "hypothesis": txt,
                                             "wer": round(w, 4), "path": paths[i]})
                            print(f"  WER {intent}/{val}: {w:.3f}  {txt!r}", flush=True)
            else:
                print("  WER skipped: whisper unavailable", flush=True)

        wer_df = pd.DataFrame(wer_rows)
        wer_summary = {}
        if len(wer_df):
            wer_summary = {
                "overall_wer": round(float(wer_df["wer"].mean()), 4),
                "by_intent": {k: round(float(g["wer"].mean()), 4) for k, g in wer_df.groupby("intent")},
                "by_value": {f"{iv[0]}/{iv[1]}": round(float(g["wer"].mean()), 4)
                             for iv, g in wer_df.groupby(["intent", "value"])},
                "n_utterances": int(len(wer_df)),
            }
            wer_df.to_csv(os.path.join(OUTDIR, "wer_details.csv"), index=False)
            # WER is an ASR-leg metric, identical for all models -> attach to each
            for nm in ["logistic", "dnn", "cnn1d", "crnn"]:
                if nm in results:
                    results[nm]["wer"] = wer_summary

        r = {
            "model": name, "kind": kind, "epochs": a.epochs, "seed": a.seed,
            "params": nparams,
            "size_fp32_mb": round(fp32_mb, 3), "size_int8_mb": round(int8_mb, 3),
            "macs_per_clip": int(macs), "macs_millions": round(macs/1e6, 2),
            "train_wall_seconds": round(train_wall, 1),
            "best_val_epoch": best_epoch, "best_val_acc": round(best_acc, 5),
            "final_train_acc": history[-1]["train_acc"],
            "test_accuracy": round(test_acc, 5),
            "per_intent_recall": rec,
            "confusion_matrix": cm.tolist(),
            "task_completion": {
                "overall_success_rate": round(task_success, 5),
                "command_success_rate": round(cmd_recall_overall, 5),
                "commands_total": int(is_cmd.sum()), "commands_ok": cmd_ok,
                "rejects_total": int(is_rej.sum()), "rejects_ok": rej_ok,
                "false_triggers": false_trigger, "missed_commands": missed_cmd,
            },
            "latency": lat,
            "wer": wer_summary,
        }
        torch.save(best_state, os.path.join(OUTDIR, f"{tag}_state.pt"))
        with open(os.path.join(OUTDIR, f"{tag}_history.json"), "w") as f:
            json.dump({"result": r, "history": history}, f, indent=1)
        results[name] = r
        print(f"[{name}] TEST={test_acc:.4f}  task_success={task_success:.4f}  "
              f"p50={lat['single_p50_ms']}ms  p95={lat['single_p95_ms']}ms  "
              f"throughput={lat['batch_clips_per_sec']} clips/s  "
              f"WER={wer_summary.get('overall_wer','n/a')}", flush=True)

    with open(os.path.join(OUTDIR, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("\n===== FINAL SUMMARY =====")
    for n, r in results.items():
        print(f"{n:9s} test={r['test_accuracy']:.4f}  task={r['task_completion']['overall_success_rate']:.4f}  "
              f"p50={r['latency']['single_p50_ms']}ms  thr={r['latency']['batch_clips_per_sec']}/s  "
              f"params={r['params']:,}  WER={r['wer'].get('overall_wer','n/a')}")

if __name__ == "__main__":
    main()
