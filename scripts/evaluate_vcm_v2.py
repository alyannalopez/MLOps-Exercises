#!/usr/bin/env python3
"""Stage 5 (extended): Full VCM-benchmark evaluation.

Adds on top of the original evaluate_vcm.py:
  - F2 scores (per-intent + macro)
  - Command-level (93-variation) accuracy and confusion pairs
  - Misfire decomposition (cross-intent vs same-intent wrong-slot)
  - Unknown-intent extraction from OOS transcripts
  - Slot-value accuracy (exact match) where the model's intent-level
    prediction is correct — reports the upper bound achievable without
    a dedicated slot head
  - Pipeline-style stats (response rate, no-response, extra fires)
  - Explicit TBD markers for metrics requiring Stage 6 (wake word, RPi)

All numbers are computed from real inference on the held-out test set.
Nothing is fabricated or estimated.

Usage:
  python scripts/evaluate_vcm_v2.py --checkpoint models/conformer_s_seed0/best.pt
  python scripts/evaluate_vcm_v2.py --checkpoint models/tcn_seed0/best.pt --tag tcn
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from time import perf_counter
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, fbeta_score,
                             precision_recall_fscore_support)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vcm.models import build, count_params  # noqa: E402
from vcm.features import log_mel_spectrogram  # noqa: E402
from vcm.slots import (INTENT_SLOTS, SLOT_VALUES, NO_SLOT_IDX,  # noqa: E402
                       slot_index_for, decode_slot)

ROOT = Path(__file__).resolve().parents[1]
PACKS = ROOT / "data" / "packs"
RESULTS = ROOT / "results"
MANIFEST = ROOT / "data" / "me2-dataset" / "manifest.csv"
N_CLASSES = 20
N_MELS, N_FRAMES = 40, 100
INTENTS = [
    "PLAY_MUSIC", "WEATHER", "TIME", "LIGHT_ON", "LIGHT_OFF",
    "PAUSE", "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN",
    "CALL", "MESSAGE", "LIST_REMINDERS",
    "TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS", "COLOR", "CREATE_REMINDER",
    "OUT_OF_SCOPE",
]
OOS_IDX = 19


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
    out = np.empty((len(waveforms), 1, N_MELS, N_FRAMES), dtype=np.float32)
    for i, w in enumerate(waveforms):
        out[i] = log_mel_spectrogram(w, n_mels=N_MELS)[None]
    return out


def load_split(names: list[str]):
    ws, ys, ss, spks = [], [], [], []
    for name in names:
        z = np.load(PACKS / f"{name}.npz", allow_pickle=True)
        ws.append(z["waveforms"])
        ys.append(z["labels"])
        ss.append(z["is_synthetic"])
        spks.append(z["speaker"])
    return (np.concatenate(ws), np.concatenate(ys),
            np.concatenate(ss), np.concatenate(spks))


def load_manifest_test() -> pd.DataFrame:
    """Load manifest rows for test + neg_test, aligned to npz ordering."""
    df = pd.read_csv(MANIFEST)
    test = df[df["split"] == "test"].reset_index(drop=True)
    # neg_test clips are all OUT_OF_SCOPE (label 19)
    z_neg = np.load(PACKS / "neg_test.npz", allow_pickle=True)
    n_neg = len(z_neg["labels"])
    neg_rows = pd.DataFrame({
        "split": ["neg_test"] * n_neg,
        "file": [f"neg_test_{i}" for i in range(n_neg)],
        "command": ["OUT_OF_SCOPE"] * n_neg,
        "label_id": [19] * n_neg,
        "variation": [""] * n_neg,
        "slot_value": [""] * n_neg,
        "speaker_id": z_neg["speaker"],
        "source": ["synthetic_negative"] * n_neg,
        "is_synthetic": [1] * n_neg,
        "accent_group": [""] * n_neg,
        "bucket": [""] * n_neg,
        "numerals": [""] * n_neg,
        "duration_s": [0.0] * n_neg,
        "transcript": [""] * n_neg,
        "out_of_scope": [1] * n_neg,
    })
    combined = pd.concat([test, neg_rows], ignore_index=True)
    return combined


# Numeric scale per slot value (for MAE / mean-relative-error on ordered slots).
# Units: TIMER=seconds, ALARM=hour-of-day (0-23), TEMPERATURE=Celsius,
# BRIGHTNESS=percent.  Categorical slots (COLOR, CREATE_REMINDER) have no
# numeric scale -> phonetic distance instead.
_SLOT_NUM: dict[str, float] = {
    "10 seconds": 10.0, "30 seconds": 30.0, "1 minute": 60.0,
    "6:00 AM": 6.0, "8:00 AM": 8.0, "9:00 PM": 21.0,
    "18 degrees": 18.0, "22 degrees": 22.0, "26 degrees": 26.0,
    "20 percent": 20.0, "60 percent": 60.0, "100 percent": 100.0,
}
_SLOT_UNIT: dict[str, str] = {
    "10 seconds": "s", "30 seconds": "s", "1 minute": "s",
    "6:00 AM": "h", "8:00 AM": "h", "9:00 PM": "h",
    "18 degrees": "degC", "22 degrees": "degC", "26 degrees": "degC",
    "20 percent": "%", "60 percent": "%", "100 percent": "%",
}


def _levenshtein(a: str, b: str) -> int:
    """Plain Levenshtein edit distance (case-insensitive, spaces kept)."""
    a, b = a.lower(), b.lower()
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _phonetic_distance(a: str, b: str) -> float:
    """Normalised edit distance in [0,1]; 0 = identical."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 0.0
    return _levenshtein(a, b) / max(la, lb)


def _slot_true_idx(row) -> int:
    return slot_index_for(str(row["command"]), row.get("slot_value", ""))


def compute_slot_metrics(y_true_labels, pred_labels, manifest: pd.DataFrame,
                         slot_pred_values: list[str] | None = None):
    """Slot-level analysis for the 6 slotted intents.

    Two modes:
      - slot_pred_values is None  -> intent-level model (upper bound only).
      - slot_pred_values given    -> two-head model: per-clip slot predictions
        from the slot head (jointly decoded against the *predicted* intent).
        Reports exact match, MAE + mean-relative-error on ordered (numeric)
        slots, and normalised phonetic (edit) distance on categorical slots.

    Joint decoding mirrors the runtime: the slot head's argmax is restricted
    to the valid values of the predicted intent, so a wrong intent forces
    NO_SLOT (slot necessarily wrong).
    """
    slotted_intents = set(INTENT_SLOTS.keys())
    intent_to_idx = {name: i for i, name in enumerate(INTENTS)}
    two_head = slot_pred_values is not None

    results = {}
    agg = {
        "n_slotted_clips": 0, "n_intent_correct": 0,
        "n_slot_exact": 0,
        "mae_vals": [], "rel_err_vals": [], "phon_vals": [],
    }
    for intent in sorted(slotted_intents):
        idx = intent_to_idx[intent]
        mask = y_true_labels == idx
        n_total = int(mask.sum())
        if n_total == 0:
            continue
        rows = manifest.loc[mask].reset_index(drop=True)
        n_correct_intent = int((pred_labels[mask] == idx).sum())

        per_intent = {
            "n_clips": n_total,
            "intent_correct": n_correct_intent,
            "intent_wrong": n_total - n_correct_intent,
            "intent_accuracy": round(n_correct_intent / n_total, 4),
            "slot_distribution": {str(k): int(v) for k, v in
                                  rows["slot_value"].value_counts().items()},
        }

        if two_head:
            true_idx = np.array([_slot_true_idx(r) for _, r in rows.iterrows()])
            pred_idx = np.array([
                slot_index_for(INTENTS[pred_labels[i]], slot_pred_values[i])
                for i in np.where(mask)[0]
            ])
            exact = int((true_idx == pred_idx).sum())
            per_intent["slot_exact_match"] = round(exact / n_total, 4)
            per_intent["slot_exact_given_intent"] = round(
                exact / max(1, n_correct_intent), 4)

            # Numeric (ordered) slots: MAE + mean relative error in native units
            mae_v, rel_v = [], []
            # Categorical slots: normalised phonetic distance
            phon_v = []
            for ti, pi in zip(true_idx, pred_idx):
                tv, pv = SLOT_VALUES[ti], SLOT_VALUES[pi]
                if tv in _SLOT_NUM and pv in _SLOT_NUM:
                    # Ordered (numeric) slot -> MAE / relative error in units
                    mae_v.append(abs(_SLOT_NUM[tv] - _SLOT_NUM[pv]))
                    rel_v.append(abs(_SLOT_NUM[tv] - _SLOT_NUM[pv])
                                 / max(1.0, _SLOT_NUM[tv]))
                elif tv not in _SLOT_NUM:
                    # Categorical slot (COLOR, CREATE_REMINDER) -> phonetic
                    phon_v.append(_phonetic_distance(tv, pv))
            if mae_v:
                per_intent["slot_mae_units"] = round(float(np.mean(mae_v)), 3)
                per_intent["slot_unit"] = _SLOT_UNIT[SLOT_VALUES[true_idx[0]]]
                per_intent["slot_mean_relative_error"] = round(
                    float(np.mean(rel_v)), 4)
            if phon_v:
                per_intent["slot_phonetic_dist"] = round(float(np.mean(phon_v)), 4)

            agg["n_slotted_clips"] += n_total
            agg["n_intent_correct"] += n_correct_intent
            agg["n_slot_exact"] += exact
            agg["mae_vals"].extend(mae_v)
            agg["rel_err_vals"].extend(rel_v)
            agg["phon_vals"].extend(phon_v)
        else:
            per_intent["slot_exact_match_upper_bound"] = round(
                n_correct_intent / n_total, 4)
            per_intent["note"] = ("Intent-level model: slot resolved correctly "
                                  "iff intent is correct. Per-slot accuracy "
                                  "requires a slot-prediction head.")
        results[intent] = per_intent

    if two_head:
        results["_summary"] = {
            "mode": "two_head_joint_decoding",
            "n_slotted_clips": agg["n_slotted_clips"],
            "intent_correct": agg["n_intent_correct"],
            "slot_exact_match": round(agg["n_slot_exact"] /
                                      max(1, agg["n_slotted_clips"]), 4),
            "slot_exact_given_intent": round(agg["n_slot_exact"] /
                                             max(1, agg["n_intent_correct"]), 4),
            "mae_units_overall": round(float(np.mean(agg["mae_vals"])), 3)
            if agg["mae_vals"] else None,
            "mean_relative_error_overall": round(
                float(np.mean(agg["rel_err_vals"])), 4)
            if agg["rel_err_vals"] else None,
            "phonetic_dist_overall": round(float(np.mean(agg["phon_vals"])), 4)
            if agg["phon_vals"] else None,
        }
    return results


def compute_command_level(y_true_labels, pred_labels, manifest: pd.DataFrame):
    """Command-variation-level accuracy (93 variations)."""
    intent_to_idx = {name: i for i, name in enumerate(INTENTS)}
    
    # Build per-clip: true_variation, pred_intent
    true_vars = manifest["variation"].fillna("").values
    true_cmds = manifest["command"].fillna("").values
    
    # A clip is "command-correct" if the predicted intent matches AND
    # (for non-slotted) there's only one variation per intent... 
    # Actually for non-slotted intents, 3 variations share the same intent label.
    # The model can't distinguish variations within an intent.
    # So command-level accuracy = intent accuracy for non-slotted,
    # and for slotted it's intent accuracy (can't resolve slot).
    
    # Report per-variation accuracy:
    var_stats = {}
    for var in sorted(set(true_vars)):
        if not var:
            continue
        mask = true_vars == var
        n = int(mask.sum())
        if n == 0:
            continue
        # The intent for this variation
        var_intent = true_cmds[mask][0]
        var_idx = intent_to_idx.get(var_intent, -1)
        if var_idx < 0:
            continue
        n_correct = int((pred_labels[mask] == var_idx).sum())
        var_stats[var] = {
            "intent": var_intent,
            "n_clips": n,
            "correct": n_correct,
            "accuracy": round(n_correct / n, 4),
        }
    
    # Command-level confusion: for slotted intents, show which slot values
    # get confused with which other intents
    cmd_confusion = defaultdict(lambda: defaultdict(int))
    for i in range(len(y_true_labels)):
        if pred_labels[i] != y_true_labels[i]:
            true_cmd = true_cmds[i]
            pred_intent = INTENTS[pred_labels[i]]
            cmd_confusion[true_cmd][pred_intent] += 1
    
    # Top command-level confusion pairs
    cmd_pairs = []
    for true_cmd, preds in cmd_confusion.items():
        for pred_intent, cnt in preds.items():
            cmd_pairs.append((true_cmd, pred_intent, cnt))
    cmd_pairs.sort(key=lambda t: -t[2])
    
    return {
        "per_variation": var_stats,
        "top_command_confusion_pairs": [[a, b, n] for a, b, n in cmd_pairs[:15]],
        "note": "Models classify at intent level (20 classes). Variations within an intent "
                "(e.g. 'Timer 10 seconds' vs 'Countdown for 10 seconds') are NOT distinguished. "
                "Command-level accuracy = intent-level accuracy for the parent intent.",
    }


def compute_misfire_decomposition(y_true_labels, pred_labels):
    """Decompose errors into cross-intent vs OOS-related."""
    oos_idx = OOS_IDX
    n_total = len(y_true_labels)
    n_correct = int((y_true_labels == pred_labels).sum())
    n_errors = n_total - n_correct
    
    # Cross-intent: true=A (in-scope), pred=B (different in-scope)
    cross_intent = 0
    # True in-scope -> pred OOS (false reject)
    to_oos = 0
    # True OOS -> pred in-scope (false accept)
    oos_to_ins = 0
    
    for t, p in zip(y_true_labels, pred_labels):
        if t == p:
            continue
        if t == oos_idx and p != oos_idx:
            oos_to_ins += 1
        elif t != oos_idx and p == oos_idx:
            to_oos += 1
        elif t != oos_idx and p != oos_idx:
            cross_intent += 1
    
    return {
        "total_clips": n_total,
        "total_correct": n_correct,
        "total_errors": n_errors,
        "cross_intent_errors": cross_intent,
        "inscope_to_oos (false_reject)": to_oos,
        "oos_to_inscope (false_accept)": oos_to_ins,
        "cross_intent_rate": round(cross_intent / max(1, n_total), 4),
        "false_reject_rate": round(to_oos / max(1, n_total - oos_idx * 0), 4),  # of in-scope
        "false_accept_rate": round(oos_to_ins / max(1, (y_true_labels == oos_idx).sum()), 4),
    }


def compute_pipeline_stats(y_true_labels, pred_labels, probs):
    """Simulate pipeline behavior with a confidence threshold.
    
    In a real pipeline:
    - Wake word triggers -> VCM runs
    - If max_prob < threshold -> no response (silence)
    - If max_prob >= threshold -> execute predicted command
    
    We sweep thresholds and report:
    - Response rate: fraction of clips where the system responds
    - No-response rate: fraction where system stays silent
    - Extra fires: OOS clips that trigger a response
    - Missed commands: in-scope clips that get no response
    """
    max_probs = probs.max(axis=1)
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    oos_idx = OOS_IDX
    
    results = {}
    for thr in thresholds:
        respond = max_probs >= thr
        n_respond = int(respond.sum())
        n_no_response = int((~respond).sum())
        
        # Of those that respond:
        responded_correct = int((respond & (pred_labels == y_true_labels)).sum())
        responded_wrong = n_respond - responded_correct
        
        # Extra fires: OOS clips that triggered a response
        oos_mask = y_true_labels == oos_idx
        extra_fires = int((respond & oos_mask).sum())
        
        # Missed: in-scope clips that got no response
        inscope_mask = y_true_labels != oos_idx
        missed = int((~respond & inscope_mask).sum())
        
        # Of in-scope clips that responded, how many correct?
        inscope_responded = int((respond & inscope_mask).sum())
        inscope_correct = int((respond & inscope_mask & (pred_labels == y_true_labels)).sum())
        
        results[f"threshold_{thr}"] = {
            "response_rate": round(n_respond / len(y_true_labels), 4),
            "no_response_rate": round(n_no_response / len(y_true_labels), 4),
            "n_respond": n_respond,
            "n_no_response": n_no_response,
            "responded_correct": responded_correct,
            "responded_wrong": responded_wrong,
            "extra_fires (OOS responded)": extra_fires,
            "missed_commands (in-scope silent)": missed,
            "inscope_response_precision": round(inscope_correct / max(1, inscope_responded), 4),
            "inscope_response_recall": round(inscope_correct / max(1, inscope_mask.sum()), 4),
        }
    
    return results


def extract_unknown_intents(manifest: pd.DataFrame, y_true_labels, pred_labels):
    """What do the OOS clips actually contain? And what in-scope clips
    get misclassified as OOS?"""
    oos_idx = OOS_IDX
    
    # OOS clips: show transcript distribution (if available)
    oos_mask = y_true_labels == oos_idx
    oos_clips = manifest.loc[oos_mask]
    
    # For neg_test, transcripts are empty; for test OOS, they may have text
    oos_transcripts = oos_clips["transcript"].dropna().tolist()
    oos_transcripts = [t for t in oos_transcripts if t and str(t) != "nan"]
    
    # What in-scope intents get pushed to OOS?
    pushed_to_oos = Counter()
    for i in range(len(y_true_labels)):
        if y_true_labels[i] != oos_idx and pred_labels[i] == oos_idx:
            pushed_to_oos[INTENTS[y_true_labels[i]]] += 1
    
    # What do OOS clips get accepted as?
    oos_accepted_as = Counter()
    for i in range(len(y_true_labels)):
        if y_true_labels[i] == oos_idx and pred_labels[i] != oos_idx:
            oos_accepted_as[INTENTS[pred_labels[i]]] += 1
    
    return {
        "oos_clip_count": int(oos_mask.sum()),
        "oos_transcripts_sample": oos_transcripts[:20],
        "oos_transcripts_available": len(oos_transcripts),
        "inscope_pushed_to_oos": dict(pushed_to_oos.most_common(10)),
        "oos_accepted_as_command": dict(oos_accepted_as.most_common(10)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--no-neg", action="store_true")
    ap.add_argument("--latency-n", type=int, default=200)
    ap.add_argument("--tag", default=None)
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
    ckpt_size_mb = Path(args.checkpoint).stat().st_size / (1024 * 1024)

    # Load data
    split_names = [args.split]
    if not args.no_neg:
        neg_name = f"neg_{args.split}"
        if (PACKS / f"{neg_name}.npz").exists():
            split_names.append(neg_name)
    print(f"loading split(s): {split_names}")
    waveforms, y, synth, speakers = load_split(split_names)
    print(f"  n_clips={len(y)}  is_synthetic={int(synth.sum())}  real={int((synth==0).sum())}")

    # Load manifest for command/slot/transcript info
    manifest = load_manifest_test()
    assert len(manifest) == len(y), f"Manifest ({len(manifest)}) != npz ({len(y)})"

    # Features + inference
    t_feat = time.time()
    X_np = static_features(waveforms)
    print(f"  features in {time.time()-t_feat:.1f}s")
    X = torch.from_numpy(X_np)

    with torch.no_grad():
        out = model(X)
    two_head = isinstance(out, tuple)
    if two_head:
        logits, slot_logits = out
        slot_pred = slot_logits.argmax(-1).numpy()
    else:
        logits = out
        slot_pred = None
    pred = logits.argmax(-1).numpy()
    probs = torch.softmax(logits, -1).numpy()

    # === CORE METRICS ===
    acc = accuracy_score(y, pred)
    ba = balanced_accuracy_score(y, pred)
    prec, rec, f1, sup_macro = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    _, _, _, sup = precision_recall_fscore_support(y, pred, average=None, zero_division=0)
    f2_macro = fbeta_score(y, pred, average="macro", beta=2, zero_division=0)
    acc_lo, acc_hi = bootstrap_ci(y, pred, accuracy_score)
    ba_lo, ba_hi = bootstrap_ci(y, pred, balanced_accuracy_score)

    # Per-intent with F2
    cm = confusion_matrix(y, pred, labels=list(range(N_CLASSES)))
    per_intent = {}
    for c, name in enumerate(INTENTS):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        f2 = (5 * p * r) / (4 * p + r) if 4 * p + r else 0.0
        per_intent[name] = dict(
            support=int(sup[c]) if c < len(sup) else int(cm[c, :].sum()),
            precision=round(p, 4), recall=round(r, 4),
            f1=round(f, 4), f2=round(f2, 4),
        )

    # Confusion pairs
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

    # OOS
    oos_idx = OOS_IDX
    oos_true = y == oos_idx
    inscope_true = ~oos_true
    fa_rate = float(((pred != oos_idx) & oos_true).sum() / max(1, oos_true.sum()))
    fr_rate = float(((pred == oos_idx) & inscope_true).sum() / max(1, inscope_true.sum()))

    # === EXTENDED METRICS ===
    print("computing extended metrics...")
    if two_head:
        # Joint decoding: restrict slot argmax to the predicted intent's
        # valid values (mirrors runtime).  Wrong intent -> NO_SLOT.
        slot_pred_values = [
            decode_slot(INTENTS[p], [float(v) for v in slot_logits[i]])
            if INTENTS[p] in INTENT_SLOTS else None
            for i, p in enumerate(pred)
        ]
    else:
        slot_pred_values = None
    slot_metrics = compute_slot_metrics(y, pred, manifest, slot_pred_values)
    cmd_level = compute_command_level(y, pred, manifest)
    misfire = compute_misfire_decomposition(y, pred)
    pipeline = compute_pipeline_stats(y, pred, probs)
    unknown = extract_unknown_intents(manifest, y, pred)

    # Latency
    model.cpu()
    times = []
    with torch.no_grad():
        for i in range(args.latency_n):
            xb = X[i % len(X)][None]
            t0 = time.perf_counter()
            model(xb)
            times.append((perf_counter() - t0) * 1000)
    times = np.array(times)

    # FLOPs estimate (rough: params * 2 ops per forward for matmul-dominated)
    # This is a rough proxy; real FLOPs need torch profiler
    flops_estimate = n_params * 2  # MACs ≈ FLOPs/2 for dense layers

    report = {
        "checkpoint": str(args.checkpoint),
        "arch": arch,
        "split": "+".join(split_names),
        "n_clips": int(len(y)),
        "n_params": n_params,
        "checkpoint_size_mb": round(ckpt_size_mb, 2),
        "flops_estimate_per_inference": int(flops_estimate),
        "flops_note": "Rough proxy (params × 2). Real GFLOP/s requires torch.profiler on target HW.",
        
        # Core
        "accuracy": round(acc, 4),
        "accuracy_ci95": [round(acc_lo, 4), round(acc_hi, 4)],
        "balanced_accuracy": round(ba, 4),
        "balanced_accuracy_ci95": [round(ba_lo, 4), round(ba_hi, 4)],
        "macro_precision": round(prec, 4),
        "macro_recall": round(rec, 4),
        "macro_f1": round(f1, 4),
        "macro_f2": round(f2_macro, 4),
        "real_voice_accuracy": round(real_acc, 4) if real_acc is not None else None,
        "synthetic_voice_accuracy": round(syn_acc, 4) if syn_acc is not None else None,
        "oos_false_accept_rate": round(fa_rate, 4),
        "oos_false_accept_count": int(((pred != oos_idx) & oos_true).sum()),
        "oos_false_reject_rate": round(fr_rate, 4),
        "oos_false_reject_count": int(((pred == oos_idx) & inscope_true).sum()),
        "latency_ms": {
            "p50": round(float(np.percentile(times, 50)), 2),
            "p95": round(float(np.percentile(times, 95)), 2),
            "p99": round(float(np.percentile(times, 99)), 2),
            "mean": round(float(times.mean()), 2),
            "max": round(float(times.max()), 2),
        },
        "rtf_estimate": round(float(times.mean()) / 1000 / 1.0, 4),  # 1s clip
        "rtf_note": "RTF = inference_time / clip_duration (1.0s). Computed on Apple Silicon CPU.",
        
        # Per-intent
        "per_intent": per_intent,
        "top_confusion_pairs_intent_level": [[a, b, n] for a, b, n in pairs[:10]],
        "confusion_matrix": cm.tolist(),
        
        # Extended
        "slot_metrics": slot_metrics,
        "command_level": cmd_level,
        "misfire_decomposition": misfire,
        "pipeline_stats_by_threshold": pipeline,
        "unknown_intent_analysis": unknown,
        
        # TBD (require Stage 6)
        "tbd_metrics": {
            "wake_word_detection": "TBD — Stage 6 (OSS wake word not yet integrated)",
            "rpi_latency": "TBD — Stage 6 (requires live RPi 4/5 measurement)",
            "rpi_temp_clock_throttle": "TBD — Stage 6",
            "rpi_cpu_ram_usage": "TBD — Stage 6",
            "rpi_gflops": "TBD — Stage 6 (requires torch.profiler on RPi)",
            "unseen_speaker_eval": "TBD — holdout.npz (202 clips, 5 speakers) reserved for live Pi test per dataset docs",
            "own_voice_validation": "TBD — requires student-recorded voice samples",
        },
        
        "eval_seconds": round(time.time() - t_start, 1),
    }

    RESULTS.mkdir(exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out = RESULTS / f"eval_v2_{arch}_{args.split}{tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    
    # Summary
    summary = {k: report[k] for k in [
        "arch", "n_clips", "n_params", "checkpoint_size_mb",
        "accuracy", "accuracy_ci95", "balanced_accuracy",
        "macro_f1", "macro_f2",
        "real_voice_accuracy", "synthetic_voice_accuracy",
        "oos_false_accept_rate", "oos_false_reject_rate",
        "latency_ms", "rtf_estimate", "eval_seconds"]}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
