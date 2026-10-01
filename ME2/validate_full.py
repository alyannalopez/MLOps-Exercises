#!/usr/bin/env python3
"""Full held-out test-set validation of the deployed CRNN INT8 model.

Uses the EXACT Pi inference path (pi_bundle/vcm_infer.classify_pcm) and the
official test split (data/processed/splits/test.csv). Produces:
  - overall accuracy + latency percentiles
  - per-intent accuracy
  - confusion pairs (where the model gets confused)
  - per-intent breakdown by condition (clean vs noisy) and by source
  - the 100 worst misclassified clips with their text (for failure analysis)

Run: python3 validate_full.py
"""
from __future__ import annotations
import csv, os, sys, time, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "pi_bundle")
sys.path.insert(0, BUNDLE)
import vcm_infer as V  # the deployed inference module

MODEL = os.path.join(BUNDLE, "model_int8.onnx")
TEST_CSV = os.path.join(HERE, "data", "processed", "splits", "test.csv")


def load_wav_16k(path):
    import wave
    with wave.open(path, "rb") as w:
        sr = w.getframerate(); n = w.getnframes()
        raw = np.frombuffer(w.readframes(n), dtype=np.int16)
    return raw, sr


def main():
    rows = []
    with open(TEST_CSV) as f:
        for r in csv.DictReader(f):
            if r["abs_path"] and os.path.exists(r["abs_path"]):
                rows.append(r)
    print(f"test split: {len(rows)} clips (with files on disk)")

    vcm = V.VCM(MODEL)

    # ---- feature parity check (deployed path vs training path) ----
    import features as F
    fn = F.logmel_features
    worst = 0.0
    for r in rows[:8]:
        raw, sr = load_wav_16k(r["abs_path"])
        mine = vcm._logmel_train(raw, sr)
        ref = fn(r["abs_path"])
        m = min(mine.shape[0], ref.shape[0])
        worst = max(worst, float(np.abs(mine[:m] - ref[:m]).max()))
    print(f"[feature parity] max |diff| over 8 clips: {worst:.4f} dB "
          f"({'PASS' if worst < 3.0 else 'FAIL'})\n")

    # ---- full run ----
    t0 = time.time()
    per = collections.defaultdict(lambda: [0, 0])          # intent -> [correct, total]
    per_cond = collections.defaultdict(lambda: [0, 0])     # (intent, cond) -> [c, t]
    per_src = collections.defaultdict(lambda: [0, 0])      # (intent, src) -> [c, t]
    conf_pairs = collections.Counter()                     # (true, pred) -> n
    lat = []
    wrong = []
    total = correct = 0

    for i, r in enumerate(rows):
        try:
            raw, sr = load_wav_16k(r["abs_path"])
        except Exception:
            continue
        res = vcm.classify_pcm(raw, sr)
        lat.append(res["timings_ms"]["total"])
        true = r["intent_name"]; pred = res["intent"]
        total += 1
        per[true][1] += 1
        per_cond[(true, r["condition"])][1] += 1
        per_src[(true, r["source"])][1] += 1
        if pred == true:
            correct += 1
            per[true][0] += 1
            per_cond[(true, r["condition"])][0] += 1
            per_src[(true, r["source"])][0] += 1
        else:
            conf_pairs[(true, pred)] += 1
            wrong.append((res.get("confidence", 0.0), r))
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  ...{i+1}/{len(rows)} ({el:.0f}s elapsed)", flush=True)

    lat = np.array(lat)
    wall = time.time() - t0
    print(f"\nran {total} clips in {wall:.0f}s wall-clock\n")

    print("=" * 72)
    print(f"OVERALL ACCURACY: {correct}/{total} = {correct/total*100:.2f}%")
    print(f"LATENCY ms: mean={lat.mean():.1f}  p50={np.percentile(lat,50):.1f}  "
          f"p95={np.percentile(lat,95):.1f}  p99={np.percentile(lat,99):.1f}  "
          f"max={lat.max():.1f}")
    print("=" * 72)

    print("\nPER-INTENT ACCURACY (sorted weakest first):")
    print(f"{'intent':<20}{'acc':>8}{'n':>7}  bar")
    for intent, (c, t) in sorted(per.items(), key=lambda kv: kv[1][0]/kv[1][1]):
        bar = "#" * int(c / t * 40)
        print(f"{intent:<20}{c/t*100:>7.1f}%{t:>7}  {bar}")

    print("\nTOP CONFUSION PAIRS (true -> predicted):")
    for (t_, p_), n in conf_pairs.most_common(15):
        print(f"  {t_:<20} -> {p_:<20} x{n}")

    print("\nBY CONDITION (clean vs noisy):")
    for (intent, cond), (c, t) in sorted(per_cond.items()):
        print(f"  {intent:<20} {cond:<8} {c/t*100:6.1f}%  ({c}/{t})")

    print("\nBY SOURCE:")
    for (intent, src), (c, t) in sorted(per_src.items(), key=lambda kv: kv[1][0]/kv[1][1]):
        print(f"  {intent:<20} {src:<12} {c/t*100:6.1f}%  ({c}/{t})")

    print(f"\nWORST {min(100, len(wrong))} MISCLASSIFIED CLIPS (highest-confidence wrong first):")
    wrong.sort(key=lambda x: -x[0])
    for conf, r in wrong[:100]:
        txt = (r.get("text") or "").strip()
        print(f"  conf={conf:.2f}  true={r['intent_name']:<20} "
              f"src={r['source']:<10} cond={r['condition']:<6} "
              f"{txt[:70]}")

    # dump full confusion matrix to json for reference
    import json
    out = {
        "overall_acc": correct / total,
        "n": total,
        "latency_ms": {"mean": float(lat.mean()), "p50": float(np.percentile(lat, 50)),
                        "p95": float(np.percentile(lat, 95)), "p99": float(np.percentile(lat, 99)),
                        "max": float(lat.max())},
        "per_intent": {k: {"correct": c, "total": t, "acc": c / t} for k, (c, t) in per.items()},
        "confusion_top": {f"{t_}->{p_}": n for (t_, p_), n in conf_pairs.most_common(30)},
    }
    with open(os.path.join(HERE, "validation_full.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {os.path.join(HERE, 'validation_full.json')}")


if __name__ == "__main__":
    main()
