"""Full-test-set validation of a deployed VCM model (INT8 ONNX).

Runs the ENTIRE held-out test split through the exact inference path used on
the Pi (vcm_infer.VCM.classify_wav) and reports overall accuracy, per-intent
accuracy, and end-to-end latency percentiles.

Usage:
    python validate_full.py [model.onnx]
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from vcm_infer import VCM

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/crnn_1790586409_int8.onnx"
vcm = VCM(MODEL)
print("loaded", MODEL, "kind=", getattr(vcm, "kind", "?"))

t = pd.read_csv("data/processed/splits/test.csv", low_memory=False)
t = t.dropna(subset=["abs_path"]).reset_index(drop=True)

names = {0: "REJECT", 1: "PLAY_MUSIC", 2: "QUESTION_SEARCH", 3: "LIGHTS_ON_OFF",
         4: "DIM_COLOR_LIGHTS", 5: "SET_TIMER", 6: "SET_ALARM", 7: "THERMOSTAT",
         8: "MEDIA_CONTROL", 9: "REMINDERS_LISTS", 10: "CALLS_MESSAGING"}

def cls_of(i): return 0 if int(i) == -1 else int(i)

correct = 0; total = 0; lat = []
per = {k: [0, 0] for k in range(11)}
for _, row in t.iterrows():
    try:
        r = vcm.classify_wav(row["abs_path"])
    except Exception:
        continue
    pred = r["class"]; gold = cls_of(row["intent_10"])
    total += 1
    per[gold][1] += 1
    if pred == gold:
        correct += 1; per[gold][0] += 1
    lat.append(r["timings_ms"]["total"])

lat = np.array(lat)
print(f"\nTOTAL  n={total}  ACC={100*correct/total:.2f}%")
print(f"LATENCY p50={np.percentile(lat,50):.1f}ms p95={np.percentile(lat,95):.1f}ms max={lat.max():.1f}ms")
print("\nPer-intent accuracy:")
for k in range(11):
    c, n = per[k]
    print(f"  {names[k]:<16} {100*c/n:6.1f}%  ({c}/{n})")

out = MODEL.replace(".onnx", "_fulltest.json")
json.dump({"model": MODEL, "n": total, "acc": correct/total,
           "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
           "per_intent": {names[k]: round(per[k][0]/per[k][1], 4) for k in range(11)}},
          open(out, "w"), indent=2)
print("\nsaved", out)
