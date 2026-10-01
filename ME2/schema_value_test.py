#!/usr/bin/env python3
"""Schema-value recognition test: Option B slotted values, end-to-end.

For every slotted intent in the Option B schema (TIMER, ALARM, TEMPERATURE,
BRIGHTNESS, COLOR, CREATE_REMINDER), this runs REAL audio clips through the
exact production path:

    wav -> vcm_infer.classify_pcm (INT8 ONNX) -> intent + confidence
         -> whisper.cpp tiny.en transcript -> numparse.extract_slot -> value

and checks the recovered VALUE against the ground truth encoded in the clip
folder name (e.g. TIMER_1m, COLOR_GREEN, CREATE_REMINDER_STUDY).

Run:  python3 schema_value_test.py [--n-per-value 20]
"""
import argparse, csv, json, os, random, sys, time
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
PI = os.path.join(HERE, "pi_bundle")
sys.path.insert(0, PI)

import vcm_infer, numparse
from asr import Transcriber, default_model_path

MANIFEST = os.path.join(HERE, "data", "processed", "manifest.csv")

# fine-grained label -> (coarse intent, expected value check)
def expected_value(label: str):
    if label.startswith("TIMER_"):
        return "SET_TIMER", {"TIMER_10s": 10, "TIMER_30s": 30, "TIMER_1m": 60}[label]
    if label.startswith("ALARM_"):
        return "SET_ALARM", {"ALARM_4_00AM": "04:00", "ALARM_8_00AM": "08:00",
                             "ALARM_9_00PM": "21:00"}[label]
    if label.startswith("TEMPERATURE_"):
        return "THERMOSTAT", int(label.split("_")[1])
    if label.startswith("BRIGHTNESS_"):
        return "DIM_COLOR_LIGHTS", int(label.split("_")[1])
    if label.startswith("COLOR_"):
        return "DIM_COLOR_LIGHTS", label.split("_")[1].lower()
    if label.startswith("CREATE_REMINDER_"):
        return "REMINDERS_LISTS", label.replace("CREATE_REMINDER_", "").replace("_", " ").title()
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-value", type=int, default=20,
                    help="clips to test per fine-grained value (default 20)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=os.path.join(PI, "model_int8.onnx"))
    args = ap.parse_args()
    random.seed(args.seed)

    model = vcm_infer.VCM(args.model)
    tr = Transcriber(default_model_path())
    assert tr.available, "whisper-cli or model missing"

    rows = [r for r in csv.DictReader(open(MANIFEST)) if r["source"] == "optionB"]
    by_label = collections.defaultdict(list)
    for r in rows:
        by_label[r["intent_raw"]].append(r)

    results = []
    t0 = time.time()
    for label in sorted(by_label):
        intent, truth = expected_value(label)
        if intent is None:
            continue
        clips = random.sample(by_label[label], min(args.n_per_value, len(by_label[label])))
        for r in clips:
            res = model.classify_wav(r["abs_path"])
            pred, conf = res["intent"], res["confidence"]
            text = tr.transcribe_wav(r["abs_path"])
            slot = numparse.extract_slot(text, pred)
            # evaluate value only when the coarse intent was right
            if pred != intent:
                status = "WRONG_INTENT"
                got = None
            else:
                if intent == "SET_TIMER":
                    got = (slot.get("duration") or {}).get("seconds")
                    status = "OK" if got == truth else "WRONG_VALUE"
                elif intent == "SET_ALARM":
                    got = (slot.get("clock") or {}).get("time")
                    status = "OK" if got == truth else "WRONG_VALUE"
                elif intent == "THERMOSTAT":
                    got = slot.get("temp")
                    status = "OK" if got == truth else "WRONG_VALUE"
                elif intent == "DIM_COLOR_LIGHTS":
                    if isinstance(truth, int):
                        got = slot.get("percent")
                        status = "OK" if got == truth else "WRONG_VALUE"
                    else:  # color
                        got = slot.get("color")
                        status = "OK" if got == truth else "WRONG_VALUE"
                elif intent == "REMINDERS_LISTS":
                    got = slot.get("task")
                    # task match: case-insensitive substring either way
                    ok = bool(got) and (got.lower() in truth.lower() or truth.lower() in got.lower())
                    status = "OK" if ok else "WRONG_VALUE"
            results.append(dict(value=label, intent=intent, pred=pred,
                                conf=round(float(conf), 3), transcript=text,
                                got=got, truth=truth, status=status,
                                file=os.path.basename(r["abs_path"])))
            print(f"  {label:28s} {status:12s} conf={conf:.2f} "
                  f"heard={text!r:45.45s} got={got!r:15s} truth={truth!r}", flush=True)

    # ---- summary ----
    print("\n" + "=" * 78)
    agg = collections.defaultdict(lambda: collections.Counter())
    for r in results:
        agg[r["value"]][r["status"]] += 1
    print(f"{'VALUE':28s} {'n':>3s} {'intent':>7s} {'value':>6s} {'overall':>8s}")
    tot = collections.Counter()
    for v in sorted(agg):
        c = agg[v]
        n = sum(c.values())
        ni, nv = c["OK"] + c["WRONG_VALUE"], c["OK"]
        ov = (c["OK"]) / n * 100
        print(f"{v:28s} {n:3d} {ni/n*100:6.1f}% {nv/n*100:5.1f}% {ov:7.1f}%")
        tot.update(c)
    n = sum(tot.values())
    print("-" * 78)
    print(f"{'TOTAL':28s} {n:3d} {(tot['OK']+tot['WRONG_VALUE'])/n*100:6.1f}% "
          f"{tot['OK']/n*100:5.1f}% {tot['OK']/n*100:7.1f}%")
    print(f"(intent% = coarse intent correct | value% = value given intent right | "
          f"overall = fully correct)")
    print(f"\nwall time: {time.time()-t0:.0f}s")

    # dump failures for inspection
    fails = [r for r in results if r["status"] != "OK"]
    out = os.path.join(HERE, "schema_value_test.json")
    json.dump({"summary": {k: int(v) for k, v in tot.items()}, "results": results},
              open(out, "w"), indent=1)
    print(f"\n{len(fails)} non-OK cases -> {out}")
    for r in fails[:25]:
        print(f"  {r['value']:28s} {r['status']:12s} pred={r['pred']:16s} "
              f"heard={r['transcript']!r:45.45s} got={r['got']!r}")


if __name__ == "__main__":
    main()
