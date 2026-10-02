#!/usr/bin/env python3
"""Step 1: build the Option B manifest by scanning the raw dataset on disk.

Walks ME2/data/raw/optionB_v2/MEX2/Data/<INTENT>/<file>.wav, parses the
filename convention  <INTENT>_s<SPEAKER>_v<VARIANT>_<clean|noisy>.wav ,
reads the WAV header for duration, and maps the folder to the 10-intent
taxonomy via label_map.py.

Output: ME2/data/manifest.csv
Columns: abs_path, rel_path, intent_raw, intent_10, intent_name,
         speaker, variant, condition, duration_s
"""
from __future__ import annotations
import os, re, sys, wave
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # ME2/
sys.path.insert(0, HERE)
from label_map import RAW_TO_INTENT, CLASSES  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data", "raw", "optionB_v2", "MEX2", "Data")
OUT = os.path.join(ROOT, "data", "manifest.csv")

FN_RE = re.compile(r"^(?P<intent>.+?)_s(?P<spk>\d+)_v(?P<var>\d+)_(?P<cond>clean|noisy)\.wav$")


def wav_duration(path: str) -> float:
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return float("nan")


def main():
    rows = []
    bad = 0
    for intent_raw in sorted(os.listdir(DATA_DIR)):
        d = os.path.join(DATA_DIR, intent_raw)
        if not os.path.isdir(d) or intent_raw.startswith("."):
            continue
        if intent_raw not in RAW_TO_INTENT:
            print(f"  !! unmapped folder: {intent_raw}")
            bad += 1
            continue
        for fn in sorted(os.listdir(d)):
            m = FN_RE.match(fn)
            p = os.path.join(d, fn)
            if not m:
                bad += 1
                continue
            rows.append({
                "abs_path": p,
                "rel_path": os.path.relpath(p, ROOT),
                "intent_raw": intent_raw,
                "intent_10": RAW_TO_INTENT[intent_raw],
                "intent_name": CLASSES[RAW_TO_INTENT[intent_raw]],
                "speaker": int(m.group("spk")),
                "variant": int(m.group("var")),
                "condition": m.group("cond"),
                "duration_s": round(wav_duration(p), 3),
            })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"Manifest: {len(df)} clips -> {OUT}")
    print(f"Unmapped/unparseable: {bad}")
    print(f"Speakers: {df.speaker.nunique()}")
    print(f"Conditions: {df.condition.value_counts().to_dict()}")
    print("\nPer 10-intent:")
    print(df.groupby(["intent_10", "intent_name"]).agg(
        clips=("abs_path", "count"),
        raw_folders=("intent_raw", "nunique"),
        speakers=("speaker", "nunique"),
    ).to_string())


if __name__ == "__main__":
    main()
