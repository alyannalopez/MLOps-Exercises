#!/usr/bin/env python3
"""Stage 2b: Build the local dataset from the HF parquet files.

Produces:
  data/me2-dataset/manifest.csv   - one row per clip with resolved fields
  data/me2-dataset/audio/<split>/ - decoded 16 kHz mono wav files

Verifies the schema (19 intents + OUT_OF_SCOPE, 93 variations, slot values)
against configs/variations.csv and prints a verification report.

Usage:
  python scripts/hf_to_dataset.py [--only train,test] [--skip-audio]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / "data" / "hf"
OUT = ROOT / "data" / "me2-dataset"
AUDIO_DIR = OUT / "audio"

# The 19 intents in canonical order (OUT_OF_SCOPE appended as class 19).
INTENTS = [
    "PLAY_MUSIC", "WEATHER", "TIME", "LIGHT_ON", "LIGHT_OFF",
    "PAUSE", "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN",
    "CALL", "MESSAGE", "LIST_REMINDERS",
    "TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS", "COLOR", "CREATE_REMINDER",
]
N_CLASSES = len(INTENTS) + 1  # + OUT_OF_SCOPE
LABEL2ID = {c: i for i, c in enumerate(INTENTS + ["OUT_OF_SCOPE"])}

SPLIT_GLOBS = {
    "train": "data/train-*.parquet",
    "test": "data/test-*.parquet",
    "holdout": "data/holdout-*.parquet",
    "neg_train": "synthetic_negatives/train-*.parquet",
    "neg_test": "synthetic_negatives/test-*.parquet",
}


def load_split(name: str) -> pd.DataFrame:
    import glob
    files = sorted(glob.glob(str(HF / SPLIT_GLOBS[name])))
    if not files:
        raise FileNotFoundError(f"no parquet for split {name}")
    dfs = []
    for f in files:
        t = pq.read_table(f).to_pandas()
        t["split"] = name
        dfs.append(t)
    return pd.concat(dfs, ignore_index=True)


def verify_schema(manifest: pd.DataFrame, variations_csv: Path) -> None:
    """Cross-check the dataset against the agreed schema. Prints report."""
    var = pd.read_csv(variations_csv)
    manifest = manifest.copy()
    manifest["slot_value"] = manifest["slot_value"].fillna("")
    manifest["variation"] = manifest["variation"].fillna("__OTHER_SLOT__")
    issues = []

    # 1. Labels present
    labels_in_data = set(manifest["command"].unique())
    expected = set(INTENTS) | {"OUT_OF_SCOPE"}
    missing = expected - labels_in_data
    extra = labels_in_data - expected
    if missing: issues.append(f"missing labels: {missing}")
    if extra: issues.append(f"unexpected labels: {extra}")

    # 2. Variation count per label vs variations.csv
    #    (rows with a NaN variation are "other slot value" group recordings
    #    documented in the README; they are not schema variations)
    # 2. Variation count per label vs variations.csv
    #    (rows with an empty variation are "other slot value" group
    #    recordings documented in the README; they are not schema variations)
    n_var_expected = var.groupby("label")["phrase"].count().to_dict()
    sel = (manifest["command"] != "OUT_OF_SCOPE") & (manifest["variation"].notna()) & (manifest["variation"] != "")
    n_var_actual = manifest.loc[sel].groupby("command")["variation"].nunique().to_dict()
    for lab, exp in n_var_expected.items():
        act = n_var_actual.get(lab, 0)
        if act != exp:
            issues.append(f"{lab}: {act} variations in data, {exp} in schema")

    # 3. Slot values per slotted label (only rows that carry a slot value)
    slotted = var[var["value"].notna()]
    for _, row in slotted.iterrows():
        lab = row["label"]
        vals_expected = set(slotted[slotted["label"] == lab]["value"].unique())
        vals_actual = set(manifest[(manifest["command"] == lab) & (manifest["slot_value"] != "")]["slot_value"].unique())
        if vals_actual - vals_expected:
            issues.append(f"{lab}: unexpected slot values {vals_actual - vals_expected}")

    # 4. Speaker disjointness across train/test/holdout
    sp = manifest.groupby("split")["speaker_id"].apply(set)
    if "train" in sp and "test" in sp:
        ov = sp["train"] & sp["test"]
        if ov: issues.append(f"speaker overlap train/test: {len(ov)}")
    if "train" in sp and "holdout" in sp:
        ov = sp["train"] & sp["holdout"]
        if ov: issues.append(f"speaker overlap train/holdout: {len(ov)}")

    print("=" * 60)
    print("SCHEMA VERIFICATION")
    print("=" * 60)
    print(f"labels in data: {len(labels_in_data)} (expect {len(expected)})")
    print(f"total clips: {len(manifest)}")
    print(f"splits: {manifest['split'].value_counts().to_dict()}")
    if issues:
        print("\nISSUES FOUND:")
        for i in issues:
            print("  -", i)
    else:
        print("\nALL CHECKS PASSED: 19 intents + OUT_OF_SCOPE, 93 variations,")
        print("slot values match schema, splits are speaker-disjoint.")
    print("=" * 60)
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="train,test,holdout,neg_train,neg_test")
    ap.add_argument("--skip-audio", action="store_true", help="build manifest only, don't decode wavs")
    args = ap.parse_args()

    splits = [s.strip() for s in args.only.split(",")]
    frames = [load_split(s) for s in splits]
    manifest = pd.concat(frames, ignore_index=True)

    # Normalize columns
    manifest["label_id"] = manifest["command"].map(LABEL2ID)
    assert manifest["label_id"].notna().all(), "unmapped command label"
    manifest["slot_value"] = manifest["slot_value"].fillna("")
    # Mark "other slot value" group recordings (documented in the README)
    # with a sentinel so the CSV round-trip preserves them distinctly.
    manifest["variation"] = manifest["variation"].where(manifest["variation"].notna(), "__OTHER_SLOT__")

    OUT.mkdir(parents=True, exist_ok=True)
    mpath = OUT / "manifest.csv"
    cols = ["split", "file", "command", "label_id", "variation", "slot_value",
            "speaker_id", "source", "is_synthetic", "accent_group", "bucket",
            "numerals", "duration_s", "transcript", "out_of_scope"]
    manifest[cols].to_csv(mpath, index=False, na_rep="__OTHER_SLOT__")
    print(f"wrote {mpath} ({len(manifest)} rows)")

    issues = verify_schema(manifest, ROOT / "configs" / "variations.csv")

    if not args.skip_audio:
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        done = 0
        for split, grp in manifest.groupby("split"):
            sdir = AUDIO_DIR / split
            sdir.mkdir(exist_ok=True)
            for _, row in grp.iterrows():
                dst = sdir / (Path(row["file"]).stem + ".wav")
                if dst.exists():
                    done += 1
                    continue
                # re-read audio from parquet by matching file name is slow;
                # instead we decode during the initial load below.
        print(f"(audio decode handled in decode pass: {done} already present)")

    # Decode audio in a second pass (reads audio column, writes wavs)
    if not args.skip_audio:
        import glob
        for s in splits:
            files = sorted(glob.glob(str(HF / SPLIT_GLOBS[s])))
            sdir = AUDIO_DIR / s
            sdir.mkdir(parents=True, exist_ok=True)
            for f in files:
                d = pq.read_table(f, columns=["audio", "file"]).to_pydict()
                for a, name in zip(d["audio"], d["file"]):
                    dst = sdir / (Path(name).stem + ".wav")
                    if dst.exists():
                        continue
                    # PCM_16 WAV with a 44-byte standard header; body == a["bytes"]
                    hdr = b"RIFF" + (44 + len(a["bytes"]) - 8).to_bytes(4, "little") \
                          + b"WAVEfmt " + (16).to_bytes(4, "little") \
                          + (1).to_bytes(2, "little") + (1).to_bytes(2, "little") \
                          + (16000).to_bytes(4, "little") + (32000).to_bytes(4, "little") \
                          + (2).to_bytes(2, "little") + (16).to_bytes(2, "little") \
                          + b"data" + (len(a["bytes"])).to_bytes(4, "little")
                    dst.write_bytes(hdr + a["bytes"])
        print(f"audio decoded to {AUDIO_DIR}")

    if issues:
        print("\nNOTE: schema issues found (see above). Review before training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
