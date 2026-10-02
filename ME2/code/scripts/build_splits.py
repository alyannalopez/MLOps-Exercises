#!/usr/bin/env python3
"""Step 3: speaker-disjoint train/val/test split + reject class.

Option B has 100 real speakers. We do a SPEAKER-DISJOINT split: every
speaker's clips go wholly into exactly one of train/val/test. This is the
strongest leak-safe split — no speaker appears in more than one split —
which is exactly the generalization the exercise cares about
(unknown-voice commands).

The REJECT class (synthesized noise, no speaker identity) is split row-wise.

Output: ME2/data/splits/{train,val,test}.csv + stats.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MAN = os.path.join(ROOT, "data", "manifest.csv")
REJ_MAN = os.path.join(ROOT, "data", "reject", "reject_manifest.csv")
OUTDIR = os.path.join(ROOT, "data", "splits")


def spk_disjoint(df: pd.DataFrame, frac=(0.70, 0.15), seed=0) -> pd.Series:
    """Assign each speaker's clips wholly to one split."""
    df = df.copy()
    uniq = df["speaker"].unique()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n = len(uniq)
    tr = int(frac[0] * n)
    va = int(frac[1] * n)
    spk2 = {}
    for i, k in enumerate(perm):
        spk2[k] = "train" if i < tr else ("val" if i < tr + va else "test")
    return df["speaker"].map(spk2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, nargs=2, default=[0.70, 0.15])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rej-per-split", type=int, default=300)
    a = ap.parse_args()

    ob = pd.read_csv(MAN)
    ob["_split"] = spk_disjoint(ob, frac=tuple(a.frac), seed=a.seed)

    rej = pd.read_csv(REJ_MAN)
    rej = rej.sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    n = a.rej_per_split
    rej = rej.iloc[:3 * n]
    rej["_split"] = np.repeat(["train", "val", "test"], n)
    rej["intent_raw"] = "REJECT"
    rej["speaker"] = np.nan
    rej["variant"] = np.nan

    full = pd.concat([ob, rej], ignore_index=True)
    cols = ["abs_path", "rel_path", "intent_raw", "intent_10", "intent_name",
            "speaker", "variant", "condition", "duration_s", "_split"]
    full = full[[c for c in cols if c in full.columns]]

    os.makedirs(OUTDIR, exist_ok=True)

    # leak check
    pos = full[full.intent_name != "REJECT"]
    leaks = 0
    for s in pos.speaker.dropna().unique():
        sp = set(pos[pos.speaker == s]["_split"])
        if len(sp) > 1:
            leaks += 1

    stats = {
        "seed": a.seed, "frac": a.frac,
        "total": len(full),
        "positive_total": len(ob), "reject_total": len(rej),
        "speakers": int(ob["speaker"].nunique()),
        "per_split": {s: int((full._split == s).sum()) for s in ("train", "val", "test")},
        "speaker_leaks": leaks,
    }
    for s in ("train", "val", "test"):
        sub = full[full._split == s]
        stats[f"{s}_positive"] = int((sub.intent_name != "REJECT").sum())
        stats[f"{s}_reject"] = int((sub.intent_name == "REJECT").sum())
        stats[f"{s}_per_intent"] = sub[sub.intent_name != "REJECT"].intent_name.value_counts().to_dict()
        sub.to_csv(os.path.join(OUTDIR, f"{s}.csv"), index=False)

    with open(os.path.join(OUTDIR, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps({k: v for k, v in stats.items() if not k.endswith("_per_intent")}, indent=2))
    print(f"\nSpeaker leaks: {leaks} (must be 0)")
    print("Wrote", ", ".join(f"{s}.csv" for s in ("train", "val", "test")), "->", OUTDIR)


if __name__ == "__main__":
    main()
