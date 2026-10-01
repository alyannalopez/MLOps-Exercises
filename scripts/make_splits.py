"""Dataset construction for the VCM (v2).

Balanced, leak-safe train/val/test split.

Key fixes vs v1:
  * Speaker-disjointness is applied ONLY where a real speaker id exists
    (optionB=100 spk, snips=55 spk). Sources without speaker ids
    (slurp, speechcommands, fleurs) fall back to ROW-level 70/15/15,
    otherwise they collapse to a single group and vanish from val/test.
  * OOV (intent_10 == -1, all from FLEURS) is retained as the REJECTION
    class, capped, and split row-wise.

Per-intent capping prefers optionB (primary) first, then aux by weight.

Outputs:
  data/processed/splits/{train,val,test}.csv
  data/processed/splits/stats.json

Usage:
  python make_splits.py [--cap 2500] [--oov-cap 400] [--seed 0]
"""
from __future__ import annotations
import argparse, json, os
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MANIFEST = os.path.join(ROOT, "data", "processed", "manifest.csv")
OUTDIR   = os.path.join(ROOT, "data", "processed", "splits")

SRC_RANK = {"optionB":0,"snips":1,"slurp":2,"speechcommands":3,"fleurs":4,"librispeech":5}

def load() -> pd.DataFrame:
    df = pd.read_csv(MANIFEST, low_memory=False)
    return df.dropna(subset=["abs_path"]).reset_index(drop=True)

def _cap_positives(pos: pd.DataFrame, cap: int, rng) -> pd.DataFrame:
    parts = []
    for intent, g in pos.groupby("intent_10"):
        g = g.copy()
        g["_src"] = g.source.map(SRC_RANK).fillna(9)
        g["_w"]   = g.weight.fillna(0.1)
        g = g.sort_values(["_src","_w"], ascending=[True,False])
        g = g.sample(frac=1.0, random_state=int(rng.integers(1_000_000)))
        parts.append(g.head(cap))
    return pd.concat(parts, ignore_index=True)

def _row_split(df: pd.DataFrame, frac=(0.70,0.15), seed=0) -> pd.Series:
    idx = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n = len(df); tr = int(frac[0]*n); va = int(frac[1]*n)
    lab = np.empty(n, dtype=object)
    lab[idx[:tr]] = "train"; lab[idx[tr:tr+va]] = "val"; lab[idx[tr+va:]] = "test"
    return pd.Series(lab, index=df.index)

def _spk_split(df: pd.DataFrame, frac=(0.70,0.15), seed=0) -> pd.Series:
    """Speaker-disjoint: every (source,speaker) goes wholly into one split."""
    df = df.copy()
    df["_spk"] = df.source.astype(str) + "|" + df.speaker.astype(str)
    uniq = df._spk.unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq); tr = int(frac[0]*n); va = int(frac[1]*n)
    spk2 = {k:("train" if i<tr else ("val" if i<tr+va else "test"))
            for i,k in enumerate(uniq)}
    return df._spk.map(spk2)

def _load_reject(oov_cap:int) -> pd.DataFrame:
    """Synthesized rejection class (noise windows). Built if missing."""
    rp = os.path.join(ROOT, "data", "processed", "reject", "reject_manifest.csv")
    if not os.path.exists(rp):
        import subprocess, sys
        subprocess.run([sys.executable, os.path.join(HERE, "make_reject.py"),
                        "--n", str(oov_cap)], check=True)
    rj = pd.read_csv(rp)
    # map to manifest schema
    rj = rj.rename(columns={"intent_10":"intent_10"})
    for c in ["rel_path","intent_raw","speaker","variant"]:
        if c not in rj.columns:
            rj[c] = None
    return rj

def build(cap:int, oov_cap:int, seed:int):
    rng = np.random.default_rng(seed)
    df = load()
    # NOTE: manifest OOV (fleurs) rows have no abs_path -> unusable. We use the
    # synthesized rejection class instead (intent_10 == -1).
    pos = df[(df.intent_10 != -1)].copy()

    pos_c = _cap_positives(pos, cap, rng)
    oov_c = _load_reject(oov_cap).copy()

    # split positives: per-source (speaker-disjoint if ids exist, else row-wise)
    pos_parts = []
    for src, g in pos_c.groupby("source"):
        has_spk = g.speaker.notna().sum() > 0 and g.speaker.nunique() > 1
        if has_spk:
            g = g.copy(); g["_split"] = _spk_split(g, seed=seed+hash(src)%997)
        else:
            g = g.copy(); g["_split"] = _row_split(g, seed=seed+hash(src)%997)
        pos_parts.append(g)
    pos_c = pd.concat(pos_parts, ignore_index=True)

    # split OOV row-wise (fleurs has no speaker)
    oov_c = oov_c.copy(); oov_c["_split"] = _row_split(oov_c, seed=seed+12345)

    full = pd.concat([pos_c, oov_c], ignore_index=True)
    cols = ["source","rel_path","abs_path","intent_raw","intent_10","intent_name",
            "speaker","variant","condition","duration_s","weight","_split"]
    full = full[[c for c in cols if c in full.columns]]

    os.makedirs(OUTDIR, exist_ok=True)
    stats = {"cap":cap,"oov_cap":oov_cap,"seed":seed,"total":len(full),
             "per_split":full._split.value_counts().to_dict(),
             "oov_total":len(oov_c)}
    for split in ("train","val","test"):
        sub = full[full._split==split]
        sub.to_csv(os.path.join(OUTDIR,f"{split}.csv"), index=False)
        stats[f"{split}_per_intent"] = sub.intent_10.value_counts().sort_index().to_dict()
    with open(os.path.join(OUTDIR,"stats.json"),"w") as f:
        json.dump(stats,f,indent=2,default=str)
    return full, stats

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=2500)
    ap.add_argument("--oov-cap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    full, stats = build(a.cap, a.oov_cap, a.seed)
    print(json.dumps({k:v for k,v in stats.items() if not k.endswith("_per_intent")}, indent=2))
    for s in ("train","val","test"):
        print(f"{s:5s} per-intent:", stats[f"{s}_per_intent"])
