#!/usr/bin/env python3
"""
Unified ETL for the VCM project.

Walks every downloaded source, maps each clip to the 10-intent taxonomy via
label_map.yaml, and emits ONE flat manifest (CSV) that the training pipeline
consumes. Option B is the primary source; the rest are auxiliary.

Output:
  vcm/data/processed/manifest.csv   — one row per clip
  vcm/data/processed/REPORT.md      — human-readable summary + balance table
"""
import os, re, csv, glob, sys, json
import yaml
import pyarrow.parquet as pq
import wave, struct

ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW    = os.path.join(ROOT, "data", "raw")
OUT    = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(ROOT, "scripts", "label_map.yaml")) as f:
    LM = yaml.safe_load(f)

TAX = {int(k): v for k, v in LM["taxonomy"].items()}   # {1:"PLAY_MUSIC", ...}
INV = TAX                                              # int -> NAME
W   = LM["weights"]
# coerce optionB/snips/slurp keys to int
for _k in ("optionB", "snips", "slurp"):
    LM[_k] = {int(k) if str(k).isdigit() else k: v for k, v in LM[_k].items()}

rows = []

def add(source, rel, abs_path, intent_raw, intent_10, speaker, variant,
        condition, text, dur):
    # intent_10 may be int (1-10) or the string "OOV"
    if isinstance(intent_10, int):
        label = INV[intent_10]
    else:
        label = intent_10
    rows.append({
        "source": source,
        "rel_path": rel,
        "abs_path": abs_path,
        "intent_raw": intent_raw,
        "intent_10": intent_10 if isinstance(intent_10, int) else -1,
        "intent_name": label,
        "speaker": speaker,
        "variant": variant,
        "condition": condition,
        "text": text or "",
        "duration_s": round(dur, 3) if dur else "",
        "weight": W.get(source, 0.1),
    })

def wav_dur(p):
    try:
        with wave.open(p, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0

# ---------------------------------------------------------------------------
# 1) OPTION B  — PRIMARY
#    <INTENT>/<INTENT>_s<spk>_v<var>_<clean|noisy>.wav
# ---------------------------------------------------------------------------
ob_root = os.path.join(RAW, "optionB", "MEX2", "OptionB")
ob_map  = LM["optionB"]
ob_re   = re.compile(r"^(?P<int>\w+)_s(?P<spk>\d+)_v(?P<var>\d+)_(?P<cond>clean|noisy)\.wav$")
n_ob = 0
for folder in sorted(os.listdir(ob_root)):
    fdir = os.path.join(ob_root, folder)
    if not os.path.isdir(fdir):
        continue
    i10 = ob_map.get(folder)
    if i10 is None:
        print(f"[WARN] OptionB folder not in label map: {folder}")
        continue
    for fn in sorted(os.listdir(fdir)):
        m = ob_re.match(fn)
        if not m:
            continue
        ap = os.path.join(fdir, fn)
        add("optionB", os.path.relpath(ap, RAW), ap, folder, i10,
            m["spk"], m["var"], m["cond"], "", wav_dur(ap))
        n_ob += 1
print(f"OptionB: {n_ob} clips")

# ---------------------------------------------------------------------------
# 2) SNIPS  — parquet, intent 0-5
# ---------------------------------------------------------------------------
snips_map = LM["snips"]
n_snips = 0
for split, pat in [("train", "train*"), ("val", "validation*"), ("test", "test*")]:
    for p in sorted(glob.glob(os.path.join(RAW, "snips", "data", f"{pat}.parquet"))):
        df = pq.ParquetFile(p).read().to_pandas()
        for _, r in df.iterrows():
            i10 = snips_map.get(int(r["intent"]))
            if i10 is None:
                continue
            add("snips", f"snips/{split}/{r['uttid']}", "", int(r["intent"]), i10,
                str(r.get("speaker", "")), "", "clean", str(r["text"]), 0.0)
            n_snips += 1
print(f"SNIPS: {n_snips} clips")

# ---------------------------------------------------------------------------
# 3) SLURP  — parquet, intent 0-120 (subset mapped)
# ---------------------------------------------------------------------------
slurp_map = LM["slurp"]
n_slurp = 0
for p in sorted(glob.glob(os.path.join(RAW, "slurp", "data", "*.parquet"))):
    df = pq.ParquetFile(p).read().to_pandas()
    for _, r in df.iterrows():
        iid = int(r["intent"])
        i10 = slurp_map.get(iid)
        if i10 is None:
            continue
        add("slurp", f"slurp/{r['slurp_id']}", "", iid, i10, "", "", "clean",
            str(r["sentence"]), 0.0)
        n_slurp += 1
print(f"SLURP: {n_slurp} clips (mapped subset)")

# ---------------------------------------------------------------------------
# 4) SPEECH COMMANDS v2  — extracted dir, folder == command
# ---------------------------------------------------------------------------
sc_map = LM["speechcommands"]
sc_root = os.path.join(RAW, "extracted")   # speech_commands extracted flat
n_sc = 0
if os.path.isdir(sc_root):
    for cmd in sorted(os.listdir(sc_root)):
        cdir = os.path.join(sc_root, cmd)
        if not os.path.isdir(cdir):
            continue
        i10 = sc_map.get(cmd)
        if i10 is None:
            continue
        for fn in sorted(os.listdir(cdir)):
            if not fn.endswith(".wav"):
                continue
            ap = os.path.join(cdir, fn)
            add("speechcommands", os.path.relpath(ap, RAW), ap, cmd, i10,
                "", "", "clean", "", wav_dur(ap))
            n_sc += 1
print(f"SpeechCommands: {n_sc} clips (mapped subset)")

# ---------------------------------------------------------------------------
# 5) FLEURS fil_ph  — OOV (acoustic diversity only)
# ---------------------------------------------------------------------------
n_fl = 0
for p in sorted(glob.glob(os.path.join(RAW, "fleurs", "parquet-data", "fil_ph", "*.parquet"))):
    df = pq.ParquetFile(p).read().to_pandas()
    for _, r in df.iterrows():
        add("fleurs", f"fleurs/{r['id']}", "", "", "OOV", "", "", "clean",
            str(r.get("transcription", "")), 0.0)
        n_fl += 1
print(f"FLEURS fil_ph: {n_fl} clips (OOV)")

# ---------------------------------------------------------------------------
# 6) LIBRISPEECH  — OOV (acoustic pretraining only); count from transcripts
# ---------------------------------------------------------------------------
n_lib = 0
lib_root = os.path.join(RAW, "extracted", "librispeech")
for split in ["train-clean-100", "dev-clean", "test-clean"]:
    sp = os.path.join(lib_root, split)
    if not os.path.isdir(sp):
        continue
    for tr in glob.glob(os.path.join(sp, "**", "*.trans"), recursive=True):
        with open(tr) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                uid = parts[0]
                txt = parts[1] if len(parts) > 1 else ""
                add("librispeech", f"librispeech/{split}/{uid}", "", "", "OOV",
                    "", "", "clean", txt, 0.0)
                n_lib += 1
print(f"LibriSpeech: {n_lib} clips (OOV)")

# ---------------------------------------------------------------------------
# WRITE MANIFEST
# ---------------------------------------------------------------------------
cols = ["source", "rel_path", "abs_path", "intent_raw", "intent_10",
        "intent_name", "speaker", "variant", "condition", "text",
        "duration_s", "weight"]
mpath = os.path.join(OUT, "manifest.csv")
with open(mpath, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
print(f"\nWrote {len(rows)} rows -> {mpath}")

# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------
from collections import Counter, defaultdict
by_src_intent = defaultdict(Counter)
for r in rows:
    by_src_intent[r["source"]][r["intent_name"]] += 1

intent_tot = Counter(r["intent_name"] for r in rows)
src_tot = Counter(r["source"] for r in rows)

lines = ["# VCM Unified ETL Report", ""]
lines.append(f"**Total clips in manifest: {len(rows):,}**")
lines.append("")
lines.append("## Clips per source")
lines.append("")
lines.append("| Source | Total | Role | Weight |")
lines.append("|--------|-------|------|--------|")
role = {"optionB": "PRIMARY", "snips": "aux", "slurp": "aux",
        "speechcommands": "aux", "fleurs": "OOV/acoustic", "librispeech": "OOV/pretrain"}
for s in ["optionB", "snips", "slurp", "speechcommands", "fleurs", "librispeech"]:
    lines.append(f"| {s} | {src_tot.get(s,0):,} | {role[s]} | {W.get(s)} |")
lines.append("")
lines.append("## 10-intent balance (mapped clips only)")
lines.append("")
lines.append("| Intent | Name | optionB | snips | slurp | speechcmd | TOTAL |")
lines.append("|--------|------|---------|-------|-------|-----------|-------|")
for i in range(1, 11):
    nm = INV[i]
    a = by_src_intent["optionB"].get(nm, 0)
    b = by_src_intent["snips"].get(nm, 0)
    c = by_src_intent["slurp"].get(nm, 0)
    d = by_src_intent["speechcommands"].get(nm, 0)
    lines.append(f"| {i} | {nm} | {a:,} | {b:,} | {c:,} | {d:,} | {a+b+c+d:,} |")
oo = sum(1 for r in rows if r["intent_name"] == "OOV")
lines.append(f"| - | OOV | - | - | - | - | {oo:,} |")
lines.append("")
lines.append("## Option B detail (primary)")
lines.append("")
lines.append("| Raw intent | -> 10-intent | clips |")
lines.append("|------------|--------------|-------|")
ob_raw = Counter(r["intent_raw"] for r in rows if r["source"] == "optionB")
for raw, c in sorted(ob_raw.items()):
    i10 = ob_map.get(raw, "?")
    lines.append(f"| {raw} | {i10} ({INV.get(i10,'?')}) | {c:,} |")
lines.append("")
lines.append("_Generated by scripts/etl.py. Label map: scripts/label_map.yaml_")

with open(os.path.join(OUT, "REPORT.md"), "w") as f:
    f.write("\n".join(lines))
print(f"Wrote {os.path.join(OUT, 'REPORT.md')}")
