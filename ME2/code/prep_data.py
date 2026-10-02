#!/usr/bin/env python3
"""Extract audio from HF parquet, build manifest, compute features. Sequential, robust."""
from __future__ import annotations
import json, time, io
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / "data" / "hf"
AUDIO_DIR = ROOT / "data" / "audio"
FEAT_DIR = ROOT / "data" / "features"
SR = 16000
N_FRAMES = 100
N_MFCC = 40
N_MELS = 80

def loudest_crop(y, sr=SR, dur=1.0):
    win = int(sr * dur)
    if len(y) <= win:
        return np.pad(y, (0, win - len(y))) if len(y) < win else y
    hop = win // 4
    best_start, best_amp = 0, -1
    for start in range(0, len(y) - win + 1, hop):
        amp = np.max(np.abs(y[start:start+win]))
        if amp > best_amp:
            best_amp, best_start = amp, start
    return y[best_start:best_start+win]

def extract_and_compute(row, idx, split):
    """Extract audio for one row and compute features. Returns (mfcc, mel)."""
    a = row["audio"]
    # Audio is stored as WAV bytes in the parquet
    data, sr = sf.read(io.BytesIO(a["bytes"]))
    if data.ndim > 1:
        data = data.mean(axis=1)  # mono
    data = data.astype(np.float32)
    if sr != SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=SR)
    data = loudest_crop(data)
    
    # Save wav
    fname = f"{split}_{idx:06d}.wav"
    out = AUDIO_DIR / split / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), data, SR)
    
    # Features
    mfc = librosa.feature.mfcc(y=data, sr=SR, n_mfcc=N_MFCC, hop_length=512)
    mel = librosa.feature.melspectrogram(y=data, sr=SR, n_mels=N_MELS, hop_length=512)
    mel_db = librosa.power_to_db(mel)
    
    def pad1(x):
        # x is (n_feats, n_frames) from librosa → transpose to (n_frames, n_feats)
        xf = x.T
        if xf.shape[0] >= N_FRAMES:
            return xf[:N_FRAMES]
        return np.pad(xf, ((0, N_FRAMES - xf.shape[0]), (0, 0)))
    
    return pad1(mfc).astype(np.float32), pad1(mel).astype(np.float32)

def main():
    t0 = time.time()
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all splits
    splits = {
        "train": list(HF.glob("data/train-*.parquet")),
        "test": list(HF.glob("data/test-*.parquet")),
        "holdout": list(HF.glob("data/holdout-*.parquet")),
    }
    neg_train = list((HF / "synthetic_negatives").glob("train-*.parquet"))
    neg_test = list((HF / "synthetic_negatives").glob("test-*.parquet"))
    supp = list((HF / "supplemental_synth").glob("train-*.parquet"))
    
    all_rows = []
    for split_name, paths in splits.items():
        for p in paths:
            df = pd.read_parquet(p)
            df["_split"] = split_name
            all_rows.append(df)
    for p in neg_train:
        df = pd.read_parquet(p)
        df["_split"] = "train"
        df["command"] = "OUT_OF_SCOPE"
        all_rows.append(df)
    for p in neg_test:
        df = pd.read_parquet(p)
        df["_split"] = "test"
        df["command"] = "OUT_OF_SCOPE"
        all_rows.append(df)
    for p in supp:
        df = pd.read_parquet(p)
        if "voice_split" in df.columns:
            df = df[df["voice_split"] == "train"]
        df["_split"] = "train"
        all_rows.append(df)
    
    full = pd.concat(all_rows, ignore_index=True)
    print(f"Total clips: {len(full)}")
    print(full["_split"].value_counts().to_string())
    
    # Manifest
    cols = ["_split","command","variation","slot_value","speaker_id","is_synthetic","duration_s","out_of_scope"]
    avail = [c for c in cols if c in full.columns]
    full[avail].to_csv(ROOT / "data" / "manifest.csv", index=False)
    print(f"Manifest: {len(full)} rows")
    
    # Process per split
    for split_name in ["train", "test", "holdout"]:
        sub = full[full["_split"] == split_name].reset_index(drop=True)
        print(f"\nProcessing {split_name}: {len(sub)} clips")
        
        mfcc_list, mel_list = [], []
        for i in range(len(sub)):
            row = sub.iloc[i]
            try:
                m, mel = extract_and_compute(row, i, split_name)
                mfcc_list.append(m)
                mel_list.append(mel)
            except Exception as e:
                print(f"  SKIP {i}: {e}")
                mfcc_list.append(np.zeros((N_FRAMES, N_MFCC), dtype=np.float32))
                mel_list.append(np.zeros((N_FRAMES, N_MELS), dtype=np.float32))
            if (i+1) % 2000 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{len(sub)} ({elapsed:.0f}s)")
        
        mfcc_arr = np.stack(mfcc_list)
        mel_arr = np.stack(mel_list)
        out = FEAT_DIR / f"{split_name}.npz"
        np.savez_compressed(str(out), mfcc=mfcc_arr, mel=mel_arr)
        print(f"  Saved {out.name}: mfcc={mfcc_arr.shape}, mel={mel_arr.shape}")
    
    print(f"\nTotal time: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
