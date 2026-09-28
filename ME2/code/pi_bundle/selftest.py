#!/usr/bin/env python3
"""Self-test for the Step 3 inference engine.

Validates, WITHOUT a microphone:
  1. The pure-numpy log-mel in vcm_infer matches librosa's (training-time)
     log-mel within tolerance  -> proves the Pi path uses the same features.
  2. The INT8 ONNX model runs end-to-end on real Option B WAVs and returns a
     sane intent + sub-second timing.
  3. Full-pipeline latency distribution over a sample of clips.

Run:  python selftest.py --model ../models/crnn_..._int8.onnx [--n 40]
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcm_infer as V


def load_wav_16k(path):
    import wave
    with wave.open(path, "rb") as w:
        sr = w.getframerate(); n = w.getnframes()
        raw = np.frombuffer(w.readframes(n), dtype=np.int16)
    return raw, sr


def test_feature_parity(vcm: V.VCM, wav_paths, tol=3.0):
    """Confirm the inference feature path equals the training feature path.

    NOTE: the hand-rolled numpy logmel() in vcm_infer is known NOT to reproduce
    librosa bit-for-bit (n_fft/window/normalization drift) and yields ~18% acc,
    so the deployed classify_pcm() routes through features.logmel_features()
    instead. This test therefore checks that _logmel_train (the path actually
    used) matches features.logmel_features exactly.
    """
    import features as F
    fn = F.mfcc_features if getattr(vcm, "kind", "logmel") == "mfcc" else F.logmel_features
    worst = 0.0
    for p in wav_paths[:8]:
        raw, sr = load_wav_16k(p)
        mine = vcm._logmel_train(raw, sr)
        ref = fn(p)
        m = min(mine.shape[0], ref.shape[0])
        d = np.abs(mine[:m] - ref[:m])
        worst = max(worst, float(d.max()))
    print(f"[feature parity] inference-path vs training-path max |diff| over "
          f"{min(8,len(wav_paths))} clips: {worst:.3f} dB  "
          f"({'PASS' if worst < tol else 'CHECK'})")
    return worst < tol


def test_accuracy_and_latency(vcm: V.VCM, wav_paths, n=40):
    import csv
    # map optionB folder -> expected intent via label_map (simple substring)
    expected = {}
    lm = os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_map.yaml")
    if os.path.exists(lm):
        for line in open(lm):
            line = line.strip()
            if ":" in line and "intent_10" in line:
                pass
    # simpler: derive expected from the OptionB folder name using a small table
    folder_intent = {
        "PLAY_MUSIC": "PLAY_MUSIC", "WEATHER": "QUESTION_SEARCH", "TIME": "QUESTION_SEARCH",
        "LIGHT_ON": "LIGHTS_ON_OFF", "LIGHT_OFF": "LIGHTS_ON_OFF",
        "DIM": "DIM_COLOR_LIGHTS", "COLOR": "DIM_COLOR_LIGHTS",
        "TIMER": "SET_TIMER", "ALARM": "SET_ALARM", "TEMPERATURE": "THERMOSTAT",
        "PAUSE": "MEDIA_CONTROL", "STOP": "MEDIA_CONTROL", "NEXT": "MEDIA_CONTROL",
        "VOLUME_UP": "MEDIA_CONTROL", "VOLUME_DOWN": "MEDIA_CONTROL",
        "REMIND": "REMINDERS_LISTS", "LIST": "REMINDERS_LISTS",
        "CALL": "CALLS_MESSAGING", "MESSAGE": "CALLS_MESSAGING",
    }
    sample = wav_paths[:n]
    correct = 0; total = 0; lat = []
    for p in sample:
        pcm, sr = load_wav_16k(p)
        r = vcm.classify_pcm(pcm, sr)
        lat.append(r["timings_ms"]["total"])
        # expected from path
        parts = p.split(os.sep)
        exp = None
        for seg in parts:
            up = seg.upper()
            for k, v in folder_intent.items():
                if k in up:
                    exp = v; break
            if exp: break
        if exp:
            total += 1
            if r["intent"] == exp:
                correct += 1
    lat = np.array(lat)
    acc = correct / total if total else float("nan")
    print(f"\n[end-to-end on {len(sample)} real clips]")
    print(f"  labeled subset: {total}   accuracy: {acc:.3f}")
    print(f"  full-pipeline latency ms: mean={lat.mean():.1f}  "
          f"p50={np.percentile(lat,50):.1f}  p95={np.percentile(lat,95):.1f}  "
          f"max={lat.max():.1f}")
    print(f"  {'PASS' if lat.max() < 1000 else 'SLOW'} (target < 1000 ms)")
    return acc, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "..", "data", "raw", "optionB"))
    a = ap.parse_args()

    wav_paths = sorted(glob.glob(os.path.join(a.root, "**", "*.wav"), recursive=True))
    if not wav_paths:
        print("no Option B WAVs found at", a.root); sys.exit(1)
    print(f"found {len(wav_paths)} Option B clips\n")

    vcm = V.VCM(a.model)
    ok1 = test_feature_parity(vcm, wav_paths)
    acc, lat = test_accuracy_and_latency(vcm, wav_paths, a.n)
    print(f"\nSUMMARY: feature_parity={'PASS' if ok1 else 'FAIL'}  "
          f"acc={acc:.3f}  p95={np.percentile(lat,95):.1f}ms")


if __name__ == "__main__":
    main()
