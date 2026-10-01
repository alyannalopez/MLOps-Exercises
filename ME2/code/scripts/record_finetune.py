#!/usr/bin/env python3
"""Record YOUR OWN voice for fine-tuning the VCM CRNN.

This is the data-collection half of the accuracy fix. The model was trained
on 141k TTS clips from other speakers in a clean booth; your voice is a
different speaker + room + mic, so it needs to hear YOU.

How it works
------------
1. It shows the canonical phrase for each of the 10 intents (from
   scripts/label_map.yaml + the CANONICAL value groups).
2. For each phrase you press Enter to arm the mic, speak, press Enter again.
   It records until you stop, checks the RMS level, and saves a WAV.
3. Files land in  <root>/data/finetune/<INTENT>/<n>.wav  (16 kHz mono).
4. It writes a manifest.csv next to them so the fine-tuner can pick them up.

You do NOT need to say the exact script words perfectly — natural variation
("lights up", "can you play some music") is GOOD and is exactly what makes the
model robust to how you actually talk. Aim for ~5-10 clips per intent.

Run (on the machine with the mic — the Pi or your Mac):
    python record_finetune.py                      # all 10 intents
    python record_finetune.py --intents SET_TIMER  # just one intent
    python record_finetune.py --target 8           # ask for 8 clips/intent
    python record_finetune.py --min-sec 0.8        # reject clips shorter than 0.8s

Then fine-tune:
    python finetune.py --data ../data/finetune --base ../models_retrain/crnn_retrain_1790670993_state.pt
"""
from __future__ import annotations
import argparse, os, sys, csv, time, json
import numpy as np

SR = 16000
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# 10 intents in taxonomy order (index = class id, matches models.py N_CLASSES-1).
INTENTS = [
    "PLAY_MUSIC", "QUESTION_SEARCH", "LIGHTS_ON_OFF", "DIM_COLOR_LIGHTS",
    "SET_TIMER", "SET_ALARM", "THERMOSTAT", "MEDIA_CONTROL",
    "REMINDERS_LISTS", "CALLS_MESSAGING",
]
CLASS_OF = {name: i for i, name in enumerate(INTENTS)}  # 1-based intent -> 0-based class? see note
# NOTE: models.py uses class 0..9 = intents 1..10, class 10 = REJECT.
# So class id for intent N (1-based) is N-1.
def class_id(intent_name: str) -> int:
    return CLASS_OF[intent_name]  # already 0-based via enumerate order


# Example prompt per intent. The user can say ANY natural phrasing; these are
# just prompts so they know what to say.
PROMPTS = {
    "PLAY_MUSIC":       "say a play-music command, e.g. 'play music'",
    "QUESTION_SEARCH":  "ask a question, e.g. 'what's the weather'",
    "LIGHTS_ON_OFF":    "turn the lights on or off, e.g. 'turn on the lights'",
    "DIM_COLOR_LIGHTS": "set brightness/color, e.g. 'set the lights to fifty percent'",
    "SET_TIMER":        "set a timer, e.g. 'set a timer for five minutes'",
    "SET_ALARM":        "set an alarm, e.g. 'set an alarm for seven am'",
    "THERMOSTAT":       "set the temperature, e.g. 'set the thermostat to twenty two degrees'",
    "MEDIA_CONTROL":    "pause/stop/next/volume, e.g. 'pause'",
    "REMINDERS_LISTS":  "make a reminder, e.g. 'remind me to buy groceries'",
    "CALLS_MESSAGING":  "call or message someone, e.g. 'call mom'",
}


def _rms_db(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms <= 1e-9:
        return -120.0
    return 20.0 * np.log10(rms)


def _open_mic(device: str | None, sr: int = SR):
    """Return (read_fn, close_fn). Tries sounddevice, falls back to PyAudio."""
    try:
        import sounddevice as sd
        q = sd.RawInputStream(samplerate=sr, channels=1, dtype="int16",
                              device=device, blocksize=int(sr * 0.03))
        q.start()
        def read(n_blocks: int) -> np.ndarray:
            raw, _ = q.read(n_blocks)
            return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return read, q.stop
    except Exception:
        import pyaudio
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=sr,
                         input=True, frames_per_buffer=int(sr * 0.03),
                         input_device_index=_pa_index(pa, device))
        def read(n_blocks: int) -> np.ndarray:
            buf = b"".join(stream.read(int(sr * 0.03), exception_on_overflow=False)
                           for _ in range(n_blocks))
            return np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
        def close():
            stream.stop_stream(); stream.close(); pa.terminate()
        return read, close


def _pa_index(pa, device: str | None) -> int | None:
    if not device:
        return None
    if device.isdigit():
        return int(device)
    for i in range(pa.get_device_count()):
        if device.lower() in pa.get_device_info_by_index(i)["name"].lower():
            return i
    raise SystemExit(f"mic not found: {device}")


def _save_wav(path: str, x: np.ndarray):
    import soundfile as sf
    sf.write(path, x, SR)


def _existing(outdir: str, intent: str) -> int:
    d = os.path.join(outdir, intent)
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith(".wav")])


def record_one(read, outdir: str, intent: str, idx: int,
               min_sec: float, min_db: float, max_sec: float) -> bool:
    """Record a single clip. Returns True if accepted."""
    print(f"\n  ▶  {PROMPTS[intent]}")
    print("     press ENTER to start, then speak, then ENTER to stop.")
    input("     >> ")
    t0 = time.time()
    frames = []
    block_ms = 30
    try:
        while time.time() - t0 < max_sec:
            x = read(1)
            frames.append(x)
            # live level meter
            db = _rms_db(x)
            bar = "#" * max(0, int((db + 60) / 3))
            sys.stdout.write(f"\r     lvl {db:6.1f} dB |{bar:<30}| {time.time()-t0:4.1f}s   ")
            sys.stdout.flush()
            if time.time() - t0 >= 0.4 and db < -55:
                # short quiet tail after speech -> auto-stop
                pass
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\n")
    if not frames:
        print("     (no audio captured)")
        return False
    x = np.concatenate(frames)
    dur = len(x) / SR
    db = _rms_db(x)
    if dur < min_sec:
        print(f"     ✗ too short ({dur:.2f}s < {min_sec}s) — try again")
        return False
    if db < min_db:
        print(f"     ✗ too quiet ({db:.1f} dB < {min_db} dB) — get closer / try again")
        return False
    d = os.path.join(outdir, intent)
    os.makedirs(d, exist_ok=True)
    _save_wav(os.path.join(d, f"{idx}.wav"), x)
    print(f"     ✓ saved {os.path.join(intent, idx)}.wav  ({dur:.2f}s, {db:.1f} dB)")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "data", "finetune"))
    ap.add_argument("--intents", default=None,
                    help="comma-separated intents to record (default: all 10)")
    ap.add_argument("--target", type=int, default=6,
                    help="clips to collect per intent (default: 6)")
    ap.add_argument("--min-sec", type=float, default=0.5,
                    help="reject clips shorter than this (default 0.5s)")
    ap.add_argument("--min-db", type=float, default=-45.0,
                    help="reject clips quieter than this dB (default -45)")
    ap.add_argument("--max-sec", type=float, default=6.0,
                    help="hard cap per clip (default 6s)")
    ap.add_argument("--device", default=None, help="input device (name or index)")
    ap.add_argument("--resume", action="store_true",
                    help="keep existing clips and only record what's missing "
                         "(default already does this; forces it even if target met)")
    a = ap.parse_args()

    outdir = os.path.abspath(a.outdir)
    intents = ([s.strip() for s in a.intents.split(",") if s.strip()]
               if a.intents else INTENTS)
    bad = [i for i in intents if i not in INTENTS]
    if bad:
        raise SystemExit(f"unknown intents {bad}; valid: {INTENTS}")

    print("=" * 64)
    print("VCM fine-tune recorder — say YOUR commands in YOUR voice")
    print("=" * 64)
    print(f"output dir : {outdir}")
    print(f"intents    : {len(intents)}   target/intent: {a.target}")
    print("(natural phrasing is fine — you do NOT need the exact script words)")

    read, close = _open_mic(a.device)
    manifest_rows = []
    try:
        for intent in intents:
            have = _existing(outdir, intent)
            print(f"\n{'='*64}\n{intent}  (already have {have}, target {a.target})\n{'='*64}")
            need = max(0, a.target - have)
            if have >= a.target:
                print("  ✓ target already met — skipping (use --intents to force)")
            for k in range(need):
                idx = have + k + 1
                ok = False
                attempts = 0
                while not ok and attempts < 5:
                    ok = record_one(read, outdir, intent, idx,
                                    a.min_sec, a.min_db, a.max_sec)
                    attempts += 1
                    if ok:
                        break
                if ok:
                    manifest_rows.append({
                        "path": os.path.join(outdir, intent, f"{idx}.wav"),
                        "intent": intent, "class": class_id(intent),
                    })
    finally:
        close()

    # write/merge manifest
    mpath = os.path.join(outdir, "manifest.csv")
    if os.path.exists(mpath):
        with open(mpath) as f:
            existing = list(csv.DictReader(f))
    else:
        existing = []
    seen = {r["path"] for r in existing}
    for r in manifest_rows:
        if r["path"] not in seen:
            existing.append(r)
    with open(mpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "intent", "class"])
        w.writeheader()
        w.writerows(existing)

    # summary
    print("\n" + "=" * 64)
    print("Recording complete. Counts per intent:")
    for intent in INTENTS:
        n = _existing(outdir, intent)
        mark = "✓" if n >= 3 else ("~" if n > 0 else "✗")
        print(f"  {mark} {intent:18s} {n:3d}")
    print("=" * 64)
    print(f"manifest: {mpath}")
    print("\nNext step — fine-tune the CRNN on these recordings:")
    print(f"  python {os.path.join(HERE,'finetune.py')} --data {outdir} \\")
    print(f"        --base {os.path.join(ROOT,'models_retrain','crnn_retrain_1790670993_state.pt')}")


if __name__ == "__main__":
    main()
