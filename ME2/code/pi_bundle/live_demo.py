#!/usr/bin/env python3
"""VCM live demo (Step 3) — Raspberry Pi 4/5.

Streams the USB microphone, detects a spoken command with energy VAD,
classifies it with the INT8 ONNX model, prints the intent + timing, and
loops. This is the on-device real-time proof the exercise asks for.

Run on the Pi:
    pip install onnxruntime numpy
    python live_demo.py --model ../models/crnn_..._int8.onnx --device /dev/snd/pcm0.0

The same file runs on the Mac for a quick sanity check (uses the default mic).
"""
from __future__ import annotations
import argparse, time, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcm_infer import VCM, TARGET_SR, WINDOW_S, rms_db

FRAME_MS   = 30
MIN_SPEECH = 0.25      # s of voiced audio before we commit
MAX_CMD    = 3.0       # s cap per command
NOISE_DB   = -45.0     # adaptive floor start
TRIGGER_DB = 12.0      # dB above floor to count as speech


def open_stream(device: str | None, sr: int = TARGET_SR, block: int = 480):
    """Open a 16 kHz mono int16 stream. Returns (reader_fn, closer_fn)."""
    try:
        import sounddevice as sd
        q = []
        def cb(indata, frames, t, status):
            q.append(indata.copy())
        stream = sd.RawInputStream(samplerate=sr, blocksize=block, dtype="int16",
                                   channels=1, callback=cb,
                                   device=device)
        stream.start()
        def read(n_blocks=1):
            out = []
            while len(q) < n_blocks and len(out) < n_blocks:
                time.sleep(0.005)
            while q:
                out.append(q.pop(0))
            if not out:
                return np.zeros(block, dtype=np.int16)
            return np.concatenate(out)[:block * n_blocks]
        return read, lambda: stream.stop() or stream.close()
    except Exception as e:
        print(f"[warn] sounddevice unavailable ({e}); falling back to arecord")
        import subprocess
        proc = subprocess.Popen(
            ["arecord", "-f", "S16_LE", "-r", str(sr), "-c", "1",
             "-d", "0", "-q"] + (["-D", device] if device else []),
            stdout=subprocess.PIPE)
        def read(n_blocks=1):
            chunk = proc.stdout.read(block * 2 * n_blocks)
            if not chunk:
                return np.zeros(block, dtype=np.int16)
            return np.frombuffer(chunk, dtype=np.int16)
        return read, lambda: (proc.terminate(), proc.wait())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default=None, help="sounddevice index or ALSA name")
    ap.add_argument("--trigger-db", type=float, default=TRIGGER_DB)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    vcm = VCM(a.model)
    read, close = open_stream(a.device)
    block = 480  # 30 ms @16k
    print(f"Listening on {a.device or 'default'} mic. Say a command...\n")

    buf = np.array([], dtype=np.int16)
    voiced = 0.0
    floor = NOISE_DB
    t_start = None

    try:
        while True:
            blk = read(1)
            if len(blk) == 0:
                continue
            db = rms_db(blk)
            # slow-adapt noise floor on non-speech
            if db < floor + a.trigger_db * 0.5:
                floor = 0.95 * floor + 0.05 * db

            if len(buf) == 0:
                if db > floor + a.trigger_db:
                    buf = blk
                    voiced = FRAME_MS / 1000
                    t_start = time.perf_counter()
                    if not a.quiet:
                        print("\r>> speech detected, capturing...", end="", flush=True)
                continue

            buf = np.concatenate([buf, blk])
            voiced += FRAME_MS / 1000
            elapsed = time.perf_counter() - t_start
            # commit when we have >=1 s of audio OR hit the cap
            if len(buf) >= int(WINDOW_S * TARGET_SR) or elapsed > MAX_CMD:
                r = vcm.classify_pcm(buf)
                ms = r["timings_ms"]
                if r["intent"] == "REJECT":
                    line = f"  (rejected  conf={r['confidence']:.2f})"
                else:
                    line = (f"  [{r['intent']}]  conf={r['confidence']:.2f}  "
                            f"feat={ms['feature']:.1f}ms  infer={ms['infer']:.2f}ms  "
                            f"TOTAL={ms['total']:.1f}ms")
                print(line)
                buf = np.array([], dtype=np.int16)
                voiced = 0.0
                t_start = None
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        close()


if __name__ == "__main__":
    main()
