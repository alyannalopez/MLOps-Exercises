#!/usr/bin/env python3
"""Kokoro TTS worker — runs INSIDE the isolated 3.12 venv (.venv-kokoro).

Invoked as a subprocess by KokoroTTS (src/voiceduplex/tts/kokoro.py). One
process per synthesis keeps the main 3.14 pipeline free of kokoro-onnx (which
caps Python <3.14).

Protocol (JSON over stdio):
  stdin  : {"text": "...", "voice": "af_heart", "speed": 1.0,
            "model": "...", "voices": "...", "out": "/abs/path.wav"}
  stdout : {"ok": true, "sample_rate": 24000, "duration_audio": 3.04,
            "duration_compute": 1.21}
  stderr : human-readable logs / tracebacks
  exit 0 on success, non-zero on failure.

Model load cost is paid per call (a fresh process each time). For the demo the
replies are short and the load is ~1s, so this is acceptable; a persistent
daemon is a later optimization if per-call latency becomes a bottleneck.
"""

from __future__ import annotations

import json
import sys
import time


def main() -> int:
    req = json.loads(sys.stdin.read())
    text = req["text"]
    voice = req.get("voice", "af_heart")
    speed = float(req.get("speed", 1.0))
    model = req["model"]
    voices = req["voices"]
    out = req["out"]

    import numpy as np
    import soundfile as sf
    import kokoro_onnx

    t0 = time.perf_counter()
    k = kokoro_onnx.Kokoro(model, voices)
    audio, sr = k.create(text, voice=voice, speed=speed)
    audio = np.asarray(audio, dtype=np.float32)
    # Kokoro's float output is NOT centered on 0 — it carries a positive DC
    # offset (measured +0.019..+0.04 of full scale, i.e. +600..+1300 int16
    # counts). Scaling straight to int16 bakes that offset into every sample,
    # which the speaker renders as a constant low static/hum. Remove the DC
    # (subtract the mean) before converting, then normalize the peak to a safe
    # level so the voice is clean and not clipped.
    audio = audio - float(audio.mean())
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = audio * (0.98 / peak)
    # Write 16-bit PCM (not float32). The parent (kokoro.py) reads the file
    # back with np.fromfile(dtype=np.int16); a float32 WAV reinterpreted as
    # int16 is garbage -> loud static. int16 is also what the pipeline plays.
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    sf.write(out, pcm, sr, subtype="PCM_16")
    wall = time.perf_counter() - t0

    result = {
        "ok": True,
        "sample_rate": int(sr),
        "duration_audio": float(len(audio) / sr),
        "duration_compute": float(wall),
    }
    sys.stdout.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - report any failure to parent
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stdout.write(json.dumps({"ok": False, "error": repr(exc)}) + "\n")
        raise SystemExit(1)
