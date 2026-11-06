#!/usr/bin/env python3
"""Step 6 success check — run the FULL-DUPLEX voice loop (barge-in).

Usage:
  PYTHONPATH=src python3 scripts/run_full_duplex.py
  PYTHONPATH=src python3 scripts/run_full_duplex.py --tts piper
  PYTHONPATH=src python3 scripts/run_full_duplex.py --no-barge-in   # behave like half-duplex

What to do:
  1. Run it. Wait for the banner.
  2. NORMAL TURN: say "what time is it", wait for the reply to finish.
  3. BARGE-IN: say "what's 12 times 8" and, WHILE it is answering, start a NEW
     sentence ("stop, what date is it today?"). It should CUT OFF mid-word and
     answer the new question.
  4. BACKCHANNEL: ask it to say something long ("tell me a joke"), and while it
     talks say "yeah" or "mm-hm". It should acknowledge and KEEP talking (not
     stop).
  5. Quit: Ctrl-C, or type q + Enter.

Afterwards, inspect the timeline:
  cat data/timeline.jsonl | python3 -m json.tool --compact
  Barge-in turns carry O0_overlap_onset / O1_overlap_classified / O2_agent_reaction.
  (O0->O2 is the barge-in reaction latency Step 9 scores.)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from voiceduplex.asr import make_asr            # noqa: E402
from voiceduplex.audioio import make_audio_io   # noqa: E402
from voiceduplex.loop import FullDuplexLoop, TimelineLog  # noqa: E402
from voiceduplex.delegate import build_backend  # noqa: E402
from voiceduplex.tts import make_tts            # noqa: E402
from voiceduplex.vad import make_vad            # noqa: E402


def load_cfg() -> dict:
    p = ROOT / "config" / "config.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="ME3 full-duplex loop (barge-in)")
    ap.add_argument("--vad", choices=["silero", "webrtc"], default=None)
    ap.add_argument("--tts", choices=["kokoro", "piper", "fallback"], default=None)
    ap.add_argument("--no-barge-in", action="store_true",
                    help="disable barge-in (behaves like the half-duplex loop)")
    ap.add_argument("--log", default=str(ROOT / "data" / "timeline.jsonl"))
    args = ap.parse_args()

    cfg = load_cfg()
    a, v, asrcfg, tcfg = cfg["audio"], cfg["vad"], cfg["asr"], cfg["tts"]
    ag = cfg.get("agent", {})
    dx = cfg.get("duplex", {})

    print("Loading models...", flush=True)
    t0 = time.time()

    audio = make_audio_io(
        a.get("backend"),
        input_device=a.get("input_device"),
        output_device=a.get("output_device"),
        sample_rate=int(a.get("sample_rate", 16000)),
        block=int(a.get("block", 512)),
        playback_rate=int(a.get("playback_rate", 48000)),
    )

    vad = make_vad(
        args.vad or v.get("engine", "silero"),
        threshold=float(v.get("threshold", 0.5)),
        min_speech_frames=int(v.get("min_speech_frames", 4)),
        min_silence_frames=int(v.get("min_silence_frames", 12)),
    )

    asr = make_asr(
        asrcfg.get("backend", "whispercpp"),
        model_path=ROOT / asrcfg.get("model", "models/ggml-base.en.bin"),
        binary=asrcfg.get("binary", "whisper-cli"),
        language=asrcfg.get("language", "en"),
        threads=int(asrcfg.get("threads", 0)),
        beam_size=int(asrcfg.get("beam_size", 5)),
    )

    tts = _make_tts(args.tts or tcfg.get("backend", "piper"), tcfg, ROOT)

    print(f"  VAD={vad.__class__.__name__}  ASR={asr.__class__.__name__}  "
          f"TTS={tts.backend_name}   ({time.time()-t0:.1f}s)", flush=True)

    print("Warming up...", flush=True)
    asr.warm()
    tts.warm()
    print("Ready.\n", flush=True)

    audio.open()
    log = TimelineLog(args.log)

    barge_in = bool(dx.get("barge_in", True)) and not args.no_barge_in

    # Step 8 — delegation backend (OnIt by default; see config `delegate:`).
    delegate = build_backend(cfg)
    dcfg = cfg.get("delegate") or {}
    if delegate.is_available():
        print(f"  delegate: {delegate.name}  (on)", flush=True)
    else:
        print(f"  delegate: {delegate.name}  (MISSING runner — will stub)", flush=True)

    loop = FullDuplexLoop(
        audio, vad, asr, tts, log,
        max_utterance_sec=float(ag.get("max_utterance_sec", 15.0)),
        barge_in=barge_in,
        barge_min_energy=float(dx.get("barge_min_energy", 300.0)),
        backchannel_during_speech=bool(dx.get("backchannel_during_speech", True)),
        play_chunk_sec=float(dx.get("play_chunk_sec", 0.15)),
        delegate=delegate,
        delegate_wait_sec=float(dcfg.get("wait_sec", 120.0)),
    )

    print(f"  barge-in: {'ON' if barge_in else 'OFF'}\n", flush=True)
    _mic_sanity_check(audio, vad, a)
    try:
        loop.run()
    finally:
        audio.close()
        log.close()
        print(f"\nTimeline written to: {args.log}")
    return 0


def _mic_sanity_check(audio, vad, a: dict, duration: float = 5.0) -> None:
    """Sample the live mic + VAD for a few seconds and warn if it looks dead.

    Catches the "wrong mic" failure mode: a mic that delivers loud non-speech
    noise (Silero score ~0) makes the loop sit silently at the banner with no
    diagnostic. We sample the SAME stream the loop will use, so this reflects
    reality. Non-fatal — we always continue (the user may just not speak in
    the window).
    """
    import numpy as np
    import sounddevice as sd

    # Resolve the actual device name for the message.
    dev = getattr(audio, "input_device", None)
    dev_name = "system default"
    try:
        if dev is not None:
            dev_name = sd.query_devices(dev)["name"]
    except Exception:
        pass

    print(f"  mic: {dev_name}  — listening {duration:.0f}s, SPEAK NOW "
          f"(say 'hello')…", flush=True)
    t0 = time.time()
    voiced = 0
    n_frames = 0
    amp_max = 0
    score_max = 0.0
    saw_start = False
    while time.time() - t0 < duration:
        try:
            frame = audio.read()
        except Exception:
            break
        if frame is None:
            continue
        pcm = frame.pcm
        n_frames += 1
        amp = float(np.max(np.abs(pcm))) if len(pcm) else 0.0
        amp_max = max(amp_max, amp)
        try:
            s = float(vad._score(pcm))
        except Exception:
            s = 0.0
        score_max = max(score_max, s)
        if s >= vad.threshold:
            voiced += 1
        try:
            _, event = vad.process(pcm)  # pass the ndarray, like the loop does
            if event == "speech_start":
                saw_start = True
        except Exception:
            pass

    ok = saw_start or voiced >= vad.min_speech_frames
    if ok:
        print(f"  mic OK — detected speech (score_max {score_max:.2f}, "
              f"{voiced} voiced frames). Ready.\n", flush=True)
    else:
        print("\n  ⚠️  MIC WARNING: no speech detected in the last "
              f"{duration:.0f}s (score_max {score_max:.2f}, "
              f"amp_max {amp_max:.0f}).", flush=True)
        print("     The loop will still run, but it may not hear you.", flush=True)
        print("     Fix: pick a working mic in the macOS menu bar (Input), or", flush=True)
        print(f'     set input_device in config/config.yaml. Current: "{a.get("input_device")}"', flush=True)
        print("     Continuing in 3s… (Ctrl-C to abort)", flush=True)
        try:
            time.sleep(3.0)
        except KeyboardInterrupt:
            raise
        print("", flush=True)


def _make_tts(backend: str, tcfg: dict, ROOT: Path):
    piper_kw = dict(
        model=str(ROOT / tcfg.get("piper_model", "data/voices/en_US-lessac-medium.onnx")),
        speed=float(tcfg.get("speed", 1.0)),
    )
    kokoro_kw = dict(
        python=str(ROOT / tcfg.get("kokoro_python", ".venv-kokoro/bin/python")),
        model=str(ROOT / tcfg.get("kokoro_model", "models/kokoro-v1.0.onnx")),
        voices=str(ROOT / tcfg.get("kokoro_voices", "models/voices-v1.0.bin")),
        voice=tcfg.get("voice", "af_heart"),
        speed=float(tcfg.get("speed", 1.0)),
    )
    if backend == "piper":
        return make_tts("piper", **piper_kw)
    if backend == "kokoro":
        return make_tts("kokoro", **kokoro_kw)
    # fallback: primary (default piper) with automatic backup (default kokoro)
    return make_tts(
        "fallback",
        primary=tcfg.get("primary", "piper"),
        backup=tcfg.get("backup", "kokoro"),
        primary_kwargs=piper_kw if tcfg.get("primary", "piper") == "piper" else kokoro_kw,
        backup_kwargs=kokoro_kw if tcfg.get("backup", "kokoro") == "kokoro" else piper_kw,
    )


if __name__ == "__main__":
    raise SystemExit(main())
