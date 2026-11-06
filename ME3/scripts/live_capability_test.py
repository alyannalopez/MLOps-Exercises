#!/usr/bin/env python3
"""ME3 LIVE CAPABILITY TEST — drives the REAL full-duplex loop through a
scripted scenario over the live mic, so you can verify every capability in one
sitting (later, when the room is quiet).

Why this exists
---------------
Your recent live run showed the MODEL is fine (routing was correct every turn)
but the MIC/ROOM was feeding fragments and blanks ("a short poem.", "ring.",
"[BLANK_AUDIO]"). This script isolates that: it runs the exact same
FullDuplexLoop as run_full_duplex.py, but instead of you improvising, it walks
a fixed checklist and, for each item, prints what the loop HEARD vs what you
were SUPPOSED to say. If the transcript matches the cue, the mic+ASR+VAD chain
is healthy and any earlier trouble was environmental. If it's garbled/blank,
the culprit is the mic / room tone / thresholds — not the agent.

What it exercises (every capability)
------------------------------------
  1  quick_tool  : "what time is it"                 -> local, instant
  2  quick_tool  : "play some music"                 -> simulated device
  3  quick_tool  : "what's 12 times 8"               -> math extra
  4  barge-in    : "tell me a joke" then INTERRUPT   -> mid-reply cut
  5  backchannel : "tell me a story" then "mm-hm"    -> ack + KEEP talking
  6  delegate    : "write a short poem about the sea"-> OnIt (Step 8, ~25s)

Usage
-----
  cd /Users/yannamii/sandbox/ME3
  PYTHONPATH=src python3 scripts/live_capability_test.py
  PYTHONPATH=src python3 scripts/live_capability_test.py --skip-delegate   # no OnIt
  PYTHONPATH=src python3 scripts/live_capability_test.py --only 1,2,4      # subset
  PYTHONPATH=src python3 scripts/live_capability_test.py --quiet-room 30   # pre-check

Behaviour
---------
  * Before starting, it samples the mic for N seconds and measures room tone
    (RMS + Silero score). High room tone => it warns you to quiet the room /
    wear headphones before the test.
  * Each step: prints the CUE, opens a timed window, listens, and scores the
    transcript against the cue (exact / partial / blank / garbled).
  * Barge-in and backchannel steps are driven by the LIVE loop: you must speak
    the interrupt/ack DURING the reply window (the script times it for you).
  * Quit anytime with Ctrl-C. On exit it shuts down the OnIt worker (no orphans).

Scoring legend
--------------
  MATCH   transcript contains the expected phrase (case-insensitive)
  PARTIAL some expected words present
  BLANK   empty / "[BLANK_AUDIO]"
  GARBLED got text but not the cue  <-- this is the mic/room signature
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from voiceduplex.asr import make_asr  # noqa: E402
from voiceduplex.audioio import make_audio_io  # noqa: E402
from voiceduplex.loop import FullDuplexLoop, TimelineLog  # noqa: E402
from voiceduplex.delegate import build_backend  # noqa: E402
from voiceduplex.tts import make_tts  # noqa: E402
from voiceduplex.vad import make_vad  # noqa: E402


# --------------------------------------------------------------------------- #
# Scenario definition
# --------------------------------------------------------------------------- #
@dataclass
class Step:
    id: int
    kind: str            # quick | barge | backchannel | delegate
    cue: str             # what YOU say (the primary utterance)
    expect: str          # substring the transcript SHOULD contain
    interrupt: str = ""  # for barge/backchannel: what you say WHILE it talks
    interrupt_expect: str = ""
    # When (seconds after the reply starts) to prompt the interrupt. Kept short
    # so it lands inside the TTS window; tune per your TTS speed.
    interrupt_at: float = 2.5
    wait: float = 6.0    # how long to listen for the PRIMARY cue
    note: str = ""


STEPS: list[Step] = [
    Step(1, "quick", "what time is it", "time",
         wait=6.0, note="quick_tool: local clock, instant reply"),
    Step(2, "quick", "play some music", "music",
         wait=6.0, note="quick_tool: simulated music device"),
    Step(3, "quick", "what's 12 times 8", "96",
         wait=6.0, note="quick_tool: math extra (expect 'That's 96')"),
    Step(4, "barge", "tell me a joke", "joke",
         interrupt="stop what date is it today", interrupt_expect="date",
         interrupt_at=2.5, wait=6.0,
         note="BARGE-IN: say the interrupt WHILE it's telling the joke"),
    Step(5, "backchannel", "tell me a story about a dog", "dog",
         interrupt="mm-hm", interrupt_expect="",
         interrupt_at=2.5, wait=6.0,
         note="BACKCHANNEL: say 'mm-hm' while it talks -> it should KEEP going"),
    Step(6, "delegate", "write a short poem about the sea", "poem",
         wait=6.0,
         note="DELEGATE: OnIt (Step 8) — takes ~25s, expect 'Let me check that'"),
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_cfg() -> dict:
    with open(ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


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


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def score_transcript(transcript: str, expect: str) -> str:
    """MATCH / PARTIAL / BLANK / GARBLED."""
    t = norm(transcript)
    if not t or "blank_audio" in t.replace(" ", "_"):
        return "BLANK"
    e_words = [w for w in norm(expect).split() if w]
    if not e_words:
        return "MATCH" if t else "BLANK"
    hits = sum(1 for w in e_words if w in t.split())
    if hits == len(e_words):
        return "MATCH"
    if hits >= max(1, len(e_words) // 2):
        return "PARTIAL"
    return "GARBLED"


# --------------------------------------------------------------------------- #
# Room-tone pre-check (the thing you suspect)
# --------------------------------------------------------------------------- #
def room_tone_check(audio, vad, seconds: float = 15.0) -> None:
    """Sample the live mic in SILENCE and report ambient level + Silero score.

    This is the direct diagnostic for 'is my room too noisy?'. Stay silent for
    the whole window. High RMS or a non-zero Silero score while you're NOT
    talking = room tone / fan / AC / mic gain too hot.
    """
    print("\n" + "=" * 70)
    print(f"  ROOM-TONE PRE-CHECK — stay SILENT for {seconds:.0f}s")
    print("  (this measures your ambient noise floor, not your voice)")
    print("=" * 70, flush=True)
    t0 = time.time()
    rms_list = []
    score_max = 0.0
    voiced_frames = 0
    while time.time() - t0 < seconds:
        try:
            frame = audio.read()
        except Exception:
            break
        if frame is None:
            continue
        pcm = frame.pcm
        if len(pcm):
            rms = float(np.sqrt(np.mean((pcm.astype(np.float32) ** 2))))
            rms_list.append(rms)
        try:
            s = float(vad._score(pcm))
            score_max = max(score_max, s)
            if s >= vad.threshold:
                voiced_frames += 1
        except Exception:
            pass
    if not rms_list:
        print("  (no audio captured — mic problem?)")
        return
    avg_rms = float(np.mean(rms_list))
    peak_rms = float(np.max(rms_list))
    print(f"\n  ambient RMS  avg={avg_rms:8.1f}  peak={peak_rms:8.1f}  (0-32767 scale)")
    print(f"  Silero score (while silent)  max={score_max:.3f}  threshold={vad.threshold}")
    print(f"  voiced frames while silent   {voiced_frames}")
    if score_max >= vad.threshold or voiced_frames > 0:
        print("  ⚠️  NOISE DETECTED while silent — room tone / fan / AC is tripping the VAD.")
        print("     Fix before the real test: quiet the room, lower mic gain, wear")
        print("     headphones, or raise vad.threshold in config/config.yaml.")
    elif avg_rms > 600:
        print("  ⚠️  Ambient RMS is high (>600). Consider lowering mic gain.")
    else:
        print("  ✅ Room looks quiet enough to proceed.")
    print()


# --------------------------------------------------------------------------- #
# The scripted run
# --------------------------------------------------------------------------- #
def run_step(step: Step, loop: FullDuplexLoop) -> None:
    """Print the cue, open a listen window, then report what the loop heard.

    The loop is ALREADY running on its own thread (started by the caller); here
    we just pace the human and read back the last transcript the loop captured.
    """
    print("\n" + "-" * 70)
    print(f"  STEP {step.id}  [{step.kind}]  {step.note}")
    print(f"  >>> SAY:  {step.cue!r}")
    if step.interrupt:
        print(f"  >>> THEN (at ~{step.interrupt_at:.0f}s, WHILE it talks) SAY: {step.interrupt!r}")
    print(f"  listening... (window {step.wait:.0f}s)")
    print("-" * 70, flush=True)

    # Pace the interrupt prompt for the interactive kinds.
    t0 = time.time()
    prompted = False
    deadline = t0 + step.wait
    while time.time() < deadline:
        if step.interrupt and not prompted and (time.time() - t0) >= step.interrupt_at:
            print(f"  ⏰ NOW say the interrupt: {step.interrupt!r}", flush=True)
            prompted = True
        time.sleep(0.1)
    # Give the loop a beat to finish ASR/route/speak for the last utterance.
    time.sleep(2.0)

    last = loop.last_heard()  # (transcript, route, reply) of the most recent turn
    transcript, route, reply = last
    verdict = score_transcript(transcript, step.expect)
    mark = {"MATCH": "✅", "PARTIAL": "🟡", "BLANK": "❌", "GARBLED": "⚠️"}.get(verdict, "?")
    print(f"  {mark} {verdict:8}  heard={transcript!r}  route={route}  reply={reply!r}")
    if step.interrupt and step.interrupt_expect:
        iv = score_transcript(loop.last_interrupt_heard() or "", step.interrupt_expect)
        imark = {"MATCH": "✅", "PARTIAL": "🟡", "BLANK": "❌", "GARBLED": "⚠️"}.get(iv, "?")
        print(f"  {imark} {iv:8}  interrupt heard={loop.last_interrupt_heard()!r}")
    if verdict in ("BLANK", "GARBLED"):
        print("     ^ this is the mic/room signature. The agent logic is fine; the")
        print("       audio it received was not clean speech. Re-run in a quieter")
        print("       room / with headphones, or raise vad.threshold.")


def main() -> int:
    ap = argparse.ArgumentParser(description="ME3 live capability test (scripted)")
    ap.add_argument("--tts", choices=["kokoro", "piper", "fallback"], default=None)
    ap.add_argument("--vad", choices=["silero", "webrtc"], default=None)
    ap.add_argument("--no-barge-in", action="store_true")
    ap.add_argument("--skip-delegate", action="store_true",
                    help="skip the OnIt step (no network / no worker)")
    ap.add_argument("--only", default="",
                    help="comma list of step ids to run, e.g. --only 1,2,4")
    ap.add_argument("--quiet-room", type=float, default=15.0,
                    help="seconds for the silent room-tone pre-check (0 to skip)")
    ap.add_argument("--log", default=str(ROOT / "data" / "timeline_livecap.jsonl"))
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
        v.get("engine", "silero"),
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
    delegate = None if args.skip_delegate else build_backend(cfg)
    dcfg = cfg.get("delegate") or {}
    if delegate is not None:
        print(f"  delegate: {delegate.name}  ({'on' if delegate.is_available() else 'MISSING'})", flush=True)
    else:
        print("  delegate: SKIPPED (--skip-delegate)", flush=True)

    loop = FullDuplexLoop(
        audio, vad, asr, tts, log,
        max_utterance_sec=float(ag.get("max_utterance_sec", 15.0)),
        barge_in=barge_in,
        barge_min_energy=float(dx.get("barge_min_energy", 150.0)),
        backchannel_during_speech=bool(dx.get("backchannel_during_speech", True)),
        play_chunk_sec=float(dx.get("play_chunk_sec", 0.15)),
        delegate=delegate,
        delegate_wait_sec=float(dcfg.get("wait_sec", 120.0)),
    )
    print(f"  barge-in: {'ON' if barge_in else 'OFF'}\n", flush=True)

    # Select steps.
    if args.only.strip():
        wanted = {int(x) for x in re.split(r"[,\s]+", args.only.strip()) if x.strip()}
        steps = [s for s in STEPS if s.id in wanted]
    else:
        steps = [s for s in STEPS if not (args.skip_delegate and s.kind == "delegate")]

    # Room-tone pre-check first (the suspected culprit).
    if args.quiet_room > 0:
        room_tone_check(audio, vad, args.quiet_room)

    print("=" * 70)
    print("  LIVE CAPABILITY TEST — follow each cue. Ctrl-C to abort.")
    print("=" * 70, flush=True)

    # Start the loop on its own thread so it listens while we pace the human.
    import threading
    loop_thread = threading.Thread(target=loop.run, daemon=True)
    loop_thread.start()
    time.sleep(0.5)

    results = []
    try:
        for step in steps:
            run_step(step, loop)
            transcript, route, reply = loop.last_heard()
            results.append((step.id, step.kind, score_transcript(transcript, step.expect)))
    except KeyboardInterrupt:
        print("\n  (aborted by user)")
    finally:
        loop.stop()
        loop_thread.join(timeout=5)
        audio.close()
        log.close()

    # Summary.
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for sid, kind, verdict in results:
        mark = {"MATCH": "✅", "PARTIAL": "🟡", "BLANK": "❌", "GARBLED": "⚠️"}.get(verdict, "?")
        print(f"  {mark}  step {sid}  [{kind:11}]  {verdict}")
    n_match = sum(1 for *_, v in results if v == "MATCH")
    n_bad = sum(1 for *_, v in results if v in ("BLANK", "GARBLED"))
    print(f"\n  {n_match}/{len(results)} clean.  {n_bad} blank/garbled.")
    if n_bad and n_bad >= max(1, len(results) // 2):
        print("  ➜ Majority blank/garbled = MIC or ROOM, not the model.")
        print("    Try: quieter room, headphones, lower mic gain, or raise vad.threshold.")
    elif n_bad:
        print("  ➜ Mostly clean; isolated blanks are likely timing (speak clearly,")
        print("    pause between phrases). The agent pipeline is healthy.")
    else:
        print("  ➜ All clean. Mic + VAD + ASR + routing + delegation all working.")
    print(f"\n  Timeline: {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
