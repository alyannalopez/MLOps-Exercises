#!/usr/bin/env python3
"""ME3 OFFLINE FULL CHECK — verifies the ENTIRE model pipeline with NO mic,
NO speaker, and NO network. This is the deterministic regression gate: if this
passes, the agent's brain is healthy and any live-conversation trouble is the
mic / room / thresholds (use scripts/live_capability_test.py to confirm).

Pipeline covered (all offline)
------------------------------
  1  VAD     : feed data/audio/test_speech.wav + silence_3s.wav through Silero
               -> speech must be detected, silence must NOT be.
  2  ASR     : transcribe test_speech.wav -> non-empty, sane text; RTF reported.
  3  TTS     : synthesize a sentence -> valid int16 audio, RTF reported.
  4  ROUTER  : run the ME2 top-10 quick-tool cases + math/echo + backchannel +
               delegate-fallthrough through brain.route() and assert each lands
               on the right route with a sane reply. (Same table as
               scripts/test_quick_tools.py, extended.)
  5  DELEGATE: build the configured backend and assert it's available
               (OnIt worker runner present) WITHOUT dispatching — no network.

Pass/fail
---------
  Exit code 0 = all green. Non-zero = at least one check failed. Each check
  prints PASS/FAIL with the evidence, so you can see exactly what broke.

Run (one line)
--------------
  cd /Users/yannamii/sandbox/ME3 && PYTHONPATH=src python3 scripts/offline_full_check.py
"""
from __future__ import annotations

import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, evidence: str) -> None:
    RESULTS.append((name, ok, evidence))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<10} {evidence}")


def load_wav_16k(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if sr != 16000:
        idx = np.linspace(0, len(pcm) - 1, int(len(pcm) * 16000 / sr))
        pcm = np.interp(idx, np.arange(len(pcm)), pcm.astype(np.float64)).astype(np.int16)
    return pcm


def main() -> int:
    cfg = yaml.safe_load(open(ROOT / "config" / "config.yaml"))
    v, acfg, tcfg = cfg["vad"], cfg["asr"], cfg["tts"]
    audio_dir = ROOT / "data" / "audio"
    speech = audio_dir / "test_speech.wav"
    silence = audio_dir / "silence_3s.wav"

    print("=" * 70)
    print("  ME3 OFFLINE FULL CHECK  (no mic / speaker / network)")
    print("=" * 70)

    # ------------------------------------------------------------------ 1. VAD
    # Import the engine class directly so a missing *other* engine's native dep
    # (e.g. webrtcvad) can't block the check. Falls back to Silero if the
    # configured engine's module isn't importable in this environment.
    engine = v.get("engine", "silero")
    try:
        from voiceduplex.vad import make_vad
        vad = make_vad(
            engine,
            threshold=float(v.get("threshold", 0.5)),
            min_speech_frames=int(v.get("min_speech_frames", 4)),
            min_silence_frames=int(v.get("min_silence_frames", 12)),
        )
    except (ImportError, ModuleNotFoundError):
        from voiceduplex.vad.silero_vad import SileroVAD
        engine = "silero"
        vad = SileroVAD(
            threshold=float(v.get("threshold", 0.5)),
            min_speech_frames=int(v.get("min_speech_frames", 4)),
            min_silence_frames=int(v.get("min_silence_frames", 12)),
        )
        print(f"  (note: {engine} engine unavailable here, using Silero)")

    def vad_events(pcm: np.ndarray) -> set:
        evs = set()
        block = 512
        for i in range(0, len(pcm) - block, block):
            _, ev = vad.process(pcm[i:i + block])
            if ev:
                evs.add(ev)
        return evs

    if speech.exists():
        sp_ev = vad_events(load_wav_16k(speech))
        check("VAD-speech", "speech_start" in sp_ev,
              f"events={sorted(sp_ev)} (want speech_start)")
    else:
        check("VAD-speech", False, f"missing {speech}")

    if silence.exists():
        vad.reset() if hasattr(vad, "reset") else None
        si_ev = vad_events(load_wav_16k(silence))
        check("VAD-silence", "speech_start" not in si_ev,
              f"events={sorted(si_ev)} (want NONE)")
    else:
        check("VAD-silence", False, f"missing {silence}")

    # ------------------------------------------------------------------ 2. ASR
    from voiceduplex.asr import make_asr
    asr = make_asr(
        acfg.get("backend", "whispercpp"),
        model_path=ROOT / acfg.get("model", "models/ggml-base.en.bin"),
        binary=acfg.get("binary", "whisper-cli"),
        language=acfg.get("language", "en"),
        threads=int(acfg.get("threads", 0)),
        beam_size=int(acfg.get("beam_size", 5)),
    )
    asr.warm()
    if speech.exists():
        t0 = time.perf_counter()
        tr = asr.transcribe(load_wav_16k(speech), lang="en")
        dt = time.perf_counter() - t0
        ok = bool(tr.text.strip())
        check("ASR", ok,
              f"text={tr.text.strip()!r}  RTF={tr.rtf:.2f}  ({dt:.2f}s wall)")
    else:
        check("ASR", False, f"missing {speech}")

    # ------------------------------------------------------------------ 3. TTS
    from voiceduplex.tts import make_tts
    if tcfg.get("backend", "kokoro") == "piper":
        tts = make_tts("piper", model=str(ROOT / tcfg.get("piper_model")),
                       speed=float(tcfg.get("speed", 1.0)))
    else:
        tts = make_tts(
            "kokoro",
            python=str(ROOT / tcfg.get("kokoro_python", ".venv-kokoro/bin/python")),
            model=str(ROOT / tcfg.get("kokoro_model", "models/kokoro-v1.0.onnx")),
            voices=str(ROOT / tcfg.get("kokoro_voices", "models/voices-v1.0.bin")),
            voice=tcfg.get("voice", "af_heart"),
            speed=float(tcfg.get("speed", 1.0)),
        )
    tts.warm()
    synth = tts.synthesize("The quick brown fox jumps over the lazy dog.")
    ok = (synth.audio is not None and len(synth.audio) > 0
          and synth.audio.dtype == np.int16 and synth.duration_audio > 0)
    check("TTS", ok,
          f"backend={synth.backend}  {synth.duration_audio:.2f}s audio  "
          f"RTF={synth.rtf:.2f}  {len(synth.audio)} samples")

    # ------------------------------------------------------------------ 4. ROUTER
    # Import brain.py DIRECTLY (not via voiceduplex.loop) so we don't drag in
    # audioio/sounddevice — the router is pure-python and needs neither.
    import importlib.util
    import types
    _src = str(ROOT / "src")
    for _pkg, _sub in (("voiceduplex", "voiceduplex"),
                       ("voiceduplex.loop", "voiceduplex/loop"),
                       ("voiceduplex.smarthome", "voiceduplex/smarthome")):
        if _pkg not in sys.modules:
            _m = types.ModuleType(_pkg)
            _m.__path__ = [f"{_src}/{_sub}"]
            sys.modules[_pkg] = _m
    for _sub in ("device", "weather"):
        _spec = importlib.util.spec_from_file_location(
            f"voiceduplex.smarthome.{_sub}", f"{_src}/voiceduplex/smarthome/{_sub}.py")
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[f"voiceduplex.smarthome.{_sub}"] = _mod
        _spec.loader.exec_module(_mod)
    _spec = importlib.util.spec_from_file_location(
        "voiceduplex.loop.brain", f"{_src}/voiceduplex/loop/brain.py")
    _brain = importlib.util.module_from_spec(_spec)
    sys.modules["voiceduplex.loop.brain"] = _brain
    _spec.loader.exec_module(_brain)
    route = _brain.route
    CASES = [
        ("quick_tool", "what time is it", "It's"),
        ("quick_tool", "what's the date today", "Today is"),
        ("quick_tool", "what's the weather in metro manila", "Simulated weather"),
        ("quick_tool", "play some music", "Now playing"),
        ("quick_tool", "turn on the lights", "Lights on"),
        ("quick_tool", "turn off the light", "Lights off"),
        ("quick_tool", "dim the lights to fifty percent", "brightness to 50"),
        ("quick_tool", "make the lights blue", "blue"),
        ("quick_tool", "set a timer for ten seconds", "Timer set for 10"),
        ("quick_tool", "set an alarm for six am", "Alarm set for 6:00 AM"),
        ("quick_tool", "set the temperature to twenty two degrees", "temperature to 22"),
        ("quick_tool", "pause the music", "Paused"),
        ("quick_tool", "turn it up", "Volume up"),
        ("quick_tool", "remind me to buy milk", "remind you to buy milk"),
        ("quick_tool", "call mom", "Calling mom"),
        ("quick_tool", "what's twelve times eight", "That's 96"),
        ("quick_tool", "echo hello world", "You said: hello world"),
        ("backchannel", "mm-hm", "Mm-hm"),
        ("delegate", "tell me a joke", None),  # must NOT be a quick_tool
    ]
    r_pass = 0
    failures = []
    for want_route, text, expect_sub in CASES:
        d = route(text)
        if want_route == "delegate":
            ok = d.route == "delegate"
        else:
            ok = (d.route == want_route) and (
                expect_sub is None or expect_sub.lower() in d.reply.lower())
        if ok:
            r_pass += 1
        else:
            failures.append(f"{text!r}->{d.route} (want {want_route})")
    check("ROUTER", r_pass == len(CASES),
          f"{r_pass}/{len(CASES)} routed correctly"
          + ("" if not failures else f"  FAIL: {failures[:3]}"))

    # ------------------------------------------------------------------ 5. DELEGATE
    from voiceduplex.delegate import build_backend
    backend = build_backend(cfg)
    avail = backend.is_available()
    check("DELEGATE", avail,
          f"backend={backend.name}  available={avail}"
          + ("" if avail else "  (runner/venv missing — Step 8 will stub)"))

    # ------------------------------------------------------------------ summary
    total = len(RESULTS)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print("=" * 70)
    print(f"  {passed}/{total} checks passed")
    if passed == total:
        print("  ✅ ALL GREEN — the model pipeline is healthy.")
        print("     If live conversation still mishears, it's the mic/room, not the model.")
        print("     Confirm with:  PYTHONPATH=src python3 scripts/live_capability_test.py")
    else:
        print("  ❌ FAILURES:")
        for name, ok, ev in RESULTS:
            if not ok:
                print(f"     - {name}: {ev}")
    print("=" * 70)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
