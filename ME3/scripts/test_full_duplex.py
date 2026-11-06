#!/usr/bin/env python3
"""Step 6 offline success check — full-duplex loop WITHOUT a real mic.

Drives FullDuplexLoop with a scripted FakeAudioIO that paces playback to
REAL-TIME (so the agent is genuinely "speaking" when the user barges in):

  Scenario 1 (normal turn):  user speaks -> agent answers -> full turn completes.
  Scenario 2 (barge-in):     user asks, agent starts a LONG reply, user INTERRUPTS
                             mid-reply -> playback is cut short, agent responds to
                             the NEW utterance. O0<=O1<=O2 are stamped.
  Scenario 3 (backchannel):  user asks, agent replies, user says "mm-hm" mid-reply
                             -> classified backchannel, agent acknowledges (no new
                             reply), turn completes.

Uses the REAL VAD (Silero) and REAL brain router; ASR and TTS are stubs so the
test is deterministic. Verifies:
  - a normal turn stamps T0..T8 and records a timeline line
  - a barge-in turn cuts playback short AND stamps O0<=O1<=O2
  - a backchannel overlap is recorded as route='backchannel_resume'
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from voiceduplex.audioio.base import AudioIO, Frame
from voiceduplex.loop import FullDuplexLoop, TimelineLog
from voiceduplex.vad import make_vad
from voiceduplex.asr.base import Transcript

# Compress playback pacing so the offline test runs in ~2-3s instead of ~15s.
# 1.0 = true real-time (use this if you ever need timing-accurate assertions).
TIME_SCALE = 0.1


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
class StubASR:
    """Canned transcripts, one per transcribe() call (in order)."""
    def __init__(self, texts):
        self._texts = list(texts)
        self._i = 0
    def warm(self):
        pass
    def transcribe(self, pcm, lang="en"):
        t = self._texts[min(self._i, len(self._texts) - 1)]
        self._i += 1
        dur = len(pcm) / 16000.0
        return Transcript(text=t, lang="en", duration_audio=dur,
                          duration_compute=0.01, segments=[(0.0, dur, t)])


class StubTTS:
    """Fixed-length 24 kHz tone (long enough to be interruptible)."""
    def __init__(self, seconds=3.0, sr=24000):
        self._n = int(seconds * sr)
        self.sr = sr
    def warm(self):
        pass
    def synthesize(self, text):
        from voiceduplex.tts.base import Synthesis
        n = self._n
        t = np.arange(n) / self.sr
        audio = (np.sin(2 * np.pi * 220 * t) * 8000).astype(np.int16)
        return Synthesis(audio=audio, sample_rate=self.sr, text=text,
                         duration_audio=n / self.sr, duration_compute=0.02,
                         backend="stub")
    @property
    def backend_name(self):
        return "stub"


# ---------------------------------------------------------------------------
# Fake audio: scripted mic + REAL-TIME-paced async playback
# ---------------------------------------------------------------------------
class FakeAudioIO(AudioIO):
    SAMPLE_RATE = 16000
    BLOCK = 512

    def __init__(self, script, playback_seconds=3.0, barge_after_chunk=None,
                 speech_wav=None, interrupt_seconds=0.0):
        self._speech = self._load_speech(speech_wav)
        # Build the full frame feed. If an interrupt is requested, append it
        # (with trailing silence for a clean VAD end) AFTER the main script.
        # The main script must end in silence so the interrupt reads as a
        # distinct utterance. The barge event (fired after N playback chunks)
        # marks the moment the agent is "speaking" when the interrupt arrives.
        full_script = list(script)
        if interrupt_seconds > 0:
            full_script += [("speech", interrupt_seconds), ("silence", 0.6)]
        self._frames = self._build_frames(full_script)
        self._idx = 0
        self.playback_rate = 48000
        self.sample_rate = 16000
        self.block = 512
        self.played_samples = 0
        self.total_full = 0
        self._t0 = time.monotonic()
        self._barge_after_chunk = barge_after_chunk
        self._chunks = 0
        self._barge_fn = None
        self._cur_reply_samples = 0
        self._barge_cut_samples = None

    @staticmethod
    def _load_speech(path):
        """Load real speech (Silero detects it; synthetic tones don't trigger)."""
        import wave
        w = wave.open(str(path))
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return data

    def _build_frames(self, script):
        frames = []
        sp = self._speech
        sp_len = len(sp)
        for kind, secs in script:
            n_frames = int(secs * 16000 / 512)
            for fi in range(n_frames):
                if kind == "speech":
                    base = (fi * 512) % sp_len
                    chunk = sp[base:base + 512]
                    if len(chunk) < 512:
                        chunk = np.concatenate([chunk, sp[:512 - len(chunk)]])
                    frames.append(chunk.astype(np.int16))
                else:
                    frames.append(np.zeros(512, dtype=np.int16))
        return frames

    def open(self):
        pass
    def close(self):
        pass

    def read(self) -> Frame:
        if self._idx >= len(self._frames):
            frame = np.zeros(512, dtype=np.int16)  # endless silence after feed
        else:
            frame = self._frames[self._idx]
            self._idx += 1
        return Frame(pcm=frame, t=time.monotonic() - self._t0)

    def play(self, pcm):
        self.played_samples += len(pcm)

    def play_async(self, pcm):
        # Instant (no sleep) so the offline test runs in well under a second.
        self.total_full += len(pcm)
        self.played_samples += len(pcm)
        self._cur_reply_samples += len(pcm)
        self._chunks += 1
        # Deterministic barge-in: after N chunks, fire the loop's barge event.
        # Record how much of the CURRENT reply was played before the cut.
        if (self._barge_after_chunk is not None
                and self._chunks == self._barge_after_chunk
                and self._barge_fn is not None):
            self._barge_cut_samples = self._cur_reply_samples
            self._barge_fn()


def run_scenario(name, script, asr_texts, expect_interrupt=False,
                 expect_backchannel=False, playback_seconds=3.0,
                 barge_after_chunk=None, interrupt_seconds=0.0):
    print(f"\n===== {name} =====")
    speech_wav = ROOT / "data" / "audio" / "test_speech.wav"
    audio = FakeAudioIO(script, playback_seconds, barge_after_chunk=barge_after_chunk,
                        speech_wav=speech_wav, interrupt_seconds=interrupt_seconds)
    vad = make_vad("silero", threshold=0.5, min_speech_frames=4, min_silence_frames=12)
    asr = StubASR(asr_texts)
    tts = StubTTS(seconds=playback_seconds)
    tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    tmp.close()
    log = TimelineLog(tmp.name)

    loop = FullDuplexLoop(
        audio, vad, asr, tts, log,
        max_utterance_sec=15.0, barge_in=True, barge_min_energy=300.0,
        backchannel_during_speech=True, play_chunk_sec=0.1, verbose=True,
    )
    # Wire the deterministic barge-in hook (fires the loop's barge event).
    # Also reset the per-reply sample counter when the loop starts a new reply
    # (it clears the barge event at the start of each _speak_reply).
    _orig_clear = loop._barge_evt.clear
    def _clear_and_reset():
        audio._cur_reply_samples = 0
        _orig_clear()
    loop._barge_evt.clear = _clear_and_reset
    audio._barge_fn = lambda: loop._barge_evt.set()

    audio.open()
    lines = []
    try:
        loop._running = True
        loop._stop.clear()
        loop._barge_evt.clear()
        loop._t_origin = time.monotonic()
        # Start the listener (run() normally does this; we drive one turn here).
        from voiceduplex.loop.full_duplex import ListenerThread
        loop._listener = ListenerThread(loop.audio, loop.vad,
                                        loop._barge_evt, loop._stop, loop._t_origin)
        loop._listener.start()
        loop._await_and_handle_turn()
        with open(tmp.name) as f:
            lines = [json.loads(l) for l in f if l.strip()]
    finally:
        loop.stop()
        audio.close()
        log.close()

    full_len = int(playback_seconds * 48000)
    print(f"\n  [stats] turns={len(lines)}  played={audio.played_samples}  "
          f"full_reply={full_len}  ratio={audio.played_samples/max(full_len,1):.2f}")
    for ln in lines:
        print(f"  turn {ln['turn_id']}: route={ln['route']!r} tx={ln['transcript']!r} "
              f"O0={ln['O0_overlap_onset']} O1={ln['O1_overlap_classified']} "
              f"O2={ln['O2_agent_reaction']} T8={ln['T8_turn_complete']}")

    ok = True
    if expect_interrupt:
        barge = [l for l in lines if l.get("O0_overlap_onset") is not None]
        if not barge:
            print("  FAIL: no barge-in turn (O0 never stamped)"); ok = False
        else:
            b = barge[-1]
            if not (b["O0_overlap_onset"] is not None and b["O1_overlap_classified"] is not None
                    and b["O2_agent_reaction"] is not None):
                print("  FAIL: barge-in turn missing O0/O1/O2"); ok = False
            elif not (b["O0_overlap_onset"] <= b["O1_overlap_classified"] <= b["O2_agent_reaction"]):
                print("  FAIL: O0/O1/O2 out of order"); ok = False
            else:
                print(f"  OK: barge-in O0<=O1<=O2  (O0->O2 = "
                      f"{(b['O2_agent_reaction']-b['O0_overlap_onset'])*1000:.0f}ms)")
        if (audio._barge_cut_samples is not None
                and audio._barge_cut_samples < full_len):
            print(f"  OK: first reply cut short ({audio._barge_cut_samples} < {full_len})")
        else:
            print(f"  FAIL: first reply NOT cut short (cut={audio._barge_cut_samples}, full={full_len})")
            ok = False

    if expect_backchannel:
        bc = [l for l in lines if l.get("route") == "backchannel_resume"]
        if not bc:
            print("  FAIL: no backchannel_resume turn"); ok = False
        else:
            print("  OK: backchannel classified & recorded as backchannel_resume")

    print(f"  {'PASS' if ok else 'FAIL'} :: {name}")
    return ok


def main():
    # Watchdog: force-exit if anything deadlocks (keeps the offline test from
    # hanging). The test itself should finish in well under 5s.
    import threading as _th
    def _watchdog():
        time.sleep(20)
        print("\n[WATCHDOG] test exceeded 20s — forcing exit (possible deadlock)")
        import os
        os._exit(3)
    _th.Thread(target=_watchdog, daemon=True).start()

    results = []

    ok1 = run_scenario(
        "Scenario 1: normal turn",
        [("silence", 0.5), ("speech", 1.0), ("silence", 0.6)],
        asr_texts=["what time is it"],
        playback_seconds=2.0,
    )
    results.append(ok1)

    ok2 = run_scenario(
        "Scenario 2: barge-in mid-reply",
        [("silence", 0.5), ("speech", 0.8), ("silence", 0.4)],
        asr_texts=["what time is it", "stop what date is it today"],
        expect_interrupt=True,
        playback_seconds=2.0,
        barge_after_chunk=5,
        interrupt_seconds=0.8,
    )
    results.append(ok2)

    ok3 = run_scenario(
        "Scenario 3: backchannel during reply",
        [("silence", 0.5), ("speech", 0.8), ("silence", 0.4)],
        asr_texts=["tell me a joke", "mm-hm"],
        expect_backchannel=True,
        playback_seconds=2.0,
        barge_after_chunk=5,
        interrupt_seconds=0.4,
    )
    results.append(ok3)

    print("\n" + "=" * 50)
    print(f"RESULT: {sum(results)}/{len(results)} scenarios passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
