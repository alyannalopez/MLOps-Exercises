"""Half-duplex conversation loop (Step 5).

Strict turn-taking:
  LISTEN  -> VAD detects speech_start, buffers the utterance
  (silence) -> VAD detects speech_end, flush buffer to ASR
  THINK   -> route(transcript)
  SPEAK   -> synthesize reply, play it. Mic keeps streaming but those frames
            are DISCARDED (this is what makes it half-duplex, not full-duplex).
  -> back to LISTEN

Every turn is stamped T0..T8 and appended to the timeline JSONL.

Run:  PYTHONPATH=src python3 scripts/run_loop.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..asr import make_asr
from ..audioio import make_audio_io
from ..tts import make_tts
from ..vad import make_vad, VADState
from .brain import route
from .timeline import Timeline, TimelineLog
from ..delegate import DelegateBackend


def _resample_to_int16(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample mono float/int PCM to sr_out, return int16."""
    x = x.astype(np.float64)
    if len(x) == 0:
        return np.zeros(0, dtype=np.int16)
    if sr_in != sr_out:
        n_out = int(round(len(x) * sr_out / sr_in))
        xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
        xk = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        x = np.interp(xk, xp, x)
    return np.clip(x, -32768, 32767).astype(np.int16)


class HalfDuplexLoop:
    def __init__(
        self,
        audio,
        vad,
        asr,
        tts,
        log: TimelineLog,
        *,
        max_utterance_sec: float = 15.0,
        verbose: bool = True,
        delegate: "Optional[DelegateBackend]" = None,
        delegate_wait_sec: float = 120.0,
    ) -> None:
        self.audio = audio
        self.vad = vad
        self.asr = asr
        self.tts = tts
        self.log = log
        self.max_utterance_frames = int(max_utterance_sec * audio.SAMPLE_RATE / audio.BLOCK)
        self.verbose = verbose
        self.delegate = delegate
        self.delegate_wait_sec = delegate_wait_sec
        self._turn = 0
        self._running = False

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def run(self) -> None:
        self._running = True
        self._log("\n=== ME3 half-duplex loop — talk, then stay silent to end your turn ===")
        self._log("Type 'q' in the console and press Enter to quit.\n")
        try:
            while self._running:
                self._one_turn()
        except KeyboardInterrupt:
            self._log("\n[interrupted]")
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    def _one_turn(self) -> None:
        self._turn += 1
        tl = Timeline(turn_id=self._turn, t_origin=time.monotonic())

        # ---- LISTEN: wait for speech_start --------------------------
        buf: list[np.ndarray] = []
        speaking = False
        while self._running:
            frame = self.audio.read()
            state, event = self.vad.process(frame.pcm)
            if event == "speech_start":
                tl.stamp("T0_user_speech_start")
                self._log(f"[{tl.turn_id}] 🎤 you started speaking @ {tl.T0_user_speech_start:.2f}s")
                speaking = True
                buf.append(frame.pcm)
                break
            # while idle, allow a console 'q' without blocking the mic
        if not self._running:
            return

        # ---- collect until speech_end (or cap) ----------------------
        while self._running:
            frame = self.audio.read()
            state, event = self.vad.process(frame.pcm)
            if speaking:
                buf.append(frame.pcm)
            if event == "speech_end":
                tl.stamp("T1_user_speech_end")
                self._log(f"[{tl.turn_id}] 🔇 you stopped @ {tl.T1_user_speech_end:.2f}s "
                          f"({len(buf) * self.audio.BLOCK / self.audio.SAMPLE_RATE:.2f}s audio)")
                break
            if len(buf) >= self.max_utterance_frames:
                tl.stamp("T1_user_speech_end")
                self._log(f"[{tl.turn_id}] ⏱ utterance capped @ {tl.T1_user_speech_end:.2f}s")
                break
        if not self._running:
            return

        # ---- THINK: ASR -> route ------------------------------------
        pcm = np.concatenate(buf) if buf else np.zeros(self.audio.BLOCK, dtype=np.int16)
        try:
            tr = self.asr.transcribe(pcm)
        except Exception as e:  # noqa: BLE001
            tl.error = f"asr: {e}"
            self._log(f"[{tl.turn_id}] ASR error: {e}")
            self.log.record(tl)
            return
        tl.stamp("T2_asr_final")
        tl.transcript = tr.text
        self._log(f"[{tl.turn_id}] 📝 heard: “{tr.text}”  (ASR {tr.duration_compute:.2f}s, RTF {tr.rtf:.2f})")

        decision = route(tr.text, delegate=self.delegate)
        tl.stamp("T3_route_decision")
        tl.route = decision.route
        tl.reply = decision.reply
        self._log(f"[{tl.turn_id}] 🧠 route={decision.route:<12} ({decision.detail})")

        # Step 8: delegated task -> speak the ack, then the result. Half-duplex
        # does NOT listen while speaking, so we just wait (no barge-in here).
        if decision.pending is not None:
            self._log(f"[{tl.turn_id}] ⏳ waiting for {getattr(self.delegate, 'name', 'harness')} …")
            res = decision.pending.wait(self.delegate_wait_sec)
            if res.ok:
                tl.reply = res.text
                tl.stamp("T6_delegate_result")
                self._log(f"[{tl.turn_id}] 📦 delegate done in {res.elapsed:.1f}s")
            else:
                tl.error = f"delegate: {res.error}"
                tl.reply = (
                    "Sorry, my research brain timed out on that one."
                    if res.cancelled
                    else "Sorry, I couldn't finish that task. Let's try something simpler."
                )
                self._log(f"[{tl.turn_id}] ⚠️ delegate failed: {res.error}")

        # ---- SPEAK: synthesize + play (mic frames discarded) --------
        try:
            synth = self.tts.synthesize(decision.reply)
        except Exception as e:  # noqa: BLE001
            tl.error = f"tts: {e}"
            self._log(f"[{tl.turn_id}] TTS error: {e}")
            self.log.record(tl)
            return
        # drain any mic frames that arrived during synthesis (half-duplex: ignore them)
        self._drain_mic()
        # Resample TTS (24k/22.05k) up to the playback rate (48k) so Bluetooth
        # A2DP sinks negotiate. Built-in speakers are fine at 16k too.
        pb_rate = getattr(self.audio, "playback_rate", self.audio.SAMPLE_RATE)
        out = _resample_to_int16(synth.audio, synth.sample_rate, pb_rate)
        tl.stamp("T4_agent_speech_start")
        self._log(f"[{tl.turn_id}] 🔊 speaking: “{decision.reply}”  (TTS {synth.duration_compute:.2f}s, RTF {synth.rtf:.2f})")
        self.audio.play(out)
        tl.stamp("T5_agent_speech_end")

        # ---- close the turn -----------------------------------------
        # reset VAD so the next turn starts clean
        self.vad.reset()
        self._drain_mic()
        tl.stamp("T8_turn_complete")
        lat = tl.latencies_ms()
        self._log(f"[{tl.turn_id}] ✔ turn done — TTFA={lat['A4_TTFA']}ms  ASR={lat['A2_L_asr']}ms  "
                  f"route={lat['A3_L_route']}ms\n")
        self.log.record(tl)

    def _drain_mic(self) -> None:
        """Discard buffered mic frames so stale audio doesn't trigger a false turn."""
        # sounddevice InputStream buffers a handful of blocks; give it a beat.
        time.sleep(0.05)
        try:
            while not self.audio._q.empty():  # noqa: SLF001 — pragmatic drain
                self.audio._q.get_nowait()
        except Exception:  # noqa: BLE001
            pass
