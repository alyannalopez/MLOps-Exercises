"""Full-duplex conversation loop (Step 6).

Unlike the half-duplex loop (Step 5), the microphone stays LIVE while the
agent is talking. That enables two behaviours:

  BARGE-IN        The user interrupts the agent mid-reply. We stop speaking
                  immediately and respond to the new utterance.
  BACKCHANNEL     The user says "yeah"/"mm-hm" while we talk. That is NOT an
                  interruption — we briefly acknowledge and RESUME our reply.

Design — one mic owner, an utterance FIFO, and a barge-in signal
----------------------------------------------------------------
The ListenerThread is the ONLY reader of the audio device (you can't have two
callbacks pulling the same InputStream). It runs forever (idle AND during
playback):
  - runs every frame through the VAD (the single authority on "is a human
    talking right now"),
  - as soon as a start->end pair completes, stores the captured audio as a
    Utterance in a FIFO,
  - sets the barge-in Event on every "start" so the playback chunk loop can
    stop within one chunk.

The worker (main thread) does all thinking and NEVER reads the mic:
  - idle: pulls the next Utterance from the FIFO -> ASR -> route -> TTS.
  - speaking: plays the reply in short async chunks, checking the barge-in
    Event between chunks. On a barge-in it pops the overlapping Utterance from
    the FIFO, classifies it (backchannel vs interrupt), and acts.

Because the listener captures utterances as they happen, an overlap that occurs
mid-playback is already in the FIFO by the time the worker needs it — which is
what makes barge-in deterministic (no re-capture, no waiting on the VAD).

Playback is ASYNC (audio.play_async) so the mic InputStream keeps delivering
frames while we talk — that is what makes it full-duplex.

Every turn is stamped T0..T8 plus the overlap events O0..O3 and appended to
the timeline JSONL (Step 9 reads O0->O2 as the barge-in reaction latency).

Run:  PYTHONPATH=src python3 scripts/run_full_duplex.py
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..audioio import AudioIO
from ..tts import TTS
from ..vad import VAD
from .brain import route
from .classifier import classify_overlap
from ..delegate import DelegateBackend
from .timeline import Timeline, TimelineLog
from .half_duplex import _resample_to_int16


# ---------------------------------------------------------------------------
# Utterances
# ---------------------------------------------------------------------------
@dataclass
class Utterance:
    """One VAD-delimited speech segment, captured by the listener."""
    start_t: float        # seconds since loop origin
    end_t: float
    energy: float         # RMS at onset (proxy for loudness)
    pcm: np.ndarray       # the actual audio of this utterance

    @property
    def duration(self) -> float:
        return self.end_t - self.start_t


class ListenerThread(threading.Thread):
    """Sole owner of the mic. Runs the VAD and captures every utterance.

    It reads the mic forever (idle AND during playback), runs each frame through
    the VAD, and as soon as a start->end pair completes it stores the captured
    audio as an Utterance in a FIFO. It also sets `barge_evt` on every "start"
    so the playback chunk loop can stop within one chunk.

    The worker simply consumes utterances from the FIFO — it never reads the
    mic and never waits for a VAD event, which is what makes barge-in
    deterministic: an utterance that overlaps playback is already captured by
    the time the worker needs it.
    """

    def __init__(self, audio: AudioIO, vad: VAD,
                 barge_evt: threading.Event, stop_evt: threading.Event,
                 t_origin: float, event_cb=None) -> None:
        super().__init__(daemon=True, name="listener")
        self.audio = audio
        self.vad = vad
        self.barge_evt = barge_evt
        self.stop_evt = stop_evt
        self.t_origin = t_origin
        self.event_cb = event_cb
        self.utterances: "queue.Queue[Utterance]" = queue.Queue(maxsize=32)
        self._speaking_since: Optional[float] = None
        self._buf: list[np.ndarray] = []
        self._start_energy = 0.0

    def run(self) -> None:
        while not self.stop_evt.is_set():
            try:
                frame = self.audio.read()
            except Exception:
                continue
            now = time.monotonic() - self.t_origin
            state, event = self.vad.process(frame.pcm)
            # Observer-only: expose live mic level + VAD score for the web UI.
            # Read-only -- does not alter VAD state or capture.
            if self.event_cb is not None:
                try:
                    amp = float(np.max(np.abs(frame.pcm))) if len(frame.pcm) else 0.0
                    sf = getattr(self.vad, "_score", None)
                    score = float(sf(frame.pcm)) if callable(sf) else 0.0
                    self.event_cb("mic", {"level": amp, "score": score,
                                          "speaking": bool(state)})
                except Exception:  # noqa: BLE001
                    pass
            if event == "speech_start":
                self._speaking_since = now
                self._buf = [frame.pcm]
                self._start_energy = float(np.sqrt(np.mean(frame.pcm.astype(np.float64) ** 2)))
                self.barge_evt.set()
            elif event == "speech_end":
                self._buf.append(frame.pcm)
                pcm = np.concatenate(self._buf) if self._buf else np.zeros(self.audio.BLOCK, dtype=np.int16)
                utt = Utterance(self._speaking_since or now, now, self._start_energy, pcm)
                self._speaking_since = None
                self._buf = []
                try:
                    self.utterances.put_nowait(utt)
                except queue.Full:
                    pass  # drop oldest rather than block the mic
            else:
                if self._speaking_since is not None:
                    self._buf.append(frame.pcm)

    def get_utterance(self, timeout: float = 0.1) -> Optional[Utterance]:
        try:
            return self.utterances.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> None:
        while True:
            try:
                self.utterances.get_nowait()
            except queue.Empty:
                return


class FullDuplexLoop:
    def __init__(
        self,
        audio: AudioIO,
        vad: VAD,
        asr,
        tts: TTS,
        log: TimelineLog,
        *,
        max_utterance_sec: float = 15.0,
        barge_in: bool = True,
        barge_min_energy: float = 300.0,
        backchannel_during_speech: bool = True,
        play_chunk_sec: float = 0.15,
        ring_seconds: float = 8.0,
        verbose: bool = True,
        delegate: Optional[DelegateBackend] = None,
        delegate_wait_sec: float = 120.0,
        event_cb: Optional[Callable[[str, object], None]] = None,
    ) -> None:
        self.audio = audio
        self.vad = vad
        self.asr = asr
        self.tts = tts
        self.log = log
        # Optional observer (web UI / live test). Called as event_cb(kind, payload)
        # on state changes. When None (the default) the loop behaves exactly as
        # before -- run_full_duplex.py is unaffected.
        self.event_cb = event_cb
        self._listening = False
        self._thinking = False

        self.max_utterance_frames = int(max_utterance_sec * audio.SAMPLE_RATE / audio.BLOCK)
        self.max_utterance_sec = max_utterance_sec
        self.barge_in_enabled = barge_in
        self.barge_min_energy = barge_min_energy
        self.backchannel_during_speech = backchannel_during_speech
        self.chunk_samples = max(1, int(play_chunk_sec * audio.SAMPLE_RATE))

        self.verbose = verbose
        self.delegate = delegate
        self.delegate_wait_sec = delegate_wait_sec
        self._turn = 0
        self._running = False
        self._stop = threading.Event()
        self._barge_evt = threading.Event()
        self._t_origin = 0.0
        self._playing = False
        self._listener: Optional[ListenerThread] = None
        # Introspection hooks (read by scripts/live_capability_test.py). Record
        # the most-recent primary turn and the most-recent barge-in/interrupt so
        # a scripted test can score "what did the loop actually hear?".
        self._last_heard: tuple[str, str, str] = ("", "", "")
        self._last_interrupt: str = ""

    # ------------------------------------------------------------------
    def _emit(self, kind: str, payload: object = None) -> None:
        """Notify the optional observer. Never raises, never blocks the loop."""
        cb = self.event_cb
        if cb is None:
            return
        try:
            cb(kind, payload)
        except Exception:  # noqa: BLE001 - observer must never break the loop
            pass

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)
        self._emit("log", msg)

    def run(self) -> None:
        self._running = True
        self._stop.clear()
        self._barge_evt.clear()
        self._t_origin = time.monotonic()
        self._log("\n=== ME3 FULL-DUPLEX loop — you can interrupt me mid-sentence ===")
        self._log("Talk normally. To BARGE IN, start a new sentence while I'm talking.")
        self._log("To just ACKNOWLEDGE (not interrupt), say 'yeah'/'mm-hm' while I talk.\n")

        self._listener = ListenerThread(self.audio, self.vad,
                                        self._barge_evt, self._stop, self._t_origin,
                                        event_cb=self.event_cb)
        self._listener.start()
        try:
            while self._running:
                self._await_and_handle_turn()
        except KeyboardInterrupt:
            self._log("\n[interrupted]")
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False
        self._stop.set()
        if self._listener is not None:
            self._listener.join(timeout=2.0)
            self._listener = None
        # Kill the persistent OnIt worker so Ctrl-C leaves no orphaned agent.
        if self.delegate is not None:
            shut = getattr(self.delegate, "shutdown", None)
            if callable(shut):
                try:
                    shut()
                except Exception as e:  # noqa: BLE001 - never crash on teardown
                    self._log(f"[cleanup] delegate shutdown: {e}")

    # ------------------------------------------------------------------
    # Introspection (used by scripts/live_capability_test.py)
    # ------------------------------------------------------------------
    def last_heard(self) -> tuple[str, str, str]:
        """(transcript, route, reply) of the most recent primary turn."""
        return self._last_heard

    def last_interrupt_heard(self) -> str:
        """Transcript of the most recent barge-in/interrupt ('' if none)."""
        return self._last_interrupt

    # ------------------------------------------------------------------
    # One turn: wait for the user's utterance, then think + speak.
    # ------------------------------------------------------------------
    def _await_and_handle_turn(self) -> None:
        # Clear any stale barge-in signal from the previous turn so it doesn't
        # instantly "interrupt" the next playback. The listener will re-set it
        # on a genuine speech_start during playback.
        self._barge_evt.clear()
        # Wait for a captured utterance from the listener FIFO.
        utt: Optional[Utterance] = None
        waited = 0.0
        self._listening = True
        self._emit("state", "listening")
        while self._running and utt is None:
            utt = self._listener.get_utterance(timeout=0.05)
            if utt is None:
                waited += 0.05
                if waited > 60.0:  # safety: don't spin forever in tests
                    return
        if utt is None:
            self._listening = False
            self._emit("state", "idle")
            return

        self._listening = False
        self._thinking = True
        self._emit("state", "thinking")
        self._turn += 1
        tl = Timeline(turn_id=self._turn, t_origin=time.monotonic())
        tl.T0_user_speech_start = 0.0
        tl.T1_user_speech_end = utt.duration

        try:
            tr = self.asr.transcribe(utt.pcm)
        except Exception as e:  # noqa: BLE001
            tl.error = f"asr: {e}"
            self._log(f"[{tl.turn_id}] ASR error: {e}")
            self.log.record(tl)
            return
        tl.stamp("T2_asr_final")
        tl.transcript = tr.text
        self._emit("heard", {"turn": tl.turn_id, "text": tr.text,
                             "asr_ms": round(tr.duration_compute * 1000, 1)})
        self._log(f"[{tl.turn_id}] 📝 heard: “{tr.text}”  (ASR {tr.duration_compute:.2f}s, RTF {tr.rtf:.2f})")

        decision = route(tr.text, delegate=self.delegate)
        tl.stamp("T3_route_decision")
        tl.route = decision.route
        tl.reply = decision.reply
        self._last_heard = (tr.text, decision.route, decision.reply)
        self._emit("route", {"turn": tl.turn_id, "route": decision.route,
                             "detail": decision.detail})
        self._log(f"[{tl.turn_id}] 🧠 route={decision.route:<12} ({decision.detail})")

        self._speak_reply(tl, decision.reply)

        # Step 8: if we delegated, the ack ("let me check that") is already
        # playing. Now wait for the harness result and speak it. The mic stays
        # LIVE during this wait, so a barge-in still cuts us off.
        if decision.pending is not None:
            self._finish_delegate(tl, decision.pending)

    # ------------------------------------------------------------------
    def _speak_reply(self, tl: Timeline, reply: str) -> None:
        try:
            synth = self.tts.synthesize(reply)
        except Exception as e:  # noqa: BLE001
            tl.error = f"tts: {e}"
            self._log(f"[{tl.turn_id}] TTS error: {e}")
            self.log.record(tl)
            return

        pb_rate = getattr(self.audio, "playback_rate", self.audio.SAMPLE_RATE)
        out = _resample_to_int16(synth.audio, synth.sample_rate, pb_rate)

        tl.stamp("T4_agent_speech_start")
        self._log(f"[{tl.turn_id}] 🔊 speaking: “{reply}”  (TTS {synth.duration_compute:.2f}s, RTF {synth.rtf:.2f})")

        self._playing = True
        self._thinking = False
        self._barge_evt.clear()
        self._emit("state", "speaking")
        self._emit("reply", {"turn": tl.turn_id, "text": reply})
        interrupted = self._play_with_barge_in(out)
        self._playing = False

        tl.stamp("T5_agent_speech_end")
        if interrupted:
            self._log(f"[{tl.turn_id}] ⏹ playback cut short by barge-in")
            self._handle_interrupt(tl)
        tl.stamp("T8_turn_complete")
        lat = tl.latencies_ms()
        self._emit("turn_done", {"turn": tl.turn_id,
                                 "ttfa_ms": lat.get("A4_TTFA"),
                                 "asr_ms": lat.get("A2_L_asr"),
                                 "route_ms": lat.get("A3_L_route"),
                                 "interrupted": bool(interrupted)})
        self._log(f"[{tl.turn_id}] ✔ turn done — TTFA={lat['A4_TTFA']}ms  ASR={lat['A2_L_asr']}ms  "
                  f"route={lat['A3_L_route']}ms\n")
        self.log.record(tl)

    def _finish_delegate(self, tl: Timeline, pending) -> None:
        """Wait for a delegated task to finish, then speak the result.

        The mic is live the whole time (full-duplex), so the user can barge in
        while the harness runs. On a barge-in we abandon the wait and let
        ``_handle_interrupt`` take over the conversation.
        """
        if self.verbose:
            self._log(f"[{tl.turn_id}] ⏳ waiting for {getattr(self.delegate, 'name', 'harness')} …")
        res = pending.wait(self.delegate_wait_sec)

        # A barge-in fired while we waited: stop chasing the result.
        if self._barge_evt.is_set() or not self._running:
            pending.cancel()
            self._log(f"[{tl.turn_id}] ⏸ delegate abandoned (barge-in / stopped)")
            return

        if res.ok:
            tl.reply = res.text
            tl.stamp("T6_delegate_result")
            self._log(f"[{tl.turn_id}] 📦 delegate done in {res.elapsed:.1f}s")
            self._speak_reply(tl, res.text)
        else:
            msg = (
                "Sorry, my research brain timed out on that one. Try asking again."
                if res.cancelled
                else "Sorry, I couldn't finish that task. Let's try something simpler."
            )
            tl.error = f"delegate: {res.error}"
            self._log(f"[{tl.turn_id}] ⚠️ delegate failed: {res.error}")
            self._speak_reply(tl, msg)

    def _play_with_barge_in(self, out: np.ndarray) -> bool:
        """Queue `out` in short async chunks; return True if a barge-in cut it
        short. play_async() is non-blocking so the mic keeps streaming."""
        n = len(out)
        pos = 0
        pb_rate = getattr(self.audio, "playback_rate", self.audio.SAMPLE_RATE)
        chunk_dur = self.chunk_samples / pb_rate
        while pos < n and self._running:
            seg = out[pos:pos + self.chunk_samples]
            pos += len(seg)
            self.audio.play_async(seg)
            if self.barge_in_enabled and self._barge_evt.is_set():
                # Cut the audio off NOW (flush the queue + close the stream) so
                # the user hears the agent stop, not just stop queuing.
                stop = getattr(self.audio, "stop_playback", None)
                if callable(stop):
                    stop()
                return True
            time.sleep(chunk_dur)  # pace to real-time
        return False

    # ------------------------------------------------------------------
    # The user barged in: consume the overlapping utterance, classify, act.
    # ------------------------------------------------------------------
    def _handle_interrupt(self, parent_tl: Timeline) -> None:
        # The overlapping utterance was captured by the listener during
        # playback. Consume it from the FIFO (with a short grace period in case
        # the VAD hasn't emitted the "end" yet).
        utt: Optional[Utterance] = None
        deadline = time.monotonic() + 2.0
        while self._running and utt is None and time.monotonic() < deadline:
            utt = self._listener.get_utterance(timeout=0.05)
        if utt is None:
            return

        # Ignore a faint overlap (our own voice leaking into the mic / a cough).
        if utt.energy < self.barge_min_energy:
            self._log(f"[bc] faint overlap (energy {utt.energy:.0f}) -> ignoring")
            return

        self._emit("barge_in", {"energy": round(utt.energy, 1),
                                "during": "speaking"})

        self._turn += 1
        tl = Timeline(turn_id=self._turn, t_origin=time.monotonic())
        tl.T0_user_speech_start = 0.0
        tl.T1_user_speech_end = utt.duration
        tl.stamp("O0_overlap_onset")

        try:
            tr = self.asr.transcribe(utt.pcm)
            text = tr.text
        except Exception as e:  # noqa: BLE001
            self._log(f"[overlap] ASR error: {e}; treating as backchannel (resume)")
            text = ""

        verdict = classify_overlap(text, utt.duration)
        tl.stamp("O1_overlap_classified")
        tl.transcript = text

        if verdict == "backchannel" and self.backchannel_during_speech:
            self._emit("backchannel", {"text": text, "action": "resume"})
            self._log(f"[bc] backchannel: “{text}” -> acknowledging, resuming")
            tl.route = "backchannel_resume"
            tl.reply = "(acknowledged, resuming)"
            tl.stamp("O2_agent_reaction")
            tl.stamp("T8_turn_complete")
            self.log.record(tl)
            return

        # Empty / unusable transcript: the barge-in was noise or a misfire
        # (e.g. our own voice leaking into the mic). Don't hand an empty task
        # to the delegate — that's what produces the phantom "Let me check
        # that." -> "I don't have a task" loop. Just stay quiet and wait.
        if not text.strip():
            self._log("[bc] blank overlap -> ignoring (no new command)")
            tl.route = "backchannel_resume"
            tl.reply = "(blank overlap ignored)"
            tl.stamp("O2_agent_reaction")
            tl.stamp("T8_turn_complete")
            self.log.record(tl)
            return

        self._log(f"[⚡ barge-in] interrupt: “{text}” -> responding")
        self._emit("interrupt", {"text": text})
        self._last_interrupt = text
        decision = route(text, delegate=self.delegate)
        tl.stamp("O2_agent_reaction")
        tl.stamp("T3_route_decision")
        tl.route = decision.route
        tl.reply = decision.reply
        self._log(f"[{tl.turn_id}] 🧠 route={decision.route:<12} ({decision.detail})")
        self._speak_reply(tl, decision.reply)
        if decision.pending is not None:
            self._finish_delegate(tl, decision.pending)
