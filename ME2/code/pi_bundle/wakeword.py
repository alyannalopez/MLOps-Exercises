#!/usr/bin/env python3
"""Wakeword gate for the VCM live demo.

Sits IN FRONT of the intent model: the mic streams continuously, and the
intent model only fires AFTER a wakeword is detected ("Alexa", "Hey Jarvis",
etc.). Without a wakeword, the system ignores everything.

Usage:
    python live_demo.py --model model_int8.onnx --simulate --wakeword
    python live_demo.py --model model_int8.onnx --simulate --wakeword --wake alexa,hey_jarvis

Dependencies:
    pip install openwakeword
    (first run downloads ~5 MB of ONNX models to ~/.openwakeword/)

Architecture:
    mic → [80 ms frame] → openwakeword.predict()
                          ├─ score < 0.5 → keep listening (idle)
                          └─ score >= 0.5 → WAKE
                              → drain residual audio
                              → record command (energy VAD, up to 5 s)
                              → VCM.classify_pcm() → intent
                              → hub.handle_intent() → repaint dashboard
                              → reset wakeword → back to idle

The wakeword model runs at 2.1 ms per 80 ms frame on a Mac — negligible.
On a Pi 4 it's ~5–8 ms per frame, still well within real-time budget.
"""
from __future__ import annotations
import time
import numpy as np


# openwakeword expects 1280 samples (80 ms) per predict() call at 16 kHz
OWW_FRAME = 1280
OWW_SR = 16000

# Default wakeword models to load (all small, <2 MB each)
DEFAULT_WAKEWORDS = ["alexa", "hey_jarvis", "hey_mycroft", "hey_rhasspy", "timer"]

# Score threshold: silence scores ~0.002, real wakewords score >0.7
WAKE_THRESHOLD = 0.5

# How long to wait for the command after waking (seconds)
COMMAND_WINDOW = 5.0

# Minimum speech duration to commit a command (seconds)
MIN_SPEECH = 0.25

# --- Anti-TV / ambient-speech guard ---------------------------------------
# After a wake, the LISTENING state used to commit on pure energy: ANY loud
# sound (a TV, a radio, someone in the next room) got classified. We now
# transcribe the captured clip with a tiny whisper.cpp model and require it to
# look like a real smart-home command. TV chatter transcribes to unrelated
# prose (or empty) and is discarded -> back to idle.
#
# Words that appear in our 10 intents (plus a couple of connectives). If the
# transcript contains NONE of these, it is almost certainly not a command.
# NOTE: deliberately EXCLUDES generic connectives (to/the/in/of/at/for/my) --
# those appear in ordinary TV/news speech and would defeat the gate. Only
# domain-specific smart-home words count as a command signal.
CMD_KEYWORDS = {
    "play", "music", "song", "turn", "on", "off", "lights", "light",
    "dim", "brighter", "darker", "color", "colour", "blue", "red", "green",
    "yellow", "white", "warm", "cool", "set", "timer", "alarm", "minutes",
    "minute", "hours", "hour", "call", "message", "text", "remind",
    "reminder", "add", "list", "search", "find", "what", "where", "who",
    "how", "weather", "temperature", "thermostat", "degrees",
    "pause", "resume", "next", "previous", "volume", "louder", "quieter",
    "stop", "please",
}


class WakeGate:
    """State-machine gate: IDLE → (wakeword) → LISTENING → (command) → IDLE.

    Feeds 80 ms frames to openwakeword. When a wakeword fires, switches to
    command-capture mode (energy VAD + buffer). Returns the captured PCM
    when a command is complete, or None otherwise.
    """

    IDLE = "idle"
    LISTENING = "listening"

    def __init__(self, wakeword_names: list[str] | None = None,
                 threshold: float = WAKE_THRESHOLD,
                 quiet: bool = False,
                 transcriber=None):
        self.threshold = threshold
        self.quiet = quiet
        self.state = self.IDLE
        # Optional ASR gate: a Transcriber instance (asr.py) or None. When set,
        # captured commands are filtered through it to reject TV/ambient audio.
        self.transcriber = transcriber
        self._sample_buf = np.array([], dtype=np.int16)  # accumulator for 1280-sample frames

        # Load the openwakeword model
        names = wakeword_names or DEFAULT_WAKEWORDS
        try:
            import openwakeword
            self._oww = openwakeword.Model(
                wakeword_models=names,
                inference_framework="onnx",
            )
            self._names = names
            if not quiet:
                print(f"[wakeword] loaded: {', '.join(names)}  "
                      f"(threshold={threshold})")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load openwakeword: {e}\n"
                f"Fix: pip install openwakeword && "
                f"python -c \"import openwakeword; "
                f"openwakeword.utils.download_models({names!r})\""
            ) from e

        # Command-capture state
        self._cmd_buf = np.array([], dtype=np.int16)
        self._cmd_floor = -45.0
        self._cmd_voiced = 0.0
        self._cmd_start = None
        self._drain = 0  # frames to discard after wake (residual audio)

    # ------------------------------------------------------------------
    def feed(self, pcm: np.ndarray) -> np.ndarray | None:
        """Feed raw 16 kHz int16 PCM (any length).

        Returns the captured command PCM when a full cycle completes,
        or None while still waiting.
        """
        if self.state == self.IDLE:
            return self._feed_idle(pcm)
        elif self.state == self.LISTENING:
            return self._feed_listening(pcm)
        return None

    # ------------------------------------------------------------------
    def _feed_idle(self, pcm: np.ndarray) -> np.ndarray | None:
        """IDLE: accumulate samples, feed 80 ms frames to openwakeword."""
        self._sample_buf = np.concatenate([self._sample_buf, pcm])
        fired_word = None
        fired_score = 0.0

        while len(self._sample_buf) >= OWW_FRAME:
            frame = self._sample_buf[:OWW_FRAME].copy()
            self._sample_buf = self._sample_buf[OWW_FRAME:]
            scores = self._oww.predict(frame)
            # scores: dict {model_name: score}
            best = max(scores.values()) if scores else 0.0
            if best >= self.threshold:
                fired_word = max(scores, key=scores.get)
                fired_score = best
                break

        if fired_word is None:
            return None

        # ---- WAKE EVENT ----
        self._oww.reset()
        if not self.quiet:
            print(f"\n\033[92m[WAKE] {fired_word!r} detected "
                  f"(score={fired_score:.2f}) — say your command...\033[0m")

        # Switch to LISTENING; discard residual audio in the buffer
        self.state = self.LISTENING
        self._cmd_buf = np.array([], dtype=np.int16)
        self._cmd_floor = -45.0
        self._cmd_voiced = 0.0
        self._cmd_start = time.perf_counter()
        # Drain: discard the rest of the accumulated samples (they're
        # part of the wakeword utterance, not the command)
        self._drain = len(self._sample_buf)
        self._sample_buf = np.array([], dtype=np.int16)
        return None

    # ------------------------------------------------------------------
    def _feed_listening(self, pcm: np.ndarray) -> np.ndarray | None:
        """LISTENING: energy-VAD capture of the command after the wakeword."""
        # Discard residual wakeword audio
        if self._drain > 0:
            skip = min(self._drain, len(pcm))
            pcm = pcm[skip:]
            self._drain -= skip
            if len(pcm) == 0:
                return None

        if len(pcm) == 0:
            return None

        # Timeout: if no speech started within COMMAND_WINDOW, go back to idle
        if len(self._cmd_buf) == 0:
            elapsed = time.perf_counter() - self._cmd_start
            if elapsed > COMMAND_WINDOW:
                if not self.quiet:
                    print("\033[90m[wakeword] no command heard, back to idle\033[0m")
                self._go_idle()
                return None

        from vcm_infer import rms_db
        db = rms_db(pcm)

        # Adapt noise floor on silence
        if db < self._cmd_floor + 6.0:
            self._cmd_floor = 0.95 * self._cmd_floor + 0.05 * db

        # Start capturing when speech begins
        if len(self._cmd_buf) == 0:
            if db > self._cmd_floor + 12.0:
                self._cmd_buf = pcm
                self._cmd_voiced = 0.03
            return None

        # Accumulate speech
        self._cmd_buf = np.concatenate([self._cmd_buf, pcm])
        self._cmd_voiced += 0.03
        elapsed = time.perf_counter() - self._cmd_start

        # Commit: >=1 s of audio OR hit the cap
        if len(self._cmd_buf) >= 16000 or elapsed > COMMAND_WINDOW:
            cmd = self._cmd_buf.copy()
            self._go_idle()
            if not self._passes_command_gate(cmd):
                return None
            return cmd

        return None

    # ------------------------------------------------------------------
    def _passes_command_gate(self, cmd: np.ndarray) -> bool:
        """Reject captured audio that is not a real command (TV, radio, etc.).

        Uses the optional ASR transcriber. If no transcriber is available we
        always pass (graceful degradation to the old energy-only behavior).
        A command passes when the transcript contains at least one known
        command keyword. Empty transcripts (pure noise/music) fail the gate.
        """
        if self.transcriber is None or not getattr(self.transcriber, "available", False):
            return True
        try:
            text = self.transcriber.transcribe_pcm(cmd, sr=OWW_SR)
        except Exception:
            return True  # ASR hiccup -> don't drop a possibly-real command
        text_l = text.lower()
        if not self.quiet:
            print(f"\033[90m[gate] heard: {text!r}\033[0m")
        if not text_l.strip():
            return False  # silence / music / non-speech -> ignore
        # Split contractions ("what's" -> "what", "don't" -> "do") so the
        # keyword match isn't defeated by apostrophes from ASR.
        norm = (text_l.replace("n't", " not ")
                       .replace("'s", " s")
                       .replace("'re", " re")
                       .replace("'ll", " will")
                       .replace("'ve", " have")
                       .replace("'", " "))
        words = set(norm.replace(",", " ").replace(".", " ").split())
        return bool(words & CMD_KEYWORDS)

    # ------------------------------------------------------------------
    def _go_idle(self):
        self.state = self.IDLE
        self._cmd_buf = np.array([], dtype=np.int16)
        self._cmd_voiced = 0.0
        self._cmd_start = None
        self._sample_buf = np.array([], dtype=np.int16)
        self._drain = 0

    # ------------------------------------------------------------------
    @property
    def is_idle(self) -> bool:
        return self.state == self.IDLE

    @property
    def status_line(self) -> str:
        if self.state == self.IDLE:
            return "🔇 idle — say a wakeword"
        else:
            return "🎤 listening for command..."
