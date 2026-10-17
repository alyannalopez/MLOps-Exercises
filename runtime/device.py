"""Simulated smart-home device.

Every actuator that would need real hardware (lights, bulbs, thermostat,
speakers, display, cellular/radio for calls and messages) is simulated in
memory. Time and date come from the local system clock (datetime.now()),
so the device answers "what time is it?" correctly with zero network.

The device keeps a public `state` dict so tests and the demo can inspect
exactly what changed, and every action returns a human-readable response
string (what a real device would speak over the speaker).
"""
from __future__ import annotations

import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

# Copyright-free demo tracks (brief tasks 1 & 8). Synthesized by
# scripts/generate_demo_mp3s.py; the runtime plays whichever of .mp3/.wav
# exists, so real MP3s can be dropped in later with no code change.
# Real songs live in pi_bundle/assets/music (the bundle ships them there); the
# repo-root assets/music is a fallback. Search both so playback works whether
# you run from the repo root or from inside pi_bundle.
def _find_music_dir() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "pi_bundle" / "assets" / "music",
        here.parent / "assets" / "music",
        here.parent.parent / "pi_bundle" / "assets" / "music",
        here.parent.parent / "assets" / "music",
    ]
    for c in candidates:
        if c.is_dir() and any(c.glob("*.mp3")):
            return c
    return candidates[0]


_MUSIC_DIR = _find_music_dir()
_MUSIC_TRACKS = [
    "track1_relaxing",
    "track2_upbeat",
    "track3_lofi",
    "track4_demo_song",
    "track5_alarm",
    "na_para_bang",
]

_TRACK_TITLES: dict[str, str] = {
    "track1_relaxing": "Isang Daan – Cup of Joe",
    "track2_upbeat":   "Kung Payagan – Cup of Joe",
    "track3_lofi":     "Lalayo – Cup of Joe",
    "track4_demo_song":"Pamahiin – Cup of Joe",
    "track5_alarm":    "Sandali – Cup of Joe",
    "na_para_bang":    "Na Para Bang – Mariah Deborah",
}


def _track_title(stem: str) -> str:
    """Human-readable song title for a track stem."""
    return _TRACK_TITLES.get(stem, stem)


def _pick_audio_file(stem: str) -> Path | None:
    for ext in (".mp3", ".wav"):
        p = _MUSIC_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _audio_player_cmd(path: Path) -> list[str] | None:
    """Return the OS command to play `path`, or None if no player is found."""
    if path.suffix == ".wav":
        if shutil.which("afplay"):          # macOS
            return ["afplay", str(path)]
        if shutil.which("aplay"):           # Linux / Pi
            return ["aplay", "-q", str(path)]
    elif path.suffix == ".mp3":
        # afplay (macOS) decodes MP3 natively and is always present, so try it
        # first; fall back to mpg123/ffplay on Linux/Pi.
        if shutil.which("afplay"):
            return ["afplay", str(path)]
        for cmd in (["mpg123", "-q", str(path)], ["ffplay", "-nodisp",
                 "-autoexit", str(path)]):
            if shutil.which(cmd[0]):
                return cmd
    return None

# ---------------------------------------------------------------------------
# Slot parsing helpers (regex over the spoken phrase; no NLU service)
# ---------------------------------------------------------------------------

_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "ninety": 90, "hundred": 100,
}


def parse_number(text: str) -> int | None:
    """Extract a small integer from a phrase ('turn it up to twenty five')."""
    digits = re.findall(r"\d+", text)
    if digits:
        return int(digits[0])
    words = text.lower().split()
    total, i = 0, 0
    while i < len(words):
        w = words[i].strip(",.")
        if w in _NUM_WORDS:
            v = _NUM_WORDS[w]
            if v == 100 and i + 1 < len(words) and words[i + 1] in _NUM_WORDS:
                total += v * _NUM_WORDS[words[i + 1]]
                i += 2
                continue
            total += v
        i += 1
    return total if total > 0 else None


def parse_time_of_day(text: str) -> tuple[int, int] | None:
    """'Alarm 6:00 AM', 'Wake me up at 6:00 AM', 'at 7', 'at 12:30 pm',
    'wait me up at 8 a.m.', 'alert 9 p.m.', 'at eight am'."""
    t = text.lower().replace("a.m.", "am").replace("p.m.", "pm")
    # explicit H:MM with optional am/pm
    m = re.search(r"(\d{1,2}):(\d{2})\s*(am|pm)?", t)
    if m:
        h, mn, ap = int(m.group(1)), int(m.group(2)), m.group(3)
        if ap == "pm" and h < 12:
            h += 12
        if ap == "am" and h == 12:
            h = 0
        if 0 <= h <= 23 and 0 <= mn <= 59:
            return h, mn
    # bare hour (digit or word), optional am/pm: 'at seven', '6 am', '8 a.m.'
    hour_words = (r"zero|one|two|three|four|five|six|seven|eight|nine|"
                  r"ten|eleven|twelve")
    m = re.search(rf"\b(\d{{1,2}}|{hour_words})\s*(am|pm)\b", t)
    if m:
        h = int(m.group(1)) if m.group(1).isdigit() else _NUM_WORDS.get(m.group(1))
        if h is None:
            return None
        ap = m.group(2)
        if ap == "pm" and h < 12:
            h += 12
        if ap == "am" and h == 12:
            h = 0
        if 0 <= h <= 23:
            return h, 0
    # bare hour without am/pm: 'at 7', 'at seven'
    m = re.search(rf"\bat\s+(\d{{1,2}}|{hour_words})\b", t)
    if m:
        h = int(m.group(1)) if m.group(1).isdigit() else _NUM_WORDS.get(m.group(1))
        if h is not None and 0 <= h <= 23:
            return h, 0
    return None


_COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "white",
           "pink", "cyan", "violet"]


def parse_color(text: str) -> str | None:
    t = text.lower()
    for c in _COLORS:
        if c in t:
            return c
    return None


def parse_duration(text: str) -> int | None:
    """'Timer 10 seconds', 'Countdown for 1 minute', 'in five minutes' -> seconds."""
    t = text.lower()
    m = re.search(r"(\d+)\s*(second|sec|minute|min|hour|hr)s?", t)
    if not m:
        m = re.search(r"(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|ninety)\s*(second|sec|minute|min|hour|hr)s?", t)
        if not m:
            return None
        n = _NUM_WORDS.get(m.group(1))
        if n is None:
            return None
    else:
        n = int(m.group(1))
    unit = m.group(2)
    if unit.startswith("sec"):
        return n
    if unit.startswith("min"):
        return n * 60
    return n * 3600


# ---------------------------------------------------------------------------
# Simulated device
# ---------------------------------------------------------------------------

@dataclass
class _Timer:
    id: int
    label: str
    fire_at: float  # time.monotonic()
    seconds: int


@dataclass
class _Alarm:
    id: int
    label: str
    hour: int
    minute: int


@dataclass
class _Reminder:
    id: int
    text: str
    due: datetime | None


@dataclass
class _Call:
    contact: str
    incoming: bool
    answered: bool = False


class SimulatedSmartHome:
    """In-memory smart home. All side effects are recorded in `state` and in
    the `events` log (useful for the demo transcript and for tests)."""

    def __init__(self):
        self.state: dict = {
            "lights": False,
            "brightness": 80,          # 0-100
            "color": "white",
            "thermostat": {"mode": "off", "target_c": 22.0, "current_c": 24.0},
            "volume": 50,              # 0-100
            "music": {"playing": False, "track": None, "paused": False},
            "timers": {},
            "alarms": {},
            "reminders": [],
            "calls": [],
            "messages": [],
        }
        self.events: list[str] = []
        self._next_id = 1
        self._lock = threading.Lock()
        self._player_proc: subprocess.Popen | None = None
        self._install_cleanup()

    def _install_cleanup(self) -> None:
        """Ensure the audio player is killed when the process exits —
        Ctrl-C, SIGTERM, normal return, or crash. Without this, afplay
        becomes an orphan and keeps playing after the demo is gone."""
        import atexit

        def _shutdown():
            self._stop_player()
            self.restore_volume()

        atexit.register(_shutdown)

        # Also catch SIGINT/SIGTERM so the player dies immediately,
        # before the interpreter begins its shutdown sequence.
        def _sig_handler(signum, frame):
            _shutdown()
            sys.exit(0)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _sig_handler)
            except (OSError, ValueError):
                pass  # not in main thread (e.g. test runner) — atexit still covers us

    # -- bookkeeping -------------------------------------------------------
    def _log(self, msg: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        self.events.append(line)

    def _id(self) -> int:
        with self._lock:
            i = self._next_id
            self._next_id += 1
        return i

    # -- lights / brightness / color ---------------------------------------
    def light_on(self) -> str:
        self.state["lights"] = True
        self._log(f"lights ON (brightness {self.state['brightness']}%, color {self.state['color']})")
        return "Turning the lights on."

    def light_off(self) -> str:
        self.state["lights"] = False
        self._log("lights OFF")
        return "Turning the lights off."

    def set_brightness(self, phrase: str) -> str:
        n = parse_number(phrase)
        if n is None:
            t = phrase.lower()
            if "up" in t or "increase" in t or "brighter" in t:
                n = min(100, self.state["brightness"] + 10)
            elif "down" in t or "decrease" in t or "dimmer" in t or "dim" in t:
                n = max(0, self.state["brightness"] - 10)
            else:
                return "Sorry, I couldn't hear a level. Try 'set brightness to fifty'."
        self.state["brightness"] = max(0, min(100, n))
        self.state["lights"] = True
        self._log(f"brightness -> {self.state['brightness']}%")
        return f"Setting brightness to {self.state['brightness']} percent."

    def set_color(self, phrase: str) -> str:
        c = parse_color(phrase)
        if c is None:
            return "Sorry, I didn't catch a color. Try 'set the lights to blue'."
        self.state["color"] = c
        self.state["lights"] = True
        self._log(f"color -> {c}")
        return f"Setting the lights to {c}."

    # -- thermostat ---------------------------------------------------------
    def set_temperature(self, phrase: str) -> str:
        t = phrase.lower()
        m = re.search(r"(\d{1,3}(?:\.\d)?)\s*(degrees?|deg|c|celsius)?", t)
        if m:
            target = float(m.group(1))
        else:
            n = parse_number(phrase)
            target = float(n) if n is not None else None
        if target is None:
            return "Sorry, I couldn't hear a temperature. Try 'set the thermostat to twenty two'."
        if "off" in t:
            self.state["thermostat"]["mode"] = "off"
            self._log("thermostat OFF")
            return "Turning the thermostat off."
        self.state["thermostat"]["target_c"] = target
        self.state["thermostat"]["mode"] = "cool" if "cool" in t or target < self.state["thermostat"]["current_c"] else "heat"
        self._log(f"thermostat -> {target}°C ({self.state['thermostat']['mode']})")
        return f"Setting the thermostat to {target:g} degrees."

    # -- media ---------------------------------------------------------------
    def _play_audio_bg(self, path: Path, track_label: str) -> bool:
        """Launch a non-blocking audio player for `path`. Returns True if a
        player was found and started. Playback runs in a daemon thread so the
        REPL / mic loop stays responsive; stop() kills the current player."""
        cmd = _audio_player_cmd(path)
        if cmd is None:
            return False
        try:
            self._player_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            return False
        self._log(f"audio PLAYING {path.name} ({track_label})")
        return True

    def _stop_player(self) -> None:
        proc = getattr(self, "_player_proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._player_proc = None

    # -- volume ducking (so the mic can hear the wake word over music) ------
    def duck_volume(self) -> None:
        """Lower system audio so the mic can pick up the wake word while
        music is playing. Saves the original volume for restore_volume()."""
        if sys.platform != "darwin":
            return
        if getattr(self, "_ducked", False):
            return
        try:
            # Read current output volume (0-100)
            out = subprocess.check_output(
                ["osascript", "-e",
                 "output volume of (get volume settings)"],
                text=True, timeout=3).strip()
            self._saved_volume = int(out)
            # Duck to 15% — quiet enough for the mic to hear speech,
            # still audible so the user knows music is playing.
            subprocess.run(
                ["osascript", "-e",
                 "set volume output volume 15"],
                timeout=3, check=False)
            self._ducked = True
            self._log(f"volume ducked {self._saved_volume}% -> 15%")
        except Exception:
            pass

    def restore_volume(self) -> None:
        """Restore system audio to the pre-duck level."""
        if sys.platform != "darwin":
            return
        if not getattr(self, "_ducked", False):
            return
        vol = getattr(self, "_saved_volume", 50)
        try:
            subprocess.run(
                ["osascript", "-e",
                 f"set volume output volume {vol}"],
                timeout=3, check=False)
            self._log(f"volume restored -> {vol}%")
        except Exception:
            pass
        self._ducked = False

    def play_music(self, phrase: str) -> str:
        m = re.search(r"(?:play|start)\s+(?:some\s+|the\s+)?(.+)", phrase.lower())
        requested = m.group(1).strip(" .!") if m else ""
        # Pick a random copyright-free demo track (brief: "play randomly").
        stem = random.choice(_MUSIC_TRACKS)
        title = _track_title(stem)
        path = _pick_audio_file(stem)
        self.state["music"].update(playing=True, paused=False, track=stem)
        if path is not None and self._play_audio_bg(path, title):
            self._log(f"music PLAYING: {title}")
        else:
            # No player available (headless / CI) -> pure simulation.
            self._log(f"music PLAYING (simulated): {title}")
        return f"Now playing: {title}"

    def pause(self) -> str:
        if not self.state["music"]["playing"]:
            return "Nothing is playing."
        self._stop_player()
        self.state["music"]["paused"] = True
        self._log("music PAUSED")
        return "Paused."

    def stop(self) -> str:
        self._stop_player()
        self.state["music"].update(playing=False, paused=False, track=None)
        self._log("music STOPPED")
        return "Stopped."

    def next_track(self) -> str:
        if not self.state["music"]["playing"]:
            return "Nothing is playing."
        cur = self.state["music"]["track"]
        pool = [s for s in _MUSIC_TRACKS if s != cur]
        nxt = random.choice(pool) if pool else random.choice(_MUSIC_TRACKS)
        title = _track_title(nxt)
        path = _pick_audio_file(nxt)
        self.state["music"]["track"] = nxt
        if path is not None:
            self._stop_player()
            self._play_audio_bg(path, title)
        self._log(f"music NEXT: {title}")
        return f"Now playing: {title}"

    def volume_up(self) -> str:
        self.state["volume"] = min(100, self.state["volume"] + 10)
        self._log(f"volume -> {self.state['volume']}")
        return f"Volume up, now at {self.state['volume']}."

    def volume_down(self) -> str:
        self.state["volume"] = max(0, self.state["volume"] - 10)
        self._log(f"volume -> {self.state['volume']}")
        return f"Volume down, now at {self.state['volume']}."

    # -- timers / alarms ------------------------------------------------------
    def set_timer(self, phrase: str) -> str:
        secs = parse_duration(phrase)
        if secs is None:
            return "Sorry, I couldn't hear how long. Try 'set a timer for five minutes'."
        tid = self._id()
        self.state["timers"][tid] = _Timer(tid, phrase.strip(), time.monotonic() + secs, secs)
        self._log(f"timer #{tid} set for {secs}s")
        return f"Timer set for {_human_dur(secs)}."

    def set_alarm(self, phrase: str) -> str:
        hm = parse_time_of_day(phrase)
        if hm is None:
            return "Sorry, I couldn't hear a time. Try 'set an alarm for seven am'."
        aid = self._id()
        self.state["alarms"][aid] = _Alarm(aid, phrase.strip(), hm[0], hm[1])
        self._log(f"alarm #{aid} set for {hm[0]:02d}:{hm[1]:02d}")
        return f"Alarm set for {hm[0]:02d}:{hm[1]:02d}."

    def cancel_timers(self) -> str:
        n = len(self.state["timers"])
        self.state["timers"].clear()
        self._log(f"{n} timer(s) cancelled")
        return f"Cancelled {n} timer{'s' if n != 1 else ''}."

    # -- reminders / lists -----------------------------------------------------
    def create_reminder(self, phrase: str) -> str:
        t = phrase.lower()
        # strip leading command words to get the reminder body
        body = re.sub(r"^(please\s+)?(create\s+a\s+reminder|remind\s+me\s+to|reminder)\s*", "", t)
        body = body.strip(" .!")
        if not body:
            body = "something"
        due = None
        dm = re.search(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", t)
        if dm:
            h = int(dm.group(1))
            ap = dm.group(3)
            if ap == "pm" and h < 12:
                h += 12
            due = datetime.now().replace(hour=h, minute=int(dm.group(2) or 0), second=0)
        elif "tomorrow" in t:
            due = datetime.now() + timedelta(days=1)
        rid = self._id()
        self.state["reminders"].append(_Reminder(rid, body, due))
        self._log(f"reminder #{rid}: {body} (due {due})")
        when = f" at {due.strftime('%H:%M')}" if due else ""
        return f"Okay, I'll remind you to {body}{when}."

    def list_reminders(self) -> str:
        rs = self.state["reminders"]
        if not rs:
            return "You have no reminders."
        parts = [f"{r.text}" + (f" at {r.due.strftime('%H:%M')}" if r.due else "") for r in rs]
        return "Your reminders: " + "; ".join(parts) + "."

    # -- calls / messages (simulated radio) ------------------------------------
    def call(self, phrase: str) -> str:
        t = phrase.lower()
        m = re.search(r"call\s+(?:to\s+|the\s+)?([a-z]+)", t)
        contact = m.group(1).strip(" .!") if m and m.group(1) not in ("a", "me", "phone", "number") else "home"
        self.state["calls"].append(_Call(contact, incoming=False))
        self._log(f"CALL (simulated) -> {contact}")
        return f"Calling {contact}."

    def message(self, phrase: str) -> str:
        t = phrase.lower()
        m = re.search(r"message\s+(?:to\s+)?([a-z]+)", t)
        contact = m.group(1).strip(" .!") if m and m.group(1) not in ("a", "me", "my") else "home"
        self.state["messages"].append({"to": contact, "body": phrase, "ts": datetime.now()})
        self._log(f"MESSAGE (simulated) -> {contact}")
        return f"Sending a message to {contact}."

    def simulate_incoming_call(self, contact: str = "Mom") -> str:
        self.state["calls"].append(_Call(contact, incoming=True))
        self._log(f"INCOMING CALL (simulated) <- {contact}")
        return f"Incoming call from {contact}."

    # -- time / date (local resources, no network) ------------------------------
    def tell_time(self) -> str:
        now = datetime.now()
        return "It's " + now.strftime("%-I:%M %p") + "."

    def tell_date(self) -> str:
        now = datetime.now()
        return "Today is " + now.strftime("%A, %B %-d, %Y") + "."

    # -- periodic tick (called by the runtime loop) --------------------------------
    def tick(self) -> list[str]:
        """Check timers/alarms against the clock; returns responses due now."""
        due: list[str] = []
        now_mono = time.monotonic()
        for tid in list(self.state["timers"]):
            t = self.state["timers"][tid]
            if now_mono >= t.fire_at:
                del self.state["timers"][tid]
                self._log(f"timer #{tid} FIRED")
                due.append("Your timer is done.")
        now_dt = datetime.now()
        for aid in list(self.state["alarms"]):
            a = self.state["alarms"][aid]
            if (now_dt.hour, now_dt.minute) == (a.hour, a.minute) and a.id not in getattr(self, "_alarmed", set()):
                self._alarmed = getattr(self, "_alarmed", set()) | {a.id}
                self._log(f"alarm #{aid} RANG")
                due.append(f"It's {a.hour:02d}:{a.minute:02d}. Your alarm is going off.")
        return due

    # -- snapshot for reporting ----------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "lights": self.state["lights"],
            "brightness": self.state["brightness"],
            "color": self.state["color"],
            "thermostat": dict(self.state["thermostat"]),
            "volume": self.state["volume"],
            "music": dict(self.state["music"]),
            "n_timers": len(self.state["timers"]),
            "n_alarms": len(self.state["alarms"]),
            "n_reminders": len(self.state["reminders"]),
            "n_calls": len(self.state["calls"]),
            "n_messages": len(self.state["messages"]),
        }


def _human_dur(secs: int) -> str:
    if secs % 3600 == 0:
        return f"{secs // 3600} hour{'s' if secs // 3600 != 1 else ''}"
    if secs % 60 == 0:
        return f"{secs // 60} minute{'s' if secs // 60 != 1 else ''}"
    return f"{secs} seconds"
