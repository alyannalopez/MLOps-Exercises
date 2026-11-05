"""Simulated smart-home device (ported from ME2 runtime/device.py).

Every actuator that would need real hardware (lights, bulbs, thermostat,
speaker, phone) is simulated in memory. Time and date come from the local
system clock (datetime.now()), so the device answers "what time is it?"
correctly with zero network. Weather is served by weather.weather_response().

Adaptations vs ME2 for the ME3 full-duplex voice loop:
  * MUSIC is SIMULATED (track state + title announcement), not played through
    afplay/aplay. Playing a real file would open a second audio output stream
    and fight the duplex playback path (the very static/latency issue we are
    avoiding). The track still "changes" in state, so pause/stop/next behave.
  * No signal/atexit handlers — ME2 registered them to reap an afplay child
    process; with no real player there is nothing to reap.

The device keeps a public `state` dict so tests can inspect exactly what
changed, and every action returns a human-readable response string (what a
real device would speak over the speaker).
"""
from __future__ import annotations

import random
import re
import threading
import time
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Demo track catalogue (brief tasks 1 & 8). Titles only — no audio is played.
# ---------------------------------------------------------------------------
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
    return _TRACK_TITLES.get(stem, stem)


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
        w = words[i].strip(".,")
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
    'wake me up at 8 a.m.', 'alarm 9 p.m.', 'at eight am'."""
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
        m = re.search(
            r"(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
            r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
            r"nineteen|twenty|thirty|forty|fifty|sixty|ninety)\s*"
            r"(second|sec|minute|min|hour|hr)s?", t)
        if m:
            n = _NUM_WORDS.get(m.group(1))
            if n is None:
                return None
            unit = m.group(2)
        else:
            return None
    else:
        n = int(m.group(1))
        unit = m.group(2)
    mult = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600}
    return n * mult[unit]


# ---------------------------------------------------------------------------
# The device
# ---------------------------------------------------------------------------
class SimulatedSmartHome:
    """In-memory smart home. All state is public; every action returns a
    human-readable response string."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_id = 1
        self.state: dict = {
            "lights_on": False,
            "brightness": 100,     # 0..100
            "color": "white",
            "temperature_c": 24,   # current setpoint
            "volume": 50,          # 0..100
            "music": {"playing": False, "paused": False, "track": None},
            "timers": {},          # id -> {"label":..., "fire_at": monotonic}
            "alarms": {},          # id -> {"label":..., "hour":..., "minute":...}
            "reminders": [],       # list of {"id":..., "text":...}
            "calls": [],           # log of simulated calls
            "messages": [],        # log of simulated messages
        }
        self.events: list[str] = []

    # -- bookkeeping -------------------------------------------------------
    def _log(self, msg: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.events.append(f"[{stamp}] {msg}")

    def _id(self) -> int:
        with self._lock:
            i = self._next_id
            self._next_id += 1
            return i

    # -- 1. music ----------------------------------------------------------
    def play_music(self, phrase: str = "") -> str:
        track = random.choice(_MUSIC_TRACKS)
        title = _track_title(track)
        self.state["music"].update(playing=True, paused=False, track=track)
        self._log(f"music PLAY: {title}")
        return f"Now playing {title}."

    # -- 8. media controls -------------------------------------------------
    def pause(self) -> str:
        if not self.state["music"]["playing"]:
            return "Nothing is playing."
        self.state["music"]["paused"] = True
        self._log("music PAUSE")
        return "Paused."

    def resume(self) -> str:
        if not self.state["music"]["playing"]:
            return "Nothing is playing."
        self.state["music"]["paused"] = False
        self._log("music RESUME")
        return "Resuming."

    def stop(self) -> str:
        if not self.state["music"]["playing"]:
            return "Nothing is playing."
        self.state["music"].update(playing=False, paused=False, track=None)
        self._log("music STOP")
        return "Stopped."

    def next_track(self) -> str:
        if not self.state["music"]["playing"]:
            return "Nothing is playing."
        cur = self.state["music"]["track"]
        pool = [s for s in _MUSIC_TRACKS if s != cur]
        nxt = random.choice(pool) if pool else random.choice(_MUSIC_TRACKS)
        title = _track_title(nxt)
        self.state["music"]["track"] = nxt
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

    # -- 3. lights ---------------------------------------------------------
    def light_on(self) -> str:
        self.state["lights_on"] = True
        self._log("lights ON")
        return "Lights on."

    def light_off(self) -> str:
        self.state["lights_on"] = False
        self._log("lights OFF")
        return "Lights off."

    # -- 4. brightness / color --------------------------------------------
    def set_brightness(self, phrase: str) -> str:
        n = parse_number(phrase)
        if n is None:
            return "How bright should I make it?"
        n = max(0, min(100, n))
        self.state["brightness"] = n
        self.state["lights_on"] = True
        self._log(f"brightness -> {n}")
        return f"Setting brightness to {n} percent."

    def set_color(self, phrase: str) -> str:
        c = parse_color(phrase)
        if c is None:
            return "Which color would you like?"
        self.state["color"] = c
        self.state["lights_on"] = True
        self._log(f"color -> {c}")
        return f"Setting the lights to {c}."

    # -- 5. timers ---------------------------------------------------------
    def set_timer(self, phrase: str) -> str:
        secs = parse_duration(phrase)
        if secs is None:
            return "How long should the timer run?"
        tid = self._id()
        self.state["timers"][tid] = {
            "label": phrase.strip(),
            "fire_at": time.monotonic() + secs,
        }
        self._log(f"timer #{tid} set for {secs}s")
        unit = "second" if secs == 1 else "seconds"
        return f"Timer set for {secs} {unit}."

    # -- 6. alarms ---------------------------------------------------------
    def set_alarm(self, phrase: str) -> str:
        tm = parse_time_of_day(phrase)
        if tm is None:
            return "What time should the alarm be set for?"
        h, mn = tm
        aid = self._id()
        self.state["alarms"][aid] = {"label": phrase.strip(), "hour": h, "minute": mn}
        self._log(f"alarm #{aid} set for {h:02d}:{mn:02d}")
        ap = "AM" if h < 12 else "PM"
        disp_h = h % 12 or 12
        return f"Alarm set for {disp_h}:{mn:02d} {ap}."

    # -- 7. thermostat -----------------------------------------------------
    def set_temperature(self, phrase: str) -> str:
        n = parse_number(phrase)
        if n is None:
            return "What temperature should I set it to?"
        n = max(10, min(40, n))
        self.state["temperature_c"] = n
        self._log(f"temperature -> {n}C")
        return f"Setting the temperature to {n} degrees."

    # -- 9. reminders / lists ---------------------------------------------
    def create_reminder(self, phrase: str) -> str:
        # strip leading trigger words
        text = re.sub(r"^\s*(remind me to|remind me|reminder to|add reminder to)\s*",
                      "", phrase, flags=re.IGNORECASE).strip()
        if not text:
            text = phrase.strip()
        rid = self._id()
        self.state["reminders"].append({"id": rid, "text": text})
        self._log(f"reminder #{rid}: {text}")
        return f"Got it, I'll remind you to {text}."

    def list_reminders(self) -> str:
        rs = self.state["reminders"]
        if not rs:
            return "You have no reminders."
        items = "; ".join(r["text"] for r in rs)
        return f"You have {len(rs)} reminders: {items}."

    # -- 10. calls / messages ----------------------------------------------
    def call(self, phrase: str) -> str:
        contact = re.sub(r"^\s*(call|phone|dial|ring)\s+(me\s+)?", "",
                         phrase, flags=re.IGNORECASE).strip() or "contact"
        self.state["calls"].append(contact)
        self._log(f"CALL (simulated) -> {contact}")
        return f"Calling {contact}."

    def message(self, phrase: str) -> str:
        m = re.search(r"\bto\b\s+(.+)$", phrase, flags=re.IGNORECASE)
        contact = m.group(1).strip() if m else "contact"
        self.state["messages"].append(contact)
        self._log(f"MESSAGE (simulated) -> {contact}")
        return f"Sending a message to {contact}."

    # -- 2. time / date (local resources, no network) ----------------------
    def tell_time(self) -> str:
        now = datetime.now()
        return "It's " + now.strftime("%-I:%M %p") + "."

    def tell_date(self) -> str:
        now = datetime.now()
        return "Today is " + now.strftime("%A, %B %-d, %Y") + "."

    # -- periodic tick (called by the runtime loop) ------------------------
    def tick(self) -> list[str]:
        """Check timers/alarms against the clock; returns responses due now."""
        due: list[str] = []
        now_mono = time.monotonic()
        for tid in list(self.state["timers"]):
            t = self.state["timers"][tid]
            if now_mono >= t["fire_at"]:
                del self.state["timers"][tid]
                self._log(f"timer #{tid} FIRED")
                due.append("Your timer is done.")
        now_dt = datetime.now()
        for aid in list(self.state["alarms"]):
            a = self.state["alarms"][aid]
            if (now_dt.hour, now_dt.minute) == (a["hour"], a["minute"]):
                del self.state["alarms"][aid]
                self._log(f"alarm #{aid} FIRED")
                due.append("Your alarm is going off.")
        return due
