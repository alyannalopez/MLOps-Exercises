#!/usr/bin/env python3
"""Slot-filling for the VCM: turn a spoken command's transcript into a number.

The 340 KB INT8 intent model only says WHICH intent ("SET_TIMER"); it does not
transcribe the digits. This module takes the transcript produced by a small
ASR pass (whisper.cpp) and extracts a concrete duration:

    "set a timer for 5 minutes"   -> {"value": 5,  "unit": "minute", "seconds": 300}
    "remind me in ten minutes"    -> {"value": 10, "unit": "minute", "seconds": 600}
    "alarm at 7"                  -> {"time": "07:00", "seconds": None}

Design goals:
  * stdlib only (re) -- runs on the Pi with zero extra packages
  * forgiving of ASR typos ("five" vs "5", "mins" vs "minutes", "hr" vs "hour")
  * returns None / {} when nothing is understood so callers can fall back
    gracefully (e.g. a default 5-minute timer) rather than guessing wildly
"""
from __future__ import annotations
import re
from typing import Optional

# --------------------------------------------------------------------------- #
#  Word -> digit (covers the digits people actually say in commands)          #
# --------------------------------------------------------------------------- #
_WORD_NUM = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100,
}

# --------------------------------------------------------------------------- #
#  Unit synonyms -> canonical unit                                            #
# --------------------------------------------------------------------------- #
_UNIT_CANON = {
    "sec": "second", "secs": "second", "secnd": "second", "second": "second",
    "seconds": "second",
    "min": "minute", "mins": "minute", "minit": "minute", "minute": "minute",
    "minutes": "minute",
    "hr": "hour", "hrs": "hour", "hour": "hour", "hours": "hour",
    "h": "hour",
    "day": "day", "days": "day",
}

_UNIT_SECONDS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}

# "half hour" / "quarter hour" style phrases -> (value, unit)
_FRAC = {
    "half hour": (30, "minute"),
    "quarter hour": (15, "minute"),
    "half an hour": (30, "minute"),
    "a quarter": (15, "minute"),
}


def _norm(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _words_to_num(tokens: list[str]) -> Optional[int]:
    """Convert a small run of number words to an int.

    Handles "five", "twenty five", "one hundred twenty", "one hundred".
    Returns None if the run isn't a recognizable number.
    """
    if not tokens:
        return None
    # pure-digit token?
    if all(t.isdigit() for t in tokens):
        return int("".join(tokens))
    total, current = 0, 0
    valid = True
    for t in tokens:
        if t in _WORD_NUM:
            n = _WORD_NUM[t]
            if n == 100:
                if current == 0:
                    current = 1
                total += current * 100
                current = 0
            elif n >= 20:  # tens
                current += n
            else:          # units
                current += n
        else:
            valid = False
            break
    if not valid:
        return None
    return total + current if (total + current) > 0 else None


def parse_duration(transcript: str) -> Optional[dict]:
    """Extract a duration (value, unit, seconds) from a transcript.

    Returns None if no duration is found.
    """
    t = _norm(transcript)
    if not t:
        return None

    # 1) fractional-hour phrases (checked first, they don't use a bare number)
    for phrase, (val, unit) in _FRAC.items():
        if phrase in t:
            return {"value": val, "unit": unit, "seconds": val * _UNIT_SECONDS[unit],
                    "source": f"'{phrase}'"}

    # 2) find a number followed (within a couple tokens) by a unit word
    toks = t.split()
    for i, tok in enumerate(toks):
        # candidate number start: digit or number-word
        if not (tok.isdigit() or tok in _WORD_NUM):
            continue
        # collect the number run (digits stay glued; words run until non-num)
        num_tokens = [tok]
        j = i + 1
        while j < len(toks) and toks[j] in _WORD_NUM:
            num_tokens.append(toks[j]); j += 1
        value = _words_to_num(num_tokens)
        if value is None or value <= 0:
            continue
        # look ahead up to 2 tokens for a unit
        unit = None
        for k in range(j, min(j + 3, len(toks))):
            if toks[k] in _UNIT_CANON:
                unit = _UNIT_CANON[toks[k]]
                break
        if unit is None:
            # bare number with no unit -> assume minutes for timers/alarms
            # (most common; caller can override)
            unit = "minute"
        return {"value": value, "unit": unit,
                "seconds": value * _UNIT_SECONDS[unit],
                "source": " ".join(num_tokens)}

    return None


def parse_clock_time(transcript: str) -> Optional[dict]:
    """Extract a clock time for alarms: '7', '7 pm', '7:30', 'seven thirty'.

    Returns {"time": "HH:MM", "seconds": None} or None.
    """
    t = _norm(transcript)
    if not t:
        return None

    # "4am" / "8PM" / "7pm" -- whisper often glues the meridiem to the hour.
    # Handle BEFORE the generic H MM pass so "4am" isn't missed as a glued pair.
    m = re.search(r"\b(\d{1,2})(am|pm)\b", t)
    if m:
        h, mer = int(m.group(1)), m.group(2)
        if 1 <= h <= 12:
            if mer == "pm" and h < 12:
                h += 12
            if mer == "am" and h == 12:
                h = 0
            return {"time": f"{h:02d}:00", "seconds": None}

    # explicit H:MM
    m = re.search(r"(\d{1,2}):(\d{2})", t)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 23 and 0 <= mi <= 59:
            return {"time": f"{h:02d}:{mi:02d}", "seconds": None}

    # "9 p.m." / "8 a.m." -- whisper writes the meridiem with periods;
    # _norm turned them into "p m" / "a m" tokens. Re-glue them FIRST so the
    # am/pm detection below sees "pm"/"am".
    t = re.sub(r"\bp m\b", "pm", t)
    t = re.sub(r"\ba m\b", "am", t)

    # meridiem?
    pm = bool(re.search(r"\bpm\b", t))
    am = bool(re.search(r"\bam\b", t))

    # H MM (e.g. "seven thirty", "7 30")
    toks = t.split()
    for i, tok in enumerate(toks):
        h = _words_to_num([tok]) if (tok in _WORD_NUM) else (int(tok) if tok.isdigit() else None)
        if h is None or h > 24:
            continue
        mi = 0
        if i + 1 < len(toks):
            nxt = toks[i + 1]
            if nxt.isdigit() and nxt[0] != "0" or (nxt.isdigit()):
                cand = int(nxt)
                if 0 <= cand <= 59:
                    mi = cand
            elif nxt in _WORD_NUM and _WORD_NUM[nxt] < 60:
                mi = _WORD_NUM[nxt]
        # apply am/pm
        if pm and h < 12:
            h += 12
        if am and h == 12:
            h = 0
        if 0 <= h <= 23:
            return {"time": f"{h:02d}:{mi:02d}", "seconds": None}
    return None


def parse_percent(transcript: str) -> Optional[dict]:
    """Extract a brightness/level percentage from a transcript.

    Handles:
      * "40 percent", "40%", "set to 40"
      * word numbers: "dim to fifty percent", "set to ninety"
      * relative: "brighter", "dimmer", "louder", "quieter"  -> direction only
      * "max"/"full" -> 100, "half" -> 50

    Returns {"percent": int} or {"direction": "up"|"down"} or None.
    """
    t = _norm(transcript)
    if not t:
        return None

    # explicit percent sign or word
    m = re.search(r"(\d{1,3})\s*(?:percent|%)", t)
    if m:
        return {"percent": max(0, min(100, int(m.group(1))))}

    # word-number + percent
    toks = t.split()
    for i, tok in enumerate(toks):
        if tok in _WORD_NUM and _WORD_NUM[tok] <= 100:
            # look ahead for "percent"
            for k in range(i + 1, min(i + 3, len(toks))):
                if toks[k] in ("percent", "pct"):
                    return {"percent": _WORD_NUM[tok]}
            # bare word-number in a brightness context -> treat as percent
            if any(w in t for w in ("dim", "bright", "set", "to", "level")):
                return {"percent": _WORD_NUM[tok]}

    # bare digit in a brightness context ("set to 40", "dim to 40")
    m = re.search(r"\b(\d{1,3})\b", t)
    if m and any(w in t for w in ("dim", "bright", "set", "to", "level")):
        return {"percent": max(0, min(100, int(m.group(1))))}

    # relative directions (adjectives first, then bare up/down in command context)
    if re.search(r"\b(brighter|louder|higher|increase|raise|boost)\b", t):
        return {"direction": "up"}
    if re.search(r"\b(dimmer|quieter|lower|darker|decrease|reduce)\b", t):
        return {"direction": "down"}
    # bare "up"/"down" — only in a command context (turn/set/make + up/down)
    if re.search(r"\b(turn|set|make|bring|crank)\w*\s+\w+\s+up\b", t) or \
       re.search(r"\bup\b\s*$", t) and re.search(r"\b(turn|heat|light)\b", t):
        return {"direction": "up"}
    if re.search(r"\b(turn|set|make|bring|crank)\w*\s+\w+\s+down\b", t) or \
       re.search(r"\bdown\b\s*$", t) and re.search(r"\b(turn|heat|light)\b", t):
        return {"direction": "down"}

    # special values
    if re.search(r"\b(max|maximum|full|brightest)\b", t):
        return {"percent": 100}
    if re.search(r"\b(half|mid|medium)\b", t):
        return {"percent": 50}

    return None


# --------------------------------------------------------------------------- #
#  Color names (Option B schema: Red, Blue, Green -- we also accept common    #
#  extras so real speech doesn't fall through)                                #
# --------------------------------------------------------------------------- #
_COLORS = {
    "red": "red", "blue": "blue", "green": "green", "yellow": "yellow",
    "white": "white", "orange": "orange", "purple": "purple", "pink": "pink",
    "cyan": "cyan", "amber": "amber", "violet": "violet", "magenta": "magenta",
    "teal": "teal", "brown": "brown", "black": "black",
}


def parse_color(transcript: str) -> Optional[dict]:
    """Extract a color name from a transcript.

    "change color to red" -> {"color": "red"}
    "lights blue"         -> {"color": "blue"}
    Returns None if no known color appears.
    """
    t = _norm(transcript)
    if not t:
        return None
    for tok in t.split():
        if tok in _COLORS:
            return {"color": _COLORS[tok]}
    return None


# --------------------------------------------------------------------------- #
#  Reminder tasks (Option B schema: Drink water, Study, Exercise)             #
# --------------------------------------------------------------------------- #
# Canonical tasks from the schema, longest-first so "drink water" beats
# "water" if we ever added a bare noun. ASR typos tolerated via the
# _TYPO_MAP below.
_TASKS = [
    "drink water", "study", "exercise",
    # common extras so real speech still lands somewhere sensible
    "take medicine", "take medication", "medication", "medicine",
    "call mom", "buy groceries", "groceries", "workout", "stretch",
    "eat lunch", "lunch", "breakfast", "dinner", "sleep", "nap",
    "water the plants", "plants", "email", "meeting", "homework",
    "read a book", "reading", "shower", "brush teeth", "dentist",
    "pay bills", "bills", "laundry", "clean the house", "cleaning",
    "walk the dog", "dog", "yoga", "meditate", "practice piano", "piano",
    "cook dinner", "cooking", "iron clothes", "vacuum", "dishes",
]
# frequent whisper typos -> canonical token
_TASK_TYPO = {
    "studdy": "study", "studying": "study", "studie": "study",
    "exersice": "exercise", "exercize": "exercise", "exersise": "exercise",
    "drinkwater": "drink water",
    "medition": "medication",
    "grocerys": "groceries",
    "meetin": "meeting",
    "medicatoin": "medication",
}


def parse_reminder_task(transcript: str) -> Optional[dict]:
    """Extract a reminder task from a transcript.

    "remind me to study"        -> {"task": "Study"}
    "reminder drink water"      -> {"task": "Drink water"}
    "create a reminder to exercise" -> {"task": "Exercise"}
    Falls back to a cleaned-up verbatim tail (after "remind me to" /
    "reminder" / "to") when no canonical task matches, so ANY task the
    user says still produces a usable reminder. Returns None only when
    the transcript is empty or contains nothing after the cue words.
    """
    t = _norm(transcript)
    if not t:
        return None
    # fix known typos
    for bad, good in _TASK_TYPO.items():
        t = re.sub(rf"\b{bad}\b", good, t)
    # 1) canonical task present anywhere?
    for task in _TASKS:
        if task in t:
            return {"task": task.title()}
    # 2) fallback: strip cue words and keep the remainder as the task
    tail = t
    for cue in ("create a reminder to", "create a reminder", "create reminder",
                "remind me to", "remind me", "remind", "reminder",
                "set a reminder to", "set a reminder"):
        idx = tail.find(cue)
        if idx >= 0:
            tail = tail[idx + len(cue):]
            break
    tail = tail.strip(" ,.")
    # drop leading filler
    tail = re.sub(r"^(please\s+|i want to\s+|i'd like to\s+|to\s+|a\s+)", "", tail)
    tail = tail.strip()
    if tail and len(tail) <= 40:
        return {"task": tail.title()}
    return None


def extract_slot(transcript: str, intent: str) -> dict:
    """Unified entry point. Given an intent + transcript, return the slot(s).

    Supported intents:
      SET_TIMER       -> {"duration": {value, unit, seconds}}
      SET_ALARM       -> {"clock": {time}}
      DIM_COLOR_LIGHTS -> {"percent": int} and/or {"color": str} / {"direction"}
      THERMOSTAT      -> {"temp": int} (absolute) or {"direction": "up"|"down"}
      REMINDERS_LISTS -> {"task": str}  ("Study", "Drink water", ...)

    Always returns a dict; callers check for keys.
    """
    out: dict = {"transcript": transcript}
    if intent in ("SET_ALARM",):
        # Alarms take a CLOCK time ("7", "7:30 am"), not a countdown. A bare
        # number is the hour, so we deliberately do NOT run duration parsing --
        # otherwise "7" would be misread as "7 minutes".
        clk = parse_clock_time(transcript)
        if clk:
            out["clock"] = clk
    elif intent in ("DIM_COLOR_LIGHTS",):
        pct = parse_percent(transcript)
        if pct:
            out.update(pct)
        col = parse_color(transcript)
        if col:
            out.update(col)
    elif intent in ("THERMOSTAT",):
        pct = parse_percent(transcript)
        if pct and "percent" in pct:
            # "set to 22 degrees" -> treat the number as temp
            out["temp"] = pct["percent"]
        elif pct and "direction" in pct:
            out["direction"] = pct["direction"]
        else:
            # try to grab a bare number as absolute temp
            t = _norm(transcript)
            m = re.search(r"\b(\d{1,2})\b", t)
            if m:
                out["temp"] = int(m.group(1))
    elif intent in ("REMINDERS_LISTS",):
        task = parse_reminder_task(transcript)
        if task:
            out.update(task)
    else:
        dur = parse_duration(transcript)
        if dur:
            out["duration"] = dur
    return out


# --------------------------------------------------------------------------- #
#  Self-test                                                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cases = [
        # (transcript, intent, expected_check)
        ("set a timer for 5 minutes", "SET_TIMER", lambda s: s.get("duration", {}).get("seconds") == 300),
        ("set a timer for 5 mins", "SET_TIMER", lambda s: s.get("duration", {}).get("seconds") == 300),
        ("set a timer for ten minutes", "SET_TIMER", lambda s: s.get("duration", {}).get("seconds") == 600),
        ("set a timer for 30 seconds", "SET_TIMER", lambda s: s.get("duration", {}).get("seconds") == 30),
        ("set a timer for half an hour", "SET_TIMER", lambda s: s.get("duration", {}).get("seconds") == 1800),
        ("set a timer for 1 hour", "SET_TIMER", lambda s: s.get("duration", {}).get("seconds") == 3600),
        ("set a timer for 2 hours", "SET_TIMER", lambda s: s.get("duration", {}).get("seconds") == 7200),
        ("set a timer for 25 minutes", "SET_TIMER", lambda s: s.get("duration", {}).get("seconds") == 1500),
        ("set a timer for one minute", "SET_TIMER", lambda s: s.get("duration", {}).get("seconds") == 60),
        ("set a timer", "SET_TIMER", lambda s: "duration" not in s),
        ("set a timer for 100 minutes", "SET_TIMER", lambda s: s.get("duration", {}).get("seconds") == 6000),
        ("set an alarm for 7", "SET_ALARM", lambda s: s.get("clock", {}).get("time") == "07:00"),
        ("set an alarm for 7:30 am", "SET_ALARM", lambda s: s.get("clock", {}).get("time") == "07:30"),
        ("set an alarm for 7 pm", "SET_ALARM", lambda s: s.get("clock", {}).get("time") == "19:00"),
        ("set an alarm at seven thirty", "SET_ALARM", lambda s: s.get("clock", {}).get("time") == "07:30"),
        ("wake me up at 4am", "SET_ALARM", lambda s: s.get("clock", {}).get("time") == "04:00"),
        ("set an alarm for 9pm", "SET_ALARM", lambda s: s.get("clock", {}).get("time") == "21:00"),
        ("set an alarm for 9 p.m.", "SET_ALARM", lambda s: s.get("clock", {}).get("time") == "21:00"),
        ("wake me up at 8 a.m.", "SET_ALARM", lambda s: s.get("clock", {}).get("time") == "08:00"),
        ("alarm 8AM", "SET_ALARM", lambda s: s.get("clock", {}).get("time") == "08:00"),
        # --- brightness / percent ---
        ("dim the lights to 40 percent", "DIM_COLOR_LIGHTS", lambda s: s.get("percent") == 40),
        ("set the lights to 75%", "DIM_COLOR_LIGHTS", lambda s: s.get("percent") == 75),
        ("dim to fifty percent", "DIM_COLOR_LIGHTS", lambda s: s.get("percent") == 50),
        ("make the lights brighter", "DIM_COLOR_LIGHTS", lambda s: s.get("direction") == "up"),
        ("make the lights dimmer", "DIM_COLOR_LIGHTS", lambda s: s.get("direction") == "down"),
        ("set lights to max", "DIM_COLOR_LIGHTS", lambda s: s.get("percent") == 100),
        ("set lights to half", "DIM_COLOR_LIGHTS", lambda s: s.get("percent") == 50),
        ("dim the lights to 30", "DIM_COLOR_LIGHTS", lambda s: s.get("percent") == 30),
        # --- thermostat ---
        ("set the thermostat to 22 degrees", "THERMOSTAT", lambda s: s.get("temp") == 22),
        ("turn the heat up", "THERMOSTAT", lambda s: s.get("direction") == "up"),
        ("turn the heat down", "THERMOSTAT", lambda s: s.get("direction") == "down"),
        ("set temperature to 18", "THERMOSTAT", lambda s: s.get("temp") == 18),
        # --- colors (Option B schema: Red, Blue, Green) ---
        ("change color to red", "DIM_COLOR_LIGHTS", lambda s: s.get("color") == "red"),
        ("switch color to blue", "DIM_COLOR_LIGHTS", lambda s: s.get("color") == "blue"),
        ("set color to green", "DIM_COLOR_LIGHTS", lambda s: s.get("color") == "green"),
        ("dim the lights to 40 percent green", "DIM_COLOR_LIGHTS",
         lambda s: s.get("percent") == 40 and s.get("color") == "green"),
        ("lights yellow please", "DIM_COLOR_LIGHTS", lambda s: s.get("color") == "yellow"),
        # --- reminder tasks (Option B schema: Drink water, Study, Exercise) ---
        ("remind me to study", "REMINDERS_LISTS", lambda s: s.get("task") == "Study"),
        ("reminder drink water", "REMINDERS_LISTS", lambda s: s.get("task") == "Drink Water"),
        ("create a reminder to exercise", "REMINDERS_LISTS", lambda s: s.get("task") == "Exercise"),
        ("remind me to study later", "REMINDERS_LISTS", lambda s: s.get("task") == "Study"),
        ("remind me to call mom", "REMINDERS_LISTS", lambda s: s.get("task") == "Call Mom"),
        ("remind me to take out the trash", "REMINDERS_LISTS",
         lambda s: s.get("task") == "Take Out The Trash"),
        ("remind me to studdy", "REMINDERS_LISTS", lambda s: s.get("task") == "Study"),
    ]
    ok = 0
    for text, intent, check in cases:
        slot = extract_slot(text, intent)
        passed = check(slot)
        mark = "OK " if passed else "FAIL"
        if passed:
            ok += 1
        print(f"[{mark}] {text!r:48} -> {slot}")
    print(f"\n{ok}/{len(cases)} passed")
