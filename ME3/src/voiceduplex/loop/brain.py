"""Agent brain — "what do we do with what they said?"

A deterministic, OFFLINE router so the loop is fully testable without any
LLM/API. It classifies a transcript into one of three routes:

  backchannel : short filler ("mm", "yeah", "uh-huh") -> brief acknowledgment
  quick_tool  : a self-contained easy task -> answer now
  delegate    : anything else -> hand to a bigger harness (Step 8 stubs this)

Step 7: quick_tool now covers the SAME ten "easy tasks" the ME2 brief ranked
by real Alexa/Google-Home command logs (see smarthome/__init__.py). Intent
detection is keyword/regex over the transcript (ME2 used a trained VCM on raw
audio; ME3 already has ASR, so we classify the text instead — same intents,
same simulated device, same response strings).

  1  play music            6  set an alarm
  2  questions / search    7  adjust thermostat
     (time, date, weather)
  3  lights on / off       8  media controls (pause/stop/next/volume)
  4  dim / color lights    9  reminders & lists
  5  set a timer          10  calls & messages

Plus two ME3 extras that were already here: simple math and echo.

The interface (route(text) -> RouteDecision) is unchanged, so the half-duplex
and full-duplex loops never change. Step 8 replaces delegate's stub with a real
MCP call to Onit/Hermes/OpenCode.
"""

from __future__ import annotations

import datetime as _dt
import itertools
import re
import threading
from dataclasses import dataclass

from ..smarthome.device import SimulatedSmartHome
from ..smarthome.weather import weather_response


@dataclass
class RouteDecision:
    route: str          # "backchannel" | "quick_tool" | "delegate"
    reply: str          # the text the TTS will speak
    detail: str = ""    # human-readable note for the console/log
    pending: object = None  # Step 8: a delegate.PendingTask handle (delegate route
                            #   only). The loop waits on it, then speaks the result.


# ---- backchannel detection -----------------------------------------------
_BACKCHANNEL = re.compile(
    r"^\s*(mm+h*|mhm+m*|uh+ ?h+ ?uh*|yeah+|yep|yes|mm-hm|mhm-hm|right|okay|ok|"
    r"sure|got it|i see|exactly|cool|nice|alright|roger|affirmative|uh-huh|"
    r"mmm|hm+|mhm)\s*[!.]?\s*$",
    re.IGNORECASE,
)


def _is_backchannel(text: str) -> bool:
    return bool(_BACKCHANNEL.match(text))


# ---- quick-tool handlers (offline, no network) ---------------------------
# operator words -> symbol. Longest phrases first so "multiplied by" wins
# over "multiply", "divided by" over "divide".
_OP_WORDS = [
    ("multiplied by", "*"),
    ("divided by", "/"),
    ("plus", "+"),
    ("minus", "-"),
    ("times", "*"),
    ("multiply", "*"),
    ("divided", "/"),
    ("divide", "/"),
    ("and", "+"),
]
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# small English number words -> value (so "twelve times eight" works by voice)
_WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}
_WORD_NUM_ALT = "|".join(sorted(_WORD_NUMS, key=len, reverse=True))
# Compiled once. (Can't use re.match(pattern, s, pos) with a literal "L" in the
# alternation — at a non-zero pos CPython parses a leading "L" as the LOCALE
# flag. A compiled pattern avoids that entirely.)
_WORD_NUM_RE = re.compile(rf"({_WORD_NUM_ALT})(?:\s+({_WORD_NUM_ALT}))?")


def _do_math(text: str) -> str | None:
    """Evaluate a simple arithmetic expression if the text is essentially one.

    Token-based, not string-scrubbing: we walk the text left to right and pull
    out numbers (digits OR small English words) and operator words in order,
    ignoring every other word. This sidesteps the failure modes of the old
    regex-pipeline:
      * "that's 12 times 8."  -> the stray "that's" is simply ignored
      * "3.5 plus 1.5"        -> the decimal dot is preserved (never space-split)
      * "twelve times eight"  -> word numbers resolve to 12 * 8
    """
    low = text.lower()
    tokens: list[str] = []
    i = 0
    n = len(low)
    while i < n:
        c = low[i]
        # number (int or decimal)
        m = _NUM_RE.match(low, i)
        if m:
            tokens.append(m.group(0))
            i = m.end()
            continue
        # word number? (e.g. "twelve", "twenty five") — accumulate a phrase
        if c.isalpha():
            wm = _WORD_NUM_RE.match(low, i)
            if wm and (i == 0 or not low[i - 1].isalnum()):
                val = _WORD_NUMS[wm.group(1)]
                if wm.group(2):
                    # "twenty five" -> 25 ; "two hundred" -> 200
                    second = _WORD_NUMS[wm.group(2)]
                    if wm.group(1) == "hundred":
                        val = second * 100
                    elif second < 20 or val % 100 == 0:
                        val = val + second
                tokens.append(str(val))
                i = wm.end()
                continue
        # operator word?
        matched_op = False
        for word, sym in _OP_WORDS:
            if low.startswith(word, i) and (
                i + len(word) >= n or not (low[i + len(word)].isalnum())
            ):
                # ensure the char before wasn't a letter (avoid "and" inside "sand")
                if i == 0 or not low[i - 1].isalpha():
                    tokens.append(sym)
                    i += len(word)
                    matched_op = True
                    break
        if matched_op:
            continue
        # explicit symbol (user typed "*" etc.)
        if c in "+-*/":
            tokens.append(c)
            i += 1
            continue
        i += 1  # skip any other char (letters, punctuation, spaces)

    if not tokens:
        return None
    # must be a VALID expression: number (op number)+  — i.e. at least two
    # operands. This rejects dangling-operator fragments like "plus 5."
    # (likely a mis-transcription) and lone numbers.
    expr = " ".join(tokens)
    if not re.fullmatch(r"\d+(?:\.\d+)?(?:\s+[+\-*/]\s+\d+(?:\.\d+)?)+", expr):
        return None
    # final safety: only digits, dots, spaces, and the four operators
    if not re.fullmatch(r"[\d\s.+\-*/]+", expr):
        return None
    try:
        val = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 — restricted charset
        if isinstance(val, float) and val.is_integer():
            val = int(val)
        return f"That's {val}."
    except Exception:
        return None


def _do_echo(text: str) -> str | None:
    m = re.search(r"(?i)^(?:echo|repeat|say)\s+(.+)$", text)
    if m:
        return f"You said: {m.group(1).strip()}"
    return None


# ============================================================================
# Step 7 — the ME2 top-10 intent set.
#
# Each classifier returns a (category, reply) pair, or None if the text is not
# that intent. They are tried in _QUICK_HANDLERS order (specific before general)
# so a phrase like "turn off the lights" hits lights, not media-stop.
#
# The shared SimulatedSmartHome instance is a module-level singleton so state
# (lights on, timer running, reminders) persists across turns — exactly like
# ME2's device. A lock guards it because the full-duplex loop may route from
# the barge-in path and the idle path.
# ============================================================================
_DEVICE = SimulatedSmartHome()
_DEVICE_LOCK = threading.Lock()


def _act(fn, *args) -> str:
    """Run a device method under the lock and return its response string."""
    with _DEVICE_LOCK:
        return fn(*args)


# -- 1. play music ----------------------------------------------------------
_MUSIC_PLAY = re.compile(r"\b(play|put on|start|resume)\b.*\b(music|song|songs|tune|tunes|playlist|album|track|something)\b", re.I)
_MUSIC_BARE = re.compile(r"^\s*(play|put on|start)\s+(music|some music)\s*[!.]?\s*$", re.I)


def _h_music(text: str):
    if _MUSIC_BARE.match(text) or _MUSIC_PLAY.search(text):
        return ("music", _act(_DEVICE.play_music, text))
    return None


# -- 2. questions / search: time, date, weather ----------------------------
def _h_questions(text: str):
    t = text.lower()
    if "weather" in t or "forecast" in t or re.search(r"\b(rain|hot|cold|warm|sunny|cloudy)\b.*\b(today|tomorrow|outside|manila)\b", t):
        return ("questions/search", _act(weather_response))
    if re.search(r"\b(time|clock|late)\b", t):
        return ("questions/search", _act(_DEVICE.tell_time))
    if re.search(r"\b(date|today|day is it|what day|calendar)\b", t):
        return ("questions/search", _act(_DEVICE.tell_date))
    return None


# -- 3. lights on / off -----------------------------------------------------
_LIGHT_ON = re.compile(r"\b(turn|switch|make)\b.*\b(on)\b.*\b(lights?|light|lamp|lamps)\b|\b(lights?|lamp)\b.*\b(on)\b", re.I)
_LIGHT_OFF = re.compile(r"\b(turn|switch|make)\b.*\b(off)\b.*\b(lights?|light|lamp|lamps)\b|\b(lights?|lamp)\b.*\b(off)\b", re.I)


def _h_lights(text: str):
    if _LIGHT_ON.search(text):
        return ("lights", _act(_DEVICE.light_on))
    if _LIGHT_OFF.search(text):
        return ("lights", _act(_DEVICE.light_off))
    return None


# -- 4. dim / color lights --------------------------------------------------
_DIM = re.compile(r"\b(dim|bright(er|ness)?|darken|lighten|lower|raise)\b.*\b(lights?|light|brightness)\b|\b(lights?|light)\b.*\b(dim|bright|dark)\b", re.I)
_COLOR = re.compile(r"\b(color|colour)\b|\b(make|set|change|turn)\b.*\b(red|blue|green|yellow|orange|purple|white|pink|cyan|violet)\b", re.I)


def _h_dim_color(text: str):
    if _DIM.search(text):
        return ("brightness/color", _act(_DEVICE.set_brightness, text))
    if _COLOR.search(text):
        return ("brightness/color", _act(_DEVICE.set_color, text))
    return None


# -- 5. timers --------------------------------------------------------------
_TIMER = re.compile(r"\btimer\b|\bcountdown\b|\bcount down\b|\bin (a )?(few|couple|some) (seconds?|minutes?|hours?)\b", re.I)


def _h_timer(text: str):
    if _TIMER.search(text):
        return ("timers", _act(_DEVICE.set_timer, text))
    return None


# -- 6. alarms --------------------------------------------------------------
_ALARM = re.compile(r"\balarm\b|\bwake (me )?up\b|\bring (me )?up\b", re.I)


def _h_alarm(text: str):
    if _ALARM.search(text):
        return ("alarms", _act(_DEVICE.set_alarm, text))
    return None


# -- 7. thermostat ----------------------------------------------------------
_THERMO = re.compile(r"\b(thermostat|temperature|temp|heat|cool|ac|a/c|air conditioning|heating)\b", re.I)


def _h_thermo(text: str):
    if _THERMO.search(text):
        return ("thermostat", _act(_DEVICE.set_temperature, text))
    return None


# -- 8. media controls ------------------------------------------------------
_VOL_UP = re.compile(r"\b(volume up|turn (it )?up|louder|speak up)\b", re.I)
_VOL_DOWN = re.compile(r"\b(volume down|turn (it )?down|quieter|softer|too loud)\b", re.I)
_NEXT = re.compile(r"\b(next|skip)\b", re.I)
_PAUSE = re.compile(r"\bpause\b", re.I)
_STOP = re.compile(r"\b(stop|halt)\b", re.I)
_RESUME = re.compile(r"\b(resume|continue|unpause|keep playing)\b", re.I)


def _h_media(text: str):
    t = text.lower()
    if _VOL_UP.search(text):
        return ("media controls", _act(_DEVICE.volume_up))
    if _VOL_DOWN.search(text):
        return ("media controls", _act(_DEVICE.volume_down))
    if _NEXT.search(text):
        return ("media controls", _act(_DEVICE.next_track))
    if _RESUME.search(text):
        return ("media controls", _act(_DEVICE.resume))
    if _PAUSE.search(text):
        return ("media controls", _act(_DEVICE.pause))
    if _STOP.search(text):
        return ("media controls", _act(_DEVICE.stop))
    return None


# -- 9. reminders / lists ---------------------------------------------------
_LIST_REM = re.compile(r"\b(list|show|what are|what's|whats|my) (reminders?|to-dos?|todos?|tasks?|notes?)\b", re.I)
_CREATE_REM = re.compile(r"\bremind(me)?\b|\badd (a )?reminder\b|\breminder\b", re.I)


def _h_reminders(text: str):
    if _LIST_REM.search(text):
        return ("reminders/lists", _act(_DEVICE.list_reminders))
    if _CREATE_REM.search(text):
        return ("reminders/lists", _act(_DEVICE.create_reminder, text))
    return None


# -- 10. calls / messages ---------------------------------------------------
_CALL = re.compile(r"\b(call|phone|dial|ring)\b", re.I)
_MESSAGE = re.compile(r"\b(message|text|sms)\b", re.I)


def _h_calls(text: str):
    if _CALL.search(text):
        return ("calls/messages", _act(_DEVICE.call, text))
    if _MESSAGE.search(text):
        return ("calls/messages", _act(_DEVICE.message, text))
    return None


# Ordered: specific intents first. (math/echo are ME3 extras, kept last so a
# real command like "call 555" isn't mistaken for arithmetic.)
_QUICK_HANDLERS = [
    _h_music,
    _h_questions,
    _h_lights,
    _h_dim_color,
    _h_timer,
    _h_alarm,
    _h_thermo,
    _h_media,
    _h_reminders,
    _h_calls,
    _do_math,
    _do_echo,
]


def _run_quick(text: str):
    """Try every quick-tool handler; return (category, reply) or None.

    The ten ME2 handlers return (category, reply); the two ME3 extras
    (_do_math, _do_echo) return a bare reply string. Normalize both.
    """
    for h in _QUICK_HANDLERS:
        r = h(text)
        if r is None:
            continue
        if isinstance(r, tuple):
            cat, reply = r
        else:  # _do_math / _do_echo return a bare string
            cat, reply = "quick_tool", r
        if reply:
            return cat, reply
    return None


# ---- delegate replies (rotate so back-to-back turns don't sound canned) --
_DELEGATE_LINES = [
    "I'm not sure how to help with that yet. I'll learn more tasks soon.",
    "I'm still figuring that out, but I'll be able to help with more things soon.",
    "I can't help with that right now. I'll learn how soon.",
    "I'm not able to assist with that just yet, but I'm continuously expanding what I can do.",
    "That's beyond what I can do at the moment, but I'm getting better every day.",
    "I haven't picked that skill up yet. Give me a little more time.",
    "I can't take that on just yet, but it's on my list of things to learn.",
    "I'm not equipped for that one yet. I'll be ready for more soon.",
    "That's a bit outside my range right now, but I'm steadily growing my abilities.",
    "I can't handle that one yet, but I'll keep learning so I can help with more.",
]
_delegate_cycle = itertools.cycle(range(len(_DELEGATE_LINES)))


def _next_delegate_reply() -> str:
    return _DELEGATE_LINES[next(_delegate_cycle)]


# ---- the router ----------------------------------------------------------
def route(text: str, delegate=None) -> RouteDecision:
    """Classify a transcript and produce a spoken reply.

    ``delegate`` is an optional ``voiceduplex.delegate.DelegateBackend``. When
    present, a task that falls through to the ``delegate`` route is handed to
    the backend *now* (in a background thread) and the returned decision
    carries an immediate "let me check that" reply plus a ``pending`` handle.
    The loop speaks the ack, waits on ``pending`` while the mic stays live
    (barge-in still works), then speaks the result. Without a backend this
    behaves exactly as before (canned stub line).
    """
    text = (text or "").strip()
    if not text:
        return RouteDecision(route="backchannel", reply="Hmm?", detail="empty transcript")

    if _is_backchannel(text):
        return RouteDecision(route="backchannel", reply="Mm-hm.", detail="filler ack")

    quick = _run_quick(text)
    if quick is not None:
        category, reply = quick
        return RouteDecision(route="quick_tool", reply=reply, detail=category)

    # default: hand to a bigger harness (Step 8)
    if delegate is not None:
        from ..delegate import PendingTask

        pending = PendingTask(delegate, text)
        return RouteDecision(
            route="delegate",
            reply="Let me check that.",
            detail=f"no local intent matched -> delegate ({getattr(delegate, 'name', 'harness')})",
            pending=pending,
        )

    return RouteDecision(
        route="delegate",
        reply=_next_delegate_reply(),
        detail="no local intent matched -> delegate (stub)",
    )
