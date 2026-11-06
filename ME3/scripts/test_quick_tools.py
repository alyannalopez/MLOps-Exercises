#!/usr/bin/env python3
"""Step 7 offline test — the ME2 top-10 quick tasks.

No microphone, no audio, no network. Feeds representative transcripts straight
into the brain router and asserts each one lands on route="quick_tool" with a
sane, intent-specific reply. Mirrors ME2's ten command categories (brief tasks
1-10) so we can confirm the port is faithful.

Run:  PYTHONPATH=src python3 scripts/test_quick_tools.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceduplex.loop.brain import route  # noqa: E402

# (category, transcript, substring the reply MUST contain)
CASES = [
    # 1. play music
    ("music", "play some music", "Now playing"),
    ("music", "put on a song", "Now playing"),
    # 2. questions / search (time, date, weather)
    ("questions/search", "what time is it", "It's"),
    ("questions/search", "what's the date today", "Today is"),
    ("questions/search", "what's the weather in metro manila", "Simulated weather"),
    # 3. lights on / off
    ("lights", "turn on the lights", "Lights on"),
    ("lights", "turn off the light", "Lights off"),
    # 4. dim / color
    ("brightness/color", "dim the lights to fifty percent", "brightness to 50"),
    ("brightness/color", "make the lights blue", "blue"),
    # 5. timer
    ("timers", "set a timer for ten seconds", "Timer set for 10"),
    ("timers", "countdown for two minutes", "Timer set for 120"),
    # 6. alarm
    ("alarms", "set an alarm for six am", "Alarm set for 6:00 AM"),
    ("alarms", "wake me up at 7 pm", "Alarm set for 7:00 PM"),
    # 7. thermostat
    ("thermostat", "set the temperature to twenty two degrees", "temperature to 22"),
    ("thermostat", "turn up the heat to 25", "temperature to 25"),
    # 8. media controls
    ("media controls", "play some music", "Now playing"),  # seed playback
    ("media controls", "pause the music", "Paused"),
    ("media controls", "resume", "Resuming"),
    ("media controls", "next track", "Now playing"),
    ("media controls", "stop the song", "Stopped"),
    ("media controls", "turn it up", "Volume up"),
    ("media controls", "turn it down", "Volume down"),
    # 9. reminders / lists
    ("reminders/lists", "remind me to buy milk", "remind you to buy milk"),
    ("reminders/lists", "what are my reminders", "reminders"),
    # 10. calls / messages
    ("calls/messages", "call mom", "Calling mom"),
    ("calls/messages", "send a message to dad", "message to dad"),
    # ME3 extras (still supported)
    ("extra/math", "what's twelve times eight", "That's 96"),
    ("extra/math", "3.5 plus 1.5", "That's 5"),
    ("extra/echo", "echo hello world", "You said: hello world"),
]

# Things that must NOT be quick_tools -> delegate (sanity: no over-matching)
DELEGATE_CASES = [
    "tell me a joke",
    "who won the world cup last year",
    "book me a flight to tokyo",
]


def main() -> int:
    passed = 0
    failed = 0
    print(f"{'CAT':<18} {'ROUTE':<12} OK  reply")
    print("-" * 78)
    for cat, text, expect in CASES:
        d = route(text)
        ok = (d.route == "quick_tool") and (expect.lower() in d.reply.lower())
        mark = "OK" if ok else "XX"
        print(f"{cat:<18} {d.route:<12} {mark}  {d.reply!r}   <- {text!r}")
        if ok:
            passed += 1
        else:
            failed += 1
            print(f"{'':<38} expected route=quick_tool and {expect!r} in reply")

    print("-" * 78)
    for text in DELEGATE_CASES:
        d = route(text)
        ok = d.route == "delegate"
        mark = "OK" if ok else "XX"
        print(f"{'delegate':<18} {d.route:<12} {mark}  {d.reply!r}   <- {text!r}")
        if ok:
            passed += 1
        else:
            failed += 1

    print("-" * 78)
    total = passed + failed
    print(f"{passed}/{total} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
