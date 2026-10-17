"""Skill dispatcher: maps the 20 VCM intents to simulated-device actions.

Covers all ten command categories from the brief:

  1. music            -> PLAY_MUSIC
  2. questions/search -> WEATHER (offline sim), TIME (local clock)
  3. lights           -> LIGHT_ON, LIGHT_OFF
  4. brightness/color -> BRIGHTNESS, COLOR
  5. timers           -> TIMER
  6. alarms           -> ALARM
  7. thermostat       -> TEMPERATURE
  8. media controls   -> PAUSE, STOP, NEXT, VOLUME_UP, VOLUME_DOWN
  9. reminders/lists  -> CREATE_REMINDER, LIST_REMINDERS
 10. calls/messages   -> CALL, MESSAGE

OUT_OF_SCOPE -> polite decline (no action).
"""
from __future__ import annotations

from dataclasses import dataclass

from runtime.device import SimulatedSmartHome
from runtime.offline_weather import weather_response
from runtime.vcm_runtime import OOS_IDX


@dataclass
class SkillOutcome:
    intent: str
    category: str
    response: str
    acted: bool


CATEGORY_BY_INTENT = {
    "PLAY_MUSIC": "music",
    "WEATHER": "questions/search",
    "TIME": "questions/search",
    "LIGHT_ON": "lights",
    "LIGHT_OFF": "lights",
    "BRIGHTNESS": "brightness/color",
    "COLOR": "brightness/color",
    "TIMER": "timers",
    "ALARM": "alarms",
    "TEMPERATURE": "thermostat",
    "PAUSE": "media controls",
    "STOP": "media controls",
    "NEXT": "media controls",
    "VOLUME_UP": "media controls",
    "VOLUME_DOWN": "media controls",
    "CREATE_REMINDER": "reminders/lists",
    "LIST_REMINDERS": "reminders/lists",
    "CALL": "calls/messages",
    "MESSAGE": "calls/messages",
    "OUT_OF_SCOPE": "out_of_scope",
}


class SkillDispatcher:
    def __init__(self, device: SimulatedSmartHome):
        self.device = device

    def dispatch(self, intent: str, phrase: str = "", slot: str | None = None) -> SkillOutcome:
        d = self.device
        # Use the model's slot value if available; fall back to phrase parsing.
        # The slot is the ground truth from the two-head model (no ASR).
        effective_phrase = phrase or ""
        if slot:
            effective_phrase = slot

        if intent == "PLAY_MUSIC":
            resp = d.play_music(effective_phrase)
        elif intent == "WEATHER":
            resp = weather_response()
        elif intent == "TIME":
            resp = d.tell_time()
        elif intent == "LIGHT_ON":
            resp = d.light_on()
        elif intent == "LIGHT_OFF":
            resp = d.light_off()
        elif intent == "BRIGHTNESS":
            resp = d.set_brightness(effective_phrase)
        elif intent == "COLOR":
            resp = d.set_color(effective_phrase)
        elif intent == "TIMER":
            resp = d.set_timer(effective_phrase)
        elif intent == "ALARM":
            resp = d.set_alarm(effective_phrase)
        elif intent == "TEMPERATURE":
            resp = d.set_temperature(effective_phrase)
        elif intent == "PAUSE":
            resp = d.pause()
        elif intent == "STOP":
            resp = d.stop()
        elif intent == "NEXT":
            resp = d.next_track()
        elif intent == "VOLUME_UP":
            resp = d.volume_up()
        elif intent == "VOLUME_DOWN":
            resp = d.volume_down()
        elif intent == "CREATE_REMINDER":
            resp = d.create_reminder(effective_phrase)
        elif intent == "LIST_REMINDERS":
            resp = d.list_reminders()
        elif intent == "CALL":
            resp = d.call(effective_phrase)
        elif intent == "MESSAGE":
            resp = d.message(effective_phrase)
        elif intent == "OUT_OF_SCOPE":
            resp = "I'm not sure I can help with that. I can control music, lights, " \
                   "temperature, timers, alarms, reminders, calls and messages."
            return SkillOutcome(intent, "out_of_scope", resp, acted=False)
        else:
            resp = f"Unknown intent {intent}."
            return SkillOutcome(intent, "unknown", resp, acted=False)
        return SkillOutcome(intent, CATEGORY_BY_INTENT[intent], resp, acted=True)
