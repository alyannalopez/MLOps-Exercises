"""Single source of truth: Option B raw folder -> 10-intent taxonomy.

The 10 intents are the most common voice commands issued to smart devices
(ranked by real-world Alexa/Google Home logs + consumer surveys). Option B's
32 fine-grained folders map onto them as below. Class indices 0..9 are the
intents; index 10 = REJECT (ambient noise / no command).
"""

# intent index (0-based) -> name
CLASSES = [
    "PLAY_MUSIC",        # 0  "play music"
    "QUESTION_SEARCH",   # 1  weather / time / general queries
    "LIGHTS_ON_OFF",     # 2  "turn on/off the lights"
    "DIM_COLOR_LIGHTS",  # 3  brightness % / light color
    "SET_TIMER",         # 4  "set a timer for ..."
    "SET_ALARM",         # 5  "set an alarm for ..."
    "THERMOSTAT",        # 6  "set temperature to ..."
    "MEDIA_CONTROL",     # 7  pause / stop / next / volume
    "REMINDERS_LISTS",   # 8  "remind me to ..." / list reminders
    "CALLS_MESSAGING",   # 9  call someone / send a message
]
REJECT = 10
N_CLASSES = 11

# Option B raw folder name -> 10-intent index
RAW_TO_INTENT = {
    # 1 PLAY_MUSIC
    "PLAY_MUSIC": 0,
    # 2 QUESTION_SEARCH
    "WEATHER": 1,
    "TIME": 1,
    # 3 LIGHTS_ON_OFF
    "LIGHT_ON": 2,
    "LIGHT_OFF": 2,
    # 4 DIM_COLOR_LIGHTS
    "BRIGHTNESS_20": 3,
    "BRIGHTNESS_60": 3,
    "BRIGHTNESS_100": 3,
    "COLOR_RED": 3,
    "COLOR_GREEN": 3,
    "COLOR_BLUE": 3,
    "COLOR_YELLOW": 3,
    # 5 SET_TIMER
    "TIMER_10s": 4,
    "TIMER_30s": 4,
    "TIMER_1m": 4,
    # 6 SET_ALARM
    "ALARM_4_00AM": 5,
    "ALARM_8_00AM": 5,
    "ALARM_9_00PM": 5,
    # 7 THERMOSTAT
    "TEMPERATURE_18": 6,
    "TEMPERATURE_22": 6,
    "TEMPERATURE_26": 6,
    # 8 MEDIA_CONTROL
    "PAUSE": 7,
    "STOP": 7,
    "NEXT": 7,
    "VOLUME_UP": 7,
    "VOLUME_DOWN": 7,
    # 9 REMINDERS_LISTS
    "CREATE_REMINDER_STUDY": 8,
    "CREATE_REMINDER_EXERCISE": 8,
    "CREATE_REMINDER_DRINK_WATER": 8,
    "LIST_REMINDERS": 8,
    # 10 CALLS_MESSAGING
    "CALL": 9,
    "MESSAGE": 9,
}

# Canonical representative phrase per 10-intent (used by the WER proxy).
CANON_PHRASE = {
    0: "play music",
    1: "what is the weather",
    2: "turn on the lights",
    3: "dim the lights to sixty percent",
    4: "set a timer for thirty seconds",
    5: "set an alarm for eight am",
    6: "set the temperature to twenty two degrees",
    7: "pause",
    8: "remind me to study",
    9: "call mom",
}

# Raw intents that carry a value slot (value-level accuracy is meaningful).
SLOTTED_RAW = {
    "ALARM_4_00AM", "ALARM_8_00AM", "ALARM_9_00PM",
    "BRIGHTNESS_20", "BRIGHTNESS_60", "BRIGHTNESS_100",
    "COLOR_RED", "COLOR_GREEN", "COLOR_BLUE", "COLOR_YELLOW",
    "TIMER_10s", "TIMER_30s", "TIMER_1m",
    "TEMPERATURE_18", "TEMPERATURE_22", "TEMPERATURE_26",
    "CREATE_REMINDER_STUDY", "CREATE_REMINDER_EXERCISE", "CREATE_REMINDER_DRINK_WATER",
}

if __name__ == "__main__":
    print(f"{len(RAW_TO_INTENT)} raw folders -> {len(CLASSES)} intents (+REJECT)")
    from collections import Counter
    c = Counter(RAW_TO_INTENT.values())
    for i, n in enumerate(CLASSES):
        print(f"  {i}: {n:18s} <- {c[i]} raw folders")
