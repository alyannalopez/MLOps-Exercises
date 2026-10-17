"""Slot vocabulary for the two-head VCM (shared slot head, tiny-alexa style).

Six slotted intents, each with a fixed set of values.  All other intents
map to NO_SLOT.  The slot head predicts one of 19 classes (18 values +
NO_SLOT).  No ASR — the model outputs the slot directly from audio.
"""
from __future__ import annotations

import numpy as np

# Ordered slot values (index = class id).  NO_SLOT is always last.
SLOT_VALUES: list[str] = [
    # TIMER (3)
    "10 seconds",       # 0
    "30 seconds",       # 1
    "1 minute",         # 2
    # ALARM (3)
    "6:00 AM",          # 3
    "8:00 AM",          # 4
    "9:00 PM",          # 5
    # TEMPERATURE (3)
    "18 degrees",       # 6
    "22 degrees",       # 7
    "26 degrees",       # 8
    # BRIGHTNESS (3)
    "20 percent",       # 9
    "60 percent",       # 10
    "100 percent",      # 11
    # COLOR (3)
    "Red",              # 12
    "Blue",             # 13
    "Green",            # 14
    # CREATE_REMINDER (3)
    "Drink water",      # 15
    "Study",            # 16
    "Exercise",         # 17
    # NO_SLOT
    "NO_SLOT",          # 18
]

N_SLOTS = len(SLOT_VALUES)  # 19
NO_SLOT_IDX = N_SLOTS - 1   # 18

# intent -> list of slot class indices (for validation / joint decoding)
INTENT_SLOTS: dict[str, list[int]] = {
    "TIMER":           [0, 1, 2],
    "ALARM":           [3, 4, 5],
    "TEMPERATURE":     [6, 7, 8],
    "BRIGHTNESS":      [9, 10, 11],
    "COLOR":           [12, 13, 14],
    "CREATE_REMINDER": [15, 16, 17],
}

# Reverse lookup: slot_value string -> slot class index
_SLOT_TO_IDX: dict[str, int] = {v: i for i, v in enumerate(SLOT_VALUES)}

# Intent names (must match vcm_runtime.INTENTS order)
INTENTS = [
    "PLAY_MUSIC", "WEATHER", "TIME", "LIGHT_ON", "LIGHT_OFF",
    "PAUSE", "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN",
    "CALL", "MESSAGE", "LIST_REMINDERS",
    "TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS", "COLOR", "CREATE_REMINDER",
    "OUT_OF_SCOPE",
]


def slot_index_for(intent: str, slot_value) -> int:
    """Map (intent, slot_value) from the manifest to a slot class index.

    Empty / missing / NaN slot_value -> NO_SLOT.
    """
    if slot_value is None:
        return NO_SLOT_IDX
    if isinstance(slot_value, float) and np.isnan(slot_value):
        return NO_SLOT_IDX
    sv = str(slot_value).strip()
    if not sv or sv.lower() == "nan":
        return NO_SLOT_IDX
    if sv in _SLOT_TO_IDX:
        return _SLOT_TO_IDX[sv]
    # Fuzzy: case-insensitive match
    lower = sv.lower()
    for v, i in _SLOT_TO_IDX.items():
        if v.lower() == lower:
            return i
    # Unknown slot value -> NO_SLOT (should not happen with clean data)
    return NO_SLOT_IDX


def decode_slot(intent: str, slot_logits: list[float]) -> str | None:
    """Decode the slot head output for a given intent.

    Returns the slot value string, or None for non-slotted intents.
    Only considers valid slot values for this intent (joint decoding).
    """
    if intent not in INTENT_SLOTS:
        return None
    valid = INTENT_SLOTS[intent]
    # Find argmax among valid slot indices only
    best_idx, best_logit = valid[0], float("-inf")
    for si in valid:
        if slot_logits[si] > best_logit:
            best_logit = slot_logits[si]
            best_idx = si
    return SLOT_VALUES[best_idx]
