"""Unit tests for the Stage 6 runtime: slot parsers, simulated device,
skill dispatcher, and the ONNX classifier's feature pipeline.

Run: python -m pytest tests/test_runtime.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.device import (  # noqa: E402
    SimulatedSmartHome, parse_color, parse_duration, parse_number,
    parse_time_of_day,
)
from runtime.skills import SkillDispatcher, CATEGORY_BY_INTENT  # noqa: E402
from runtime.offline_weather import get_weather, weather_response  # noqa: E402


# ---------------------------------------------------------------------------
# Slot parsers (against the real phrase formats from configs/variations.csv)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phrase,expected", [
    ("Timer 10 seconds", 10),
    ("Countdown for 30 seconds", 30),
    ("Timer 1 minute", 60),
    ("Start a timer for 1 minute", 60),
    ("set a timer for 2 hours", 7200),
    ("in five minutes", 300),
])
def test_parse_duration(phrase, expected):
    assert parse_duration(phrase) == expected


@pytest.mark.parametrize("phrase,expected", [
    ("Alarm 6:00 AM", (6, 0)),
    ("Alarm 9:00 PM", (21, 0)),
    ("Wake me up at 8:00 AM", (8, 0)),
    ("Set an alarm for 6:00 AM", (6, 0)),
    ("at 12:30 pm", (12, 30)),
])
def test_parse_time(phrase, expected):
    assert parse_time_of_day(phrase) == expected


@pytest.mark.parametrize("phrase,expected", [
    ("Brightness 20 percent", 20),
    ("Adjust brightness to 100 percent", 100),
    ("Brightness level 60 percent", 60),
])
def test_parse_number(phrase, expected):
    assert parse_number(phrase) == expected


@pytest.mark.parametrize("phrase,expected", [
    ("Change color to Red", "red"),
    ("Switch color to Blue", "blue"),
    ("Set color to Green", "green"),
    ("make it purple please", "purple"),
])
def test_parse_color(phrase, expected):
    assert parse_color(phrase) == expected


# ---------------------------------------------------------------------------
# Simulated device
# ---------------------------------------------------------------------------

def test_lights():
    d = SimulatedSmartHome()
    d.light_on()
    assert d.state["lights"] is True
    d.light_off()
    assert d.state["lights"] is False


def test_brightness_and_color():
    d = SimulatedSmartHome()
    d.set_brightness("Brightness 40 percent")
    assert d.state["brightness"] == 40
    d.set_color("Change color to Blue")
    assert d.state["color"] == "blue"


def test_thermostat():
    d = SimulatedSmartHome()
    d.set_temperature("Temperature 18 degrees")
    assert d.state["thermostat"]["target_c"] == 18.0
    assert d.state["thermostat"]["mode"] == "cool"  # 18 < current 24


def test_timer_set():
    d = SimulatedSmartHome()
    d.set_timer("Timer 10 seconds")
    assert len(d.state["timers"]) == 1


def test_alarm_set():
    d = SimulatedSmartHome()
    d.set_alarm("Alarm 6:00 AM")
    assert len(d.state["alarms"]) == 1
    a = list(d.state["alarms"].values())[0]
    assert (a.hour, a.minute) == (6, 0)


def test_reminder():
    d = SimulatedSmartHome()
    d.create_reminder("Remind me to Drink water")
    assert len(d.state["reminders"]) == 1
    assert d.state["reminders"][0].text == "drink water"


def test_media_controls():
    d = SimulatedSmartHome()
    d.play_music("Play music")
    assert d.state["music"]["playing"] is True
    d.pause()
    assert d.state["music"]["paused"] is True
    d.next_track()
    assert d.state["music"]["track"] is not None
    d.stop()
    assert d.state["music"]["playing"] is False
    d.volume_up()
    assert d.state["volume"] == 60
    d.volume_down()
    assert d.state["volume"] == 50


def test_call_and_message():
    d = SimulatedSmartHome()
    d.call("Call")
    assert len(d.state["calls"]) == 1
    d.message("Message")
    assert len(d.state["messages"]) == 1


def test_time_is_local():
    d = SimulatedSmartHome()
    resp = d.tell_time()
    assert "It's" in resp


# ---------------------------------------------------------------------------
# Offline weather (deterministic, no network)
# ---------------------------------------------------------------------------

def test_weather_deterministic():
    w1 = get_weather()
    w2 = get_weather()
    assert w1["forecast"] == w2["forecast"]
    assert w1["source"] == "offline_simulated"
    assert len(w1["forecast"]) == 3


def test_weather_response_string():
    r = weather_response()
    assert "Simulated weather" in r


# ---------------------------------------------------------------------------
# Skill dispatcher: every in-scope intent maps to a category and acts
# ---------------------------------------------------------------------------

def test_dispatcher_covers_all_ten_categories():
    d = SimulatedSmartHome()
    disp = SkillDispatcher(d)
    seen = set()
    for intent in [
        "PLAY_MUSIC", "WEATHER", "TIME", "LIGHT_ON", "LIGHT_OFF",
        "PAUSE", "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN",
        "CALL", "MESSAGE", "LIST_REMINDERS", "TIMER", "ALARM",
        "TEMPERATURE", "BRIGHTNESS", "COLOR", "CREATE_REMINDER",
    ]:
        out = disp.dispatch(intent, phrase="x")
        assert out.acted is True, f"{intent} did not act"
        seen.add(out.category)
    expected = {
        "music", "questions/search", "lights", "brightness/color",
        "timers", "alarms", "thermostat", "media controls",
        "reminders/lists", "calls/messages",
    }
    assert seen == expected, f"missing categories: {expected - seen}"


def test_dispatcher_oos_does_not_act():
    d = SimulatedSmartHome()
    disp = SkillDispatcher(d)
    out = disp.dispatch("OUT_OF_SCOPE", "tell me a joke")
    assert out.acted is False
    assert out.category == "out_of_scope"


# ---------------------------------------------------------------------------
# Classifier feature pipeline matches training (shape + determinism)
# ---------------------------------------------------------------------------

def test_classifier_features_shape():
    from runtime.vcm_runtime import VCMClassifier
    import tempfile, os
    onnx = ROOT / "models" / "conformer_s_float.onnx"
    if not onnx.exists():
        pytest.skip("ONNX model not exported yet")
    clf = VCMClassifier(str(onnx))
    w = np.random.randn(16000).astype(np.float32)
    f = clf.features(w)
    assert f.shape == (1, 1, 40, 100)
    # determinism
    f2 = clf.features(w)
    assert np.allclose(f, f2)


def test_classifier_classify_returns_intent():
    from runtime.vcm_runtime import VCMClassifier
    onnx = ROOT / "models" / "conformer_s_float.onnx"
    if not onnx.exists():
        pytest.skip("ONNX model not exported yet")
    clf = VCMClassifier(str(onnx))
    w = np.random.randn(16000).astype(np.float32)
    res = clf.classify(w)
    assert res.intent in clf.intents
    assert 0.0 <= res.confidence <= 1.0
    assert res.latency_ms > 0
