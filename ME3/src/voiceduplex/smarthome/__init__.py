"""Simulated smart-home device + offline weather (Step 7).

Ported from the ME2 VCM runtime (runtime/device.py, runtime/offline_weather.py)
so ME3's brain can answer the same ten "easy tasks" the ME2 brief ranked by
real Alexa/Google-Home command logs:

  1  play music            6  set an alarm
  2  questions / search    7  adjust thermostat
     (time, date, weather)
  3  lights on / off       8  media controls (pause/stop/next/volume)
  4  dim / color lights    9  reminders & lists
  5  set a timer          10  calls & messages

Everything is in-memory and OFFLINE: time/date from the system clock, weather
from a deterministic date-seeded simulator, and every actuator (lights, bulbs,
thermostat, speaker, phone) simulated. No network, no real hardware.
"""

from .device import SimulatedSmartHome
from .weather import get_weather, weather_response

__all__ = ["SimulatedSmartHome", "get_weather", "weather_response"]
