"""Offline, deterministic weather source.

The brief says to use local resources for offline weather — no network
calls. This generates a plausible, fully deterministic "forecast" from the
current date (same day -> same forecast, so the device is consistent
within a day). Clearly labeled as simulated in the response strings.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta

_CONDITIONS = ["sunny", "partly cloudy", "cloudy", "light rain", "overcast",
               "clear skies", "drizzle", "scattered showers"]


def _day_seed(day: datetime) -> int:
    key = day.strftime("%Y%m%d").encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def get_weather(city: str = "Metro Manila", days: int = 3) -> dict:
    """Deterministic simulated forecast for the next `days` days."""
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    seed = _day_seed(today)
    rng_state = seed
    forecast = []
    for d in range(days):
        rng_state = (rng_state * 1103515245 + 12345) % (2 ** 31)
        cond = _CONDITIONS[rng_state % len(_CONDITIONS)]
        hi = 24 + (rng_state >> 8) % 12          # 24-35 C
        lo = hi - 6 - (rng_state >> 16) % 5      # 5-10 deg spread
        wind = (rng_state >> 4) % 30             # 0-29 km/h
        day = today + timedelta(days=d)
        forecast.append({
            "date": day.strftime("%A, %b %d"),
            "condition": cond,
            "hi_c": hi,
            "lo_c": lo,
            "wind_kmh": wind,
        })
    return {"city": city, "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source": "offline_simulated", "forecast": forecast}


def weather_response(city: str = "Metro Manila") -> str:
    w = get_weather(city)
    f0 = w["forecast"][0]
    parts = [f"Simulated weather for {city}, Philippines: today is {f0['condition']}, "
             f"high {f0['hi_c']} degrees, low {f0['lo_c']} degrees, "
             f"wind {f0['wind_kmh']} kilometers per hour."]
    for f in w["forecast"][1:]:
        parts.append(f"{f['date']}: {f['condition']}, {f['lo_c']} to {f['hi_c']} degrees.")
    return " ".join(parts)
