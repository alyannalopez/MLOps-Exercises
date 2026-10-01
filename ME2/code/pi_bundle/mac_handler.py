#!/usr/bin/env python3
"""VCM action handler — a SIMULATED smart-home hub (the "hands").

The Pi (or Mac) runs the VCM (ears + brain) and sends each predicted intent
here. Instead of just popping a notification, this handler MAINTAINS STATE and
SIMULATES performing the action, so you can WATCH the whole voice->action loop:

    * lights actually turn on/off and change brightness/color
    * the thermostat temperature moves up/down
    * timers and alarms COUNT DOWN live
    * music "plays" with a moving progress bar
    * reminders/calls get logged

Two ways to run it:

  1) As a WebSocket server (two-device mode — the Pi broadcasts to it):
         python3 mac_handler.py --port 8765

  2) Import it and call handle_intent() directly (single-process --simulate):
         from mac_handler import HubSimulator
         hub = HubSimulator(); hub.handle_intent("PLAY_MUSIC", 0.97)

The simulator is a DUMMY: it touches no real hardware, no real apps. It just
keeps believable state and renders it, so the demo proves the pipeline end to
end. To wire in real devices later, replace the bodies in HubSimulator.act().
"""
from __future__ import annotations
import argparse, asyncio, json, random, threading, time
import numparse

# --------------------------------------------------------------------------- #
#  The simulated home state                                                   #
# --------------------------------------------------------------------------- #
class HubSimulator:
    """A dumb in-memory smart home. Each intent mutates state and returns a
    short human-readable description of what 'happened'."""

    def __init__(self):
        self.lights_on   = False
        self.brightness  = 80          # %
        self.color       = "warm white"
        self.temp        = 21.5        # °C
        self.target_temp = 21.5
        self.media_state = "stopped"   # stopped / playing / paused
        self.track       = None
        self.timer       = None        # (label, end_epoch)
        self.alarm       = None        # (label, end_epoch)
        self.reminders   = []          # list of strings
        self.last_call   = None
        self.last_search = None
        self.last_unknown = None       # (timestamp, confidence) of last REJECT
        self.events      = []          # rolling log of everything

    # -- helpers ------------------------------------------------------------ #
    def _log(self, msg: str):
        stamp = time.strftime("%H:%M:%S")
        self.events.append(f"[{stamp}] {msg}")
        self.events = self.events[-8:]

    # -- the actions -------------------------------------------------------- #
    def act(self, intent: str, confidence: float = 0.0, slot: dict | None = None) -> str:
        """Mutate state for `intent`; return a one-line description.

        `slot` (optional) carries a parsed number/time from the ASR pass, e.g.
        {"duration": {"value":5,"unit":"minute","seconds":300}} for timers or
        {"clock": {"time":"07:00"}} for alarms. When absent, a sensible default
        is used (so the demo still works without the ASR model).
        """
        c = f"{confidence:.2f}"
        slot = slot or {}
        if intent == "PLAY_MUSIC":
            self.media_state = "playing"
            self.track = random.choice(["Lo-fi Beats", "Jazz in the Rain",
                                        "Deep Focus", "Acoustic Morning"])
            self._log(f"▶ PLAY_MUSIC  '{self.track}'")
            return f"▶ Playing “{self.track}”"

        if intent == "QUESTION_SEARCH":
            q = random.choice(["Weather tomorrow", "Nearest coffee shop",
                               "Flight status BA231", "Recipe for pasta"])
            self.last_search = q
            self._log(f"🔎 QUESTION_SEARCH  “{q}”")
            return f"🔎 Searching: {q}"

        if intent == "LIGHTS_ON_OFF":
            self.lights_on = not self.lights_on
            self._log(f"💡 LIGHTS_ON_OFF  -> {'ON' if self.lights_on else 'OFF'}")
            return f"💡 Lights {'ON' if self.lights_on else 'OFF'}"

        if intent == "DIM_COLOR_LIGHTS":
            if not self.lights_on:
                self.lights_on = True
            # Use slot if available, else random
            if "percent" in slot:
                self.brightness = slot["percent"]
            elif "direction" in slot:
                delta = 15 if slot["direction"] == "up" else -15
                self.brightness = max(5, min(100, self.brightness + delta))
            else:
                self.brightness = random.choice([30, 45, 60, 75, 90])
            # Use the spoken color if present, else keep/roll a random one
            if slot and slot.get("color"):
                self.color = slot["color"]
            else:
                self.color = random.choice(["warm white", "cool white", "red",
                                            "blue", "amber", "purple"])
            self._log(f"🎨 DIM_COLOR_LIGHTS  {self.brightness}% {self.color}")
            return f"🎨 Lights {self.brightness}% · {self.color}"

        if intent == "SET_TIMER":
            dur = slot.get("duration")
            if dur:
                secs = dur["seconds"]
                self.timer = (f"{dur['value']} {dur['unit']}", time.time() + secs)
                self._log(f"⏱ SET_TIMER  {dur['value']} {dur['unit']}")
                return f"⏱ Timer set for {dur['value']} {dur['unit']}"
            mins = random.choice([1, 5, 10, 25, 60])
            self.timer = (f"{mins} min", time.time() + mins * 60)
            self._log(f"⏱ SET_TIMER  {mins} min (default)")
            return f"⏱ Timer set for {mins} min"

        if intent == "SET_ALARM":
            clk = slot.get("clock")
            if clk:
                self.alarm = (clk["time"], time.time() + 3600)
                self._log(f"⏰ SET_ALARM  {clk['time']}")
                return f"⏰ Alarm set for {clk['time']}"
            hh = random.randint(6, 9); mm = random.choice([0, 15, 30, 45])
            self.alarm = (f"{hh:02d}:{mm:02d}", time.time() + 3600)
            self._log(f"⏰ SET_ALARM  {hh:02d}:{mm:02d} (default)")
            return f"⏰ Alarm set for {hh:02d}:{mm:02d}"

        if intent == "THERMOSTAT":
            # Use slot if available, else random delta
            if "temp" in slot:
                self.target_temp = float(max(15.0, min(30.0, slot["temp"])))
            elif "direction" in slot:
                delta = 1.0 if slot["direction"] == "up" else -1.0
                self.target_temp = round(max(15.0, min(30.0, self.target_temp + delta)), 1)
            else:
                delta = random.choice([-2.0, -1.0, 1.0, 2.0])
                self.target_temp = round(max(15.0, min(30.0, self.target_temp + delta)), 1)
            self._log(f"🌡 THERMOSTAT  target -> {self.target_temp}°C")
            return f"🌡 Thermostat target → {self.target_temp}°C"

        if intent == "MEDIA_CONTROL":
            cycle = {"stopped": "playing", "playing": "paused", "paused": "stopped"}
            self.media_state = cycle.get(self.media_state, "playing")
            if self.media_state == "playing" and not self.track:
                self.track = "Lo-fi Beats"
            self._log(f"⏯ MEDIA_CONTROL  -> {self.media_state}")
            return f"⏯ Media {self.media_state}"

        if intent == "REMINDERS_LISTS":
            # Use the spoken task if present, else roll a plausible default
            if slot and slot.get("task"):
                item = slot["task"]
            else:
                item = random.choice(["Buy milk", "Call mom", "Submit report",
                                      "Water plants", "Book dentist"])
            self.reminders.append(item)
            self.reminders = self.reminders[-6:]
            self._log(f"📝 REMINDERS_LISTS  + {item}")
            return f"📝 Reminded: {item}"

        if intent == "CALLS_MESSAGING":
            person = random.choice(["Mom", "Alex", "Work", "Dr. Lee"])
            self.last_call = person
            self._log(f"📞 CALLS_MESSAGING  -> {person}")
            return f"📞 Calling {person}"

        return f"• {intent} (no action)"

    def handle_intent(self, intent: str, confidence: float = 0.0,
                      slot: dict | None = None) -> str | None:
        """Entry point used by both the WS server and --simulate mode."""
        if intent == "REJECT":
            self.last_unknown = (time.time(), confidence)
            self._log(f"\u26a0\ufe0f UNKNOWN  (didn't recognise  conf={confidence:.2f})")
            return None
        self.last_unknown = None  # a recognised command clears the warning
        desc = self.act(intent, confidence, slot)
        print(f"  [{intent}]  conf={confidence:.2f}  ->  {desc}")
        return desc

    # -- rendering ---------------------------------------------------------- #
    def render(self) -> str:
        """Return a compact multi-line snapshot of the simulated home."""
        L = []
        L.append("┌──────────────────────────── SIMULATED HOME ────────────────────────────┐")
        # lights
        if self.lights_on:
            bar = "█" * max(1, round(self.brightness / 10)) + "░" * (10 - max(1, round(self.brightness / 10)))
            L.append(f"  💡 Lights:  ON   [{bar}] {self.brightness:3d}%   {self.color}")
        else:
            L.append("  💡 Lights:  OFF  [░░░░░░░░░░]    —")
        # thermostat (current temp drifts toward target)
        self.temp = round(self.temp + (self.target_temp - self.temp) * 0.15, 1)
        L.append(f"  🌡 Temp:    {self.temp:4.1f}°C   (target {self.target_temp:4.1f}°C)")
        # media
        if self.media_state == "playing":
            tick = int(time.time() * 3) % 10
            bar = "▮" * (tick + 1) + "·" * (10 - tick - 1)
            L.append(f"  🎵 Music:   PLAYING  [{bar}]  {self.track}")
        elif self.media_state == "paused":
            L.append(f"  🎵 Music:   PAUSED   {self.track or '—'}")
        else:
            L.append("  🎵 Music:   stopped")
        # timer / alarm
        tline = "  ⏱ Timer:   —"
        if self.timer:
            remain = self.timer[1] - time.time()
            if remain <= 0:
                tline = f"  ⏱ Timer:   {self.timer[0]}  ⏰ DONE!"
                self.timer = None
            else:
                m, s = divmod(int(remain), 60)
                tline = f"  ⏱ Timer:   {self.timer[0]:>7s}  ⏳ {m:02d}:{s:02d}"
        L.append(tline)
        aline = "  ⏰ Alarm:   —"
        if self.alarm:
            aline = f"  ⏰ Alarm:   {self.alarm[0]}  (armed)"
        L.append(aline)
        # reminders / calls / search
        rem = ", ".join(self.reminders[-3:]) if self.reminders else "—"
        L.append(f"  📝 To-do:   {rem}")
        if self.last_call:
            L.append(f"  📞 Last:    called {self.last_call}")
        if self.last_search:
            L.append(f"  🔎 Query:   {self.last_search}")
        # last-unknown indicator (shows for 5 s after a REJECT)
        if self.last_unknown:
            ts, conf = self.last_unknown
            age = time.time() - ts
            if age < 5.0:
                L.append(f"  ⚠️ Status:  ❌ DIDN'T UNDERSTAND  (conf={conf:.2f})")
            else:
                self.last_unknown = None
        L.append("├" + "─" * 74 + "┤")
        for ev in self.events[-4:]:
            L.append(f"  {ev}")
        L.append("└" + "─" * 74 + "┘")
        return "\n".join(L)


# --------------------------------------------------------------------------- #
#  WebSocket server (two-device mode)                                         #
# --------------------------------------------------------------------------- #
_hub = HubSimulator()

async def handle(ws):
    async for raw in ws:
        try:
            msg = json.loads(raw)
        except Exception:
            continue
        _hub.handle_intent(msg.get("intent"), msg.get("confidence", 0.0), msg.get("slot"))

async def main():
    ap = argparse.ArgumentParser(description="VCM simulated smart-home hub (WebSocket)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    try:
        import websockets
    except ImportError:
        raise SystemExit("Missing dep. Run:  pip install websockets")
    print(f"VCM simulated hub listening on ws://{a.host}:{a.port}")
    print("Waiting for intents from the Pi...\n")
    async with websockets.serve(handle, a.host, a.port):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
