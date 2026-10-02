#!/usr/bin/env python3
"""UI Simulation for hardware-dependent VCM commands.

Simulates the actions that would be performed on physical hardware:
  - Command 3: LIGHT_ON / LIGHT_OFF  → toggle light state
  - Command 4: BRIGHTNESS            → set brightness %
  - Command 5: COLOR                 → set RGB color
  - Command 6: TIMER                 → start countdown timer
  - Command 7: ALARM                 → set alarm time
  - Command 9: TEMPERATURE           → set thermostat
  - Command 10: CREATE_REMINDER      → add reminder

Run: python3 code/ui_simulation.py
"""
from __future__ import annotations
import json, random, time
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

@dataclass
class SmartHomeState:
    lights_on: bool = False
    brightness: int = 100
    color: str = "White"
    rgb: tuple = (255, 255, 255)
    timer_active: bool = False
    timer_remaining: int = 0
    timer_total: int = 0
    alarms: list = field(default_factory=list)
    temperature: int = 22
    reminders: list = field(default_factory=list)
    music_playing: bool = False
    current_track: str = ""
    volume: int = 50
    paused: bool = False
    weather: str = ""
    time_str: str = ""

    def execute(self, command: str, slot_value: str = "") -> str:
        """Execute a command and return a status message."""
        msg = ""
        if command == "LIGHT_ON":
            self.lights_on = True
            msg = f"✅ Lights ON (brightness={self.brightness}%, color={self.color})"
        elif command == "LIGHT_OFF":
            self.lights_on = False
            msg = "🔴 Lights OFF"
        elif command == "BRIGHTNESS":
            try:
                pct = int(slot_value.replace("percent", "").strip())
                self.brightness = max(0, min(100, pct))
                msg = f"💡 Brightness → {self.brightness}%"
            except:
                msg = "⚠️  Could not parse brightness value"
        elif command == "COLOR":
            self.color = slot_value.title()
            color_map = {"Red": (255,0,0), "Blue": (0,0,255), "Green": (0,255,0),
                         "White": (255,255,255), "Yellow": (255,255,0), "Purple": (128,0,128)}
            self.rgb = color_map.get(self.color, (255,255,255))
            msg = f"🎨 Color → {self.color} RGB{self.rgb}"
        elif command == "TIMER":
            # Parse "10 seconds", "30 seconds", "1 minute"
            secs = self._parse_duration(slot_value)
            self.timer_active = True
            self.timer_total = secs
            self.timer_remaining = secs
            msg = f"⏱️  Timer started: {secs}s ({secs//60}m {secs%60}s)"
        elif command == "ALARM":
            self.alarms.append(slot_value)
            msg = f"⏰ Alarm set: {slot_value} (total: {len(self.alarms)})"
        elif command == "TEMPERATURE":
            try:
                deg = int(slot_value.replace("degrees", "").strip())
                self.temperature = max(16, min(30, deg))
                msg = f"🌡️  Thermostat → {self.temperature}°C"
            except:
                msg = "⚠️  Could not parse temperature"
        elif command == "CREATE_REMINDER":
            self.reminders.append(slot_value)
            msg = f"📝 Reminder added: '{slot_value}' (total: {len(self.reminders)})"
        elif command == "PLAY_MUSIC":
            self.music_playing = True
            self.paused = False
            tracks = ["SoundHelix-Song-1", "SoundHelix-Song-2", "SoundHelix-Song-3",
                       "SoundHelix-Song-4", "SoundHelix-Song-5"]
            self.current_track = random.choice(tracks)
            msg = f"🎵 Playing: {self.current_track}"
        elif command == "PAUSE":
            self.paused = True
            msg = "⏸️  Paused"
        elif command == "STOP":
            self.music_playing = False
            self.paused = False
            self.current_track = ""
            msg = "⏹️  Stopped"
        elif command == "NEXT":
            if self.music_playing:
                tracks = ["SoundHelix-Song-1", "SoundHelix-Song-2", "SoundHelix-Song-3",
                           "SoundHelix-Song-4", "SoundHelix-Song-5"]
                self.current_track = random.choice(tracks)
                msg = f"⏭️  Next: {self.current_track}"
            else:
                msg = "⚠️  Nothing playing"
        elif command == "VOLUME_UP":
            self.volume = min(100, self.volume + 10)
            msg = f"🔊 Volume → {self.volume}%"
        elif command == "VOLUME_DOWN":
            self.volume = max(0, self.volume - 10)
            msg = f"🔉 Volume → {self.volume}%"
        elif command == "WEATHER":
            self.weather = "Metro Manila: 31°C, partly cloudy, humidity 78%"
            msg = f"🌤️  {self.weather}"
        elif command == "TIME":
            self.time_str = datetime.now().strftime("%H:%M:%S")
            msg = f"🕐 {self.time_str}"
        elif command == "CALL":
            msg = "📞 Dialing... (simulated — no hardware connected)"
        elif command == "MESSAGE":
            msg = "💬 Opening messaging app... (simulated)"
        elif command == "LIST_REMINDERS":
            if self.reminders:
                msg = "📋 Reminders: " + ", ".join(self.reminders)
            else:
                msg = "📋 No reminders"
        elif command == "OUT_OF_SCOPE":
            msg = "❓ Sorry, I didn't understand that command."
        else:
            msg = f"❓ Unknown command: {command}"
        
        # Tick timer
        if self.timer_active and self.timer_remaining > 0:
            self.timer_remaining -= 1
            if self.timer_remaining == 0:
                self.timer_active = False
                msg += " ⏰ Timer DONE!"
        
        return msg

    def _parse_duration(self, s: str) -> int:
        s = s.lower().strip()
        if "minute" in s:
            num = int(''.join(c for c in s if c.isdigit()) or 1)
            return num * 60
        elif "second" in s:
            return int(''.join(c for c in s if c.isdigit()) or 0)
        return 0

    def status(self) -> str:
        lines = [
            f"{'='*50}",
            f"  SMART HOME STATUS  {datetime.now().strftime('%H:%M:%S')}",
            f"{'='*50}",
            f"  Lights:      {'ON' if self.lights_on else 'OFF'}  brightness={self.brightness}%  color={self.color}",
            f"  Music:       {'▶ ' + self.current_track if self.music_playing else 'stopped'}  vol={self.volume}%  {'⏸' if self.paused else ''}",
            f"  Timer:       {'active ' + str(self.timer_remaining) + 's' if self.timer_active else 'none'}",
            f"  Alarms:      {self.alarms if self.alarms else 'none'}",
            f"  Temp:        {self.temperature}°C",
            f"  Reminders:   {self.reminders if self.reminders else 'none'}",
            f"{'='*50}",
        ]
        return "\n".join(lines)


def demo():
    """Run a demo sequence of commands."""
    state = SmartHomeState()
    print(state.status())
    
    commands = [
        ("LIGHT_ON", ""),
        ("BRIGHTNESS", "60 percent"),
        ("COLOR", "Blue"),
        ("PLAY_MUSIC", ""),
        ("VOLUME_UP", ""),
        ("NEXT", ""),
        ("TIMER", "30 seconds"),
        ("ALARM", "6:00 AM"),
        ("TEMPERATURE", "22 degrees"),
        ("CREATE_REMINDER", "Drink water"),
        ("WEATHER", ""),
        ("TIME", ""),
        ("PAUSE", ""),
        ("LIST_REMINDERS", ""),
        ("OUT_OF_SCOPE", ""),
        ("STOP", ""),
        ("LIGHT_OFF", ""),
    ]
    
    for cmd, slot in commands:
        msg = state.execute(cmd, slot)
        print(f"\n  [{cmd:20s}] {msg}")
    
    print(f"\n{state.status()}")
    
    # Save demo log
    log = {"timestamp": datetime.now().isoformat(), "commands": commands, "final_state": {
        "lights_on": state.lights_on, "brightness": state.brightness, "color": state.color,
        "music_playing": state.music_playing, "volume": state.volume,
        "temperature": state.temperature, "alarms": state.alarms, "reminders": state.reminders,
    }}
    out = Path(__file__).resolve().parents[1] / "report" / "ui_simulation_log.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nDemo log saved -> {out}")


if __name__ == "__main__":
    demo()
