# VCM on Raspberry Pi — Setup Guide (beginner-friendly)

Goal: run the **CRNN INT8** voice-command model (340 KB, 94.4% test accuracy,
~5 ms inference) **fully on-device** on a Raspberry Pi 4/5, and prove it works
in real time from a USB microphone.

You do NOT need to know anything about Raspberry Pis to follow this. Each step
says exactly what to type and what you should see.

---

## 0. What you need physically
- A Raspberry Pi 4 or 5 (any RAM, 2 GB is fine — the model is 340 KB).
- A microSD card (16 GB+) **already flashed with Raspberry Pi OS** (see §1 if not).
- Power supply (official USB-C PSU), HDMI monitor + keyboard + mouse (for setup only).
- A **USB microphone** (or a USB webcam — its mic works fine).
- Internet connection (Ethernet cable is easiest; Wi-Fi also works).

> The model + all code are already in THIS folder (`pi_bundle`). You only need
> to copy this one folder onto the Pi.

---

## 1. Flash the SD card (skip if you already have a working Pi OS)
1. Plug the microSD card into your computer.
2. Download **Raspberry Pi Imager** (free): https://www.raspberrypi.com/software/
3. Open Imager → choose **"Raspberry Pi OS (64-bit)"** → select your SD card → **WRITE**.
4. Before writing, click the gear icon next to "Write" and set:
   - **Hostname:** `vcm`
   - **Username / password:** e.g. user `pi`, password `raspberry` (pick your own)
   - **Enable SSH:** ON
   - **Wi-Fi:** enter your network + password (so you can connect later)
5. Eject the card, put it in the Pi, plug in power + HDMI + keyboard.
6. The Pi boots to a desktop. Log in with the username/password you set.

Once the desktop loads, open a **Terminal** (black window, top-left menu →
"Accessories → Terminal"). Everything below runs in that Terminal.

---

## 2. Update the system & install Python packages
In the Pi's Terminal, run these ONE AT A TIME (press Enter after each, wait for
each to finish before the next):

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv
```

Then create an isolated Python environment and install the two libraries the
model needs:

```bash
python3 -m venv ~/vcm_env
source ~/vcm_env/bin/activate
pip install --upgrade pip
pip install numpy onnxruntime librosa sounddevice
```

> `onnxruntime` is what runs the model. `numpy` does the math. `librosa` is
> needed for EXACT feature parity (the model was trained on librosa features —
> using the same ones is why accuracy stays at 94% instead of dropping to ~18%).
> `sounddevice` reads the microphone.
>
> First install may take a few minutes. `librosa` pulls in a couple of extra
> wheels — that's normal.

Install the audio hardware helper (fallback mic reader):

```bash
sudo apt install -y alsa-utils
```

---

## 3. Copy the pi_bundle folder onto the Pi
Pick whichever way is easiest. The folder you need is `pi_bundle` (≈2 MB).

**Option A — USB flash drive (simplest, no internet needed):**
Copy `pi_bundle` onto a USB stick, plug it into the Pi, then in the Terminal:
```bash
mkdir -p ~/vcm
cp -r /media/$USER/*/*/pi_bundle/* ~/vcm/     # adjust path to match where it mounted
```

**Option B — scp from your Mac (needs both on the same network):**
On your **Mac**, in the folder that contains `pi_bundle`:
```bash
scp -r pi_bundle pi@vcm.local:~/vcm/
```
(`vcm` is the hostname from §1; `pi` is your username. If SSH is off, use the
Pi's IP from `hostname -I` on the Pi instead.)

**Verify it landed correctly** — in the Pi Terminal:
```bash
ls ~/vcm
```
You should see:
```
SETUP_GUIDE.md  features.py  live_demo.py  model_float.onnx
model_int8.onnx  selftest.py  test_clips/  vcm_infer.py
```

---

## 4. Test WITHOUT a microphone (do this first!)
This proves the model + code work on the Pi before you touch the mic.

```bash
cd ~/vcm
source ~/vcm_env/bin/activate
python3 selftest.py --model model_int8.onnx --n 20
```
Expected: it loads the model, checks feature parity, classifies a batch of
clips, and prints a latency distribution. If you see per-clip intents and
sub-second timings, you're good.

Quick manual check on the included clips:
```bash
python3 vcm_infer.py model_int8.onnx test_clips/LIGHT_ON.wav
```
Expected: `LIGHTS_ON_OFF  conf=1.000  ...  <- test_clips/LIGHT_ON.wav`

> Note: the FIRST clip is slow (~1–2 s) because it warms up the JIT/ONNX
> runtime. Every clip after that is ~5–25 ms. That warm-up is one-time.

---

## 5. Find your microphone
```bash
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```
Look for your USB mic in the list. Note its **number** (the index in brackets,
e.g. `[1]`) and confirm it says `16000` or `48000 Hz` and `1 ch` is supported.
You'll pass that number as `--device` in the next step. If you don't specify a
device, the demo uses the system default.

Also confirm ALSA sees it (fallback path):
```bash
arecord -l
```

---

## 6. Run the LIVE demo (the real-time proof)
```bash
cd ~/vcm
source ~/vcm_env/bin/activate
python3 live_demo.py --model model_int8.onnx
```
Add `--device 1` if your mic isn't the default (replace `1` with the index
from §5).

**How it works:** it listens continuously, waits for you to speak (energy-based
voice detection), captures ~1 second of audio, classifies it, and prints the
intent + timing. Then it loops.

**Try saying** (these are the phrases it was trained on):
- "play music"
- "turn on the lights" / "turn off the lights"
- "dim the lights to fifty percent"
- "set a timer for one minute"
- "set an alarm for eight am"
- "pause" / "stop" / "next" / "volume up"
- "what's the weather" / "what time is it"
- "call mom"
- "remind me to drink water"

**Example output:**
```
>> speech detected, capturing...
  [LIGHTS_ON_OFF]  conf=0.98  feat=18.2ms  infer=4.1ms  TOTAL=23.5ms
```
`TOTAL` well under 1000 ms = **real-time, on-device** ✓

Press `Ctrl+C` to stop.

### Tuning if it misbehaves
- **Triggers too easily (picks up background noise):** raise the trigger:
  `python3 live_demo.py --model model_int8.onnx --trigger-db 18`
- **Misses quiet speech:** lower it: `--trigger-db 8`
- **Wrong mic:** add `--device <index>`.

---

## 7. Make it start on boot (optional, for the final demo)
Create a systemd service so the demo auto-runs when the Pi powers on:

```bash
sudo nano /etc/systemd/system/vcm.service
```
Paste (adjust the username if it isn't `pi`):
```
[Unit]
Description=VCM voice command demo
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/vcm
ExecStart=/home/pi/vcm_env/bin/python live_demo.py --model model_int8.onnx
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```
Save (`Ctrl+O`, Enter, `Ctrl+X`), then enable:
```bash
sudo systemctl daemon-reload
sudo systemctl enable vcm
sudo systemctl start vcm
journalctl -u vcm -f        # watch it live (Ctrl+C to stop watching)
```

---

## Troubleshooting
| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: onnxruntime` | You forgot `source ~/vcm_env/bin/activate`. Run it, then retry. |
| `sounddevice` error / no mic | Check `python3 -c "import sounddevice as sd; print(sd.query_devices())"`. Try `--device <index>`. As a last resort the demo falls back to `arecord` automatically. |
| First clip takes ~2 s, rest are fast | Normal — one-time warm-up. Ignore it. |
| Accuracy seems lower than 94% on the mic | Mic quality + room noise matter. The 94% is on clean dataset audio. Real mic is noisier. Use a decent USB mic, speak clearly ~30 cm away. |
| `PAUSE` classified as `DIM_COLOR_LIGHTS` etc. | Known weak spot — `MEDIA_CONTROL`/short words are the hardest class. See note below. |
| Pi won't boot from SD | Re-flash with Imager; try a different card; make sure you selected "Raspberry Pi OS (64-bit)". |

### Honest expectations
- **Strong intents** (lights, timer, alarm, color, temperature, weather,
  reminders, calls): ~93–100%.
- **Weakest intents**: `SET_TIMER` (~87%) and `QUESTION_SEARCH` (~89%) — these
  are the smaller training classes. Short words like "pause"/"stop"
  (`MEDIA_CONTROL`) can occasionally bleed into neighbors. This is expected and
  is a good talking point for the project ("we identified the weak classes and
  know where to add more data").

---

## What to screenshot/report for the exercise
1. `ls ~/vcm` — shows the tiny footprint (model is 340 KB).
2. `selftest.py` output — feature parity + latency.
3. `live_demo.py` output with 3–5 real spoken commands showing `TOTAL < 1000 ms`.
4. `free -m` or `python3 -c "import psutil"` memory read — shows it runs in a
   few hundred MB, nowhere near the Pi's limit.
5. The INT8 model size: `du -h model_int8.onnx` → ~340 KB.

That's your on-device, real-time, no-cloud proof. Done.

---

## 8. Two-device mode — make the MacBook/phone DO the action (optional)

By default the Pi only *prints* the intent. To make the **MacBook or phone
actually perform the action** (play music, show a notification, toggle a light),
split the system: the Pi keeps the tiny model (ears + brain), and a small
**handler** on the Mac/phone becomes the hands.

```
Pi:  mic -> VCM -> "PLAY_MUSIC" --(WebSocket)--> MacBook: mac_handler.py
                                                        -> plays music / notifies
```

### On the MacBook (the "hands")
```bash
cd pi_bundle
pip install websockets
python3 mac_handler.py --port 8765
```
Leave it running. It prints "listening on ws://0.0.0.0:8765".

### Find the Mac's Wi-Fi IP (the Pi needs it)
```bash
ipconfig getifaddr en0        # e.g. 192.168.1.20
```

### On the Pi (the "brain")
```bash
pip install websocket-client
python3 live_demo.py --model model_int8.onnx --broadcast ws://192.168.1.20:8765
```
Now say "play music" into the Pi's mic → the **MacBook** shows a "▶ Play music"
notification. Recognition stayed on-device; the action happened on the Mac.

### Making it do REAL things
Edit the `ACTIONS` dict at the top of `mac_handler.py`. Defaults are
notifications (work everywhere). Point them at your real devices, e.g.:
```python
"LIGHTS_ON_OFF":  lambda: requests.post("http://homebridge.local/lights/toggle"),
"PLAY_MUSIC":     lambda: subprocess.run(["open","spotify:playlist:..."]),
"SET_TIMER":      lambda: _notify("VCM","Timer set"),
```
On a **phone**, run the same handler inside a tiny app/web-page (WebSockets)
and map intents to the phone's media player, notifications, or HomeKit.

> Both devices must be on the **same Wi-Fi**. If the Pi can't connect, check
> the IP, that `mac_handler.py` is still running, and that no firewall blocks
> port 8765 (System Settings → Network → Firewall on the Mac).
