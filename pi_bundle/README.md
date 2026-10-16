# ME2 VCM — Raspberry Pi 4/5 bundle

Fully local, voice-controlled smart-home device:

```
mic (16 kHz) -> openWakeWord "hey mycroft" -> 2 s command recording (endpointing)
              -> conformer_s ONNX classifier (20 intents)
              -> skill dispatcher -> simulated smart home -> text/spoken response
```

No cloud, no network at runtime. Time/date from the system clock; weather
from a deterministic offline simulator; all actuators (lights, thermostat,
speakers, calls, messages) simulated in memory.

## Contents

| Path | What |
|---|---|
| `models/conformer_s_float.onnx` | Voice-command classifier, 1.0 MB, float32, opset 14 |
| `requirements-pi.txt` | CPU-only Python deps |
| `setup_pi.sh` | One-shot install (venv + deps + wake-word models + smoke test) |
| `me2-vcm.service` | systemd unit for headless operation |
| `sample_clips/*.wav` | Real held-out test clips for `--file` testing without a mic |
| `runtime/`, `vcm/` | Runtime code (symlinked/copied by setup) |

## Install

```bash
scp -r pi_bundle pi:/home/pi/me2/
ssh pi
cd /home/pi/me2/pi_bundle
./setup_pi.sh
```

## Run

```bash
./.venv/bin/python -m runtime.main --mic                 # live microphone
./.venv/bin/python -m runtime.main --file sample_clips/play_music_sample.wav
./.venv/bin/python -m runtime.main --file sample_clips/timer_sample.wav --phrase "Timer 10 seconds"
./.venv/bin/python -m runtime.main --demo                # offline replay of held-out clips
```

`--phrase` supplies the transcript for slot parsing in `--file` mode (the
live `--mic` path uses the spoken phrase captured alongside the audio).

Headless service (after editing the paths in `me2-vcm.service`):

```bash
sudo cp me2-vcm.service /etc/systemd/system/
sudo systemctl enable --now me2-vcm
journalctl -u me2-vcm -f
```

## Ten command categories

music · questions/search (weather + time) · lights · brightness/color ·
timers · alarms · thermostat · media controls · reminders/lists ·
calls/messages (simulated)

## Hardware notes (RPi 4 / 5)

- CPU-only ONNX Runtime; the classifier is 246,055 params (~490K FLOPs per
  1 s clip). Measured on Apple Silicon M-series: 0.67 ms per clip end-to-end.
  On a Pi 4 (4x Cortex-A72 @ 1.5 GHz) expect roughly 10–50x slower per-clip
  inference (order of tens of ms) — still far below real time. **Measure on
  your board** with `--demo` and record the numbers; do not assume.
- Memory: Python + onnxruntime + openwakeword baseline is a few hundred MB
  RSS on desktop; on a 1 GB Pi 4 add swap (`dphys_swapfile`) if needed.
  The 2 GB Pi 4 and any Pi 5 run comfortably.
- Audio: USB mic or the official camera-mic HAT; PortAudio is installed by
  `setup_pi.sh`. Verify with `arecord -l` / `python -m sounddevice`.
- The openWakeWord "hey_mycroft" model triggers on the phrase *hey mycroft*;
  threshold defaults to 0.5 (tunable via `--ww-threshold`). Measured false
  triggers: 2/250 neg_test clips at 0.5 (both multi-speaker babble, peaks
  0.706/0.652); raising the threshold to 0.75 eliminates both with zero cost
  to the true positive (peak 0.9996).

## Verified on the dev machine (macOS, Apple Silicon)

- Wake word: synthetic "hey mycroft" clip peak probability 0.9996; 250
  negative test clips, 2 false triggers at threshold 0.5 (both babble;
  zero at 0.75) (see `../results/wakeword_bench.json`).
- Classifier: 0.67 ms/clip end-to-end (features + inference), 1,494 clips/s.
- Full pipeline over 4,367 held-out in-scope clips: 86.9% correct dispatch
  (see `../report/stage6_deployment.md`).
