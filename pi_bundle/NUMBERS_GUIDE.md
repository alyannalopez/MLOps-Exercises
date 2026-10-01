

---

## 10. Slot-filling: recognizing VALUES ("set a timer for 5 minutes")

By default the model only knows the **coarse intent** (`SET_TIMER`) — it does not
transcribe the digits/words, so the simulator picks a random duration/color/task.
To make it recognize the **value**, add a tiny local ASR pass that runs *only* on
slotted commands (timer, alarm, brightness, color, temperature, reminder).

### What gets recognized (all Option B schema values)
| Command you say | Slot extracted | Simulator does |
|---|---|---|
| "start a timer for 1 minute" | `duration=60s` | ⏱ counts down from 01:00 |
| "set an alarm for 8:00 am" | `clock=08:00` | ⏰ alarm at 08:00 |
| "set the temperature to 22 degrees" | `temp=22` | 🌡 target 22.0°C |
| "adjust brightness to 60 percent" | `percent=60` | 💡 60% |
| "change color to green" | `color=green` | 💡 turns green |
| "remind me to study" | `task=Study` | 📝 reminder: Study |

### Why a second model?
The 340 KB INT8 intent model is a 10-class classifier — it has no concept of
digits. The standard lightweight fix is a small English-only Whisper model
(~78 MB, CPU, ~300 ms) that transcribes the *same* utterance, plus a regex
parser that pulls out the number + unit. It's **optional**: the base demo stays
340 KB and instant; `--numbers` opts into the extra pass.

### Setup (one-time)
```bash
# 1) install whisper.cpp (provides the whisper-cli binary)
brew install whisper-cpp            # Mac
#    on the Pi: git clone https://github.com/ggml-org/whisper.cpp && cd whisper.cpp && cmake -B build && cmake --build build

# 2) download the tiny English model (~78 MB) into ../asr/
mkdir -p ../asr
curl -L -o ../asr/ggml-tiny.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin
```

### Run it
```bash
python3 live_demo.py --model model_int8.onnx --simulate --numbers
```
Now say **"set a timer for 5 minutes"** and the panel shows a **5-minute**
countdown (not a random one). Also covered: **"set an alarm for 7:30 am"** →
`07:30`, **"dim the lights to 40 percent"** → 40%, **"change color to red"** →
red, **"remind me to study"** → reminder "Study".

### What you'll see
```
  [SET_TIMER]  conf=0.99  feat=8.1ms  infer=2.9ms  TOTAL=11.0ms
  [numbers] heard: 'set a timer for 5 minutes'
  ⏱ Timer set for 5 minute
```

### Files added
| File | Role |
|------|------|
| `numparse.py` | stdlib-only parser: transcript → duration / clock / percent / **color** / **task** |
| `asr.py` | whisper.cpp CLI wrapper (zero Python deps; `transcribe_pcm`) |
| `live_demo.py` | `--numbers` / `--asr-model` flags; ASR pass on all 6 slotted intents |
| `mac_handler.py` | `act()`/`handle_intent()` accept a `slot`; uses duration/clock/%/color/temp/task |

### Graceful degradation
If `whisper-cli` or the model is missing, `--numbers` prints a warning and falls
back to the default (random) duration — the demo never crashes.

### Honest limits
- Adds ~300 ms per timer/alarm command (only those, not every command).
- Whisper *tiny* is decent but not perfect on noisy mic audio; if a number is
  garbled, the parser falls back to the default. Swap in `ggml-base.en.bin`
  (~148 MB) via `--asr-model` for better accuracy at higher latency.
- The 340 KB intent model is **unchanged** — slot-filling is a separate,
  optional layer.
