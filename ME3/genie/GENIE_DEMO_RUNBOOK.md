# Genie Live Testing and Demo Terminal Runbook

> **Date**: November 21, 2026
> **Status**: Draft — verified against actual codebase
> **Scope**: M5.4 → M5.6 → M6.5 live validation on Raspberry Pi 4

---

## Architecture Reference

| Component | Machine | Port | Protocol |
|-----------|---------|------|----------|
| Whisper ASR + Kokoro TTS server | Mac | 8300 | HTTP |
| MCP server | Raspberry Pi | 8301 | HTTP |
| OpenRouter LLM | Cloud | — | HTTPS (API) |
| Genie pipeline (Pipecat) | Raspberry Pi | — | Local audio |
| Terminal dashboard | Raspberry Pi | — | Stdout (Rich) |

**Network addresses:**
- Mac: `192.168.100.134` (verify before starting)
- Raspberry Pi: `192.168.100.131` / `raspberrypi.local`

**Project directories:**
- Mac: `~/sandbox/ME3/ME3/`
- Pi: `~/ME3/ME3/`

**Virtual environments:**
- Mac inference server: `.venv-kokoro` (Python 3.12) — at `~/sandbox/ME3/ME3/.venv-kokoro/`
- Pi Genie runtime: `onit-venv` (Python 3.13.5) — at `~/ME3/onit-venv/`

---

## A. Initial Network and SSH Setup

### A.1 Check the Mac's local IP address

```bash
# [MAC TERMINAL]
ipconfig getifaddr en0
```
**Expected**: `192.168.100.134` (or whatever your Mac's LAN IP is today — it may change).

### A.2 Check the Raspberry Pi's IP address

```bash
# [MAC TERMINAL] — ping via mDNS
ping -c 3 raspberrypi.local
```
**Expected**: Reply from `192.168.100.131` (may vary if DHCP reassigned).

### A.3 Connect to the Pi via SSH

```bash
# [MAC TERMINAL] — using mDNS hostname (preferred)
ssh mlops@raspberrypi.local

# [MAC TERMINAL] — or using IP directly
ssh mlops@192.168.100.131
```
**Credentials**: username `mlops`, password `__REPLACE_WITH_YOUR_PI_PASSWORD__`.

### A.4 Verify Mac port 8300 is reachable from the Pi

```bash
# [PI TERMINAL] — after SSH login
curl -s http://192.168.100.134:8300/health
```
**Expected**:
```json
{"status":"ok","asr":true,"tts":true,"asr_model":"mlx-community/whisper-large-v3-turbo","tts_voice":"af_bella","sample_rate":24000}
```

### A.5 Return to Mac terminal

```bash
# [PI TERMINAL]
exit
```

---

## B. Start and Verify the Mac Inference Server

### B.1 Check whether port 8300 is already occupied

```bash
# [MAC TERMINAL]
lsof -i :8300
```
**If empty**: Port is free. Proceed.
**If occupied**: Note the PID. Either wait for it to finish or `kill <PID>` if it's a stale process.

### B.2 Activate the virtual environment

```bash
# [MAC TERMINAL]
cd ~/sandbox/ME3/ME3
source .venv-kokoro/bin/activate
```
**Expected**: Prompt changes to `(.venv-kokoro)`.

### B.3 Start the server

```bash
# [MAC TERMINAL] — inside .venv-kokoro
python -m genie.server
```

**Expected healthy startup** (takes 30–90 seconds for model warmup):
```
HH:MM:SS INFO     Starting Genie model server on 0.0.0.0:8300
INFO:     Started server process [<pid>]
INFO:     Waiting for application startup.
HH:MM:SS INFO     ASR warm in XX.Xs, TTS warm in XX.Xs (asr=True tts=True)
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8300 (Press CTRL+C to quit)
```

**Common errors:**
- `ASR warm failed: ...` — mlx-whisper not installed or model download incomplete
- `TTS warm failed: ...` — Kokoro not installed or voice unavailable
- `OSError: [Errno 48] Address already in use` — port 8300 occupied (see B.1)

### B.4 Verify server health (in a SECOND Mac terminal)

```bash
# [SECOND MAC TERMINAL]
curl -s http://127.0.0.1:8300/health | python3 -m json.tool
```
**Expected**:
```json
{
    "status": "ok",
    "asr": true,
    "tts": true,
    "asr_model": "mlx-community/whisper-large-v3-turbo",
    "tts_voice": "af_bella",
    "sample_rate": 24000
}
```
Both `"asr": true` and `"tts": true` must be present.

### B.5 Inspect server logs

Logs go to **stdout** of the server terminal. No separate log file is written. To see them, watch the Mac terminal where `python -m genie.server` is running.

### B.6 Safe shutdown

```bash
# [MAC TERMINAL] — in the server terminal
# Press Ctrl+C
```
**Expected**: Clean shutdown with `INFO:     Shutting down` message.

---

## C. Raspberry Pi Preflight

### C.1 SSH into the Pi

```bash
# [MAC TERMINAL]
ssh mlops@raspberrypi.local
```

### C.2 Navigate to the Genie directory

```bash
# [PI TERMINAL]
cd ~/ME3/ME3
```

### C.3 Activate the virtual environment

```bash
# [PI TERMINAL]
source ~/ME3/onit-venv/bin/activate
```
**Expected**: Prompt changes to `(onit-venv)`.

### C.4 Check Python version

```bash
# [PI TERMINAL]
python --version
```
**Expected**: `Python 3.13.5`

### C.5 Check installed dependency versions

```bash
# [PI TERMINAL]
pip list 2>/dev/null | grep -iE "pipecat|soundfile|sounddevice|numpy|httpx|rich|loguru|pydantic|onnxruntime|webrtcvad|aiohttp|websockets|pyaudio"
```
**Expected key packages**:
- `pipecat-ai` 1.15.0
- `onnxruntime` 1.31.1
- `soundfile` (installed)
- `numpy` 2.5.4
- `httpx` (installed)
- `rich` (installed)
- `loguru` (installed)
- `pydantic` (installed)
- `aiohttp` (installed)
- `websockets` (installed)
- `pyaudio` (installed, after M5.2)

### C.6 Verify `.env` configuration (without displaying secrets)

```bash
# [PI TERMINAL]
# Show which keys are set (values masked)
grep -v '^#' ~/ME3/ME3/.env | grep -v '^$' | sed 's/=\(.*\)/=***/'
```
**Verify these keys exist and are non-empty**:
- `OPENROUTER_API_KEY` — must be set (masked above)
- `GENIE_SERVER_URL` — must point to Mac, e.g. `http://192.168.100.134:8300`
- `GENIE_LLM_MODEL` — default `openai/gpt-4o-mini`
- `GENIE_PORT` — default `8300`
- `MCP_PORT` — default `8301`
- `MCP_HOST` — default `127.0.0.1`

**Critical**: `GENIE_SERVER_URL` must use the Mac's **actual current IP** (from Section A.1), not a stale address.

### C.7 Check connectivity to Mac inference server

```bash
# [PI TERMINAL]
curl -s http://192.168.100.134:8300/health
```
**Expected**: Same health response as B.4.

### C.8 Check disk space, memory, and CPU load

```bash
# [PI TERMINAL]
df -h /
free -h
uptime
```
**Expected**:
- Disk: ≥ 5 Gi available (Pi has ~11 Gi free)
- RAM: ≥ 1 Gi available (Pi has ~1.4 Gi)
- Load: < 1.0 at idle

### C.9 Check required ports are available

```bash
# [PI TERMINAL]
ss -tlnp | grep -E ':8300|:8301'
```
**Expected**: Empty output (no process should be listening on 8300/8301 yet). Port 8300 is the Mac's server (not local). Port 8301 will be used by the MCP server when Genie starts.

---

## D. USB Microphone and AirPods Hardware Test (M5.4a)

### D.1 List all audio devices (PipeWire)

```bash
# [PI TERMINAL]
wpctl status
```
**Expected output** shows:
- **SINKS** (playback devices): Robot Airpods should appear as the default sink (marked `[VOL: xx%]`)
- **SOURCES** (capture devices): USB microphone should appear here

**If USB mic is NOT listed**: The microphone is not connected, not recognized, or PipeWire has not picked it up. Plug it in and wait 5 seconds, then re-run.

### D.2 List detailed audio devices (with indices)

```bash
# [PI TERMINAL]
wpctl info
```
This shows all nodes with their IDs, names, and states. Look for:
- A sink node for the AirPods (state: SUSPENDED or RUNNING)
- A source node for the USB microphone (state: SUSPENDED or RUNNING)

### D.3 Alternative: List devices via PyAudio (Python)

```bash
# [PI TERMINAL] — inside onit-venv
python -c "
import pyaudio
pa = pyaudio.PyAudio()
print('=== INPUT DEVICES ===')
for i in range(pa.get_device_count()):
    d = pa.get_device_info_by_index(i)
    if d['maxInputChannels'] > 0:
        print(f'  [{d["index"]}] {d["name"]} (ch={d["maxInputChannels"]}, sr={d["defaultSampleRate"]})')
print('=== OUTPUT DEVICES ===')
for i in range(pa.get_device_count()):
    d = pa.get_device_info_by_index(i)
    if d['maxOutputChannels'] > 0:
        print(f'  [{d["index"]}] {d["name"]} (ch={d["maxOutputChannels"]}, sr={d["defaultSampleRate"]})')
pa.terminate()
"
```

**Expected**: At least one input device (USB mic) and one output device (AirPods or PipeWire null).

### D.4 Alternative: List devices via sounddevice

```bash
# [PI TERMINAL]
python -c "import sounddevice; print(sounddevice.query_devices())"
```

### D.5 Identify the USB microphone capture device

From the output of D.3 or D.4, identify the USB microphone by:
- Name containing "USB", "Microphone", or the brand name
- `maxInputChannels > 0`
- **NOT** a loopback/monitor device (those contain "Monitor" or "Loopback" in the name)

Record the device index — this becomes `GENIE_MIC_DEVICE` in `.env` if auto-detection doesn't pick it up.

### D.6 Identify the AirPods playback device

From the output, the AirPods appear as a sink/output device. Genie uses `cfg.speaker_device=None` (default) which lets PipeWire/PulseAudio select the default sink, so **explicit configuration is usually not needed** for the AirPods.

### D.7 Check default input and output routing

```bash
# [PI TERMINAL]
wpctl status
```
Look at the `[✺]` markers — these indicate the default source and sink.

**If the USB mic is not the default source**, you have two options:
1. Set `GENIE_MIC_DEVICE=<index>` in `.env` (recommended — explicit, doesn't change system defaults)
2. Change the default: `wpctl set-default <source_node_name>` (changes system-wide, affects other apps)

**Recommendation**: Use option 1 (`GENIE_MIC_DEVICE` in `.env`) to avoid affecting system-wide audio.

### D.8 Record a short audio sample

```bash
# [PI TERMINAL]
python -c "
import sounddevice as sd
import numpy as np

# Replace 1 with your USB mic's actual device index
device_index = 1  # ← CHANGE THIS
duration = 3
sr = 16000

print(f'Recording {duration}s at {sr}Hz from device {device_index}...')
print('Speak now: Hello, this is a test.')
recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32', device=device_index)
sd.wait()

# Save
import wave, struct
buf = (np.clip(recording, -1, 1) * 32767).astype(np.int16).tobytes()
with open('/tmp/test_mic.wav', 'wb') as f:
    wf = wave.open(f, 'w')
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(buf)
    wf.close()

# Check RMS amplitude
rms = np.sqrt(np.mean(recording**2))
print(f'Recording saved. RMS amplitude: {rms:.4f}')
if rms < 0.001:
    print('WARNING: Recording is nearly silent. Check microphone connection and volume.')
elif rms < 0.01:
    print('WARNING: Recording is very quiet. Check microphone gain.')
else:
    print('Recording looks good.')
"
```

**Expected**: RMS amplitude > 0.01 indicates audible speech. RMS < 0.001 means silence (wrong device or no sound).

### D.9 Verify the recording contains non-silent audio

```bash
# [PI TERMINAL]
python -c "
import wave
with wave.open('/tmp/test_mic.wav', 'rb') as w:
    frames = w.readframes(w.getnframes())
    samples = list(struct.unpack('<' + 'h' * len(frames)//2, frames))
    nonzero = sum(1 for s in samples if abs(s) > 100)
    print(f'Total samples: {len(samples)}, Non-silent (>100): {nonzero} ({nonzero/len(samples)*100:.1f}%)')
" 2>/dev/null || python -c "
import wave, struct
with wave.open('/tmp/test_mic.wav', 'rb') as w:
    frames = w.readframes(w.getnframes())
    samples = struct.unpack('<' + 'h' * (len(frames)//2), frames)
    nonzero = sum(1 for s in samples if abs(s) > 100)
    print(f'Total samples: {len(samples)}, Non-silent: {nonzero} ({nonzero/len(samples)*100:.1f}%)')
"
```
**Expected**: > 10% non-silent samples if you spoke during recording.

### D.10 Play a test tone through the AirPods

```bash
# [PI TERMINAL]
python -c "
import sounddevice as sd
import numpy as np

# Play a 440 Hz tone for 1 second at 24kHz
sr = 24000
duration = 1.0
t = np.linspace(0, duration, int(sr * duration))
tone = 0.3 * np.sin(2 * np.pi * 440 * t)

print('Playing 440 Hz tone through default output...')
sd.play(tone, sr)
sd.wait()
print('Done.')
"
```
**Expected**: You hear a clear 440 Hz beep from the AirPods.

### D.11 Play the recorded sample back

```bash
# [PI TERMINAL]
python -c "
import soundfile as sf
import sounddevice as sd
data, sr = sf.read('/tmp/test_mic.wav')
print(f'Playing back {len(data)/sr:.1f}s at {sr}Hz...')
sd.play(data, sr)
sd.wait()
print('Done.')
"
```
**Expected**: You hear your voice from the AirPods.

### D.12 Troubleshooting

| Problem | Solution |
|---------|----------|
| USB mic not listed in `wpctl status` | Unplug/replug, wait 5s, check `dmesg \| tail` for USB errors |
| Recording is silent (RMS ≈ 0) | Wrong device index — try each input device from D.3 |
| Recording is very quiet (RMS < 0.01) | Increase mic volume: `wpctl set-volume @DEFAULT_SOURCE@ 100%` |
| Playback fails | AirPods disconnected — `bluetoothctl connect <MAC>` |
| PyAudio error about PortAudio | `sudo apt-get install portaudio19-dev` then reinstall pyaudio |

---

## E. Audio Service Validation (M5.4b)

### E.1 Test Pi-to-Mac Whisper transcription

```bash
# [PI TERMINAL]
curl -s -X POST http://192.168.100.134:8300/transcribe \
  -H "Content-Type: audio/wav" \
  --data-binary @/tmp/test_mic.wav | python3 -m json.tool
```
**Expected**:
```json
{
    "text": "hello this is a test",
    "latency_s": 1.5,
    "speech_detected": true
}
```
**If `"speech_detected": false` or `"text": ""`**: The recording was too short, too quiet, or contained no recognizable speech. Re-record with clearer speech.

### E.2 Test Mac Kokoro TTS synthesis (from Pi)

```bash
# [PI TERMINAL]
curl -s -X POST http://192.168.100.134:8300/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from the Raspberry Pi. The audio service is working."}' \
  -o /tmp/test_tts.wav
```
**Expected**: File `/tmp/test_tts.wav` created, size > 10KB.

### E.3 Play the synthesized audio through the Pi's output device

```bash
# [PI TERMINAL]
python -c "
import soundfile as sf
import sounddevice as sd
data, sr = sf.read('/tmp/test_tts.wav')
print(f'Playing {len(data)/sr:.1f}s TTS audio at {sr}Hz...')
sd.play(data, sr)
sd.wait()
print('Done.')
"
```
**Expected**: Clear speech "Hello from the Raspberry Pi..." through the AirPods.

### E.4 Verify TTS latency header

```bash
# [PI TERMINAL]
curl -s -D - -X POST http://192.168.100.134:8300/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Quick test."}' \
  -o /dev/null 2>&1 | grep -i "X-Synthesis-Latency"
```
**Expected**: `X-Synthesis-Latency: 0.XXX` (in seconds).

---

## F. Launch Genie (M5.5)

### F.1 Start Genie on the Raspberry Pi

```bash
# [PI TERMINAL] — inside onit-venv, in ~/ME3/ME3/
cd ~/ME3/ME3
PYTHONPATH=. python -m genie.pipeline_new
```

**Expected healthy startup** (takes 10–30 seconds):
```
HH:MM:SS INFO     Model server: http://192.168.100.134:8300
HH:MM:SS INFO     Auto-detected mic: <device name> (device X)
HH:MM:SS INFO     Using configured speaker device: None (default)
HH:MM:SS INFO     MCP tools discovered: 21
HH:MM:SS INFO       - get_time: Returns the current local time
HH:MM:SS INFO       - web_search: Search the web using DuckDuckGo
... (more tools listed)
HH:MM:SS INFO     M4.5: Notification coordinator started
HH:MM:SS INFO     Starting Genie pipeline...
HH:MM:SS INFO     Say something to begin the conversation.
```
**Plus**: The Rich terminal dashboard appears, showing "Idle" state.

### F.2 Verify the MCP server started

Look for `MCP tools discovered: 21` in the startup log. The MCP server runs **inside** the Genie process (as an asyncio task on port 8301), not as a separate process.

### F.3 Verify expected MCP tools

The startup log lists all 21 tools. Key ones for demo:
- `get_time` — real clock
- `web_search` — DuckDuckGo search
- `start_background_task` — M4.5 async task
- `get_task_status` — check task progress
- `get_task_result` — retrieve completed task result

### F.4 Confirm OpenRouter connectivity

The first user utterance triggers the first LLM call. If `OPENROUTER_API_KEY` is valid, the LLM responds. If invalid, you'll see an error in the log and the dashboard.

**Check without triggering a conversation**: The `.env` file has `OPENROUTER_API_KEY` set (verify it exists, don't display it):
```bash
# [PI TERMINAL] — in a separate SSH session
grep OPENROUTER_API_KEY ~/ME3/ME3/.env | cut -d= -f1,2 | head -c 20
```
This shows only the first 20 characters of the key, confirming it's present.

---

## G. Live Demonstration Test Sequence

### Test 1: Basic greeting
**Say**: "Hello."
**Observe in dashboard**: State transitions Idle → LISTENING → THINKING → AUDIO_OUTPUT → Idle. Transcript shows "Hello". Genie responds verbally.
**Success**: Genie greets back. Dashboard shows the exchange.

### Test 2: Identity and capabilities
**Say**: "What can you do?"
**Observe**: Genie describes its capabilities (conversation, web search, background tasks, etc.).
**Success**: Coherent response listing real features.

### Test 3: Web search
**Say**: "Search for the weather in London today."
**Observe**: Dashboard shows MCP tool call `web_search`. Genie reads search results aloud.
**Success**: Relevant weather information delivered verbally.

### Test 4: Simple MCP tool call
**Say**: "What time is it?"
**Observe**: Dashboard shows `get_time` tool call. Genie states the current time.
**Success**: Accurate time reported.

### Test 5: Start a background task
**Say**: "Start a background task to research the latest AI news."
**Observe**: Dashboard shows task created with PENDING → RUNNING state. Progress updates appear.
**Success**: Task begins, Genie confirms it's running.

### Test 6: Check task status
**Say**: "What's the status of my background task?"
**Observe**: Dashboard shows `get_task_status` tool call. Genie reports task state.
**Success**: Accurate task state reported.

### Test 7: Receive task completion
**Wait** for the background task to complete (30–120 seconds). Then ask: "Did my task finish?"
**Observe**: Dashboard shows task COMPLETED state. Genie summarizes results.
**Success**: Task result delivered verbally.

### Test 8: Interrupt Genie while speaking
**While Genie is speaking**: Start talking loudly: "Stop, stop! Tell me a joke instead."
**Observe in dashboard**: State changes to INTERRUPTED. Audio playback stops immediately. New transcript appears.
**Success**: Playback halts within ~1 second. Genie processes the new request.

### Test 9: New question after interruption
**Say**: (immediately after interrupting) "What's 25 times 17?"
**Observe**: Genie computes and answers.
**Success**: Correct answer (425) delivered after clean interruption recovery.

### Test 10: Consecutive conversational turns
**Have a 3–5 turn conversation**: Ask a question, follow up, ask something related.
**Observe**: Dashboard accumulates conversation history. State transitions are smooth. No crashes.
**Success**: Coherent multi-turn dialogue with proper context retention.

---

## H. Stability Testing (M5.6)

### H.1 Monitor CPU usage (while Genie is running)

Open a **second SSH session** to the Pi:
```bash
# [SECOND PI TERMINAL]
vmstat 1
```
**Expected**: CPU idle > 30% during idle conversation. CPU spikes during ASR/TTS/LLM processing are normal.

### H.2 Monitor RAM usage

```bash
# [SECOND PI TERMINAL]
watch -n 2 'free -h'
```
**Expected**: Used RAM < 1.6 Gi during normal operation. Watch for gradual increase (memory leak indicator).

### H.3 Monitor temperature

```bash
# [SECOND PI TERMINAL]
vcgencmd measure_temp
```
**Expected**: 50–70°C under load. Above 80°C triggers throttling.

### H.4 Monitor disk space

```bash
# [SECOND PI TERMINAL]
df -h /
```
**Expected**: No significant change from preflight.

### H.5 Check Genie process status

```bash
# [SECOND PI TERMINAL]
ps aux | grep pipeline_new | grep -v grep
```
**Expected**: One `python -m genie.pipeline_new` process running.

### H.6 Check inference server availability

```bash
# [SECOND PI TERMINAL]
curl -s http://192.168.100.134:8300/health
```
**Expected**: `{"status":"ok","asr":true,"tts":true,...}`

### H.7 View Genie logs in real-time

Logs go to **stdout** of the Genie terminal. To monitor from a second session, use the same terminal (not possible with SSH multiplexing easily). Alternatively, redirect logs at startup:

```bash
# [PI TERMINAL] — alternative startup with log file
cd ~/ME3/ME3
PYTHONPATH=. python -m genie.pipeline_new 2>&1 | tee /tmp/genie.log
```
Then from a second session:
```bash
# [SECOND PI TERMINAL]
tail -f /tmp/genie.log
```

### H.8 Long-running stability check

Run Tests G.1–G.10 repeatedly for 10–15 minutes. Monitor:
- No crashes or OOM kills
- RAM usage stable (not climbing)
- No Bluetooth disconnections
- Dashboard remains responsive
- No "connection refused" errors to the Mac server

---

## I. Benchmarking and Final Evaluation (M6.5)

### I.1 Already Implemented Measurements

| Metric | How to Collect | Tool |
|--------|---------------|------|
| **ASR latency** | `latency_s` field in `/transcribe` JSON response | `curl` to Mac server |
| **TTS latency** | `X-Synthesis-Latency` HTTP header from `/speak` response | `curl -D -` to Mac server |
| **LLM latency** | **Not instrumented** — Pipecat's `MetricsFrame` system exists but is disabled (`enable_metrics=False` by default) | Requires `StartFrame(enable_metrics=True)` + observer |
| **End-to-end latency** | **Not instrumented** — no single metric captures user-stop → first-audio-out | Requires `BenchmarkObserver` |
| **Interruption latency** | **Not instrumented** — no measurement of user-speak → playback-stop | Requires transport-level hooks |
| **Background task duration** | `Task.created_at` exists but `Task.completed_at` does **not** | Derive from `TaskEvent` pairs (STARTED → COMPLETED timestamps) — **already possible without code changes** |
| **CPU/RAM utilization** | System commands (`vmstat`, `free`, `vcgencmd`) | Manual, no code needed |

### I.2 M6-A Component Benchmarks (Already Completed)

The M6-A benchmark results are saved at:
- `genie/benchmarks/results/M6A_Benchmark_Results.md`
- `genie/benchmarks/results/m6a_raw_results.json`
- `genie/benchmarks/benchmark_a.py` (the benchmark script)

**Summary of M6-A results** (Mac-only, 20 trials each):

| Metric | Mean | P95 |
|--------|------|-----|
| Whisper ASR (1s input) | 1428ms | 1439ms |
| Whisper ASR (3s input) | 1506ms | 1555ms |
| Whisper ASR (5s input) | 1701ms | 2208ms |
| Whisper ASR (10.4s real) | 1856ms | 2330ms |
| Kokoro TTS (5 words) | 580ms | 752ms |
| Kokoro TTS (40 words) | 4624ms | 5703ms |
| Kokoro TTS (101 words) | 11584ms | 12909ms |
| OpenRouter LLM (simple) | 941ms | 1359ms |
| OpenRouter LLM (moderate) | 2933ms | 3106ms |

### I.3 Measurements Requiring Additional Instrumentation

| Metric | Required Change | Effort |
|--------|----------------|--------|
| LLM TTFB / total gen time | Enable `StartFrame(enable_metrics=True)` + catch `MetricsFrame` | Low |
| End-to-end latency | `BenchmarkObserver` FrameProcessor | Medium |
| Interruption latency | `BenchmarkObserver` + transport hooks | Medium |
| Background task duration | Add `completed_at` to `Task` class | Low |
| Persistent benchmark log | `benchmark_logger.py` | Medium |
| Benchmark orchestration | `benchmark_runner.py` | Medium |

### I.4 Network RTT (Pi-to-Mac)

```bash
# [PI TERMINAL]
ping -c 20 192.168.100.134
```
**Expected**: RTT < 5ms on same LAN. Record avg, min, max, P95.

---

## J. Shutdown and Recovery

### J.1 Gracefully stop Genie

```bash
# [PI TERMINAL] — in the Genie terminal
# Press Ctrl+C
```
**Expected** (from `WorkerRunner(handle_sigint=True)` + cleanup sequence):
```
HH:MM:SS INFO     M4.5: Notification coordinator stopped
HH:MM:SS INFO     M4.76: Dashboard observer stopped
HH:MM:SS INFO     M4.76: Dashboard stopped
HH:MM:SS INFO     MCP server stopped.
```
The shutdown sequence (verified in `pipeline_new.py`):
1. `WorkerRunner` handles SIGINT, stops the pipeline
2. `NotificationCoordinator.stop()` — stops task notifications
3. `DashboardObserver.shutdown()` — awaits pending dashboard tasks
4. `TerminalDashboard.shutdown()` — stops Rich display, restores terminal
5. MCP server task cancelled

### J.2 Stop the Mac inference server

```bash
# [MAC TERMINAL] — in the server terminal
# Press Ctrl+C
```
**Expected**: `INFO:     Shutting down` followed by `INFO:     Waiting for application shutdown.`

### J.3 Check for orphaned processes

```bash
# [PI TERMINAL]
ps aux | grep -E "pipeline_new|mcp_server" | grep -v grep
```
**Expected**: No output after clean shutdown.

```bash
# [MAC TERMINAL]
ps aux | grep "genie.server" | grep -v grep
lsof -i :8300
```
**Expected**: No output after clean shutdown.

### J.4 Restart Genie after a failure

```bash
# [PI TERMINAL] — inside onit-venv
cd ~/ME3/ME3
PYTHONPATH=. python -m genie.pipeline_new
```

### J.5 Restart the Mac server after a failure

```bash
# [MAC TERMINAL] — inside .venv-kokoro
cd ~/sandbox/ME3/ME3
python -m genie.server
```

### J.6 Reconnect to Pi after SSH disconnection

```bash
# [MAC TERMINAL]
ssh mlops@raspberrypi.local
```
If the previous session's Genie process is still running, you'll see the prompt. If not, restart (J.4).

---

## Quick Reference — Demo Day Command Sequence

### Step 1: Open Mac Terminal — Check Network

```bash
# [MAC TERMINAL]
ipconfig getifaddr en0
# Note the IP — use it throughout
```

### Step 2: Start Inference Server

```bash
# [MAC TERMINAL]
cd ~/sandbox/ME3/ME3
source .venv-kokoro/bin/activate
lsof -i :8300          # Verify port is free
python -m genie.server
# Wait for "Uvicorn running on http://0.0.0.0:8300"
```

### Step 3: Verify Server Health

```bash
# [SECOND MAC TERMINAL]
curl -s http://127.0.0.1:8300/health
# Expected: {"status":"ok","asr":true,"tts":true,...}
```

### Step 4: SSH into Raspberry Pi

```bash
# [SECOND MAC TERMINAL]
ssh mlops@raspberrypi.local
# Password: __REPLACE_WITH_YOUR_PI_PASSWORD__
```

### Step 5: Activate Environment

```bash
# [PI TERMINAL]
cd ~/ME3/ME3
source ~/ME3/onit-venv/bin/activate
```

### Step 6: Check Microphone and Speaker

```bash
# [PI TERMINAL]
wpctl status
# Verify: USB mic listed as SOURCE, AirPods listed as SINK
python -c "import sounddevice; print(sounddevice.query_devices())"
```

### Step 7: Run Audio Validation (quick)

```bash
# [PI TERMINAL]
# Record a 3-second test
python -c "
import sounddevice as sd, numpy as np
rec = sd.rec(48000, samplerate=16000, channels=1, dtype='float32', device=1)
sd.wait()
print(f'RMS: {np.sqrt(np.mean(rec**2)):.4f}')
assert np.sqrt(np.mean(rec**2)) > 0.001, 'Mic too quiet!'
print('Mic OK')
"
# Test playback
python -c "
import sounddevice as sd, numpy as np
sd.play(0.3*np.sin(2*np.pi*440*np.linspace(0,1,24000)), 24000); sd.wait()
print('Speaker OK')
"
# Test ASR round-trip
curl -s -X POST http://192.168.100.134:8300/transcribe \
  -H "Content-Type: audio/wav" --data-binary @/tmp/test_mic.wav | python3 -m json.tool
```

### Step 8: Launch Genie

```bash
# [PI TERMINAL]
cd ~/ME3/ME3
PYTHONPATH=. python -m genie.pipeline_new
# Wait for "Say something to begin the conversation."
# Dashboard appears showing "Idle"
```

### Step 9: Execute Demonstration Tests

Follow the test sequence in **Section G** (Tests 1–10). Speak naturally and observe the dashboard.

### Step 10: Monitor System Health (during demo)

```bash
# [THIRD MAC TERMINAL] — optional second SSH session
ssh mlops@raspberrypi.local
vmstat 1              # CPU
free -h               # RAM
vcgencmd measure_temp # Temperature
curl -s http://192.168.100.134:8300/health  # Mac server check
```

### Step 11: Shut Everything Down

```bash
# [PI TERMINAL] — in the Genie terminal
Ctrl+C
# Wait for cleanup messages

# [PI TERMINAL]
exit

# [MAC TERMINAL] — in the server terminal
Ctrl+C
# Wait for shutdown
```

### Step 12: Verify Clean Shutdown

```bash
# [MAC TERMINAL]
lsof -i :8300          # Should be empty
ssh mlops@raspberrypi.local "ps aux | grep pipeline_new | grep -v grep"
# Should be empty
```

---

## Appendix A: Known Issues and Workarounds

| Issue | Status | Workaround |
|-------|--------|------------|
| Robot Airpods microphone not functional | Confirmed — HFP/HSP profile fails with PipeWire 1.4.2 on Pi | Use USB microphone |
| Pipeline smoke Test 5 (GENIE_SYSTEM_PROMPT import) | Pre-existing, same on Mac and Pi | Skip Test 5 — Tests 1–4 pass |
| 48kHz audio files cause slow ASR | Server-side resampling overhead | Always record at 16kHz mono |
| Mac IP address may change | DHCP | Verify with `ipconfig getifaddr en0` before each session |
| Bluetooth disconnections | Possible during long sessions | `bluetoothctl connect <MAC>` to reconnect |

## Appendix B: File Locations Reference

| File | Purpose | Location |
|------|---------|----------|
| `pipeline_new.py` | Main Genie pipeline entry point | `genie/pipeline_new.py` |
| `server.py` | Mac inference server (ASR + TTS) | `genie/server.py` |
| `mcp_server.py` | MCP server (runs inside Genie process) | `genie/mcp_server.py` |
| `services.py` | HTTP STT/TTS services | `genie/services.py` |
| `config.py` | Configuration loader | `genie/config.py` |
| `task_manager.py` | Background task management | `genie/task_manager.py` |
| `terminal_dashboard.py` | Rich terminal UI | `genie/terminal_dashboard.py` |
| `dashboard_observer.py` | Pipeline → dashboard bridge | `genie/dashboard_observer.py` |
| `.env` | Secrets and configuration | `~/ME3/ME3/.env` (Pi), `~/sandbox/ME3/ME3/.env` (Mac) |
| `.env.example` | Template (no secrets) | `~/sandbox/ME3/ME3/.env.example` |
| `M6A_Benchmark_Results.md` | M6-A benchmark results | `genie/benchmarks/results/` |
| `m6a_raw_results.json` | M6-A raw data | `genie/benchmarks/results/` |

## Appendix C: Environment Variable Reference

| Variable | Purpose | Default | Where Set |
|----------|---------|---------|-----------|
| `OPENROUTER_API_KEY` | LLM API key | (required) | `.env` |
| `GENIE_SERVER_URL` | Mac inference server URL | Auto-detected | `.env` |
| `GENIE_PORT` | Server port | 8300 | `.env` |
| `GENIE_LLM_MODEL` | LLM model name | `openai/gpt-4o-mini` | `.env` |
| `GENIE_LLM_TEMPERATURE` | LLM temperature | 0.6 | `.env` |
| `GENIE_LLM_MAX_TOKENS` | Max LLM tokens | 200 | `.env` |
| `GENIE_ASR_MODEL` | Whisper model | `mlx-community/whisper-large-v3-turbo` | `.env` |
| `GENIE_TTS_VOICE` | Kokoro voice | `af_bella` | `.env` |
| `GENIE_MIC_DEVICE` | Mic device index | Auto-detect | `.env` |
| `GENIE_SPEAKER_DEVICE` | Speaker device index | Default | `.env` |
| `GENIE_VAD_THRESHOLD` | VAD sensitivity | 0.5 | `.env` |
| `GENIE_MIN_SPEECH_MS` | Min speech duration | 280 | `.env` |
| `GENIE_MIN_SILENCE_MS` | Min silence for endpoint | 400 | `.env` |
| `GENIE_ENDPOINTING_MS` | Endpointing aggressiveness | 600 | `.env` |
| `MCP_PORT` | MCP server port | 8301 | `.env` |
| `MCP_HOST` | MCP server bind address | 127.0.0.1 | `.env` |

## Appendix D: Verification Checklist

Before starting the demo, confirm:

- [ ] Mac IP address noted and current
- [ ] Mac inference server running on port 8300 (health check returns ok)
- [ ] Both ASR and TTS show `true` in health response
- [ ] Pi reachable via SSH (`raspberrypi.local` or IP)
- [ ] Pi `.env` has correct `GENIE_SERVER_URL` pointing to Mac's current IP
- [ ] Pi `.env` has valid `OPENROUTER_API_KEY`
- [ ] USB microphone plugged in and detected
- [ ] Robot Airpods connected and paired
- [ ] Audio test recording has non-zero RMS
- [ ] Audio test playback works through AirPods
- [ ] Pi-to-Mac ASR test returns recognized text
- [ ] Pi-to-Mac TTS test produces audible speech
- [ ] Pi has ≥ 5 Gi disk space, ≥ 1 Gi RAM available
- [ ] Ports 8300 and 8301 not occupied on Pi
- [ ] `onit-venv` activated with Python 3.13.5
- [ ] pipecat-ai 1.15.0 installed in onit-venv
