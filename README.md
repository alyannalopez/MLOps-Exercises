# ME3 — Full-Duplex Voice Agent

A real-time, **full-duplex** voice assistant: it listens and speaks at the same
time, understands **barge-in** (interrupt it mid-sentence) vs **backchannel**
("mm-hm" while it talks → it keeps going), routes quick smart-home intents
locally, and delegates anything harder to a black-box agent harness (**OnIt**)
running on your own vLLM server.

Built step-by-step (Steps 1–8 complete). See
[`ME3 step-by-step implementation plan.md`](ME3%20step-by-step%20implementation%20plan.md).

## Architecture

```
mic ─▶ VAD (Silero) ─▶ ASR (whisper.cpp, local) ─▶ router (brain.py)
                                                          │
        ┌──────────────────────┬──────────────────────────┤
        ▼                      ▼                          ▼
  backchannel             quick_tool                 delegate
  ("mm-hm" → ack)      (ME2 top-10 smart-home)     (OnIt harness, black box)
        │                      │                          │
        └──────────┬───────────┴──────────────────────────┘
                   ▼
              TTS (Kokoro) ─▶ speaker        (full-duplex: barge-in aware)
```

| Step | Component | Module |
|------|-----------|--------|
| 1 | Audio I/O (Mac + Pi backends) | `src/voiceduplex/audioio/` |
| 2 | VAD — Silero (primary), WebRTC (backup) | `src/voiceduplex/vad/` |
| 3 | ASR — whisper.cpp (offline) | `src/voiceduplex/asr/` |
| 4 | TTS — Kokoro (primary), Piper (backup) | `src/voiceduplex/tts/` |
| 5 | Half-duplex loop + timeline logging | `src/voiceduplex/loop/` |
| 6 | Full-duplex: barge-in + backchannel | `src/voiceduplex/loop/full_duplex.py` |
| 7 | Quick tools (smart-home intents) | `src/voiceduplex/smarthome/` |
| 8 | Delegation to OnIt (persistent worker) | `src/voiceduplex/delegate/` |

## Layout

```
ME3/
├── config/config.yaml          # the ONLY knobs you retune (Mac -> Pi)
├── src/voiceduplex/            # the pipeline (package: voiceduplex)
│   ├── audioio/  vad/  asr/  tts/
│   ├── smarthome/              # Step 7 quick tools
│   ├── delegate/               # Step 8 OnIt backend (persistent worker)
│   └── loop/                   # half_duplex, full_duplex, brain, classifier
├── scripts/                    # entrypoints + tests
│   ├── run_full_duplex.py      # THE product — live full-duplex loop
│   ├── run_loop.py             # half-duplex loop
│   ├── offline_full_check.py   # offline health check (no mic/network)
│   ├── live_capability_test.py # scripted live capability test
│   └── download_model.sh, setup_kokoro.sh
├── models/                     # whisper + kokoro weights (NOT committed)
├── data/                       # runtime artifacts (NOT committed)
└── requirements.txt
```

## Setup

Requires **Python 3.10–3.12** for the Kokoro TTS path (kokoro-onnx caps
`<3.14`). The main pipeline runs on 3.12+.

```bash
# 1. main env
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. native ASR binary
brew install whisper-cpp            # Mac
# Pi: build whisper.cpp for aarch64

# 3. Kokoro in an isolated 3.12 venv (it can't share the main env)
python3.12 -m venv .venv-kokoro
.venv-kokoro/bin/python -m pip install -U pip kokoro-onnx soundfile numpy

# 4. download model weights (whisper + kokoro)
./scripts/download_model.sh base.en
./scripts/setup_kokoro.sh
```

## Run

```bash
# offline health check — proves VAD/ASR/TTS/router/delegate are wired
PYTHONPATH=src python3 scripts/offline_full_check.py

# live full-duplex conversation (Ctrl-C stops everything, incl. the OnIt worker)
PYTHONPATH=src python3 scripts/run_full_duplex.py

# scripted live capability test (room-tone pre-check + scored steps)
PYTHONPATH=src python3 scripts/live_capability_test.py
```

## Step 8 — OnIt delegation

Anything that isn't a backchannel or a quick-tool intent is handed to **OnIt**
as a black box. ME3 measures turnaround / keep-alive / result-back — never the
harness internals.

- **Persistent worker** (default): one warm OnIt process spawned on first use,
  served over stdin/stdout. Subsequent tasks skip the cold boot (~25s → ~1s).
- **Kill on interrupt**: `FullDuplexLoop.stop()` tears down the worker, so
  Ctrl-C leaves no orphan processes.
- **Offline fallback**: set `delegate.backend: stub` for a canned reply with no
  network.

Configure in `config/config.yaml`:

```yaml
delegate:
  backend: onit            # onit | stub
  mode: persistent         # persistent | oneshot
  python: /path/to/onit-venv/bin/python
  worker: /path/to/onit_worker.py
  wait_sec: 120
  boot_timeout: 120
```

> **Security note:** the OnIt API key is stored in the OS keychain (via
> `onit setup`), **never** in this repo. Point the worker at your own
> OpenAI-compatible / vLLM endpoint.

## Notes

- All local + offline (VAD, ASR, TTS, routing) — no cloud dependency for the
  core loop. Only Step 8 delegation calls an external model endpoint.
- Porting Mac → Pi: change `audio.backend` and the model paths in
  `config/config.yaml`; no code changes.
