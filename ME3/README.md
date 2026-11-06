# ME3 — Full-Duplex Voice Agent

A real-time, **full-duplex** voice assistant: it listens and speaks at the same
time, understands **barge-in** (interrupt it mid-sentence) vs **backchannel**
("mm-hm" while it talks → it keeps going), routes quick smart-home intents
locally, and delegates anything harder to a black-box agent harness (**Pipecat**,
with **OnIt** as an automatic backup) running on your own vLLM server.

Built step-by-step (Steps 1–8 complete). See
[`ME3 step-by-step implementation plan.md`](ME3%20step-by-step%20implementation%20plan.md).

## Architecture

```
mic ─▶ VAD (Silero) ─▶ ASR (whisper.cpp, local) ─▶ router (brain.py)
                                                          │
        ┌──────────────────────┬──────────────────────────┤
        ▼                      ▼                          ▼
  backchannel             quick_tool                 delegate
  ("mm-hm" → ack)      (ME2 top-10 smart-home)     (Pipecat → OnIt, black box)
        │                      │                          │
        └──────────┬───────────┴──────────────────────────┘
                   ▼
              TTS (Piper) ─▶ speaker        (full-duplex: barge-in aware)
```

| Step | Component | Module |
|------|-----------|--------|
| 1 | Audio I/O (Mac + Pi backends) | `src/voiceduplex/audioio/` |
| 2 | VAD — Silero (primary), WebRTC (backup) | `src/voiceduplex/vad/` |
| 3 | ASR — whisper.cpp (offline) | `src/voiceduplex/asr/` |
| 4 | TTS — Piper (primary, fast), Kokoro (backup, natural) | `src/voiceduplex/tts/` |
| 5 | Half-duplex loop + timeline logging | `src/voiceduplex/loop/` |
| 6 | Full-duplex: barge-in + backchannel | `src/voiceduplex/loop/full_duplex.py` |
| 7 | Quick tools (smart-home intents) | `src/voiceduplex/smarthome/` |
| 8 | Delegation — Pipecat (primary) + OnIt (backup) | `src/voiceduplex/delegate/`, `onit/` |
| 11 | Localhost web UI (mic + task sims + live output) | `web/` |

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
├── onit/                       # Step 8 agent-harness workers
│   ├── pipecat_worker.py       #   Pipecat (PRIMARY) worker
│   └── onit_worker.py          #   OnIt (BACKUP) persistent worker, same protocol
├── web/                        # Step 11 localhost UI (wraps the real loop)
│   ├── server.py               #   runs the loop + mirrors it to localhost
│   └── index.html              #   mic viz, task sims, live transcript
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

## Testing & running (the four entrypoints)

All commands run from the **ME3 project root** (the `ME3/` folder that holds
`config/`, `src/`, `onit/`, `scripts/`, `web/`). `cd ME3` from the repo root
first. `PYTHONPATH=src` puts the `voiceduplex`
package on the path. Pick the interpreter that has the deps — on this Mac that's
the conda base python (`/opt/miniconda3/bin/python3`); a plain `python3` works in
an interactive shell but a background launcher may pick a different one.

### 1. Offline health check — no mic, no network

Proves every layer is wired (VAD, ASR, TTS, router, delegate) using synthetic
audio. Run this first after any change; it's the fastest way to catch a broken
import or a missing model.

```bash
PYTHONPATH=src python3 scripts/offline_full_check.py
```

### 2. Live full-duplex — the product (live mic)

Real-time conversation with barge-in + backchannel. **Ctrl-C** stops everything,
including the OnIt/Pipecat worker.

```bash
PYTHONPATH=src python3 scripts/run_full_duplex.py
```

Useful flags:

```bash
# swap engines without editing config (default TTS is the piper->kokoro fallback)
PYTHONPATH=src python3 scripts/run_full_duplex.py --tts kokoro
PYTHONPATH=src python3 scripts/run_full_duplex.py --vad webrtc
PYTHONPATH=src python3 scripts/run_full_duplex.py --no-barge-in
```

### 3. Scripted live capability test (live mic, scored)

Walks through the six capabilities (quick tool ×3, barge-in, backchannel,
delegate) with a 15 s room-tone pre-check, then scores each step MATCH / GARBLED
/ BLANK. Use `--skip-delegate` to avoid the ~25 s OnIt wait, or `--only` to run
a single step.

```bash
PYTHONPATH=src python3 scripts/live_capability_test.py
PYTHONPATH=src python3 scripts/live_capability_test.py --skip-delegate
PYTHONPATH=src python3 scripts/live_capability_test.py --tts kokoro --only 4
```

> A majority of GARBLED/BLANK lines is a **mic/room** signature (self-interference
> from the loudspeaker, background noise), not a model bug — the test says so
> explicitly. Re-run with headphones or in a quieter room.

### 4. Localhost web UI (live mic + task simulators)

Wraps the **real** full-duplex loop and mirrors it to a browser page: live mic
waveform + VAD meter, the agent state machine, transcript/route/reply, barge-in
and backchannel events, ME2-style task simulators (lights, music, thermostat,
timers, math), and a terminal-output mirror. It is **non-invasive** — the loop's
terminal output is unchanged and still printed to stdout; the server only
*observes* it via an optional `event_cb` (default `None`, so `run_full_duplex.py`
behaves byte-for-byte as before).

```bash
PYTHONPATH=src python3 web/server.py
# then open  http://localhost:8321   (WebSocket on ws://localhost:8322)
```

Flags: `--port`, `--ws-port`, `--vad {silero,webrtc}`, `--tts {piper,kokoro,fallback}`,
`--no-barge-in`, `--log`. Requires `websockets` (already in the env).

## Step 8 — Delegation (Pipecat primary + OnIt backup)

Anything that isn't a backchannel or a quick-tool intent is handed to an agent
harness as a black box. ME3 measures turnaround / keep-alive / result-back —
never the harness internals. There are **two interchangeable harnesses** behind
one interface (`delegate/__init__.py`), both speaking the same line-oriented
stdin/stdout worker protocol:

- **Pipecat** (PRIMARY) — `onit/pipecat_worker.py`, in the isolated `onit-venv`
  (3.12). Two tiers, chosen automatically: (1) a real Pipecat LLM pipeline if
  `pipecat` is importable, else (2) a zero-dependency direct OpenAI-compatible
  call — so the primary never hard-fails. Chosen as primary because it is fast.
- **OnIt** (BACKUP) — `onit/onit_worker.py`, in the **same** `onit-venv`,
  against your own vLLM / OpenAI-compatible endpoint. Slower (cold boot ~25s),
  so it is the fallback.

With `backend: fallback` (the default), the loop tries **Pipecat** first and, on
*any* failure (endpoint unset, worker dies on boot, timeout, task error),
transparently retries on **OnIt**. The loop sees one backend and never knows
which harness answered. (Same pattern as TTS: Piper primary, Kokoro backup.)

- **Persistent worker** (default): one warm process spawned on first use, served
  over stdin/stdout. Subsequent tasks skip the cold boot (~25s → ~1s).
- **Kill on interrupt**: `FullDuplexLoop.stop()` tears down the worker, so
  Ctrl-C leaves no orphan processes.
- **Offline fallback**: set `delegate.backend: stub` for a canned reply with no
  network.

Configure in `config/config.yaml`:

```yaml
delegate:
  backend: fallback        # pipecat | onit | fallback | stub
  primary: pipecat         # first harness to try
  backup: onit             # second harness, used when primary fails
  mode: persistent         # persistent | oneshot
  python: /abs/path/onit-venv/bin/python   # ABSOLUTE: the venv lives OUTSIDE the repo
  worker: onit/onit_worker.py              # relative to the ME3 project root
  pipecat_worker: onit/pipecat_worker.py   # relative to the ME3 project root
  wait_sec: 120
  boot_timeout: 120
```

> **Where the venv lives:** the isolated `onit-venv` (3.12, holds both OnIt and
> Pipecat) is created **outside** the repo — one level above the `ME3/` folder
> (e.g. `../onit-venv`), so it is never committed. `python:` therefore uses an
> absolute path; the worker scripts are relative and travel with the folder.

> **Security note:** endpoint URLs and keys are supplied via the environment
> (`ONIT_HOST`, `PIPECAT_HOST`, `OPENAI_API_KEY`), **never** in this repo. The
> OnIt key also lives in the OS keychain (via `onit setup`). Point each worker
> at your own OpenAI-compatible / vLLM endpoint.

## Notes

- All local + offline (VAD, ASR, TTS, routing) — no cloud dependency for the
  core loop. Only Step 8 delegation calls an external model endpoint.
- The Step 11 web UI is a thin observer over the real loop (`web/server.py`);
  it adds no pipeline logic and leaves `run_full_duplex.py` unchanged. It needs
  `websockets` in the env.
- Porting Mac → Pi: change `audio.backend` and the model paths in
  `config/config.yaml`; no code changes.
