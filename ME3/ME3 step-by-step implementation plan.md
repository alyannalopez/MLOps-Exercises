# ME3 — Full-Duplex Voice Agent on Raspberry Pi
## Step-by-Step Implementation Plan (NOT a build — planning only)

> **Status:** Planning document. Nothing here is implemented. Each step ends in a
> *verifiable deliverable* so you know when to move on. Order is chosen so every
> later step stands on a tested earlier step, and the two gating items
> (working Pi + a real ASR/TTS backend) are hit first.
>
> **Companion docs:** `Final spec.pdf` (requirements), `benchmark_variables_spec.md`
> (the §B.g variable list + command bank + Lin et al. injection method).

---

## 0. What we are building (one picture)

```
[ Mic 16kHz ]
     │
     ▼
┌──────────────┐   local, on Pi
│  VAD         │   "is a human speaking right now?"
└──────────────┘
     │ (audio frames while someone talks)
     ▼
┌──────────────┐   YOUR MCP server (wraps SOTA ASR)
│  ASR         │   Deepgram / Whisper / MLX ...
└──────────────┘
     │ (text)
     ▼
┌──────────────┐   agent harness: onit / pipecat / crew
│  Agent brain  │   routes: backchannel | quick-tool | delegate
└──────────────┘
     │ (reply text)  ──► for hard tasks: delegate to a big harness
     ▼                     (Onit/Hermes/OpenCode/pi.dev) via MCP,
┌──────────────┐   YOUR MCP server (wraps SOTA TTS)     possibly on DGX/A100
│  TTS         │   Cartesia / MLX / edge-tts ...
└──────────────┘
     │ (audio)
     ▼
[ Speaker ]   ← and the Mic is STILL LISTENING  (that = full-duplex)
```

**Full-duplex** = the mic keeps capturing while the speaker is playing, so the user
can interrupt ("barge in"), backchannel ("mm-hm"), or steer mid-sentence. Half/
semi-duplex = strict turn-taking (walkie-talkie). The whole project is about making
the agent behave like a human in a conversation, not a push-to-talk bot.

**Your actual build work (counts as yours):**
1. The **MCP servers** that wrap ASR and TTS (run on the Pi).
2. The **VAD** install on the Pi.
3. Wiring it into a **harness** (onit/pipecat/crew) with MCP support.
4. The **latency/naturalness benchmark** (the classwork) — command bank + injection +
   timeline capture + scoring, per `benchmark_variables_spec.md`.

You **may** use commercial APIs for the heavy models, but the servers/wrappers are yours.

---

## 1. Why this order (and not UI-first)

Same logic as the ESO project: **build the testable core first, the thin UI/demo last.**
- The graded contribution is the **full-duplex pipeline + the benchmark**, not a pretty UI.
- A UI hides failures. If you wire a GUI to a half-built pipeline, you can't tell if a
  bug is the UI or the voice loop.
- The two **gating items** are hardware (a provisioned Pi) and a working ASR/TTS backend.
  Everything else depends on those, so they come first.

```
Pi ready → VAD → ASR MCP → TTS MCP → half-duplex loop → full-duplex (barge-in)
         → easy tasks → delegation → benchmark harness → command bank → run & report
```

---

## 2. Step-by-step

### STEP 1 — Provision the Raspberry Pi (gate #1)
Get the Pi (4 or 5) actually usable for voice.
- OS flashed (Raspberry Pi OS / Lite), SSH + network working, stable power.
- **Audio I/O confirmed:** a USB mic + speaker (or a ReSpeaker HAT) — record 5 s and
  play it back. Confirm 16 kHz capture works (`arecord`/`aplay` or Python `sounddevice`).
- Python env (venv/conda), clock synced (you'll timestamp every event for the benchmark).
- Decide Pi role: it runs **VAD + orchestration + MCP servers** but **avoids heavy model
  inference** (per spec A.a). Heavy models live on the DGX/A100 or a commercial API.

✅ **Deliverable:** a Pi that records 16 kHz mono from the mic and plays audio to the
speaker, reachable over SSH, with a venv ready. *(This was the "TBD — Pi not yet
provisioned" item in your ME2 deck — it gates everything.)*

---

### STEP 2 — Install VAD on the Pi (local voice detection)
VAD is the "is anyone talking?" gatekeeper. Runs **locally** so silence never hits the cloud.
- Pick a lightweight VAD: **Silero VAD** (recommended — small, ONNX, runs on the Pi) or
  webRTC VAD.
- Wrap it in a small streaming service that emits events: `speech_start`, `speech_end`,
  with timestamps. Tune `min_silence_ms` / threshold so it doesn't fire on fan noise.
- **This VAD is also what detects barge-in** in Step 6 (user speech while TTS is playing).

✅ **Deliverable:** a script that streams the mic and prints `speech_start`/`speech_end`
with ms timestamps, reliably ignoring ambient noise. Test: talk, pause 1 s, talk again →
two clean segments.

---

### STEP 3 — Build the ASR MCP server (gate #2)
Wrap a SOTA ASR in an MCP server you run on the Pi.
- Choose the ASR backend: **Deepgram** (fast, streaming, commercial), **Whisper**
  (open, can run on DGX), or **MLX** (Apple Silicon, if that's your inference box).
  Per the notes, raw ASR/TTS on the DGX 24/7 is allowed; avoid hosting raw LLMs there.
- Implement it as an **MCP server** exposing a tool like `transcribe(audio) -> text`
  (streaming preferred for low latency). You build + run this server; the model may be an API.
- Verify it's callable through an MCP client (the harness will be that client later).

✅ **Deliverable:** an MCP server on the Pi that, given recorded audio, returns a
transcript via an MCP tool call. Latency noted (this feeds benchmark var A2).

---

### STEP 4 — Build the TTS MCP server
Mirror of Step 3 for speech-out.
- Choose TTS: **Cartesia** (low-latency, commercial), **edge-tts** (free), or **MLX** TTS.
- MCP server exposing `synthesize(text) -> audio` (streaming chunks so first audio is fast).
- **First-audio-out time here is a key benchmark var (TTFA, A4).**

✅ **Deliverable:** an MCP server on the Pi that turns text into playable audio via an
MCP tool call. Play it on the speaker.

---

### STEP 5 — Wire the HALF-duplex loop (prove the basics first)
Before full-duplex, get a clean **turn-based** conversation working end-to-end. This is
your debugging baseline — if it's broken here, full-duplex will be un-debuggable.
- Install the **harness** (onit / pipecat / crew) with MCP support on the Pi.
  - Note from spec: **Pipecat** gave good out-of-the-box prompts but was weak on agentic
    tasks (tool calling uncertain); **Crew** is enterprise-priced. Decide with the team.
- Loop: VAD detects user speech → send to ASR MCP → harness reasons → reply to TTS MCP →
  play. Strict turn-taking (agent speaks, mic effectively gated until it finishes).
- Add **acknowledgement/backchannel replies** as simple canned responses first
  ("Mhm?", "Yes?", "Let me check that…").

✅ **Deliverable:** a half-duplex voice agent on the Pi you can have a basic back-and-forth
with. Log every event with a timestamp (this log format becomes the benchmark timeline).

---

### STEP 6 — Make it FULL-duplex (barge-in + backchannel + interruption)
This is the heart of the project. The mic stays live while TTS plays.
- **Barge-in:** while the agent is speaking, VAD detects user speech → stop/pause TTS
  quickly (target < 300 ms, benchmark var C3b).
- **Classify the overlap:** is the user's overlap a **backchannel** ("mm-hm", "uh-huh" —
  keep talking) or a **genuine interruption/steer** (stop and respond)? This drives
  `Interrupt_acc` (C3a) and `Resume_correctness` (C3c).
  - Simple start: short filler patterns → backchannel; new content/keyword → steer.
  - Improve later (small classifier or LLM judgment on the partial transcript).
- **Keep-alive:** while a hard task is delegated (Step 8), the agent must still be able to
  chat with the user (benchmark var B4), then report the result back.

✅ **Deliverable:** you can interrupt the agent mid-sentence and it stops and responds;
a "mm-hm" during its speech does *not* make it stop. Both behaviors observable + logged.

---

### STEP 7 — Easy real-world tasks (the ME2 top-10)
Per the notes: **start with the ~10 most common tasks from ME2**, then top up with complex ones.
- ME2 had 20 intents / 18 slots (edge keyword+intent model). Pull the **top ~10 highest-frequency**
  intents as your starter task set (e.g., set timer, what time, weather, reminder, todo, lookup…).
- Implement each as a **quick tool call** the agent can make (timers, clock, web lookup, calc).
- Confirm routing: easy task → quick tool (not delegated).

✅ **Deliverable:** the agent correctly performs the top-10 easy tasks by voice, each via a
tool call. (Confirm the exact 10 with the team — ME2's label set is the source.)

---

### STEP 8 — Delegation to a bigger harness (complex tasks)
For hard/long tasks, the Pi agent delegates instead of doing the work itself.
- Connect the Pi agent to an available harness (**Onit / Hermes / OpenCode / pi.dev**) via MCP.
- Example: "make a webpage about X" / "write me a report" → delegate; agent says "Let me
  check that," keeps chatting while it runs, then delivers the result.
- Treat the delegated harness as a **black box** in the benchmark (measure turnaround B1,
  keep-alive B4, result-back B3 — not internals).

✅ **Deliverable:** one delegated complex task completes end-to-end, with the agent staying
conversational during the wait and reporting the result.

---

### STEP 9 — Benchmark harness (the classwork)
Build the **standardized measurement method** per `benchmark_variables_spec.md`.
- **Timeline logger:** every run emits a timestamped event log (mic speech_start/end,
  ASR done, route decided, TTS first-audio, barge-in, backchannel, delegate start/end).
- **Injection generator (Lin et al. / Full-Duplex-Bench, arXiv:2503.04721):** synthesize the
  *user* side as TTS with **programmable pauses, filler backchannels, and mid-utterance
  interruptions** at known `offset_ms` → gives you ground-truth labels for free.
- **Scorer:** compares the logged timeline vs. injected ground truth → computes all
  variables in the spec (Family A latency, B turnaround, C turn-taking, D LLM-judge quality).
- **Reporting:** median + p90 per variable over N≥30 turns per family (spec §4).
- **Live mode:** the Pi can read the script aloud to simulate the user for unattended,
  reproducible trials.

✅ **Deliverable:** a runnable benchmark that, given the agent, produces the §4 summary
tables (TTFA, Barge_in_latency, Interrupt_acc, BC precision/recall, Naturalness, …).

---

### STEP 10 — Command bank
Build the **single bank of commands** (spec B.f), sampled per trial.
- Categories (so every metric family is exercised):
  1. Chit-chat (pure latency + naturalness)
  2. Easy tool-calls (weather/time/calc/lookup → `T_tool`, routing)
  3. Delegated tasks (webpage/report/multi-step → `T_harness`, keep-alive, result-back)
  4. Interrupt scenarios (labeled backchannel vs. genuine steer → drives Family C)
  5. Pause scenarios (varying user silences → calibrates pause sensitivity / false-end)
- Each item follows the JSON schema in the spec (id, category, script, expected_route,
  injection, ground_truth).
- Target size + per-category split to be settled at the 231 meeting (open question #4).

✅ **Deliverable:** a versioned command bank (JSON) covering all 5 categories, checked into
the shared benchmark repo.

---

### STEP 11 — Run, score, report (and the half-duplex baseline)
- Run the benchmark (sampled trials) against your full-duplex agent.
- **Strong move (open question #6):** also run the **same stack in half-duplex mode** as a
  baseline, so the write-up shows the *full-duplex gain* concretely.
- Produce the §4 tables + a short write-up. Settle the open questions from the spec §6
  (N per family, LLM-judge model, tone-consistency method, bank size, repo ownership, baseline).

✅ **Deliverable:** final benchmark report (full-duplex vs. half-duplex baseline) + write-up.

---

## 3. Suggested sequencing / rough timeline

| Phase | Steps | Gate / exit condition |
|---|---|---|
| **P0 Setup** | 1, 2 | Pi records/plays 16 kHz; VAD fires cleanly on speech |
| **P1 Voice I/O** | 3, 4 | ASR + TTS MCP servers return correct transcript/audio via MCP |
| **P2 Core loop** | 5 | Half-duplex conversation works, fully timestamp-logged |
| **P3 Full-duplex** | 6 | Barge-in + backchannel-vs-interrupt both work and are logged |
| **P4 Tasks** | 7, 8 | Top-10 easy tasks + one delegated complex task succeed |
| **P5 Benchmark** | 9, 10, 11 | Benchmark runs, command bank ready, report produced |

> P0 is the critical path — **nothing else moves until the Pi is provisioned and audio works.**

---

## 4. Key decisions to lock with the team / Sir Rowel (from the spec)
1. **Harness choice** — onit vs. pipecat vs. crew (cost + agentic strength).
2. **ASR/TTS backends** — commercial API vs. raw models on the DGX/A100 24/7 (allowed for
   raw ASR/TTS; avoid raw LLMs there — use QWEN).
3. **The exact top-10 ME2 tasks** for the starter set.
4. **Benchmark open questions** (spec §6): N per family, LLM-judge model (must be
   independent of the agent's LLM), tone-consistency method, command-bank size,
   repo ownership (2–3 student maintainers + team write access), half-duplex baseline.
5. **Coordinate with SHARC** on inference/API distribution for the models.

---

## 5. What I have NOT done
- No code written, no Pi configured, no MCP servers built, no benchmark implemented.
- This is the plan only. Say the word when you want to start on a specific step.
