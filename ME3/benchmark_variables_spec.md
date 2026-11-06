# ME3 Full-Duplex Voice Agent — Final Set of Benchmark Variables

> **Purpose:** Fill the empty slot in `Final spec.pdf` → Section B.g *"Final Set of Variables: <insert here sa 231 meeting>"*.
> This is the proposed variable list to bring to the 231 meeting. It maps **every** requirement in Section B (a–f) to a concrete, measurable variable, and copies the **Lin et al. (Full-Duplex-Bench)** method for injecting pauses / backchannels / interruptions.
>
> **Sources:**
> - Class spec: `ME3/Final spec.pdf` (Section B, Benchmarking)
> - Method reference: **Lin et al., *Full-Duplex-Bench: A Benchmark to Evaluate Full-duplex Spoken Dialogue Models on Turn-taking Capabilities***, arXiv:2503.04721 (ASRU 2025), code: `github.com/DanielLin94144/Full-Duplex-Bench`
> - Naturalness metrics: **King V1.5 paper** (as referenced in spec B.e)
>
> **Design principle (from spec):** the **harness is a black box**. We measure *everything the Pi-side pipeline controls* (VAD, ASR, TTS, barge-in, backchannel, routing) and treat the harness's internal processing as one opaque interval. This keeps the benchmark comparable across teams regardless of which harness (Onit / Hermes / OpenCode / pi.dev) they wire in.

---

## 0. The one thing to agree on first (so metrics are comparable)

Before listing variables, lock the **event timeline**. Every metric below is a difference between two timestamps on this timeline. The pipeline must emit these timestamps to a log (JSONL) on every trial.

```
T0  USER_SPEECH_START        VAD detects user speech onset
T1  USER_SPEECH_END          VAD detects user speech offset (end of utterance)
T2  ASR_FINAL                ASR returns final transcript
T3  ROUTE_DECISION           agent decides: backchannel / quick-tool / delegate
T4  AGENT_SPEECH_START       TTS begins playing agent audio
T5  AGENT_SPEECH_END         TTS finishes (or is cancelled by barge-in)
T6  HARNESS_RESULT_READY     (black-box boundary) harness returns result
T7  RESULT_SPOKEN_START      agent begins speaking the result back
T8  TURN_COMPLETE            silence confirmed; turn closed

# Overlap / interruption events (only when user speaks while agent is speaking):
O0  OVERLAP_ONSET            user speech detected during [T4, T5]
O1  OVERLAP_CLASSIFIED       classifier labels overlap: BACKCHANNEL | INTERRUPT
O2  AGENT_REACTION           agent PAUSES (and resumes) or STOPS (and responds)
```

> **Black-box rule:** `T3 → T6` for a delegated task is reported as a single **harness turnaround** interval. We do *not* decompose inside it. (Spec B.b.i: "bare minimum: treat the harness as a blackbox.")

---

## 1. Variable groups (the proposed "Final Set")

Grouped into **4 families** so the 231 meeting can approve/trim family by family. Each variable has: **symbol, definition, unit, how it's computed, and which spec line it satisfies.**

### Family A — Latency (spec B.a "Real-time responses")
These are the headline numbers. All are **per-turn**, reported as median + p90 across trials.

| # | Symbol | Variable | Definition | Unit | Computed as | Spec |
|---|--------|----------|------------|------|-------------|------|
| A1 | `L_vad` | VAD detection latency | Time from user starting to speak to VAD confirming speech | ms | `T0 → (VAD fire)` | B.a |
| A2 | `L_asr` | ASR latency | Time from end of user speech to final transcript | ms | `T1 → T2` | B.a |
| A3 | `L_route` | Routing/decision latency | Time from transcript to route decision (backchannel vs tool vs delegate) | ms | `T2 → T3` | B.a |
| A4 | `L_tts_first` | Time-to-first-audio (TTFA) | Time from end of user speech to agent's first audible word — **the perceived "response speed"** | ms | `T1 → T4` | B.a, B.e.iii |
| A5 | `L_e2e` | End-to-end turn latency | Full round trip, user speech end → agent speech start | ms | `T1 → T4` (same as A4 for simple turns) | B.a |
| A6 | `L_turn` | Total turn duration | User speech end → turn fully closed | s | `T1 → T8` | B.a |

> **A4 (TTFA) is the single most important latency number** — it's what a human *feels*. Report it first.

### Family B — Turnaround time (spec B.b "Between agent harness turns, harness = black box")

| # | Symbol | Variable | Definition | Unit | Computed as | Spec |
|---|--------|----------|------------|------|-------------|------|
| B1 | `T_harness` | Harness turnaround (delegated tasks) | Opaque time the harness spends on a complex task | s | `T3 → T6` | B.b.i |
| B2 | `T_tool` | Quick-tool turnaround (easy tasks) | Time for a quick tool-call (non-harness) | ms | `T3 → T6` | B.b |
| B3 | `T_result_back` | Result-delivery latency | Time from harness result ready to agent speaking it back | ms | `T6 → T7` | B.b, B.e.iii |
| B4 | `T_keepalive` | Background-chat responsiveness | While harness runs, how fast the agent answers a *side* question (proves spec A.e.ii "nakakausap pa rin") | ms | side-question `T1 → T4` | A.e.ii |

> **B4 is the proof of the "keeps chatting while the harness works" requirement** — measure it explicitly with a scripted side-question injected during a long harness task.

### Family C — Naturalness / turn-taking (spec B.d "how natural a conversation feels" + B.e King V1.5)
This is the heart of "GPT-Live-1 feel." Copied from **Full-Duplex-Bench** (pause handling, backchanneling, turn-taking, interruption management) and **King V1.5**.

#### C1. Backchanneling (spec B.d.i, B.e.i)
| # | Symbol | Variable | Definition | Unit | Computed as | Spec |
|---|--------|----------|------------|------|-------------|------|
| C1a | `BC_rate` | Backchannel rate | How often the agent emits a backchannel ("Mhm?", "Yes?") per minute of conversation | count/min | count(`O1==BACKCHANNEL`)/duration | B.d.i, B.e.i |
| C1b | `BC_precision` | Backchannel precision | Of overlaps the agent *treated* as backchannel, how many were truly non-steering | % | TP_bc / (TP_bc+FP_bc) | B.d.iii |
| C1c | `BC_recall` | Backchannel recall | Of true non-steering overlaps, how many the agent correctly backchanneled (vs. wrongly stopping) | % | TP_bc / (TP_bc+FN_bc) | B.d.iii |
| C1d | `Annoy_index` | "Is it annoying?" (King V1.5 e.i) | Over-response rate: agent responses per user turn where no response was warranted | count/turn | count(unsolicited agent speech) | B.e.i |

#### C2. Pause detection (spec B.d.ii)
| # | Symbol | Variable | Definition | Unit | Computed as | Spec |
|---|--------|----------|------------|------|-------------|------|
| C2a | `Pause_sensitivity` | Min pause to trigger a response | Shortest user silence the agent waits before responding (too short = cuts people off; too long = feels dead) | ms | distribution of `(T1 → T4)` on non-interrupted turns | B.d.ii |
| C2b | `False_end_rate` | False end-of-turn | Times VAD declared "user finished" but the user was just pausing (agent jumped in) | % | count(false T1)/total pauses | B.d.ii |
| C2c | `Dead_air` | Excessive waiting | Times the agent stayed silent past a natural response point | ms | tail of `(T1 → T4)` | B.d.ii |

#### C3. Interruption handling (spec B.d.iii, B.e.ii) — **the hardest, most important group**
| # | Symbol | Variable | Definition | Unit | Computed as | Spec |
|---|--------|----------|------------|------|-------------|------|
| C3a | `Interrupt_acc` | Interruption classification accuracy | Correctly labeling an overlap as BACKCHANNEL vs GENUINE STEER | % | acc(`O1` vs ground-truth) | B.d.iii, B.e.ii |
| C3b | `Barge_in_latency` | Barge-in reaction speed | Time from user starting to interrupt to the agent actually stopping/pausing TTS | ms | `O0 → O2` | B.e.ii |
| C3c | `Resume_correctness` | Pause-and-resume correctness | After a *backchannel* overlap, does the agent correctly resume what it was saying (vs. dropping it)? | % | correct resumes / backchannel overlaps | B.e.ii |
| C3d | `Silence_after_overlap` | Silence after overlap (King V1.5 e.iii) | How long the agent stays silent once it detects an overlapping event, before reacting/responding | ms | `O0 → O2` (distribution) | B.e.iii |
| C3e | `Respond_speed_new` | Response speed to new command (King V1.5 e.iii) | After a genuine interruption, how fast the agent pivots to the new command | ms | `O2 → T4`(new) | B.e.iii |
| C3f | `Overlap_count` | Overlap frequency | Number of overlapping events per conversation (baseline for the above rates) | count | count(`O0`) | B.e.ii |

#### C4. Tone consistency (spec B.e.iv)
| # | Symbol | Variable | Definition | Unit | Computed as | Spec |
|---|--------|----------|------------|------|-------------|------|
| C4a | `Tone_consistency` | Prosodic/tone consistency across the conversation | Whether the agent's tone stays consistent with earlier turns (measured by LLM-as-judge on transcripts + optional prosody features) | 0–5 scale | LLM judge rubric | B.e.iv |

### Family D — Quality / correctness (spec B.c "LLM as a Judge?")
Optional but recommended — proves the agent isn't just fast but *right*.

| # | Symbol | Variable | Definition | Unit | Computed as | Spec |
|---|--------|----------|------------|------|-------------|------|
| D1 | `Task_success` | Task completion | Did the delegated/easy task actually succeed? | % | LLM judge + ground-truth | B.c |
| D2 | `Response_quality` | Content quality | LLM-as-judge score of the spoken answer (relevance, correctness, completeness) | 0–5 | LLM judge rubric | B.c |
| D3 | `Routing_correct` | Route accuracy | Did the agent pick the right path (backchannel / quick-tool / delegate)? | % | route label vs intended | B.b, A.d |
| D4 | `Conversation_naturalness` | Overall naturalness | Holistic LLM-as-judge "how natural did this feel?" | 0–5 | LLM judge rubric | B.d, B.c |

---

## 2. The command bank (spec B.f "Unified benchmarking")

One **single bank** of commands, **sampled per trial** (not all run every time). Must be **natural, conversational, interruptible.**

**Structure of each command item:**
```json
{
  "id": "CB-014",
  "category": "easy_tool | delegated_task | chit_chat | interrupt_scenario",
  "script": "Hey, what's the weather in Manila tomorrow?",
  "expected_route": "quick_tool",
  "injection": {                    // Lin et al. style injection (see §3)
    "type": "none | pause | backchannel | interruption",
    "offset_ms": 1200,             // where in the agent's speech to inject
    "inject_text": "wait, actually..."
  },
  "ground_truth": {                 // for scoring C3/D
    "overlap_is": "interruption",  // backchannel | interruption
    "expected_agent_reaction": "stop_and_respond"
  }
}
```

**Required category mix** (so every metric family is exercised):
1. **Chit-chat** (no tools) — measures pure latency + naturalness.
2. **Easy tool-calls** (weather, time, calculator, lookup) — measures `T_tool`, routing.
3. **Delegated tasks** (build a webpage, write a report, multi-step) — measures `T_harness`, `T_keepalive`, `T_result_back`.
4. **Interrupt scenarios** — scripted overlaps injected mid-agent-speech, each labeled backchannel vs. genuine steer. This is what drives all of Family C.
5. **Pause scenarios** — scripted user silences of varying length, to calibrate `Pause_sensitivity` / `False_end_rate`.

> **Live-testing mode (spec B.f):** the Pi can **read the script aloud** (TTS) to simulate the user, enabling unattended, reproducible trials. Same bank works for (a) human-live demos and (b) automated TTS-driven runs.

---

## 3. Injection method — copied from Lin et al. (Full-Duplex-Bench)

Per spec B.f: *"We copy the method of Lin et al. 2026 for testing (inserting pauses, backchanneling, interruptions, etc)."*

Full-Duplex-Bench evaluates four interactive behaviors — **pause handling, backchanneling, turn-taking, interruption management** — with automatic, reproducible metrics. We adopt its approach:

1. **Synthetic user audio with controlled timing.** Generate the user side as TTS audio with **programmable gaps** (pauses), **short filler utterances** (backchannels: "uh-huh", "mm"), and **mid-utterance cut-ins** (interruptions) at a known `offset_ms`.
2. **Known ground truth.** Because *we* generate the injections, we know exactly when each pause/backchannel/interruption occurs and what it "means" → this gives us the labels to score `C3a Interrupt_acc`, `C1b/c`, `C3c`.
3. **Automatic scoring.** Compare the agent's emitted events (from the §0 timeline) against the injected ground truth → compute the Family C percentages with no human annotation.
4. **Sampled trials.** Per spec B.f, each trial samples a **subset** of the bank (e.g., 15–20 items) so repeated runs aren't identical but remain statistically comparable.

> **Deliverable tie-in:** this is the "standard codebase for the benchmark" (Action Item) — a repo that (a) holds the command bank, (b) generates injected user audio, (c) captures the timeline log, (d) computes all variables in §1. 2–3 student maintainers + team write access.

---

## 4. Reporting format (so results are comparable across teams)

For each variable, report **median and p90** over N trials (suggest **N ≥ 30 turns** per family). One summary table per family:

| Variable | Median | p90 | Target (proposed) |
|----------|--------|-----|-------------------|
| A4 TTFA | … ms | … ms | < 800 ms feels "live" |
| B1 T_harness | … s | … s | (informational; black box) |
| C3a Interrupt_acc | … % | … % | > 80 % |
| C3b Barge_in_latency | … ms | … ms | < 300 ms |
| C1b BC_precision | … % | … % | > 75 % |
| D4 Naturalness | … /5 | … /5 | > 3.5 /5 |

*(Targets are proposals to sanity-check against in the meeting — not class requirements.)*

---

## 5. What to actually put in the `<insert here>` slot (condensed version for the spec doc)

If the spec wants a short bullet list rather than this whole doc, paste this:

> **Final Set of Variables**
> - **Latency (per-turn, median + p90):** VAD detection (A1), ASR (A2), routing (A3), **time-to-first-audio TTFA (A4)** ⭐, end-to-end (A5), total turn (A6).
> - **Turnaround (harness = black box):** harness turnaround (B1), quick-tool turnaround (B2), result-delivery (B3), **background-chat responsiveness while harness runs (B4)** ⭐.
> - **Naturalness / turn-taking:** backchannel rate + precision/recall (C1a–c), annoyance/over-response index (C1d), pause sensitivity + false-end rate (C2a–b), **interruption classification accuracy (C3a)** ⭐, barge-in latency (C3b), pause-and-resume correctness (C3c), silence-after-overlap + respond-speed (C3d–e), tone consistency (C4a).
> - **Quality (LLM-as-judge):** task success (D1), response quality (D2), routing correctness (D3), overall naturalness (D4).
> - **Method:** single sampled command bank (chit-chat / easy-tool / delegated / interrupt / pause), injected user audio per **Lin et al. Full-Duplex-Bench** (arXiv:2503.04721) for reproducible pause/backchannel/interruption injection; Pi can read scripts aloud for live testing.

---

## 6. Open questions to settle at the 231 meeting

1. **N per family** — 30 turns enough, or more for stable p90?
2. **LLM-as-judge model** — which model grades D1–D4? (Must be independent of the agent's LLM to avoid self-grading bias.)
3. **Tone consistency (C4a)** — prosody features (pitch/energy) or transcript-only LLM judge? Prosody is more rigorous but more work.
4. **Command bank size** — target total items (e.g., 100) and per-category split.
5. **Repo ownership** — confirm the 2–3 student maintainers + team write-access model (Action Item).
6. **Baseline** — do we benchmark against a half-duplex (turn-based) baseline of the same stack, to show the full-duplex gain? (Strong for the write-up.)
