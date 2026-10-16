# ME2 — Voice-Controlled Smart Device (VCM)

Tiny, standalone **Voice Command Model** for real-time smart-device command
recognition on a Raspberry Pi 4/5. No cloud inference, no LLM, **no ASR**.

## v7 — what's new

- **New architecture: two-head `conformer_s`.** A small Conformer encoder
  (246,055 params) with an `intent_head` (20-way) **and** a `slot_head` (19-way).
  Slots (timer durations, temperatures, colors, …) are decoded **directly by the
  network** — the ASR stage from earlier versions is gone entirely.
- **Fully working terminal VCM.** Live mic loop: wake word → classify intent →
  decode slot → act on a simulated device → spoken-style response.
- **Deployable ONNX** (float + int8), dual output, sub-millisecond inference.

## Quick start (terminal VCM)

```bash
# 1. environment
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt

# 2. live microphone (wake word: "hey mycroft", then a command)
./.venv/bin/python -m runtime.main --mic

# 3. or replay a recorded clip
./.venv/bin/python -m runtime.main --file pi_bundle/sample_clips/timer_sample.wav

# 4. scripted end-to-end demo (all 10 command categories)
./.venv/bin/python -m runtime.main --demo
```

Example live interaction:

```
> hey mycroft set a timer for ten seconds
  [wake] peak 0.987  [intent] TIMER (0.95)  [slot] 10 seconds
  Timer set for 10 seconds.
```

## Layout
```
vcm/        core library: features, models (5 archs), slots, augment, losses, export
runtime/    on-device runtime (wakeword + VCMClassifier + device actions + main loop)
scripts/    pipeline: hf_to_dataset -> pack_data -> train_vcm -> evaluate_vcm -> export
configs/    variations.csv (93 command phrasings, from the HF dataset)
data/       hf/ (raw HF parquet), me2-dataset/ (manifest+audio), packs/ (npz)   [gitignored]
models/     ONNX exports (committed) + seed checkpoints (gitignored)  -> models/README.md
pi_bundle/  self-contained Pi deployment: ONNX + runtime + systemd unit + sample clips
report/     evaluation JSON/figures + stage reports
results/    training/eval summaries
tests/      unit tests (39)
docs/       dataset + architecture documentation + deck
```

## Pipeline
1. `scripts/hf_to_dataset.py` — build manifest from HF parquet, verify schema
2. `scripts/pack_data.py` — decode audio, extract log-mel, pack npz bundles
3. `scripts/train_vcm.py --arch <cnn|crnn|tcn|conformer_s|mlp>` — train w/ early stopping
4. `scripts/evaluate_vcm.py --checkpoint models/<arch>/best.pt` — full metric suite
5. `scripts/export_deploy_onnx.py --checkpoint models/conformer_s_seed0/best.pt` — ONNX + parity

## Headline results (conformer_s, held-out test)
- Accuracy **85.2 %** (CI [84.2, 86.2]), balanced accuracy **83.5 %**
- Slot exact-match **69.3 %** (77.3 % given correct intent)
- Wake word: 0 false-accepts over 250 negative clips @ threshold 0.75
- Inference: **~0.67 ms/clip** (≈1,494 clips/s)

Full detail in `report/` and `models/README.md`.
