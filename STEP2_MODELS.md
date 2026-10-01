# VCM — Step 2: Tiny On-Device Voice Command Models

Trained, evaluated, and exported **five candidate architectures** for recognizing the
10 smart-home voice commands (+ a REJECT/silence class) **fully on-device**, no cloud.
All models run through the same ONNX Runtime (CPU) path that will run on the Pi.

## The 11 classes
`1 PLAY_MUSIC · 2 QUESTION_SEARCH · 3 LIGHTS_ON_OFF · 4 DIM_COLOR_LIGHTS ·
5 SET_TIMER · 6 SET_ALARM · 7 THERMOSTAT · 8 MEDIA_CONTROL · 9 REMINDERS_LISTS ·
10 CALLS_MESSAGING · REJECT (ambient noise/silence)`

## Data
- **Source:** unified ETL manifest (141,681 clips) from Option B (primary) +
  SNIPS / SLURP / SpeechCommands v2 (aux) + FLEURS/LibriSpeech (acoustic only).
- **Split:** `data/processed/splits/` — 13,550 train / 2,880 val / 2,883 test.
  Per-intent cap 2,500; **speaker-disjoint** where speaker IDs exist (Option B,
  SNIPS), row-wise otherwise.
- **REJECT class:** synthesized from SpeechCommands `_background_noise_`
  (white/pink noise, tap, etc.) — the real "no command" case — 1,500 clips.
- **Features:** 16 kHz, 1.0 s window, 10 ms hop → fixed **T=101** frames.
  - MFCC: `(101, 40)` — for the logistic/DNN baselines
  - log-mel: `(101, 80)` — for the CNN/CRNN

## Results (held-out test set)

| # | Architecture | Features | Params | Test Acc | ONNX fp32 | ONNX INT8 | Latency (CPU) |
|---|--------------|----------|--------|----------|-----------|-----------|---------------|
| A | Logistic (mean-pool) | MFCC | **451** | 45.3% | 2.1 KB | **1.7 KB** | ~0.01 ms |
| B | Small DNN | MFCC | 14.6 k | 55.1% | 60 KB | **22 KB** | ~0.01 ms |
| C | 1-D CNN | log-mel | 246 k | **92.1%** | 962 KB | **253 KB** | 0.13 / 0.28 ms |
| D | CRNN (CNN+BiLSTM) | log-mel | 332 k | **94.5%** | 1.30 MB | **340 KB** | 0.36 / 0.41 ms |
| E | PocketSphinx grammar | GMM-HMM | ~0.5 M | (fallback) | — | **<1 MB** | ~10–50 ms |

**Every model is far under the <5 M-param / <1 MB-INT8 / real-time budget.**
Model inference is in the **sub-millisecond to sub-millisecond** range; the real
on-device latency is dominated by feature extraction (~30–60 ms) + microphone
capture, not the classifier.

## The key finding
**Feature choice matters more than model size.** Swapping MFCC-mean-pool for a
log-mel spectrogram lifted accuracy from ~45–55% to ~92–94% — a bigger gain than
going from a 451-param logistic model to a 332k-param CRNN on the same features.
Once you use log-mel, even the modest 1-D CNN reaches 92%.

## Per-intent accuracy (CRNN, the winner) — remarkably balanced
| Intent | Test Acc | | Intent | Test Acc |
|--------|----------|-|--------|----------|
| REJECT | **100%** | | SET_TIMER | 87.1% |
| PLAY_MUSIC | **100%** | | QUESTION_SEARCH | 89.0% |
| LIGHTS_ON_OFF | 95.0% | | REMINDERS_LISTS | 96.5% |
| SET_ALARM | 95.2% | | CALLS_MESSAGING | 94.9% |
| THERMOSTAT | 96.3% | | MEDIA_CONTROL | 94.1% |
| DIM_COLOR_LIGHTS | 93.0% | | | |

No weak spot — the lowest intent is SET_TIMER at 87%. (The earlier "weak intents"
were an artifact of the MFCC baselines, not the log-mel CNN/CRNN.)

## Recommendation
- **Primary: CRNN (D)** — 94.5% acc, 340 KB INT8, 0.41 ms. Best accuracy, still
  tiny. Fits the Pi with room to spare.
- **Lightweight alt: 1-D CNN (C)** — 92.1% acc, 253 KB INT8. Only ~2.4 pts lower
  for 25% less memory; pick this if you want the smallest good model.
- **Baseline to beat / last resort: PocketSphinx grammar (E)** — <1 MB, no
  training data needed, but exact-phrase only (poor paraphrase robustness). Keep
  as the simplest defensible fallback and the bar the neural model must clear.

## Files
```
scripts/
  features.py            # MFCC / log-mel extraction (16 kHz, 1 s, T=101)
  make_splits.py         # balanced speaker-disjoint 70/15/15 + reject
  make_reject.py         # synthesize REJECT class from background noise
  models.py              # A–D architectures (shared 11-class contract)
  train.py               # train + eval + export (TorchScript/ONNX/INT8)
  reexport.py            # re-export a saved state dict (no retrain)
  export_and_bench.py    # ONNX Runtime latency benchmark
  pocketsphinx_note.md   # Architecture E explained (GMM-HMM keyword spotting)
  pocketsphinx_cmds.jsgf # the 10-intent grammar for PocketSphinx
models/
  {logistic,dnn,cnn1d,crnn}_TAG.onnx          # fp32
  {logistic,dnn,cnn1d,crnn}_TAG_int8.onnx     # INT8 (the Pi artifact)
  {…}_TAG_meta.json / _bench.json             # params, acc, sizes, latency
```

## Reproduce
```bash
cd vcm/scripts
python make_splits.py --cap 2500 --oov-cap 1500 --seed 0   # (splits already built)
python train.py --model crnn --epochs 15 --bs 32 --lr 5e-4 # ~2 min on Apple Silicon
python export_and_bench.py --tag crnn_<TAG>
```

## Known limitations / next
- REJECT class is **synthesized noise**, not real ambient recordings — validate
  against real silence on the Pi before trusting the rejection rate.
- Weaker intents (DIM_COLOR, REMINDERS, CALLS) would benefit from more Option-B
  variants + paraphrase augmentation.
- Latency numbers are **model-only**; the full mic→feature→classify pipeline
  benchmark happens in Step 3 on the Pi.
