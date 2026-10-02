# ME2 — Voice Command Model (VCM)

**AI231 Machine Learning Engineering | Version 6**

A tiny voice command model that classifies 19 smart-device intents (93 phrase variations) + REJECT from 1-second audio clips. Designed for Raspberry Pi 4 (4 GB) deployment with openWakeWord gating.

## Results

| Model | Params | Test Acc |
|-------|--------|----------|
| Logistic | 80K | 39.8% |
| DNN | 1.06M | 64.5% |
| 1-D CNN | 224K | 73.2% |
| **CRNN** | **719K** | **77.7%** ✅ |
| Transformer | 273K | 61.6% |

**Best model:** CRNN (CNN + BiLSTM) — 77.65% test accuracy, 2 ms inference (CPU), 702 KB INT8.

## Structure

```
ME2/
├── code/
│   ├── prep_data.py          # Download → extract → features
│   ├── train_models.py       # 5 architectures, early stopping
│   ├── evaluate.py           # Full metric battery
│   ├── plot_report.py        # 10 figures
│   ├── wakeword.py           # openWakeWord integration
│   ├── ui_simulation.py      # Hardware command simulation
│   └── benchmark/            # vcm-benchmark harness (sim mode)
├── data/
│   ├── hf/                   # HuggingFace dataset (parquet)
│   ├── audio/                # Extracted WAV clips
│   ├── features/             # MFCC + log-mel .npz
│   ├── music/                # 5 CC-licensed tracks
│   └── manifest.csv          # 20,611-row manifest
├── models/                   # Checkpoints + results JSON
├── report/
│   ├── report.md             # Full technical report
│   ├── evaluation_results.json
│   ├── wakeword_info.json
│   ├── ui_simulation_log.json
│   └── plots/                # 10 figures
└── ME2_Instructions-and-Notes.pdf
```

## Quick Start

```bash
python3 code/prep_data.py       # ~75s
python3 code/train_models.py    # ~15 min (MPS)
python3 code/evaluate.py        # ~30s
python3 code/plot_report.py     # ~10s
python3 code/wakeword.py        # ~5s
python3 code/ui_simulation.py   # instant
```

## Dataset

[`airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands) — 19 intents, 93 commands, speaker-disjoint splits, synthetic negatives for rejection.

## Wake Word

[openWakeWord](https://github.com/dscripka/openWakeWord) — "hey jarvis", ONNX, 2.2 ms/chunk, 36× real-time, Pi 4 compatible.

## Benchmark Harness

[`airimonda/vcm-benchmark`](https://github.com/airimonda/vcm-benchmark) — guided live benchmark for Pi deployment. Sim mode available for testing without hardware.
