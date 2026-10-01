# ME2 — Voice Command Model (VCM): Step 1–2 Snapshot

Pre-action snapshot of the tiny on-device voice-command model. This version
**only classifies intent** (10 classes + REJECT) — it does **not** execute any
action (no `mac_handler.py`, no `--broadcast` networking). That is intentional:
this is the checkpoint captured *before* the action-execution layer was added.

Built for the Raspberry Pi deployment (sub-millisecond inference, <1 MB INT8).

## Layout
```
ME2/
├── code/
│   ├── scripts/        # full training/ETL pipeline (etl.py, train.py, models.py,
│   │                   #   features.py, export_and_bench.py, make_splits.py,
│   │                   #   make_reject.py, validate_full.py, build_full_onnx.py, ...)
│   │   └── label_map.yaml   # single source of truth: 6 sources -> 10 intents
│   └── pi_bundle/      # deploy-ready bundle for the Pi
│       ├── live_demo.py       # CLASSIFY-ONLY live mic demo (pre-action)
│       ├── selftest.py        # no-mic self test
│       ├── vcm_infer.py       # inference engine (feature-parity w/ training)
│       ├── features.py        # log-mel / MFCC extraction
│       ├── model_int8.onnx    # 340 KB production model (INT8)
│       ├── model_float.onnx   # 1.3 MB float reference model
│       ├── SETUP_GUIDE.md     # step-by-step Pi setup
│       └── test_clips/        # 8 real Option B clips for verification
├── models/             # all trained checkpoints (.pt / .state.pt / .onnx / _int8.onnx)
│                       #   logistic, dnn, cnn1d, crnn  (24 files, ~7.6 MB)
└── dataset/
    ├── manifest.csv    # 141,681 clips, unified 10-intent manifest (21 MB)
    ├── label_map.yaml  # copy of the mapping used by ETL
    ├── REPORT.md       # ETL report
    ├── MANIFEST.md     # raw-source provenance
    ├── splits/         # train.csv / val.csv / test.csv / stats.json
    └── download_dataset.sh   # re-fetch the 33 GB raw audio (all public)
```

## Models (best-fit, smallest first)
| Arch | Params | INT8 | Test acc |
|------|--------|------|----------|
| Logistic (MFCC mean-pool) | 451 | 2 KB | 45.3% |
| DNN (MFCC) | 14.6 k | 22 KB | 55.1% |
| 1-D CNN (log-mel) | 246 k | 253 KB | 92.1% |
| **CRNN (log-mel)** | **332 k** | **340 KB** | **94.5%** ← production |

Production model = `crnn_1790586409_int8.onnx` (copied into `code/pi_bundle/model_int8.onnx`).

## Dataset
The raw audio (≈33 GB: Option B + LibriSpeech + SLURP + FLEURS fil_ph +
SpeechCommands v2 + SNIPS) is **too large for git** (2 GB/file cap). It is
reproducible: run `dataset/download_dataset.sh` to re-fetch all public sources.
The unified `manifest.csv` + `splits/` fully specify the training data and are
committed here.

Common Voice (EN) is gated (free Mozilla account) and STOP (8 domains) has no
stable public URL — both excluded from the auto-download.

## Run it (classify-only)
```bash
cd code/pi_bundle
python3 selftest.py --model model_int8.onnx      # no mic
python3 live_demo.py --model model_int8.onnx     # live mic, prints intent + confidence
```
