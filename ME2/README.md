# ME2 — Voice Command Model (VCM) for Smart Devices

A tiny, on-device voice command classifier for the most common smart-device
commands. Trained **exclusively on the Option B dataset** (Sir Mark's MEX2
corpus) — 18,375 clips, 100 speakers, 10 intents + rejection. No ASR, no
cloud: a small neural net maps 1-second audio windows directly to a command.

## Headline result

| Metric | Value |
|---|---|
| Best architecture | **1-D CNN** (log-mel, 246,443 params) |
| Test accuracy (speaker-disjoint) | **97.91%** |
| Macro F1 | 97.57% |
| Command recall | 97.68% |
| Reject recall | 100% |
| Calibration (ECE) | 0.0062 |
| Float32 / INT8 size | 0.99 MB / ~253 KB |
| CPU inference (1 s clip) | 0.42 ms (p50) |

Full numbers: [`report/VCM_Final_Report.md`](report/VCM_Final_Report.md) and
[`models/eval_results.json`](models/eval_results.json).

## Repository layout

```
ME2/
├── ME2_Instructions-and-Notes.pdf   # exercise instructions
├── README.md                        # this file
├── code/
│   └── scripts/                     # full reproducible pipeline
│       ├── label_map.py             # 32 raw folders -> 10 intents (source of truth)
│       ├── features.py              # MFCC (T,40) / log-mel (T,80) extractor
│       ├── models.py                # 4 architectures, shared (B,T,F)->(B,11) contract
│       ├── build_manifest.py        # step 1: scan raw data -> manifest.csv
│       ├── build_reject.py          # step 2: synthesize REJECT noise class
│       ├── build_splits.py          # step 3: speaker-disjoint 70/15/15 split
│       ├── build_feats.py           # step 4: extract + cache features (.npz)
│       ├── train_benchmark.py       # step 5: train all 4 archs, benchmark
│       ├── evaluate.py              # step 6: full metric battery
│       ├── plot_report.py           # step 7: all report figures
│       └── export_onnx.py           # step 8: ONNX float32 + int8 export
├── data/
│   ├── raw/optionB_v2/              # symlink -> Option B dataset (18,375 wav)
│   ├── manifest.csv                 # 18,375 rows: path, intent, speaker, condition
│   ├── reject/                      # 900 synthesized noise clips (REJECT class)
│   ├── splits/{train,val,test}.csv  # speaker-disjoint split + stats.json
│   └── feats/*.npz                  # cached MFCC + log-mel features
├── models/                          # checkpoints, histories, benchmark, eval, ONNX
└── report/
    ├── VCM_Final_Report.md          # full technical report
    └── plots/                       # 12 figures
```

## Reproduce from scratch

Requires: Python 3.10+, `torch`, `librosa`, `soundfile`, `pandas`,
`scikit-learn`, `matplotlib`, `onnx`, `onnxruntime`.

```bash
cd code/scripts
python3 build_manifest.py     # ~1 min   (scan raw data)
python3 build_reject.py       # ~10 s    (synthesize reject class)
python3 build_splits.py       # ~2 s     (speaker-disjoint split, 0 leaks)
python3 build_feats.py        # ~3 min   (MFCC + log-mel, cached)
python3 train_benchmark.py    # ~10 min  (4 archs, MPS, seed 42)
python3 evaluate.py           # ~1 min   (full metric battery)
python3 plot_report.py        # ~10 s    (12 figures)
python3 export_onnx.py        # ~30 s    (float32 + int8, parity check)
```

All steps are deterministic (seed 42) and idempotent — rerunning overwrites
outputs cleanly.

## The 10 intents

| # | Intent | Option B raw folders |
|---|--------|----------------------|
| 0 | PLAY_MUSIC | PLAY_MUSIC |
| 1 | QUESTION_SEARCH | WEATHER, TIME |
| 2 | LIGHTS_ON_OFF | LIGHT_ON, LIGHT_OFF |
| 3 | DIM_COLOR_LIGHTS | BRIGHTNESS_{20,60,100}, COLOR_{RED,GREEN,BLUE,YELLOW} |
| 4 | SET_TIMER | TIMER_{10s,30s,1m} |
| 5 | SET_ALARM | ALARM_{4_00AM,8_00AM,9_00PM} |
| 6 | THERMOSTAT | TEMPERATURE_{18,22,26} |
| 7 | MEDIA_CONTROL | PAUSE, STOP, NEXT, VOLUME_UP, VOLUME_DOWN |
| 8 | REMINDERS_LISTS | CREATE_REMINDER_{STUDY,EXERCISE,DRINK_WATER}, LIST_REMINDERS |
| 9 | CALLS_MESSAGING | CALL, MESSAGE |
| 10 | REJECT | synthesized ambient noise (no speech) |

## Data & method highlights

- **Dataset**: Option B only (per the revised instructions). 18,375 clips,
  100 speakers, clean + noisy recordings, 32 fine-grained value folders.
- **Split**: *speaker-disjoint* 70/15/15 — no speaker appears in more than
  one split (verified: 0 leaks). This tests generalization to unknown voices.
- **Features**: loudest-1-second window, 16 kHz, MFCC(40) for the small
  models, log-mel(80) for the CNN/CRNN.
- **Training**: Adam, cosine LR, early stopping (patience 25), seed 42, MPS.
- **Constraint honored**: no ASR anywhere — classification only.
