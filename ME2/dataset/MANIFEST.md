# VCM Dataset Manifest — Step 1 (collected 2026-09-28)

Total on disk: ~22.5 GB (raw/, before extraction)

## ✅ Downloaded (public, verified)

| Dataset | Dir | Size | Source | Format | Status |
|---------|-----|------|--------|--------|--------|
| **OptionB** (class) | optionB/ | 1.6 GB | github.com/markandrian30/AI231 (MEX2/OptionB) | 18,375 WAV, by intent, clean+noisy | ✅ cloned, 100% |
| **LibriSpeech** | librispeech/ | 6.6 GB | openslr.org/resources/12 | dev-clean 322M, train-clean-100 5.9G, test-clean 331M (.tar.gz) | ✅ gzip OK |
| **SLURP** | slurp/ | 6.3 GB | huggingface.co/qmeeus/slurp | 17 parquet (train/dev/test) | ✅ done |
| **SpeechCommands v2** | speechcommands/ | 2.3 GB | storage.googleapis.com/download.tensorflow.org | speech_commands_v0.02.tar.gz | ✅ gzip OK |
| **FLEURS fil_ph** | fleurs/ | 5.6 GB | huggingface.co/google/fleurs | fil_ph only: data/ 2.5G WAV + parquet-data/ 3.1G | ✅ done |
| **SNIPS SmartLights** | snips/ | 137 MB | huggingface.co/qmeeus/smart-lights-en-close-field | train/val/test parquet | ✅ done |

## ❌ NOT downloaded (need credentials / not public)

| Dataset | Why | How to get it |
|---------|-----|---------------|
| **Common Voice (EN)** | Gated — official CDN returns HTML interstitial, S3 = 403, HF = login required | Create free Mozilla account → commonvoice.mozilla.org → Download → English → 17.0 (clips.tar.gz, ~2–3 GB) |
| **STOP (8 domains)** | Not on HF/GitHub under any public id found | Likely the classmate's own build — ask for the direct link (GDrive/MEGA/HF) |
| **TimersAndSuch** | Not found on HF or GitHub | Ask classmate for the direct link |
| **56 GB classmate pool** | Private drive | Ask for the share link |

## Notes
- OptionB intents seen: PAUSE, NEXT, TIMER_1m, TIMER_30s, PLAY_MUSIC, BRIGHTNESS_20/60,
  CREATE_REMINDER_STUDY/EXERCISE, TEMPERATURE_22, VOLUME_DOWN, LIGHT_ON, LIST_REMINDERS,
  STOP, MESSAGE, ALARM_8_00AM, ALARM_9_00PM (+ more). Naming: `<INTENT>_s<speaker>_v<variant>_<clean|noisy>.wav`
- LibriSpeech is general English speech (good for acoustic pretraining / transfer), not smart-home commands.
- SLURP covers weather/time/alarm/timer/reminder/call intents (closest to VCM #2,5,6,9,10).
- SpeechCommands v2 = 35 basic commands (yes/no/up/down/left/right/on/off/go/stop, etc.) — great for
  silence-rejection + on/off/stop/next media controls.
- Next: extract the 4 tarballs, then build the unified ETL/label map (see scripts/).
