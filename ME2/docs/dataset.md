# Dataset Documentation — airimonda/ai231-me2-voice-commands

Source: https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands
License: per-source compilation, **research and education only** (see `data/hf/LICENSE.md`).
DOI: 10.57967/hf/10723

## Schema (verified against `Dataset Schema - Google Sheets.pdf` and `configs/variations.csv`)

- **19 intents + OUT_OF_SCOPE** = 20 classes (`command` column; there is no `label` column).
- **93 command variations**: 13 fixed intents × 3 phrasings + 6 slotted intents × 3 phrasings × 3 slot values.
- Slot values (3 each): TIMER {10 seconds, 30 seconds, 1 minute}, ALARM {6:00 AM, 8:00 AM, 9:00 PM},
  TEMPERATURE {18, 22, 26 degrees}, BRIGHTNESS {20, 60, 100 percent}, COLOR {Red, Blue, Green},
  CREATE_REMINDER {Drink water, Study, Exercise}.
- Audio: 16 kHz mono 16-bit PCM WAV (embedded as bytes in parquet). `duration_s` matches decoded length.

## Splits (speaker-disjoint — verified programmatically)

| Split | Clips | Speakers | Role |
|---|---|---|---|
| train | 10,733 | 260 + 202 | training (111 clips/variation; CALL 104, MESSAGE 100; +270 OOS) |
| test | 4,443 | 115 | held-out evaluation (47 clips/variation + 47 OOS) |
| holdout | 202 | 5 | **live Raspberry Pi test only — excluded from all training/selection** |
| numerals | 66,390 | — | separate set for slotted-command numbers (MLEnd), not a split |

Extra configs (separate HF configs):
- `synthetic_negatives`: train 1,000 / test 250 generated OOS clips + noise (88) + tune_oos (400)
- `supplemental_synth`: 5,856 unused synthetic clips (group_synthetic)
- `supplemental_fil`: 14,120 Filipino-accented synthetic clips (train-only; license unverified per README)

## Composition (train)
- 85.4% synthetic / 14.6% real voices → training uses a real-voice sampling boost (see train_vcm.py).
- Sources: SLURP, SpeechCommands v2, Common Voice 19, FluentSpeechCommands, SNIPS, TimersAndSuch,
  MLEnd numerals, Xela VCM (CC BY 4.0), group synthetic (LibriSpeech/SilencioPH clones), group recordings.

## Ambiguities / flags
1. **19 vs 20 classes**: spec says "19 intents"; the dataset adds OUT_OF_SCOPE as a 20th class.
   Resolution: 20-way classifier (explicit project requirement to detect out-of-scope speech).
2. **"other slot value" group recordings**: 158 train rows (TIMER/ALARM/TEMPERATURE/BRIGHTNESS/
   CREATE_REMINDER) have a valid `command` but empty `variation`/`slot_value` (e.g. "set a timer for
   5 minutes"). They are kept in training as in-scope clips of their command class; they are not
   counted as schema variations.
3. **holdout usage**: reserved exclusively for the live Pi test per the README; never used for
   training, validation, or model selection.
4. **supplemental_fil license**: consent/license not recorded upstream — excluded from the default
   training mix (opt-in only).
