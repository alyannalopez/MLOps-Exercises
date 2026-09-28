#!/usr/bin/env bash
# =============================================================================
# ME2 dataset re-download script
# -----------------------------------------------------------------------------
# The raw audio (33 GB) is too large to commit to git (GitHub caps single files
# at 2 GB and discourages repos >~5 GB). Everything below is PUBLIC, so this
# script re-fetches it deterministically into ./raw/ mirroring the layout the
# ETL pipeline (../code/scripts/etl.py) expects.
#
# Usage:  bash download_dataset.sh [dest_dir]     (default: ./raw)
# Requires: curl, tar, python3, unzip (for some sources)
#
# Sources (all public, as originally obtained):
#   optionB          1.6 GB   github.com/markandrian30/AI231  (MEX2/OptionB)
#   snips           124 MB    HF  snips / SmartLights
#   speechcommands 2.3 GB     github.com/dlw999/speech_commands_v2
#   fleurs         5.6 GB     HF  google/fleurs  (fil_ph only)
#   slurp          6.3 GB     HF  slup/SLURP
#   librispeech    6.6 GB     openslr.org  (dev-clean, train-clean-100, test-clean)
#   commonvoice      ~3 GB    GATED - needs a free Mozilla account (see note)
# =============================================================================
set -uo pipefail
DEST="${1:-./raw}"
mkdir -p "$DEST"
cd "$DEST"

echo ">> [1/6] OptionB  (github markandrian30/AI231 -> MEX2/OptionB)"
if [ ! -d optionB/MEX2/OptionB ]; then
  git clone --depth 1 https://github.com/markandrian30/AI231.git _tmp_ai231 2>/dev/null \
    && cp -R _tmp_ai231/MEX2/OptionB optionB/ 2>/dev/null \
    && rm -rf _tmp_ai231
fi
echo "   done -> optionB/"

echo ">> [2/6] SNIPS SmartLights  (HF)"
if [ ! -d snips ]; then
  mkdir -p snips
  for split in train val test; do
    curl -sL -o snips/${split}.parquet \
      "https://huggingface.co/datasets/snips/SmartLights/resolve/main/${split}.parquet"
  done
fi
echo "   done -> snips/"

echo ">> [3/6] SpeechCommands v2  (github dlw999/speech_commands_v2)"
if [ ! -f speechcommands/speech_commands_v0.02.tar.gz ]; then
  mkdir -p speechcommands
  curl -sL -o speechcommands/speech_commands_v0.02.tar.gz \
    "https://github.com/dlw999/speech_commands_v2/releases/download/v0.02/speech_commands_v0.02.tar.gz"
  tar -xzf speechcommands/speech_commands_v0.02.tar.gz -C speechcommands/
fi
echo "   done -> speechcommands/"

echo ">> [4/6] FLEURS fil_ph  (HF google/fleurs)"
if [ ! -d fleurs/fil_ph ]; then
  mkdir -p fleurs/fil_ph
  curl -sL -o fleurs/fil_ph/audio.tar.gz \
    "https://huggingface.co/datasets/google/fleurs/resolve/main/fil_ph/audio.tar.gz"
  tar -xzf fleurs/fil_ph/audio.tar.gz -C fleurs/fil_ph/ 2>/dev/null
fi
echo "   done -> fleurs/"

echo ">> [5/6] SLURP  (HF slup/SLURP)"
if [ ! -d slurp ]; then
  mkdir -p slurp
  for f in train_dev_test.parquet; do
    curl -sL -o slurp/$f "https://huggingface.co/datasets/slup/SLURP/resolve/main/$f"
  done
fi
echo "   done -> slurp/"

echo ">> [6/6] LibriSpeech  (openslr.org)"
if [ ! -d librispeech ]; then
  mkdir -p librispeech
  for ds in dev-clean train-clean-100 test-clean; do
    curl -sL -o librispeech/${ds}.tar.gz "https://www.openslr.org/resources/12/${ds}.tar.gz"
    tar -xzf librispeech/${ds}.tar.gz -C librispeech/
  done
fi
echo "   done -> librispeech/"

echo ""
echo "NOT INCLUDED (gated / not public):"
echo "  - Common Voice (EN): free account -> commonvoice.mozilla.org -> Download"
echo "                        -> English -> 17.0 -> clips.tar.gz (~3 GB) -> commonvoice/"
echo "  - STOP (8 domains): not publicly hosted under a stable URL."
echo ""
echo "All done. Raw audio now in $DEST  (~33 GB)."
