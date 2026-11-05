#!/usr/bin/env bash
# Download a whisper.cpp model. Usage:
#   ./scripts/download_model.sh base.en      # default, ~142 MB, good balance
#   ./scripts/download_model.sh small.en     # ~466 MB, more accurate
#   ./scripts/download_model.sh tiny.en      # ~75 MB, fastest (Pi low-RAM)
#   ./scripts/download_model.sh medium.en    # ~1.5 GB, very accurate (Mac)
set -euo pipefail
MODEL="${1:-base.en}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/models/ggml-$MODEL.bin"
mkdir -p "$ROOT/models"
URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-$MODEL.bin"
echo "Downloading $MODEL -> $DEST"
curl -L --fail -o "$DEST" "$URL"
echo "Done: $(du -h "$DEST" | cut -f1)"
