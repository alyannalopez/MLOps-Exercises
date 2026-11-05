#!/usr/bin/env bash
# Set up the isolated Kokoro 3.12 venv + download the model files.
# Run once on the Mac, and once on the Pi (uses whatever python3.12 is on PATH).
#
# Usage:
#   ./scripts/setup_kokoro.sh                 # python3.12 from PATH
#   ./scripts/setup_kokoro.sh /path/py3.12    # explicit interpreter
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${1:-python3.12}"
VENV="$ROOT/.venv-kokoro"

echo "==> Using interpreter: $("$PY" --version 2>&1)"
echo "==> Creating venv at $VENV"
"$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip -q
echo "==> Installing kokoro-onnx + deps"
"$VENV/bin/python" -m pip install -q kokoro-onnx soundfile numpy

mkdir -p "$ROOT/models"
cd "$ROOT/models"
BASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"
[ -f kokoro-v1.0.onnx ]  || { echo "==> Downloading kokoro-v1.0.onnx";  curl -L -sS -o kokoro-v1.0.onnx  "$BASE/kokoro-v1.0.onnx"; }
[ -f voices-v1.0.bin ]   || { echo "==> Downloading voices-v1.0.bin";   curl -L -sS -o voices-v1.0.bin   "$BASE/voices-v1.0.bin"; }

echo "==> Verifying Kokoro loads + synthesizes"
"$VENV/bin/python" - <<'PY'
import kokoro_onnx
k = kokoro_onnx.Kokoro("models/kokoro-v1.0.onnx", "models/voices-v1.0.bin")
a, sr = k.create("Setup verified.", voice="af_heart")
print(f"OK: {len(a)/sr:.2f}s audio @ {sr}Hz, {len(k.get_voices())} voices available")
PY

echo "==> Done. Test with:"
echo "    PYTHONPATH=src python3 scripts/test_tts.py --backend kokoro"
