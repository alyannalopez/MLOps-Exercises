#!/usr/bin/env bash
# Launch the VCM Smart Home web demo. Opens your browser automatically.
#   ./launch_web.sh               # http://127.0.0.1:8000
#   ./launch_web.sh --port 8010   # custom port
#   ./launch_web.sh --model crnn_int8.onnx
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
exec python3 web/app.py "$@"
