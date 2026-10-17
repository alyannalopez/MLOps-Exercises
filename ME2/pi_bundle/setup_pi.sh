#!/usr/bin/env bash
# ME2 VCM — Raspberry Pi 4/5 setup. Run from the pi_bundle directory.
#   ./setup_pi.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
echo "==> Using $($PY --version)"

sudo apt-get update
sudo apt-get install -y python3-venv python3-dev libportaudio2

$PY -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements-pi.txt

# Pretrained openWakeWord models ship inside the wheel (v0.4.0 bundles the
# ONNX assets); verify the hey_mycroft asset is present (works on 0.4.0 and 0.6.x).
./.venv/bin/python - <<'EOF'
import os, sys
sys.path.insert(0, "..")
from runtime.wakeword import WakeWordDetector
ww = WakeWordDetector(threshold=0.75)
print(f"OK: {ww.wakeword} model loaded (openwakeword {__import__('importlib.metadata', fromlist=['version']).version('openwakeword')})")
EOF

# Runtime code lives one level up (ME2 repo root); make it importable here.
# Symlink to the actual subdirectory, not to ".." itself.
ln -sfn ../runtime runtime 2>/dev/null || true
ln -sfn ../vcm vcm 2>/dev/null || true

echo "==> Verifying classifier loads..."
./.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, ".")
from runtime.vcm_runtime import VCMClassifier
clf = VCMClassifier("models/conformer_s_float.onnx")
import numpy as np
r = clf.classify(np.zeros(16000, dtype=np.float32))
print(f"OK: classifier loaded, silent-clip prediction = {r.intent} (conf {r.confidence:.2f})")
EOF

echo "==> Done. Try:"
echo "    ./.venv/bin/python -m runtime.main --file sample_clips/play_music_sample.wav"
echo "    ./.venv/bin/python -m runtime.main --mic"
echo "  Or install the service:  sudo cp me2-vcm.service /etc/systemd/system/ && sudo systemctl enable --now me2-vcm"
