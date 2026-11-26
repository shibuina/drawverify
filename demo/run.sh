#!/usr/bin/env bash
# Run the DrawVerify web demo (demo/app.py). The first run creates a private venv (about 5 GB with torch).
#   demo/run.sh             -> http://127.0.0.1:8765
#   demo/run.sh --selftest  -> load the models, score one sketch, print the timings
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
venv="${DRAWVERIFY_DEMO_VENV:-$HOME/.cache/drawverify-demo/venv}"
if [ ! -x "$venv/bin/python" ]; then
  uv venv -q "$venv"   # a venv of its own: layering on the system site-packages broke on its numpy/scipy
  uv pip install -q --python "$venv/bin/python" torch torchvision "transformers>=4.57" "peft>=0.13" "accelerate>=0.33" pillow numpy opencv-python-headless bitsandbytes
fi
exec "$venv/bin/python" "$here/app.py" "$@"
