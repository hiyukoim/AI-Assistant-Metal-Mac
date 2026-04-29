#!/usr/bin/env bash
# AI-Assistant — Apple Silicon Mac installer
# Installs the Python venv + non-heavy deps for the AI_Assistant front-end.
# Does NOT install ComfyUI (see ./install_comfyui.sh) and does NOT download models
# (see ./download_models.sh).
#
# References (do not commit env-specific paths into other docs):
#   - HANDOVER.md §3.1: pinned package set for the gradio 4.44.1 era
#   - HANDOVER.md §4.6: torch>=2.6 to avoid the 2.5.x MPS memory regression

set -euo pipefail

# --- prereq checks ---------------------------------------------------------

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "ERROR: this installer targets macOS." >&2
    exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "ERROR: this port targets Apple Silicon (arm64). Detected: $(uname -m)" >&2
    exit 1
fi

if ! xcode-select -p >/dev/null 2>&1; then
    echo "ERROR: xcode-select tools not found. Run: xcode-select --install" >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found. Install with: brew install uv" >&2
    echo "       or: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

# Move to repo root (this script lives in mac/).
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
echo "Repo root: $REPO_ROOT"

# --- venv ------------------------------------------------------------------

# Pin Python 3.10.11 (matches upstream .python-version).
if [[ ! -d .venv ]]; then
    echo "Creating Python 3.10.11 virtual environment in .venv ..."
    uv venv .venv --python 3.10.11
else
    echo "Re-using existing .venv (delete it manually if you want a clean install)."
fi

# Activate the venv for this shell so `uv pip install` targets it. uv also
# accepts the venv via VIRTUAL_ENV; setting both is harmless.
# shellcheck disable=SC1091
source .venv/bin/activate
export VIRTUAL_ENV="$REPO_ROOT/.venv"

# --- PyTorch (MPS) ---------------------------------------------------------

# Mac wheels published to the default PyPI index already include MPS support.
# Do NOT pass --index-url here; the +cu1xx channels have no Apple Silicon wheels.
# torch>=2.6 avoids the 2.5.x MPS memory regression (HANDOVER.md §4.6).
echo "Installing PyTorch with MPS support (this may take a minute) ..."
uv pip install "torch>=2.6" "torchvision>=0.21" "torchaudio"

# --- Front-end / API deps --------------------------------------------------

# Pin set tied to gradio 4.44.1 era (HANDOVER.md §3.1).
# pydantic >=2.10 emits `additionalProperties: True` (bool) which gradio_client's
# get_type() crashes on. starlette >=0.40 / fastapi >=0.116 changed
# TemplateResponse signature, breaking gradio 4.44.1. huggingface_hub >=0.27
# removed HfFolder which gradio 4.44.1's oauth.py imports (verified at runtime).
echo "Installing Gradio + FastAPI pin set ..."
uv pip install \
    "gradio==4.44.1" \
    "pydantic<2.10" \
    "fastapi==0.115.0" \
    "starlette==0.38.6" \
    "huggingface_hub<0.26"

# AI_Assistant runtime deps. utils/tagger.py needs cv2 + onnx + onnxruntime;
# utils/img_utils.py needs scikit-image (rgb2lab, deltaE_ciede2000); the
# Windows installer pins albumentations + opencv-contrib-python-headless +
# rich + onnx 1.15 + onnxruntime 1.17 (see AI_Assistant_setup.py
# packages_to_add). On Mac we let uv resolve current versions of those
# pure-Python or universally-built packages — opencv-contrib-python-headless
# has an arm64 wheel, onnx and onnxruntime publish arm64 wheels too.
echo "Installing AI_Assistant runtime deps ..."
uv pip install \
    "requests" \
    "Pillow" \
    "numpy" \
    "onnx" \
    "onnxruntime" \
    "opencv-contrib-python-headless" \
    "scikit-image" \
    "rich"
# NB: onnxruntime (CPU) only. onnxruntime-gpu has no Apple Silicon build, and
# CoreMLExecutionProvider tuning is deferred (see issue #v8).

# --- Skipped on purpose ----------------------------------------------------
#
# The Windows installer pulls these; they are intentionally NOT installed on Mac.
# Listed here so the next reader knows it was a deliberate choice, not an oversight.
#
#   xformers          — no Apple Silicon build. Use PyTorch SDPA instead.
#   bitsandbytes      — no Apple Silicon build. mps-bitsandbytes is alpha
#                       (HANDOVER.md §4.3); do not install.
#   triton-windows    — Windows-only.
#   onnxruntime-gpu   — Windows/Linux CUDA only. Use plain onnxruntime above.
#   pyinstaller       — .app bundling is out of scope for v1.
#   pygit2, pynput    — currently unused on the Mac path; revisit if a tab needs them.
#   anything +cu118 / +cu128 — no Apple Silicon wheel.

# --- final sanity check ----------------------------------------------------

python - <<'PY'
import torch
print(f"torch {torch.__version__}, MPS available: {torch.backends.mps.is_available()}")
assert torch.backends.mps.is_available(), "MPS backend unavailable — port will not work."
PY

echo
echo "Setup complete. Next: ./mac/install_comfyui.sh"
