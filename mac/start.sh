#!/usr/bin/env bash
# AI-Assistant — Apple Silicon Mac launcher
# Sets MPS-friendly env vars, activates the venv, and launches AI_Assistant.
#
# NOTE for issue #v1: this launcher only starts the AI_Assistant Python process.
# The ComfyUI sidecar startup (and the cleanup trap that goes with it) is wired
# in issue #v2 (mac/install_comfyui.sh). At this stage AI_Assistant.py is still
# expected to crash on `initialize_forge()` — the IS_MAC branch lands in #v4.

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

if [[ ! -f .venv/bin/activate ]]; then
    echo "ERROR: .venv not found. Run ./mac/install.sh first." >&2
    exit 1
fi

# --- MPS memory tuning (HANDOVER.md §3.1, §3.3, §4.7) ----------------------

# Disable PyTorch's default upper watermark, which assumes isolated GPU memory
# and is wrong for unified memory. torch.mps.set_per_process_memory_fraction()
# in the Python process becomes the single source of truth for the MPS cap.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3

# Allow CPU fallback for ops not yet implemented on MPS (e.g. SVD, some
# Conv3D paths in ControlNet preprocessors). Without this, a single missing
# kernel hard-fails mid-inference.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# --- venv ------------------------------------------------------------------

# shellcheck disable=SC1091
source .venv/bin/activate
export VIRTUAL_ENV="$REPO_ROOT/.venv"

# --- launch ----------------------------------------------------------------

# --xformers is intentionally absent (no Apple Silicon build). The IS_MAC
# branch in AI_Assistant.py (#v4) drops it from default_args automatically.
exec python AI_Assistant.py \
    --lang=jp \
    --nowebui \
    --skip-python-version-check \
    --skip-torch-cuda-test \
    "$@"
