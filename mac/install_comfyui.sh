#!/usr/bin/env bash
# AI-Assistant — ComfyUI sidecar verifier for Apple Silicon
#
# This script does NOT clone or install ComfyUI. The expected setup is that
# ComfyUI is already installed and managed by another tool (e.g. Stability
# Matrix) so the user keeps a single source of truth for ComfyUI updates,
# custom nodes, and model paths. Stability Matrix even writes the
# extra_model_paths.yaml we need.
#
# This script's job is just to:
#   1. Confirm the configured ComfyUI dir is real (main.py + a Python venv).
#   2. Confirm the model categories AI-Assistant needs are mapped in
#      extra_model_paths.yaml — warn loudly if anything is missing.
#   3. Confirm comfyui_controlnet_aux is present (used by #v7 lineart
#      preprocessor mapping). Warn if missing; do NOT install.
#   4. Create the AI-Assistant-specific Tagger subdirectory under the
#      models tree so #v3's downloader has somewhere to write.
#
# To override the defaults, set the env vars below before running. The same
# vars are picked up by mac/start.sh, so set them in your shell profile or
# a wrapper script if you want them to persist.

set -euo pipefail

# --- defaults --------------------------------------------------------------

# Yuko's setup keeps ComfyUI under Stability Matrix's Packages tree.
# Stability Matrix manages the version, custom nodes, and venv.
: "${AI_ASSISTANT_COMFY_DIR:=/Volumes/Nekochan/Stability Matrix/Data/Packages/ComfyUI}"

# Same Models tree both Stability Matrix and AI-Assistant downloads use.
: "${AI_ASSISTANT_MODELS_DIR:=/Volumes/Nekochan/Stability Matrix/Data/Models}"

# Optional: explicit Python interpreter for ComfyUI. If unset, the script
# auto-detects $AI_ASSISTANT_COMFY_DIR/venv/bin/python (Stability Matrix
# layout). The detected path is not exported because mac/start.sh re-runs
# the same detection — kept consistent in one place.
: "${AI_ASSISTANT_COMFY_PYTHON:=}"

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

echo "AI_ASSISTANT_COMFY_DIR  = $AI_ASSISTANT_COMFY_DIR"
echo "AI_ASSISTANT_MODELS_DIR = $AI_ASSISTANT_MODELS_DIR"
echo

# --- step 1: ComfyUI install present --------------------------------------

if [[ ! -f "$AI_ASSISTANT_COMFY_DIR/main.py" ]]; then
    cat <<EOF >&2
ERROR: No ComfyUI install found at:
       $AI_ASSISTANT_COMFY_DIR

Expected to find $AI_ASSISTANT_COMFY_DIR/main.py.

Either:
  - Install ComfyUI via Stability Matrix (https://lykos.ai/stability-matrix)
    and let its default Packages dir handle the install, then re-run
    this script with the same defaults; or
  - Clone ComfyUI yourself and re-run with:
      AI_ASSISTANT_COMFY_DIR=/path/to/your/ComfyUI ./mac/install_comfyui.sh
EOF
    exit 1
fi

# Detect the ComfyUI venv Python if not pre-set.
if [[ -z "$AI_ASSISTANT_COMFY_PYTHON" ]]; then
    if [[ -x "$AI_ASSISTANT_COMFY_DIR/venv/bin/python" ]]; then
        AI_ASSISTANT_COMFY_PYTHON="$AI_ASSISTANT_COMFY_DIR/venv/bin/python"
    elif [[ -x "$AI_ASSISTANT_COMFY_DIR/.venv/bin/python" ]]; then
        AI_ASSISTANT_COMFY_PYTHON="$AI_ASSISTANT_COMFY_DIR/.venv/bin/python"
    else
        cat <<EOF >&2
ERROR: ComfyUI is at $AI_ASSISTANT_COMFY_DIR but no Python venv was found.
       Expected one of:
         $AI_ASSISTANT_COMFY_DIR/venv/bin/python
         $AI_ASSISTANT_COMFY_DIR/.venv/bin/python

Stability Matrix should create venv/ automatically when you install/launch
ComfyUI through its UI at least once. If you cloned ComfyUI manually, set
AI_ASSISTANT_COMFY_PYTHON to your interpreter.
EOF
        exit 1
    fi
fi

PY_VERSION="$("$AI_ASSISTANT_COMFY_PYTHON" --version 2>&1)"
echo "ComfyUI Python   : $AI_ASSISTANT_COMFY_PYTHON ($PY_VERSION)"

# --- step 2: extra_model_paths.yaml has what we need ----------------------

YAML="$AI_ASSISTANT_COMFY_DIR/extra_model_paths.yaml"
if [[ ! -f "$YAML" ]]; then
    cat <<EOF >&2
WARNING: $YAML does not exist.

ComfyUI will look for models in its built-in default paths
($AI_ASSISTANT_COMFY_DIR/models/...) which won't see the AI-Assistant model
set unless you put them there. Stability Matrix normally writes this file
the first time you launch ComfyUI through it; do that once if you can.

For now, AI-Assistant features that depend on Lora/ControlNet/checkpoint
discovery will be empty.
EOF
fi

if [[ -f "$YAML" ]]; then
    REQUIRED_KEYS=(checkpoints loras controlnet embeddings vae)
    MISSING=()
    for key in "${REQUIRED_KEYS[@]}"; do
        # Match either "    key: ..." (indented) or "key: ..." at column 0.
        if ! grep -qE "^\s+${key}:" "$YAML"; then
            MISSING+=("$key")
        fi
    done
    if [[ ${#MISSING[@]} -gt 0 ]]; then
        echo
        echo "WARNING: extra_model_paths.yaml is missing keys: ${MISSING[*]}"
        echo "         AI-Assistant tabs depending on those will see empty dropdowns."
        echo "         Add the missing categories pointing into:"
        echo "           $AI_ASSISTANT_MODELS_DIR/<corresponding subdir>"
    else
        echo "ComfyUI YAML     : $YAML (all required keys present)"
    fi
fi

# --- step 3: comfyui_controlnet_aux present -------------------------------

CN_AUX="$AI_ASSISTANT_COMFY_DIR/custom_nodes/comfyui_controlnet_aux"
if [[ -d "$CN_AUX" ]]; then
    echo "controlnet_aux   : $CN_AUX"
else
    cat <<EOF
WARNING: $CN_AUX is missing.

This custom node provides the lineart_realistic / AnimeLineArt / canny
preprocessors AI-Assistant relies on (see #v7). Without it, those tabs
will fall back to passing raw images to ControlNet, which usually
produces poor results.

To add it via Stability Matrix:
  Open Stability Matrix → ComfyUI package → Extensions →
  search "ControlNet Auxiliary" by Fannovel16 → Install.

Or via git, into your existing install:
  git clone --depth 1 \\
    https://github.com/Fannovel16/comfyui_controlnet_aux.git \\
    "$CN_AUX"
  "$AI_ASSISTANT_COMFY_PYTHON" -m pip install -r "$CN_AUX/requirements.txt"
EOF
fi

# --- step 4: AI-Assistant-specific Tagger dir -----------------------------

if [[ ! -d "$AI_ASSISTANT_MODELS_DIR" ]]; then
    cat <<EOF >&2
ERROR: AI_ASSISTANT_MODELS_DIR does not exist:
       $AI_ASSISTANT_MODELS_DIR

Set the env var to your existing models tree, or create the dir.
EOF
    exit 1
fi

mkdir -p "$AI_ASSISTANT_MODELS_DIR/Tagger"
echo "Tagger dir       : $AI_ASSISTANT_MODELS_DIR/Tagger (ensured)"

cat <<EOF

ComfyUI sidecar verified — reusing existing install.

  ComfyUI dir    : $AI_ASSISTANT_COMFY_DIR
  ComfyUI Python : $AI_ASSISTANT_COMFY_PYTHON ($PY_VERSION)
  Models dir     : $AI_ASSISTANT_MODELS_DIR
  Tagger dir     : $AI_ASSISTANT_MODELS_DIR/Tagger

Next: ./mac/download_models.sh   (~15 GB AI-Assistant model set, STOPs for confirmation)

To launch ComfyUI standalone for testing:

  cd "\$AI_ASSISTANT_COMFY_DIR" && \\
    "\$AI_ASSISTANT_COMFY_PYTHON" main.py --listen 127.0.0.1 --port 8188 --force-fp32

(--force-fp32 because bf16/fp16 mixed precision NaNs on MPS for SDXL.
 See HANDOVER.md §4.2 for the receipts.)
EOF
