#!/usr/bin/env bash
# AI-Assistant — ComfyUI sidecar installer/verifier for Apple Silicon
#
# Default behaviour: verify that a ComfyUI install (and its venv +
# extra_model_paths.yaml + comfyui_controlnet_aux) exists at the configured
# location. Print actionable warnings for anything missing.
#
# The portable default location is ~/ComfyUI. If you have ComfyUI managed
# by another tool (e.g. Stability Matrix), point AI_ASSISTANT_COMFY_DIR
# at it via mac/config.local.env or as an env var.
#
# If no install is found, this script normally fails with hints. Setting
# AI_ASSISTANT_AUTO_CLONE=true changes that to "clone ComfyUI master into
# AI_ASSISTANT_COMFY_DIR" — opt-in to avoid surprise downloads.

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# --- load mac/config.local.env (gitignored, optional) ----------------------

# Sourced before applying defaults, so values here win against the defaults
# but lose to anything already set in the calling shell. set -a/+a marks
# everything assigned during the source as exported, so child processes see
# the resolved values too.
if [[ -f "$REPO_ROOT/mac/config.local.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/mac/config.local.env"
    set +a
fi

# --- defaults --------------------------------------------------------------

: "${AI_ASSISTANT_COMFY_DIR:=$HOME/ComfyUI}"
: "${AI_ASSISTANT_MODELS_DIR:=$REPO_ROOT/models}"
: "${AI_ASSISTANT_COMFY_PORT:=8188}"
: "${AI_ASSISTANT_COMFY_PYTHON:=}"
: "${AI_ASSISTANT_AUTO_CLONE:=false}"

echo "AI_ASSISTANT_COMFY_DIR  = $AI_ASSISTANT_COMFY_DIR"
echo "AI_ASSISTANT_MODELS_DIR = $AI_ASSISTANT_MODELS_DIR"
echo

# --- step 1: ComfyUI install present (or auto-clone) ----------------------

if [[ ! -f "$AI_ASSISTANT_COMFY_DIR/main.py" ]]; then
    if [[ "$AI_ASSISTANT_AUTO_CLONE" == "true" ]]; then
        if [[ -e "$AI_ASSISTANT_COMFY_DIR" ]]; then
            echo "ERROR: $AI_ASSISTANT_COMFY_DIR exists but isn't a ComfyUI checkout." >&2
            echo "       Move it aside or set AI_ASSISTANT_COMFY_DIR elsewhere." >&2
            exit 1
        fi
        echo "Auto-clone enabled. Cloning ComfyUI to $AI_ASSISTANT_COMFY_DIR ..."
        git clone https://github.com/comfyanonymous/ComfyUI.git "$AI_ASSISTANT_COMFY_DIR"

        # Set up a dedicated venv for ComfyUI inside its own dir, parallel
        # to the layout Stability Matrix produces (venv/bin/python). We
        # use Python 3.12 because that's ComfyUI's recommended baseline;
        # this is independent of the AI_Assistant venv (3.10.11).
        echo "Creating ComfyUI venv with Python 3.12 ..."
        if ! command -v uv >/dev/null 2>&1; then
            echo "ERROR: uv not found. Install with: brew install uv" >&2
            exit 1
        fi
        uv venv "$AI_ASSISTANT_COMFY_DIR/venv" --python 3.12

        echo "Installing ComfyUI requirements into its venv ..."
        VIRTUAL_ENV="$AI_ASSISTANT_COMFY_DIR/venv" \
            uv pip install -r "$AI_ASSISTANT_COMFY_DIR/requirements.txt"
    else
        cat <<EOF >&2
ERROR: No ComfyUI install found at:
       $AI_ASSISTANT_COMFY_DIR

You have two options:

  (a) Install ComfyUI yourself. Either via Stability Matrix
      (https://lykos.ai/stability-matrix), then point
      AI_ASSISTANT_COMFY_DIR at the resulting Packages/ComfyUI directory
      in mac/config.local.env. Or manually:
        git clone https://github.com/comfyanonymous/ComfyUI.git "$AI_ASSISTANT_COMFY_DIR"
        # ... and set up a venv with the requirements.

  (b) Have this script clone it for you (~200 MB plus ~500 MB of Python
      deps installed in a fresh venv inside the ComfyUI dir):
        AI_ASSISTANT_AUTO_CLONE=true ./mac/install_comfyui.sh
EOF
        exit 1
    fi
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

If Stability Matrix is managing this install, launch it through the SM UI
once so it can create the venv. If you cloned manually, set
AI_ASSISTANT_COMFY_PYTHON to your interpreter path.
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

ComfyUI will look for models only in its built-in default paths
($AI_ASSISTANT_COMFY_DIR/models/...) which won't see the AI-Assistant model
set unless you put them there. Stability Matrix normally writes this file
the first time you launch ComfyUI through it.

For a manual install, see https://github.com/comfyanonymous/ComfyUI's
extra_model_paths.yaml.example.
EOF
fi

if [[ -f "$YAML" ]]; then
    REQUIRED_KEYS=(checkpoints loras controlnet embeddings vae)
    MISSING=()
    for key in "${REQUIRED_KEYS[@]}"; do
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
    echo
    echo "Models dir does not exist; creating $AI_ASSISTANT_MODELS_DIR"
    mkdir -p "$AI_ASSISTANT_MODELS_DIR"
fi

mkdir -p "$AI_ASSISTANT_MODELS_DIR/Tagger"
echo "Tagger dir       : $AI_ASSISTANT_MODELS_DIR/Tagger (ensured)"

cat <<EOF

ComfyUI sidecar verified.

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
