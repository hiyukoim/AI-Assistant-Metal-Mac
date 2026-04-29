#!/usr/bin/env bash
# AI-Assistant — ComfyUI sidecar installer for Apple Silicon
# Clones ComfyUI into $AI_ASSISTANT_COMFY_DIR, installs its requirements
# into the same .venv created by mac/install.sh, installs the ControlNet
# preprocessor custom node, and renders extra_model_paths.yaml so
# ComfyUI sees the user's existing Stability Matrix model tree.

set -euo pipefail

# --- defaults --------------------------------------------------------------

# ComfyUI lives on internal disk by default — small (~200 MB) and Python
# imports benefit from internal IO. Override with $AI_ASSISTANT_COMFY_DIR
# if you keep your Python toolchain on an external SSD.
: "${AI_ASSISTANT_COMFY_DIR:=$HOME/ComfyUI}"

# Models live on external SSD by default, in the user's existing
# Stability Matrix tree. AI-Assistant downloads (~15 GB, see
# mac/download_models.sh) will be added alongside whatever's already
# there — filenames don't collide.
: "${AI_ASSISTANT_MODELS_DIR:=/Volumes/Nekochan/Stability Matrix/Data/Models}"

# Pinned ComfyUI commit (master HEAD as of 2026-04-28). Update this when
# bumping; do not float to master to avoid surprise breakage.
COMFY_COMMIT="fce0398470fe3ecdb7ab4c5c69555ad0fcbdc09e"

# --- repo root + venv ------------------------------------------------------

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

if [[ ! -f .venv/bin/activate ]]; then
    echo "ERROR: .venv not found. Run ./mac/install.sh first." >&2
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
export VIRTUAL_ENV="$REPO_ROOT/.venv"

echo "AI_ASSISTANT_COMFY_DIR  = $AI_ASSISTANT_COMFY_DIR"
echo "AI_ASSISTANT_MODELS_DIR = $AI_ASSISTANT_MODELS_DIR"
echo "ComfyUI commit          = $COMFY_COMMIT"
echo

# --- clone / update ComfyUI -----------------------------------------------

if [[ -d "$AI_ASSISTANT_COMFY_DIR/.git" ]]; then
    echo "ComfyUI directory exists; fetching and checking out pinned commit ..."
    git -C "$AI_ASSISTANT_COMFY_DIR" fetch --quiet origin
elif [[ -e "$AI_ASSISTANT_COMFY_DIR" ]]; then
    echo "ERROR: $AI_ASSISTANT_COMFY_DIR exists but isn't a git checkout." >&2
    echo "       Move it aside or set AI_ASSISTANT_COMFY_DIR elsewhere." >&2
    exit 1
else
    echo "Cloning ComfyUI to $AI_ASSISTANT_COMFY_DIR ..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$AI_ASSISTANT_COMFY_DIR"
fi
git -C "$AI_ASSISTANT_COMFY_DIR" checkout --quiet "$COMFY_COMMIT"

# --- install ComfyUI requirements into the shared venv --------------------

echo "Installing ComfyUI requirements into the AI-Assistant venv ..."
uv pip install -r "$AI_ASSISTANT_COMFY_DIR/requirements.txt"

# --- ControlNet preprocessor custom node ----------------------------------

# AI-Assistant's lineart/normalmap/lighting tabs use a1111 preprocessor
# names like 'lineart_realistic'. The shim (#v7) maps those to nodes
# from comfyui_controlnet_aux.
CN_AUX_DIR="$AI_ASSISTANT_COMFY_DIR/custom_nodes/comfyui_controlnet_aux"
if [[ -d "$CN_AUX_DIR/.git" ]]; then
    echo "comfyui_controlnet_aux already installed; updating ..."
    git -C "$CN_AUX_DIR" pull --quiet --ff-only || true
else
    echo "Installing comfyui_controlnet_aux ..."
    git clone --depth 1 https://github.com/Fannovel16/comfyui_controlnet_aux.git "$CN_AUX_DIR" || {
        echo "WARNING: failed to clone comfyui_controlnet_aux; ControlNet preprocessing will be limited."
        echo "         (See MAC_SETUP.md troubleshooting for manual install.)"
    }
fi
if [[ -f "$CN_AUX_DIR/requirements.txt" ]]; then
    uv pip install -r "$CN_AUX_DIR/requirements.txt" || {
        echo "WARNING: comfyui_controlnet_aux requirements failed; some preprocessors may not work."
    }
fi

# --- extra_model_paths.yaml -----------------------------------------------

echo "Rendering extra_model_paths.yaml ..."
TEMPLATE="$REPO_ROOT/mac/extra_model_paths.yaml.template"
TARGET="$AI_ASSISTANT_COMFY_DIR/extra_model_paths.yaml"

if [[ ! -f "$TEMPLATE" ]]; then
    echo "ERROR: $TEMPLATE missing." >&2
    exit 1
fi

# Use python rather than sed: $AI_ASSISTANT_MODELS_DIR may contain spaces,
# slashes, & or other characters that break sed escaping reliably.
python - "$TEMPLATE" "$TARGET" "$AI_ASSISTANT_MODELS_DIR" <<'PY'
import sys, pathlib
src, dst, models_dir = sys.argv[1], sys.argv[2], sys.argv[3]
text = pathlib.Path(src).read_text()
pathlib.Path(dst).write_text(text.replace("__MODELS_DIR__", models_dir))
print(f"  wrote {dst}")
PY

# --- ensure AI-Assistant-specific subdirs exist ---------------------------

# So ComfyUI doesn't error on startup if a category dir is missing, and so
# the model downloader (#v3) has somewhere to write.
for sub in StableDiffusion Lora ControlNet Embeddings Tagger; do
    mkdir -p "$AI_ASSISTANT_MODELS_DIR/$sub"
done

# --- final sanity check ----------------------------------------------------

if [[ ! -f "$AI_ASSISTANT_COMFY_DIR/main.py" ]]; then
    echo "ERROR: ComfyUI install incomplete (no main.py at $AI_ASSISTANT_COMFY_DIR)." >&2
    exit 1
fi

cat <<EOF

ComfyUI installed.

  Path     : $AI_ASSISTANT_COMFY_DIR
  Commit   : $COMFY_COMMIT
  Models   : $AI_ASSISTANT_MODELS_DIR
  YAML     : $AI_ASSISTANT_COMFY_DIR/extra_model_paths.yaml

Next: ./mac/download_models.sh   (~15 GB AI-Assistant model set; STOPs for confirmation)

To launch ComfyUI standalone for testing:

  cd "\$AI_ASSISTANT_COMFY_DIR" && python main.py --listen 127.0.0.1 --port 8188 --force-fp32

(--force-fp32 because bf16/fp16 mixed precision NaNs on MPS for SDXL.
 See HANDOVER.md §4.2 for the receipts.)
EOF
