#!/usr/bin/env bash
# AI-Assistant — Apple Silicon Mac launcher
#
# Sets MPS-friendly env vars, ensures a ComfyUI sidecar is reachable on
# 127.0.0.1:$AI_ASSISTANT_COMFY_PORT, then launches AI_Assistant.
#
# If ComfyUI is already running (e.g. you started it via Stability Matrix's
# UI), this script just connects to it. If it's not running, this script
# spawns it from $AI_ASSISTANT_COMFY_DIR using its own venv's Python and
# tears it down on Ctrl-C.
#
# AI_Assistant.py is still expected to crash on initialize_forge() until
# the IS_MAC branch lands in #v4.

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# --- defaults --------------------------------------------------------------

: "${AI_ASSISTANT_COMFY_DIR:=/Volumes/Nekochan/Stability Matrix/Data/Packages/ComfyUI}"
: "${AI_ASSISTANT_MODELS_DIR:=/Volumes/Nekochan/Stability Matrix/Data/Models}"
: "${AI_ASSISTANT_COMFY_PORT:=8188}"
: "${AI_ASSISTANT_COMFY_PYTHON:=}"

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

# --- detect / spawn ComfyUI sidecar ---------------------------------------

COMFY_URL="http://127.0.0.1:$AI_ASSISTANT_COMFY_PORT"
COMFY_PID=""

cleanup() {
    if [[ -n "$COMFY_PID" ]] && kill -0 "$COMFY_PID" 2>/dev/null; then
        echo
        echo "Stopping ComfyUI sidecar we started (pid $COMFY_PID) ..."
        kill "$COMFY_PID" 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            kill -0 "$COMFY_PID" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$COMFY_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if curl -fsS --max-time 2 "$COMFY_URL/system_stats" >/dev/null 2>&1; then
    echo "ComfyUI already reachable at $COMFY_URL — using existing instance."
else
    if [[ ! -f "$AI_ASSISTANT_COMFY_DIR/main.py" ]]; then
        cat <<EOF >&2
ERROR: ComfyUI is not running on $COMFY_URL and no install was found at
       $AI_ASSISTANT_COMFY_DIR.

       Run ./mac/install_comfyui.sh first, or start ComfyUI yourself
       (e.g. via Stability Matrix) before running this script.
EOF
        exit 1
    fi

    # Detect the ComfyUI venv Python if not pre-set (matches install_comfyui.sh).
    if [[ -z "$AI_ASSISTANT_COMFY_PYTHON" ]]; then
        if [[ -x "$AI_ASSISTANT_COMFY_DIR/venv/bin/python" ]]; then
            AI_ASSISTANT_COMFY_PYTHON="$AI_ASSISTANT_COMFY_DIR/venv/bin/python"
        elif [[ -x "$AI_ASSISTANT_COMFY_DIR/.venv/bin/python" ]]; then
            AI_ASSISTANT_COMFY_PYTHON="$AI_ASSISTANT_COMFY_DIR/.venv/bin/python"
        else
            echo "ERROR: no ComfyUI venv at $AI_ASSISTANT_COMFY_DIR/{venv,.venv}/bin/python." >&2
            echo "       Set AI_ASSISTANT_COMFY_PYTHON to your interpreter." >&2
            exit 1
        fi
    fi

    COMFY_LOG="$REPO_ROOT/mac/.comfy.log"
    mkdir -p "$REPO_ROOT/mac"
    : > "$COMFY_LOG"

    echo "Starting ComfyUI sidecar (logs → $COMFY_LOG) ..."
    echo "  python : $AI_ASSISTANT_COMFY_PYTHON"
    echo "  cwd    : $AI_ASSISTANT_COMFY_DIR"
    (
        cd "$AI_ASSISTANT_COMFY_DIR"
        # --force-fp32: bf16 NaNs on MPS for SDXL (HANDOVER.md §4.2).
        # --listen 127.0.0.1: localhost only.
        exec "$AI_ASSISTANT_COMFY_PYTHON" main.py \
            --listen 127.0.0.1 \
            --port "$AI_ASSISTANT_COMFY_PORT" \
            --force-fp32 \
            > "$COMFY_LOG" 2>&1
    ) &
    COMFY_PID=$!

    echo -n "Waiting for ComfyUI to be ready"
    for i in $(seq 1 90); do
        if ! kill -0 "$COMFY_PID" 2>/dev/null; then
            echo
            echo "ERROR: ComfyUI exited early. Last 20 log lines:" >&2
            tail -20 "$COMFY_LOG" >&2
            exit 1
        fi
        if curl -fsS --max-time 2 "$COMFY_URL/system_stats" >/dev/null 2>&1; then
            echo " ready (took ${i}s)."
            break
        fi
        echo -n "."
        sleep 1
    done
fi

# --- launch AI_Assistant ---------------------------------------------------

# --xformers is intentionally absent (no Apple Silicon build). The IS_MAC
# branch in AI_Assistant.py (#v4) drops it from default_args automatically.
python AI_Assistant.py \
    --lang=jp \
    --nowebui \
    --skip-python-version-check \
    --skip-torch-cuda-test \
    "$@"
