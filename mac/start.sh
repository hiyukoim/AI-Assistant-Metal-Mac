#!/usr/bin/env bash
# AI-Assistant — Apple Silicon Mac launcher
# Sets MPS-friendly env vars, starts the ComfyUI sidecar in the background,
# waits for it to be reachable, then launches AI_Assistant in the foreground.
# Ctrl-C cleanly tears down both processes.
#
# AI_Assistant.py is still expected to crash on initialize_forge() until #v4
# lands the IS_MAC branch.

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# --- defaults --------------------------------------------------------------

: "${AI_ASSISTANT_COMFY_DIR:=$HOME/ComfyUI}"
: "${AI_ASSISTANT_MODELS_DIR:=/Volumes/Nekochan/Stability Matrix/Data/Models}"
: "${AI_ASSISTANT_COMFY_PORT:=8188}"

if [[ ! -f .venv/bin/activate ]]; then
    echo "ERROR: .venv not found. Run ./mac/install.sh first." >&2
    exit 1
fi

if [[ ! -f "$AI_ASSISTANT_COMFY_DIR/main.py" ]]; then
    echo "ERROR: ComfyUI not installed at $AI_ASSISTANT_COMFY_DIR." >&2
    echo "       Run ./mac/install_comfyui.sh first." >&2
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

# --- ComfyUI sidecar -------------------------------------------------------

COMFY_LOG="$REPO_ROOT/mac/.comfy.log"
mkdir -p "$REPO_ROOT/mac"
: > "$COMFY_LOG"

echo "Starting ComfyUI sidecar on 127.0.0.1:$AI_ASSISTANT_COMFY_PORT (logs → $COMFY_LOG) ..."
(
    cd "$AI_ASSISTANT_COMFY_DIR"
    # --force-fp32: bf16 NaNs on MPS for SDXL (HANDOVER.md §4.2).
    # --listen 127.0.0.1: localhost only, never expose unintentionally.
    exec python main.py \
        --listen 127.0.0.1 \
        --port "$AI_ASSISTANT_COMFY_PORT" \
        --force-fp32 \
        > "$COMFY_LOG" 2>&1
) &
COMFY_PID=$!

cleanup() {
    if kill -0 "$COMFY_PID" 2>/dev/null; then
        echo
        echo "Stopping ComfyUI sidecar (pid $COMFY_PID) ..."
        kill "$COMFY_PID" 2>/dev/null || true
        # Give it 5 s to shut down cleanly before SIGKILL.
        for _ in 1 2 3 4 5; do
            kill -0 "$COMFY_PID" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$COMFY_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Wait up to 90 s for ComfyUI's /system_stats endpoint to come up.
echo -n "Waiting for ComfyUI to be ready"
for i in $(seq 1 90); do
    if ! kill -0 "$COMFY_PID" 2>/dev/null; then
        echo
        echo "ERROR: ComfyUI exited early. Last log lines:" >&2
        tail -20 "$COMFY_LOG" >&2
        exit 1
    fi
    if curl -fsS "http://127.0.0.1:$AI_ASSISTANT_COMFY_PORT/system_stats" >/dev/null 2>&1; then
        echo " ready (took ${i}s)."
        break
    fi
    echo -n "."
    sleep 1
done

# --- launch AI_Assistant ---------------------------------------------------

# --xformers is intentionally absent (no Apple Silicon build). The IS_MAC
# branch in AI_Assistant.py (#v4) drops it from default_args automatically.
python AI_Assistant.py \
    --lang=jp \
    --nowebui \
    --skip-python-version-check \
    --skip-torch-cuda-test \
    "$@"
