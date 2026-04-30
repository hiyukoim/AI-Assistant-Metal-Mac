# AI-Assistant on Apple Silicon Mac — setup guide

This is the Mac-only fork of [`tori29umai0123/AI-Assistant`](https://github.com/tori29umai0123/AI-Assistant). The original Forge backend is replaced by a ComfyUI sidecar fronted by an a1111-API-compatible HTTP shim, so the user-facing Gradio app behaves the same as Windows but runs natively on MPS.

## Requirements

- **Hardware**: Apple Silicon (M1/M2/M3/M4). Tested on M4 Pro 24 GB.
- **OS**: macOS 14 (Sonoma) or later.
- **Tools**:
  - Xcode Command Line Tools — `xcode-select --install`
  - [`uv`](https://docs.astral.sh/uv/) — `brew install uv`
  - Python 3.10.11 (uv installs it automatically)
- **Disk**: ~5 GB for the Python venv on internal disk; ~12 GB for the AI-Assistant model set on (any) volume — external SSD recommended if you already have a Stability Matrix install.
- **Memory**: 24 GB unified memory recommended for SDXL inference. 16 GB Macs may need to lower step counts and avoid ControlNet stacks.

## What gets installed

| | Where | Approx size |
|---|---|---|
| AI-Assistant Python venv | `<repo>/.venv/` | ~5 GB |
| ComfyUI sidecar | `~/ComfyUI` (default) or your existing Stability Matrix install | ~700 MB without models |
| AI-Assistant model set (animagine-xl-3.1, 12 LoRAs, 6 ControlNets, tagger) | `$AI_ASSISTANT_MODELS_DIR` | ~12 GB |
| ComfyUI custom nodes | `$AI_ASSISTANT_COMFY_DIR/custom_nodes/` | ~50 MB |

## Install

Three commands, in order. Each script is idempotent — re-running is safe.

```bash
./mac/install.sh           # 1. Python 3.10.11 venv + deps
./mac/install_comfyui.sh   # 2. ComfyUI sidecar verifier + custom nodes
./mac/download_models.sh   # 3. ~12 GB model set (STOPs for confirmation)
```

### Step 1 — `mac/install.sh`

Creates `.venv/` with Python 3.10.11 and installs:

- `torch>=2.6` + `torchvision` + `torchaudio` (MPS-capable wheels from the default PyPI index)
- `gradio==3.41.2` + `pydantic==1.10.15` + `huggingface_hub<0.20` (matches upstream pin set)
- `requests`, `Pillow<10` (Pillow 10+ removed `Image.ANTIALIAS` which `utils/img_utils.py` uses)
- `numpy`, `onnx`, `onnxruntime` (CPU; no `onnxruntime-gpu`)
- `opencv-contrib-python-headless`, `scikit-image`, `rich`

Skipped on purpose: `xformers`, `bitsandbytes`, `triton`, `pyinstaller`, `+cu1xx` wheels — none have Apple Silicon builds.

### Step 2 — `mac/install_comfyui.sh`

Default behaviour: **verifies** an existing ComfyUI install (e.g. one managed by [Stability Matrix](https://lykos.ai/stability-matrix)). Doesn't clone anything unless you opt in.

Override with environment variables (or `mac/config.local.env`, see below):

| Variable | Default | Purpose |
|---|---|---|
| `AI_ASSISTANT_COMFY_DIR` | `$HOME/ComfyUI` | Where ComfyUI lives |
| `AI_ASSISTANT_COMFY_PORT` | `8188` | Sidecar port |
| `AI_ASSISTANT_MODELS_DIR` | `<repo>/models` | Where the model set lives |
| `AI_ASSISTANT_AUTO_CLONE` | `false` | If `true`, clones ComfyUI when missing instead of failing |

The script also clones [`Kosinkadink/ComfyUI-Advanced-ControlNet`](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet) into `custom_nodes/` for full Forge `control_mode` parity (Balanced / My prompt is more important / ControlNet is more important).

`comfyui_controlnet_aux` (lineart and other preprocessors) is detected and a warning is printed if missing. Stability Matrix users typically have it already.

A `Tagger/` subdirectory is created under `$AI_ASSISTANT_MODELS_DIR/` for the wd-swinv2-tagger ONNX, and a symlink `<repo>/models/tagger → $AI_ASSISTANT_MODELS_DIR/Tagger` is set up so `utils/tagger.py` (which hardcodes `models/tagger`) finds it.

### Step 3 — `mac/download_models.sh`

Bash port of `AI_Assistant_model_DL.cmd`. Downloads 23 files via `curl` with `shasum -a 256` verification and 3-retry. Total ~12 GB.

Files land in:

```
$AI_ASSISTANT_MODELS_DIR/
├── StableDiffusion/animagine-xl-3.1.safetensors
├── Lora/  (12 LoRAs)
├── ControlNet/  (6 models, 1 from Civitai)
└── Tagger/  (wd-swinv2-tagger-v3 ONNX)
```

Re-running skips files that already match the expected hash. If Civitai blocks the `controlnet852AClone_v10` download (login wall), grab it manually from <https://civitai.com/models/443821?modelVersionId=515749> and place at the printed path; the next run accepts it.

## Configuration — `mac/config.local.env`

Optional, gitignored. Copy from `mac/config.example.env` and edit:

```bash
cp mac/config.example.env mac/config.local.env
```

Common settings:

```bash
# Use an existing Stability Matrix install instead of $HOME/ComfyUI
AI_ASSISTANT_COMFY_DIR="/Volumes/SSD/Stability Matrix/Data/Packages/ComfyUI"
AI_ASSISTANT_MODELS_DIR="/Volumes/SSD/Stability Matrix/Data/Models"

# Swap the ESRGAN upscaler used by the resize tab
# (default: 4x_NMKD-Superscale-SP_178000_G.pth — general-purpose)
# AI_ASSISTANT_UPSCALER="4x-AnimeSharp.pth"
# AI_ASSISTANT_UPSCALER=""   # set empty to disable, fall back to LANCZOS
```

The same variables can be passed inline:

```bash
AI_ASSISTANT_COMFY_DIR=/path/to/ComfyUI ./mac/start.sh
```

## Run

```bash
./mac/start.sh
```

What happens:

1. Sources `mac/config.local.env` if present.
2. Sets MPS-friendly env vars (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`, `PYTORCH_ENABLE_MPS_FALLBACK=1`).
3. If ComfyUI isn't already running on the configured port, spawns it from `$AI_ASSISTANT_COMFY_DIR` with `--force-fp32` (bf16 NaNs on MPS for SDXL — see HANDOVER.md §4.2).
4. Waits up to 90s for `/system_stats` to come up.
5. Launches AI_Assistant in the foreground with `--exui` (LoRA dropdown enabled by default).

The browser opens to `http://127.0.0.1:7860/`. If the ComfyUI sidecar was already running (e.g. via Stability Matrix's UI), `start.sh` reuses it and only kills processes it spawned itself.

Press `Ctrl-C` to stop everything.

## Performance ballpark — M4 Pro 24 GB

| Configuration | Time per generation |
|---|---|
| First run (Metal kernel compile + 6 GB SDXL load) | +30–60 s overhead one-off |
| i2i 20 steps, no ControlNet | ~150–200 s |
| i2i 20 steps + 1 ControlNet (lineart) | ~200–250 s |
| **+ Hyper-SDXL 8-step LoRA in prompt** | **~50–80 s** (≈2.5× speedup) |
| Resize tab (canonical render + 4× ESRGAN) | +5 s on top of the i2i time |
| Tagger (prompt analysis) on CPU | ~2–5 s per image |

To use Hyper-SDXL: in the i2i tab, click `LoRA更新`, pick `Hyper-SDXL-8steps-CFG-lora` from the dropdown. The shim auto-detects this in the prompt and overrides steps to 8 (and keeps your CFG because the `-CFG-` variant works at normal CFG).

`LCM-*` and `*-Lightning-Nstep` LoRAs trigger the same auto-override.

## Troubleshooting

### Generation shows "Error" almost immediately

Should not happen on this fork — the v7.1 timeout patch fixes Gradio 3.41's 5 s queue read timeout that ate every long Mac inference. If it does:

1. Check `mac/.start.log` for tracebacks.
2. Confirm the patch fired:
   ```
   [comfy_shim] patched gradio.queueing.Queue.start: queue_client.timeout=None
   ```
3. If the line is missing, `mac.comfy_shim` failed to import — re-run `./mac/install.sh`.

### ControlNet `control_mode` has no visible effect

The Kosinkadink/Advanced-ControlNet custom node may not be installed. Re-run `./mac/install_comfyui.sh` to fix. Verify with:

```bash
curl -fsS http://127.0.0.1:8188/object_info | grep -i ScaledSoftControlNet
```

If the node is missing, all CN units fall back to "Balanced" with a one-time warning in the log.

### prompt分析 says NoSuchFile

The tagger symlink is missing. Re-run `./mac/install_comfyui.sh`, or manually:

```bash
ln -snf "$AI_ASSISTANT_MODELS_DIR/Tagger" "<repo>/models/tagger"
```

### Generation slows from ~7 s/it to ~100 s/it

The OS is paging WindowServer to disk. Check Activity Monitor → Memory → Swap Used. If it's >2 GB during a session, quit Slack / Discord / Telegram / Notion / Spotify and close non-Gradio browser tabs. SDXL's working set is right at the MPS cap (≈14.2 GiB on a 24 GB Mac) and any other GPU consumer pushes us into swap.

### `MPS backend out of memory`

Lower the canonical render size by editing the input image to a smaller dimension before upload — the action layer will pick a smaller bucket. Or close other Mac apps using the GPU (browser, video calls).

### Civitai download blocked

Some Civitai models require a login token now. Grab `controlnet852AClone_v10.safetensors` manually from <https://civitai.com/models/443821?modelVersionId=515749> and drop it at `$AI_ASSISTANT_MODELS_DIR/ControlNet/controlnet852AClone_v10.safetensors`. The next `./mac/download_models.sh` run will hash-verify and accept it.

## Out of scope

| | Reason |
|---|---|
| `bf16` / `fp16` mixed precision | NaNs on MPS for SDXL as of PyTorch 2.11 (verified in CoppyLora port; documented in HANDOVER §4.2) |
| `mps-bitsandbytes` | Alpha-quality, single dev, do not install |
| PyInstaller `.app` bundling | Not needed when running from source |
| CoreML conversion | Real ~2-3× speedup possible but ControlNet support is poor in available ComfyUI bridge nodes; defer |
| Bitwise output parity with Windows | Impossible — different sampler kernels, fp32 vs fp16, MPS vs CUDA |

## Architecture

```
[Browser]
   │
   ▼
[Gradio UI on :7860]  ◄──── AI_Assistant Python (3.10) in <repo>/.venv
   │
   │ unchanged action layer calls /sdapi/v1/...
   ▼
[FastAPI on :7861] ◄──── mac/comfy_shim.py mounts a1111-compatible endpoints
   │
   │ translates payloads to ComfyUI workflow JSON
   ▼
[ComfyUI sidecar on :8188] ◄──── ComfyUI Python (3.12) in $AI_ASSISTANT_COMFY_DIR/venv
   │
   ▼
[MPS / Apple Silicon GPU]
```

The two Python processes are completely independent; they communicate over HTTP only. ComfyUI knows nothing about AI-Assistant or its action layer.

## License

AGPL-3.0, inherited from upstream.

## See also

- [README.md](README.md) — quick overview
- [HANDOVER.md](.claude/HANDOVER.md) — engineering notes from the porting work (lessons, gotchas, dead ends)
- [tori29umai0123/AI-Assistant](https://github.com/tori29umai0123/AI-Assistant) — the upstream Windows app
