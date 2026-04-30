# AI-Assistant for Apple Silicon Mac

Apple Silicon Mac (M-series) port of [`tori29umai0123/AI-Assistant`](https://github.com/tori29umai0123/AI-Assistant), an SDXL drawing-assistance Gradio app with 10 specialised tabs (img2img, lineart, normal map, lighting, anime shadow, colour scheme, coloring, resize, etc.).

This fork replaces the embedded `stable-diffusion-webui-forge` backend with a [ComfyUI](https://github.com/comfyanonymous/ComfyUI) sidecar fronted by a small a1111-API-compatible HTTP shim. The 10 original Gradio tabs and the action layer that drives them are preserved verbatim, so the user-facing workflow matches the Windows app.

## Setup

See **[MAC_SETUP.md](MAC_SETUP.md)** for the full install + run + troubleshooting guide.

Three commands:

```bash
./mac/install.sh           # Python 3.10.11 venv + deps
./mac/install_comfyui.sh   # ComfyUI sidecar verifier (or auto-clone if you don't have ComfyUI yet)
./mac/download_models.sh   # ~12 GB AI-Assistant model set
./mac/start.sh             # launches both processes + opens the UI
```

## Highlights

- **Native MPS inference.** No CUDA emulation, no Rosetta. fp32 on Apple Silicon GPU because bf16 NaNs on MPS for SDXL.
- **Hyper-SDXL / LCM / Lightning auto-detect.** Add `<lora:Hyper-SDXL-8steps-CFG-lora:1.0>` to your prompt and the shim auto-overrides steps to 8 — ~2.5× faster generation with no UI changes.
- **Full Forge → ComfyUI translation.** 22-entry sampler map; ControlNet `control_mode` parity (Balanced / My prompt is more important / ControlNet is more important) via [Kosinkadink/ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet); `pixel_perfect` honoured; LoRA dynamic loading via prompt tokens.
- **Inpainting works.** White-mask sentinel detection, `VAEEncodeForInpaint` with `grow_mask_by` from the `mask_blur` payload field.
- **ESRGAN upscale on the resize tab.** Defaults to `4x_NMKD-Superscale-SP_178000_G.pth`; overridable via `AI_ASSISTANT_UPSCALER`.
- **Gradio queue patched.** The 5 s `httpx` read timeout buried in `gradio/queueing.py:80` that turned every long Mac inference into a phantom "Error" is monkey-patched at shim mount time.
- **Reuses your existing ComfyUI install** if you have one (e.g. via [Stability Matrix](https://lykos.ai/stability-matrix)) — no duplicate downloads.

## What runs where

```
[Browser]
   │
   ▼
[Gradio UI on :7860]   ◄── AI_Assistant Python 3.10 in <repo>/.venv
   │
   │ unchanged action layer calls /sdapi/v1/...
   ▼
[FastAPI on :7861]    ◄── mac/comfy_shim.py mounts a1111-compatible endpoints
   │
   │ translates payloads to ComfyUI workflow JSON
   ▼
[ComfyUI on :8188]    ◄── ComfyUI Python 3.12 in $AI_ASSISTANT_COMFY_DIR/venv
   │
   ▼
[MPS / Apple Silicon]
```

The two Python venvs are independent (Python 3.10 for the front-end, 3.12 for ComfyUI). They communicate over HTTP only.

## Performance — M4 Pro 24 GB

| Configuration | Time |
|---|---|
| First inference (Metal compile + 6 GB SDXL load) | +30–60 s one-off |
| i2i 20 steps, no ControlNet | ~150–200 s |
| i2i 20 steps + 1 ControlNet | ~200–250 s |
| **+ Hyper-SDXL 8-step LoRA** | **~50–80 s** |
| Resize tab (canonical render + 4× ESRGAN) | +5 s |

Subsequent generations on the same checkpoint reuse the loaded weights — much faster than the first run.

## Out of scope

- bf16 / fp16 mixed precision (NaN on MPS for SDXL — verified)
- PyInstaller `.app` bundling
- Bitwise output parity with the Windows app (different sampler kernels and precision; impossible by construction)

## License

[AGPL-3.0](LICENSE.txt), inherited from upstream.

## Upstream

This is a Mac-only fork. For the Windows / CUDA / Forge version, see [`tori29umai0123/AI-Assistant`](https://github.com/tori29umai0123/AI-Assistant).
