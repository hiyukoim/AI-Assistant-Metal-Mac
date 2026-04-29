"""
mac/comfy_shim.py — placeholder for the ComfyUI ↔ AI_Assistant a1111 HTTP shim.

On Mac, this module replaces Forge's in-process API. The shim mounts
a1111-compatible /sdapi/v1/* endpoints on the AI_Assistant FastAPI app,
translates incoming a1111 payloads into ComfyUI workflow JSON, posts to
the ComfyUI sidecar at 127.0.0.1:8188, polls /history/{prompt_id}, and
returns base64-encoded results to the AI_Assistant Gradio front-end.

Implementation lands incrementally:

  #v4 (this commit) : register_shim is a no-op so AI_Assistant.py boots
                       on Mac without crashing on initialize_forge().
                       Gradio renders, every tab opens, dropdowns are
                       empty, submitting a tab returns 404.

  #v5               : trivial endpoints (/sd-models, /loras,
                       /controlnet/model_list, /options, /png-info) so
                       dropdowns populate from the local model dirs.

  #v6               : POST /sdapi/v1/img2img — base SDXL img2img path,
                       no ControlNet. First inference verification.

  #v7               : ControlNet support, single + multi-unit.

  #v8               : LoRA dynamic loading + tagger ONNX verification.
"""

from __future__ import annotations

from fastapi import FastAPI


def register_shim(app: FastAPI) -> None:
    """Mount a1111-compatible /sdapi/v1/* endpoints on the AI_Assistant FastAPI.

    No-op for #v4. The presence of this function (and a successful import)
    is the only thing AI_Assistant_gui needs to launch on Mac. Real
    routes are added in #v5+.
    """
    return None
