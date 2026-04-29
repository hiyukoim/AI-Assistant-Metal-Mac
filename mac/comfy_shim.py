"""
mac/comfy_shim.py — ComfyUI ↔ AI_Assistant a1111 HTTP shim.

On Mac this module replaces Forge's in-process API. It mounts a1111-
compatible /sdapi/v1/* endpoints on the AI_Assistant FastAPI app and
proxies them to the ComfyUI sidecar at 127.0.0.1:8188.

Implementation lands incrementally:

  #v4 : register_shim is a no-op so AI_Assistant.py boots on Mac.

  #v5 : (this commit) trivial endpoints — /sd-models, /loras,
        /controlnet/model_list, /options, /png-info — so Gradio
        dropdowns populate from the local model dirs and tab UI loads
        without 500s. /img2img returns 501 (not yet implemented).

  #v6 : POST /sdapi/v1/img2img — base SDXL img2img, no ControlNet.
        First inference verification.

  #v7 : ControlNet support, single + multi-unit.

  #v8 : LoRA dynamic loading + tagger ONNX verification.

Design notes:
  - ComfyUI's /object_info is the source of truth for "what models can
    ComfyUI load". Whatever it lists, we return. That avoids the shim
    diverging from ComfyUI's own resolution if the user adds models
    outside our default Stability Matrix tree.
  - Model filename → absolute path resolution uses two fallbacks:
      (1) $AI_ASSISTANT_MODELS_DIR/<StabilityMatrixCamelCase>/<name>
      (2) $AI_ASSISTANT_COMFY_DIR/models/<lowercase>/<name>
    First match wins. If neither matches, hash falls back to "00000000".
  - File hashes use the a1111 "AutoV1" convention (first 8 hex chars
    of SHA-256 over the full file). Cached on disk in
    mac/.hash_cache.json keyed by abspath+mtime so we re-hash only on
    file change.
  - Options store is JSON-on-disk at mac/options.json. AI-Assistant
    actions POST sd_model_checkpoint via /sdapi/v1/options; we round-
    trip the value verbatim. Compatibility quirks
    (e.g. CLIP_stop_at_last_layers ↔ ComfyUI CLIPSetLastLayer) are
    handled in #v6 when the shim translates an img2img payload to a
    ComfyUI workflow.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from PIL import Image

# --- configuration ---------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
HASH_CACHE_FILE = REPO_ROOT / "mac" / ".hash_cache.json"
OPTIONS_FILE = REPO_ROOT / "mac" / "options.json"


def _comfy_url() -> str:
    port = os.environ.get("AI_ASSISTANT_COMFY_PORT", "8188")
    return f"http://127.0.0.1:{port}"


def _models_dir() -> Path:
    raw = os.environ.get("AI_ASSISTANT_MODELS_DIR", str(REPO_ROOT / "models"))
    return Path(raw).expanduser()


def _comfy_dir() -> Path:
    raw = os.environ.get("AI_ASSISTANT_COMFY_DIR", str(Path.home() / "ComfyUI"))
    return Path(raw).expanduser()


# Stability Matrix layout maps ComfyUI's lowercase categories to CamelCase
# directory names. ComfyUI's own default layout uses lowercase category
# names directly under models/. Try SM layout first, fall back to ComfyUI
# default — covers both Yuko's setup and a vanilla install.
_SM_DIR = {
    "checkpoints": "StableDiffusion",
    "loras": "Lora",
    "controlnet": "ControlNet",
    "embeddings": "Embeddings",
    "vae": "VAE",
}


# --- ComfyUI client (cached) ----------------------------------------------

# Keep the /object_info response cached briefly so a single Gradio render
# doesn't hammer ComfyUI with 4–5 identical requests.
_OBJECT_INFO_CACHE: dict = {"data": None, "ts": 0.0}
_OBJECT_INFO_TTL_SEC = 5.0


def _comfy_object_info() -> dict:
    now = time.time()
    if _OBJECT_INFO_CACHE["data"] is not None and (now - _OBJECT_INFO_CACHE["ts"]) < _OBJECT_INFO_TTL_SEC:
        return _OBJECT_INFO_CACHE["data"]
    try:
        r = requests.get(f"{_comfy_url()}/object_info", timeout=5)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001 — keep the UI alive
        print(f"[comfy_shim] /object_info failed: {exc}", flush=True)
        data = {}
    _OBJECT_INFO_CACHE["data"] = data
    _OBJECT_INFO_CACHE["ts"] = now
    return data


def _names_from_node(node: str, field: str) -> list[str]:
    """Return the list of names ComfyUI exposes for a loader node's input.
    Returns [] if ComfyUI is unreachable or the node is missing.
    """
    info = _comfy_object_info()
    try:
        choices = info[node]["input"]["required"][field][0]
    except (KeyError, IndexError, TypeError):
        return []
    if not isinstance(choices, list):
        return []
    return [str(x) for x in choices]


# --- model file resolution -------------------------------------------------

def _resolve_model(category: str, name: str) -> Optional[Path]:
    """Map ComfyUI's (category, filename) to an absolute path on disk.

    Tries Stability Matrix CamelCase first, then ComfyUI's vanilla
    lowercase layout. Returns None if neither resolves.
    """
    candidates = []
    sm_sub = _SM_DIR.get(category, category)
    candidates.append(_models_dir() / sm_sub / name)
    candidates.append(_comfy_dir() / "models" / category / name)
    for p in candidates:
        try:
            if p.is_file():
                return p
        except OSError:
            pass
    return None


# --- hash cache ------------------------------------------------------------

def _load_hash_cache() -> dict:
    try:
        return json.loads(HASH_CACHE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_hash_cache(cache: dict) -> None:
    HASH_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    HASH_CACHE_FILE.write_text(json.dumps(cache, indent=2, sort_keys=True))


# Inline-hash files up to this size. Larger files return the cached value
# if present, otherwise fall back to "00000000" so the dropdown call
# returns within a sensible HTTP timeout. Bulk hashing of large checkpoints
# is the v6 problem (lazy/background population on first inference call).
_INLINE_HASH_MAX_BYTES = 256 * 1024 * 1024  # 256 MB — most LoRAs / small CNs


def _hash_short(path: Optional[Path]) -> str:
    """First 8 hex of SHA-256 (a1111 AutoV1 convention).

    Returns the cached value if present, computes synchronously for files
    under _INLINE_HASH_MAX_BYTES, otherwise returns '00000000' to keep
    /sd-models and /controlnet/model_list responsive (a 6.5 GB SDXL ckpt
    takes ~30 s to hash on a fast SSD; well over a typical HTTP timeout).
    Real hashes for large files are populated in #v6 by a background
    thread kicked off at shim mount time, and cached by abspath+mtime+size
    so each file is hashed at most once.
    """
    if path is None:
        return "00000000"
    try:
        st = path.stat()
    except OSError:
        return "00000000"
    cache = _load_hash_cache()
    key = f"{path}|{int(st.st_mtime)}|{st.st_size}"
    cached = cache.get(key)
    if cached:
        return cached
    if st.st_size > _INLINE_HASH_MAX_BYTES:
        return "00000000"
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError:
        return "00000000"
    short = h.hexdigest()[:8]
    cache[key] = short
    _save_hash_cache(cache)
    return short


# --- options store ---------------------------------------------------------

_DEFAULT_OPTIONS = {
    "sd_model_checkpoint": "",
    "CLIP_stop_at_last_layers": 1,
    "samples_save": True,
    "samples_format": "png",
}


def _load_options() -> dict:
    try:
        on_disk = json.loads(OPTIONS_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        on_disk = {}
    merged = dict(_DEFAULT_OPTIONS)
    merged.update(on_disk)
    return merged


def _save_options(opts: dict) -> None:
    OPTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    OPTIONS_FILE.write_text(json.dumps(opts, indent=2, sort_keys=True))


# --- shim registration -----------------------------------------------------

def register_shim(app: FastAPI) -> None:
    """Mount a1111-compatible /sdapi/v1/* endpoints on the AI_Assistant FastAPI."""

    @app.get("/sdapi/v1/sd-models")
    def get_sd_models() -> list[dict]:
        out: list[dict] = []
        for name in _names_from_node("CheckpointLoaderSimple", "ckpt_name"):
            abs_path = _resolve_model("checkpoints", name)
            short = _hash_short(abs_path)
            stem = Path(name).stem
            out.append({
                "title": f"{name} [{short}]",
                "model_name": stem,
                "hash": short,
                "sha256": None,
                "filename": str(abs_path) if abs_path else name,
                "config": None,
            })
        return out

    @app.get("/sdapi/v1/loras")
    def get_loras() -> list[dict]:
        out: list[dict] = []
        for name in _names_from_node("LoraLoader", "lora_name"):
            stem = Path(name).stem
            out.append({
                "name": stem,
                "alias": stem,
                "path": "",
                "metadata": {},
            })
        return out

    @app.get("/controlnet/model_list")
    def get_cn_model_list() -> dict:
        items: list[str] = []
        for name in _names_from_node("ControlNetLoader", "control_net_name"):
            abs_path = _resolve_model("controlnet", name)
            short = _hash_short(abs_path)
            stem = Path(name).stem
            items.append(f"{stem} [{short}]")
        return {"model_list": items}

    @app.get("/sdapi/v1/options")
    def get_options() -> dict:
        return _load_options()

    @app.post("/sdapi/v1/options")
    async def set_options(req: Request) -> dict:
        try:
            body = await req.json()
        except Exception:
            raise HTTPException(status_code=400, detail="body must be JSON")
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")
        opts = _load_options()
        opts.update(body)
        _save_options(opts)
        return {}

    @app.post("/sdapi/v1/png-info")
    async def png_info(req: Request) -> dict:
        try:
            body = await req.json()
        except Exception:
            raise HTTPException(status_code=400, detail="body must be JSON")
        b64 = body.get("image", "") if isinstance(body, dict) else ""
        if "," in b64:
            # Strip "data:image/png;base64," prefix if present.
            b64 = b64.split(",", 1)[1]
        text: dict = {}
        params = ""
        try:
            data = base64.b64decode(b64)
            img = Image.open(io.BytesIO(data))
            if hasattr(img, "text") and isinstance(img.text, dict):  # type: ignore[attr-defined]
                text = dict(img.text)  # type: ignore[attr-defined]
                params = text.get("parameters", "")
        except Exception as exc:  # noqa: BLE001 — partial info beats 500
            print(f"[comfy_shim] png-info parse failed: {exc}", flush=True)
        return {"info": params, "items": text, "parameters": params}

    @app.post("/sdapi/v1/img2img")
    async def img2img(req: Request) -> None:  # noqa: ARG001
        raise HTTPException(
            status_code=501,
            detail="img2img not yet implemented on Mac (lands in mac-v6)",
        )

    print("[comfy_shim] mounted /sdapi/v1/* endpoints + /controlnet/model_list", flush=True)
