"""
mac/comfy_shim.py — ComfyUI ↔ AI_Assistant a1111 HTTP shim.

On Mac this module replaces Forge's in-process API. It mounts a1111-
compatible /sdapi/v1/* endpoints on the AI_Assistant FastAPI app and
proxies them to the ComfyUI sidecar at 127.0.0.1:8188.

Implementation lands incrementally:

  #v4 : register_shim is a no-op so AI_Assistant.py boots on Mac.

  #v5 : trivial endpoints — /sd-models, /loras, /controlnet/model_list,
        /options, /png-info — so Gradio dropdowns populate.

  #v6 : (this commit) POST /sdapi/v1/img2img — base SDXL img2img,
        no ControlNet. First inference verification.

  #v7 : ControlNet support, single + multi-unit.

  #v8 : LoRA dynamic loading + tagger ONNX verification.

Design notes:
  - ComfyUI's /object_info is the source of truth for "what models can
    ComfyUI load". Whatever it lists, we return.
  - Model filename → absolute path resolution uses two fallbacks:
      (1) $AI_ASSISTANT_MODELS_DIR/<StabilityMatrixCamelCase>/<name>
      (2) $AI_ASSISTANT_COMFY_DIR/models/<lowercase>/<name>
  - The img2img translator builds a ComfyUI workflow graph (dict of
    nodes keyed by string ID), POSTs to /prompt, polls /history, and
    reads the result PNG from $AI_ASSISTANT_COMFY_DIR/output. The
    workflow shape mirrors a1111's stateless KSampler model rather
    than ComfyUI's reactive node-graph workflows; for one-shot
    inference the simpler shape is enough.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import random
import re
import time
import uuid
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from PIL import Image


# --- gradio queue timeout patch (must run before .queue().launch()) -------

def _patch_gradio_queue_timeout() -> None:
    """Make Gradio 3.41's queue tolerate long predicts.

    `gradio/queueing.py` line 80 constructs the queue's internal HTTP
    client as `httpx.AsyncClient(verify=ssl_verify)` — no timeout
    argument, which defaults to **5 s read timeout**. The queue worker
    uses that client to POST to its own /api/predict endpoint (lines
    374–381) where the user's predict function actually runs. Any
    predict taking longer than ~5 s raises `httpx.ReadTimeout`, which
    Gradio catches and surfaces as `process_completed` with
    `success=False` — the user sees "Error" in the output component
    while the predict keeps running on the worker thread and saves a
    correct PNG to disk.

    On Mac/MPS, SDXL inference is 100–500 s. The 5 s timeout makes
    every generation appear to fail. Patch `Queueing.start` so the
    httpx client is built with `timeout=None`.

    This patch runs on `mac.comfy_shim` import, which happens in
    `AI_Assistant_gui.py` before `gradio_tab_gui(...).queue().launch(...)`,
    so the new start() is in place when uvicorn begins serving.
    """
    try:
        import gradio.queueing as gq
        import httpx as _httpx
        from gradio.utils import run_coro_in_background as _run_bg

        async def _patched_start(self, ssl_verify=True):
            self.queue_client = _httpx.AsyncClient(verify=ssl_verify, timeout=None)
            _run_bg(self.start_processing)
            _run_bg(self.start_log_and_progress_updates)
            if not self.live_updates:
                _run_bg(self.notify_clients)

        gq.Queue.start = _patched_start
        print(
            "[comfy_shim] patched gradio.queueing.Queue.start: queue_client.timeout=None",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 — never break the app over the patch
        print(f"[comfy_shim] gradio queue patch failed (continuing without): {exc}", flush=True)


_patch_gradio_queue_timeout()


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
    under _INLINE_HASH_MAX_BYTES, otherwise returns '00000000'.
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


# --- a1111 → ComfyUI sampler map -------------------------------------------

# (sampler_name, scheduler) — ComfyUI's KSampler takes both as separate
# enums. a1111's "DPM++ 2M Karras" is one combined choice; we split it.
# Anything not in this table maps to ("euler", "normal") with a warning.
_SAMPLER_MAP: dict[str, tuple[str, str]] = {
    "Euler a":              ("euler_ancestral", "normal"),
    "Euler":                ("euler",           "normal"),
    "LMS":                  ("lms",             "normal"),
    "Heun":                 ("heun",            "normal"),
    "DPM2":                 ("dpm_2",           "normal"),
    "DPM2 a":               ("dpm_2_ancestral", "normal"),
    "DPM++ 2S a":           ("dpmpp_2s_ancestral", "normal"),
    "DPM++ 2M":             ("dpmpp_2m",        "normal"),
    "DPM++ SDE":            ("dpmpp_sde",       "normal"),
    "DPM++ 2M SDE":         ("dpmpp_2m_sde",    "normal"),
    "DPM fast":             ("dpm_fast",        "normal"),
    "DPM adaptive":         ("dpm_adaptive",    "normal"),
    "LMS Karras":           ("lms",             "karras"),
    "DPM2 Karras":          ("dpm_2",           "karras"),
    "DPM2 a Karras":        ("dpm_2_ancestral", "karras"),
    "DPM++ 2S a Karras":    ("dpmpp_2s_ancestral", "karras"),
    "DPM++ 2M Karras":      ("dpmpp_2m",        "karras"),
    "DPM++ SDE Karras":     ("dpmpp_sde",       "karras"),
    "DPM++ 2M SDE Karras":  ("dpmpp_2m_sde",    "karras"),
    "DDIM":                 ("ddim",            "normal"),
    "PLMS":                 ("plms",            "normal"),
    "UniPC":                ("uni_pc",          "normal"),
}


def _a1111_sampler_to_comfy(name: str) -> tuple[str, str]:
    if name in _SAMPLER_MAP:
        return _SAMPLER_MAP[name]
    print(f"[comfy_shim] unknown sampler '{name}', falling back to euler/normal", flush=True)
    return ("euler", "normal")


# --- title → filename helpers ----------------------------------------------

def _strip_hash_suffix(title: str) -> str:
    """Drop the trailing ' [hash8]' suffix from a sd-models title.

    Inputs we accept (no quotes, just the example string):
      "animagine-xl-3.1.safetensors [00000000]"   → "animagine-xl-3.1.safetensors"
      "animagine-xl-3.1.safetensors"              → "animagine-xl-3.1.safetensors"
      ""                                          → ""
    """
    title = (title or "").strip()
    if title.endswith("]") and " [" in title:
        return title.rsplit(" [", 1)[0]
    return title


def _resolve_checkpoint_for_workflow(payload: dict) -> str:
    """Return the ComfyUI checkpoint filename to use for this request.

    Resolution order:
      1. payload["override_settings"]["sd_model_checkpoint"]
      2. options.sd_model_checkpoint (from /sdapi/v1/options)
      3. The first ckpt ComfyUI knows about (last-resort fallback).
    """
    override = (payload.get("override_settings") or {}).get("sd_model_checkpoint")
    if override:
        return _strip_hash_suffix(override)
    chosen = _load_options().get("sd_model_checkpoint", "")
    if chosen:
        return _strip_hash_suffix(chosen)
    available = _names_from_node("CheckpointLoaderSimple", "ckpt_name")
    if available:
        print(f"[comfy_shim] no sd_model_checkpoint set, defaulting to {available[0]}", flush=True)
        return available[0]
    raise HTTPException(status_code=503, detail="No checkpoints available; set sd_model_checkpoint via /sdapi/v1/options")


# --- init image staging ----------------------------------------------------

def _save_init_image(b64: str) -> str:
    """Decode a base64 PNG/JPEG and save it to ComfyUI's input dir.
    Returns the bare filename (no path) — that's what LoadImage expects.
    """
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    fname = f"ai_assistant_{uuid.uuid4().hex[:12]}.png"
    target_dir = _comfy_dir() / "input"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / fname
    img.save(target, format="PNG")
    return fname


# --- LoRA token parsing ----------------------------------------------------

# AI_Assistant's exui mode (--exui flag) shows a LoRA dropdown that appends
# <lora:NAME:WEIGHT> tokens to the positive prompt. a1111 / Forge parse
# those tokens out and apply them as LoRAs at sample time. ComfyUI uses
# LoraLoader nodes inserted between the checkpoint and the CLIP/UNet
# consumers. We mirror that: pull tokens out of the prompt before
# CLIPTextEncode runs, insert one LoraLoader per token, chain them so each
# one's outputs feed the next.

# Match <lora:NAME:WEIGHT[:CLIP_WEIGHT]>. NAME may contain dots, dashes,
# slashes, spaces; WEIGHT is a signed float. CLIP_WEIGHT is optional —
# when present, it overrides the LoRA's CLIP strength independently of
# the model strength (a1111 / Forge convention).
_LORA_TOKEN_RE = re.compile(
    r"<lora:([^:>]+):(-?\d+(?:\.\d+)?)(?::(-?\d+(?:\.\d+)?))?>",
    re.IGNORECASE,
)


def _resolve_lora_name(stem_or_alias: str) -> Optional[str]:
    """Map an alias (no extension) or a filename to a ComfyUI LoraLoader name.

    Match by stem (case-insensitive) so users can write
    <lora:sdxl_BWLine:0.8> or <lora:sdxl_BWLine.safetensors:0.8> and
    get the same result.
    """
    target = stem_or_alias.strip()
    target_stem = Path(target).stem.lower()
    for name in _names_from_node("LoraLoader", "lora_name"):
        if Path(name).stem.lower() == target_stem:
            return name
    return None


def _extract_lora_tokens(prompt: str) -> tuple[str, list[dict]]:
    """Strip <lora:NAME:WEIGHT> tokens from `prompt` and return:
       (cleaned_prompt, [{name, model_strength, clip_strength}, …]).

    Unresolvable tokens are dropped from the prompt with a warning so
    the user doesn't get the literal text fed into CLIPTextEncode.
    """
    if not prompt or "<lora:" not in prompt.lower():
        return prompt, []

    units: list[dict] = []

    def _replace(match: re.Match) -> str:
        raw_name = match.group(1)
        weight = float(match.group(2))
        clip_weight = float(match.group(3)) if match.group(3) is not None else weight
        resolved = _resolve_lora_name(raw_name)
        if not resolved:
            print(f"[comfy_shim] LoRA '{raw_name}' not found; dropping token", flush=True)
            return ""
        units.append({
            "name": resolved,
            "model_strength": weight,
            "clip_strength": clip_weight,
        })
        return ""

    cleaned = _LORA_TOKEN_RE.sub(_replace, prompt)
    # Collapse the comma+space sequences that often surround the stripped
    # tokens (e.g. "tag, <lora:x:1>, tag" → "tag, , tag" → "tag, tag").
    cleaned = re.sub(r",\s*,", ",", cleaned)
    cleaned = re.sub(r",\s*$", "", cleaned).strip()
    return cleaned, units


def _build_lora_chain(units: list[dict]) -> tuple[list, list, dict]:
    """Build the LoraLoader chain.

    Returns:
      model_link  : ["node_id", 0]  — final MODEL output to feed KSampler
      clip_link   : ["node_id", 1]  — final CLIP output to feed CLIPSetLastLayer
      nodes       : {node_id: {…}, …} — LoraLoader node definitions

    With no LoRA units, the model_link and clip_link point at the
    checkpoint loader (node "1") directly and the nodes dict is empty.
    """
    if not units:
        return ["1", 0], ["1", 1], {}

    nodes: dict = {}
    # Start LoRA node IDs at 100 to leave room for the base (1-9) and
    # ControlNet (10+) namespaces. ComfyUI accepts any string ID.
    next_id = 100
    prev_model: list = ["1", 0]
    prev_clip: list = ["1", 1]
    summary: list[str] = []

    for u in units:
        node_id = str(next_id); next_id += 1
        nodes[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model":           prev_model,
                "clip":            prev_clip,
                "lora_name":       u["name"],
                "strength_model":  float(u["model_strength"]),
                "strength_clip":   float(u["clip_strength"]),
            },
        }
        prev_model = [node_id, 0]
        prev_clip = [node_id, 1]
        summary.append(f"{Path(u['name']).stem}@{u['model_strength']:g}")

    if summary:
        print(f"[comfy_shim] LoRA chain: {' → '.join(summary)}", flush=True)
    return prev_model, prev_clip, nodes


# --- a1111 → ComfyUI preprocessor map --------------------------------------

# AI-Assistant's actions only invoke two preprocessor names (verified by
# grepping AI_Assistant_modules/actions/ + utils/request_api.py). Each
# entry maps to a (class_type, fixed_inputs) pair; the per-request
# resolution and image input are wired up in _add_controlnet_units.
#
# "None" means "send the raw image to ControlNet without preprocessing"
# — we just don't insert a preprocessor node.
_PREPROCESSOR_MAP: dict[str, tuple[str, dict]] = {
    "lineart_realistic": ("LineArtPreprocessor", {"coarse": "disable"}),
    "lineart_anime":     ("AnimeLineArtPreprocessor", {}),
    "canny":             ("CannyEdgePreprocessor", {"low_threshold": 100, "high_threshold": 200}),
    "openpose":          ("OpenposePreprocessor", {"detect_hand": "enable", "detect_body": "enable", "detect_face": "enable"}),
    "depth_midas":       ("MiDaS-DepthMapPreprocessor", {}),
}


def _resolve_controlnet_name(field: str) -> Optional[str]:
    """Map an a1111 ControlNet model field (which carries a [hash8] suffix
    that may not match our hash) back to a ComfyUI controlnet filename.

    Strategy: drop the hash suffix and match by the leading stem against
    ComfyUI's known names (case-insensitive, stem-only). The hash on the
    a1111 side is computed by Forge using a different cached value than
    what our shim publishes, so matching by name is the only path that
    works reliably across both sides.
    """
    bare = _strip_hash_suffix(field)
    target_stem = Path(bare).stem.lower()
    for name in _names_from_node("ControlNetLoader", "control_net_name"):
        if Path(name).stem.lower() == target_stem:
            return name
    return None


# --- workflow assembly + submission ----------------------------------------

def _build_img2img_workflow(payload: dict, ckpt: str, init_filename: str) -> dict:
    """Translate an a1111 /img2img payload to a ComfyUI workflow JSON.

    Supports: prompt, negative_prompt, sampler, steps, cfg, denoise,
    seed, CLIP_stop_at_last_layers, ControlNet units (single + multi).

    Not yet supported (logged-and-ignored or rejected in caller):
    - LoRA token parsing inside the prompt (lands in #v8)
    - mask / inpainting (logged-and-ignored)
    """
    if "mask" in payload and payload.get("mask"):
        print("[comfy_shim] mask field present in img2img payload; ignored (inpainting deferred)", flush=True)

    raw_prompt = str(payload.get("prompt") or "")
    raw_negative = str(payload.get("negative_prompt") or "")

    # Strip <lora:…> tokens out of both prompts and collect them; the
    # tokens themselves shouldn't reach CLIPTextEncode. We process
    # negative-side tokens too in case a tab adds them, even though
    # AI_Assistant currently only injects LoRAs into the positive prompt.
    prompt, lora_units_pos = _extract_lora_tokens(raw_prompt)
    negative, lora_units_neg = _extract_lora_tokens(raw_negative)
    lora_units = lora_units_pos + lora_units_neg
    steps = int(payload.get("steps") or 20)
    cfg = float(payload.get("cfg_scale") or 7.0)
    denoise = float(payload.get("denoising_strength") or 0.75)
    seed_in = payload.get("seed", -1)
    try:
        seed = int(seed_in)
    except (TypeError, ValueError):
        seed = -1
    if seed < 0:
        seed = random.SystemRandom().randint(0, 2**63 - 1)
        print(f"[comfy_shim] seed=-1 → using {seed}", flush=True)

    sampler, scheduler = _a1111_sampler_to_comfy(str(payload.get("sampler_name") or "Euler a"))

    clip_skip = int((payload.get("override_settings") or {}).get("CLIP_stop_at_last_layers")
                    or _load_options().get("CLIP_stop_at_last_layers", 1))
    clip_last_layer = -max(1, clip_skip)

    # Base workflow without ControlNet. positive/negative conditioning
    # outputs at "3" and "4" feed KSampler directly. _add_controlnet_units
    # below splices ControlNetApplyAdvanced nodes into that path.
    #
    # LoRA chain: each <lora:name:weight> token in either prompt becomes a
    # LoraLoader node. The first LoraLoader takes (model, clip) from the
    # checkpoint; each subsequent one takes (model, clip) from the prior
    # LoraLoader. The final outputs feed CLIPSetLastLayer (clip) and
    # KSampler (model). _used_lora_units captures the chained outputs we
    # use as `model` and `clip` references below.
    model_link, clip_link, lora_chain_nodes = _build_lora_chain(lora_units)

    base_nodes: dict = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt},
        },
        "2": {
            "class_type": "CLIPSetLastLayer",
            "inputs": {"clip": clip_link, "stop_at_clip_layer": clip_last_layer},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["2", 0]},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["2", 0]},
        },
        "5": {
            "class_type": "LoadImage",
            "inputs": {"image": init_filename},
        },
        "6": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["5", 0], "vae": ["1", 2]},
        },
        # KSampler conditioning links are placeholders ("3", "4"); they
        # get rewritten by _add_controlnet_units when CN is in play.
        # Model link goes through the LoRA chain (or directly from the
        # checkpoint when no LoRAs are present).
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model":           model_link,
                "positive":        ["3", 0],
                "negative":        ["4", 0],
                "latent_image":    ["6", 0],
                "seed":            seed,
                "steps":           steps,
                "cfg":             cfg,
                "sampler_name":    sampler,
                "scheduler":       scheduler,
                "denoise":         denoise,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["7", 0], "vae": ["1", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": "ai_assistant"},
        },
    }
    workflow: dict = {**base_nodes, **lora_chain_nodes}

    # Splice in ControlNet units (if any).
    cn_args = ((payload.get("alwayson_scripts") or {}).get("ControlNet") or {}).get("args") or []
    cn_units_used = _add_controlnet_units(workflow, cn_args)

    workflow["__a1111__"] = {
        "ckpt": ckpt, "sampler": sampler, "scheduler": scheduler,
        "steps": steps, "cfg": cfg, "denoise": denoise, "seed": seed,
        "clip_last_layer": clip_last_layer,
        "init_filename": init_filename,
        "cn_units": cn_units_used,
    }
    return workflow


def _add_controlnet_units(workflow: dict, cn_args: list) -> list:
    """Splice ControlNetApplyAdvanced nodes into a base img2img workflow.

    For each enabled CN unit:
      - Save the unit's input image to ComfyUI/input/ via _save_init_image.
      - Add a LoadImage node for it.
      - Optionally chain a preprocessor node (per _PREPROCESSOR_MAP).
      - Add ControlNetLoader for the named model.
      - Add ControlNetApplyAdvanced linking the running positive/negative
        conditioning to the next slot.

    Mutates `workflow` in place. Re-points KSampler's positive/negative
    inputs to the last unit's output if any unit was added.

    Returns a list of {model, module, weight} dicts for diagnostic logging.
    """
    used: list = []
    if not cn_args:
        return used

    next_id = 10  # workflow nodes 1..9 are reserved for the base.
    pos_link: list = ["3", 0]
    neg_link: list = ["4", 0]

    for idx, unit in enumerate(cn_args):
        if not unit:
            continue
        if not unit.get("enabled", True):
            continue
        b64_img = unit.get("image")
        if not b64_img:
            continue
        weight = float(unit.get("weight", 1.0))
        guidance_start = float(unit.get("guidance_start", 0.0))
        guidance_end = float(unit.get("guidance_end", 1.0))
        module = str(unit.get("module") or "None")
        model_field = str(unit.get("model") or "")

        comfy_cn = _resolve_controlnet_name(model_field)
        if not comfy_cn:
            print(f"[comfy_shim] CN unit {idx}: model '{model_field}' not found in ComfyUI; skipping", flush=True)
            continue

        # 1. save image to ComfyUI/input
        try:
            unit_filename = _save_init_image(b64_img)
        except Exception as exc:  # noqa: BLE001
            print(f"[comfy_shim] CN unit {idx}: image decode failed: {exc}; skipping", flush=True)
            continue

        # 2. LoadImage
        load_id = str(next_id); next_id += 1
        workflow[load_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": unit_filename},
        }
        image_link: list = [load_id, 0]

        # 3. preprocessor (optional)
        if module != "None":
            mapping = _PREPROCESSOR_MAP.get(module)
            if mapping:
                pp_class, pp_extra = mapping
                pp_id = str(next_id); next_id += 1
                pp_inputs: dict = {"image": image_link}
                # 'resolution' is an optional input for most preprocessors;
                # pass through a1111's processor_res if the node accepts it.
                node_inputs = _comfy_object_info().get(pp_class, {}).get("input", {}) or {}
                accepted = set(node_inputs.get("required", {}) or {}) | set(node_inputs.get("optional", {}) or {})
                if "resolution" in accepted:
                    pp_inputs["resolution"] = int(unit.get("processor_res", 512))
                # Filter pp_extra to keys the node actually accepts so a
                # newer preprocessor variant doesn't 400 on an unknown key.
                for key, value in pp_extra.items():
                    if key in accepted:
                        pp_inputs[key] = value
                workflow[pp_id] = {"class_type": pp_class, "inputs": pp_inputs}
                image_link = [pp_id, 0]
            else:
                print(f"[comfy_shim] CN unit {idx}: unknown module '{module}', using raw image", flush=True)

        # 4. ControlNetLoader
        cn_loader_id = str(next_id); next_id += 1
        workflow[cn_loader_id] = {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": comfy_cn},
        }

        # 5. ControlNetApplyAdvanced
        apply_id = str(next_id); next_id += 1
        workflow[apply_id] = {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive":      pos_link,
                "negative":      neg_link,
                "control_net":   [cn_loader_id, 0],
                "image":         image_link,
                "strength":      weight,
                "start_percent": guidance_start,
                "end_percent":   guidance_end,
            },
        }
        pos_link = [apply_id, 0]
        neg_link = [apply_id, 1]

        used.append({
            "idx": idx,
            "model": comfy_cn,
            "module": module,
            "weight": weight,
            "guidance": [guidance_start, guidance_end],
        })

    if used:
        # Re-point the KSampler at the chained CN output.
        workflow["7"]["inputs"]["positive"] = pos_link
        workflow["7"]["inputs"]["negative"] = neg_link
    return used


def _submit_workflow(workflow: dict) -> str:
    """POST a workflow to ComfyUI's /prompt and return the prompt_id."""
    diag = workflow.pop("__a1111__", {})
    if diag:
        cn_units = diag.get("cn_units") or []
        cn_summary = ""
        if cn_units:
            cn_summary = " cn=[" + ", ".join(
                f"#{u['idx']}:{Path(u['model']).stem}({u['module']},w={u['weight']})"
                for u in cn_units
            ) + "]"
        print(
            f"[comfy_shim] img2img: ckpt={diag['ckpt']} "
            f"sampler={diag['sampler']}/{diag['scheduler']} "
            f"steps={diag['steps']} cfg={diag['cfg']} denoise={diag['denoise']} "
            f"seed={diag['seed']} clip_last={diag['clip_last_layer']}"
            f"{cn_summary}",
            flush=True,
        )
    body = {"prompt": workflow, "client_id": uuid.uuid4().hex}
    try:
        r = requests.post(f"{_comfy_url()}/prompt", json=body, timeout=30)
        r.raise_for_status()
    except requests.HTTPError as exc:
        # ComfyUI returns 400 with detailed validation errors; surface them.
        detail = exc.response.text if exc.response is not None else str(exc)
        raise HTTPException(status_code=502, detail=f"ComfyUI rejected the prompt: {detail}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"ComfyUI /prompt failed: {exc}") from exc
    data = r.json()
    pid = data.get("prompt_id")
    if not pid:
        raise HTTPException(status_code=502, detail=f"ComfyUI /prompt missing prompt_id: {data}")
    return pid


# Default 600 s — first inference compiles Metal kernels (~30 s) and may
# need to load a 6 GB checkpoint from a cold disk (~20 s on the SSD), so
# 60 s isn't enough for a first run. Subsequent runs return in seconds.
_HISTORY_POLL_TIMEOUT_SEC = 600


def _poll_history(prompt_id: str, timeout: int = _HISTORY_POLL_TIMEOUT_SEC) -> dict:
    """Poll /history/{prompt_id} until it has outputs, or raise."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{_comfy_url()}/history/{prompt_id}", timeout=5)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            print(f"[comfy_shim] /history poll error: {exc}", flush=True)
            time.sleep(1.0)
            continue
        entry = data.get(prompt_id)
        if entry and entry.get("outputs"):
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise HTTPException(status_code=502, detail=f"ComfyUI execution error: {status}")
            return entry["outputs"]
        time.sleep(0.5)
    raise HTTPException(status_code=504, detail=f"Timed out waiting for ComfyUI prompt {prompt_id}")


def _run_img2img_blocking(workflow: dict) -> tuple[str, Path]:
    """Submit + poll + read in a single synchronous chunk so the caller
    can put the whole thing on a worker thread via asyncio.to_thread.
    """
    prompt_id = _submit_workflow(workflow)
    outputs = _poll_history(prompt_id)
    return _read_output_image(outputs)


def _read_output_image(outputs: dict) -> tuple[str, Path]:
    """Walk a ComfyUI history outputs dict, find the first SaveImage result,
    return (base64_png, abs_path_for_logging).
    """
    for _node_id, node_out in outputs.items():
        images = node_out.get("images") or []
        if not images:
            continue
        first = images[0]
        # ComfyUI writes to subfolder "" by default ("output"), not the
        # named subdir; respect whatever filename/subfolder/type fields it
        # gave us so this works under arbitrary ComfyUI configs.
        subfolder = first.get("subfolder", "")
        filename = first.get("filename")
        if not filename:
            continue
        type_ = first.get("type", "output")  # "output" or "temp"
        base = _comfy_dir() / type_
        target = base / subfolder / filename if subfolder else base / filename
        try:
            data = target.read_bytes()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=502, detail=f"ComfyUI output not found at {target}") from exc
        return base64.b64encode(data).decode("ascii"), target
    raise HTTPException(status_code=502, detail="ComfyUI history had no images")


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
    async def img2img(req: Request) -> dict:
        try:
            payload = await req.json()
        except Exception:
            raise HTTPException(status_code=400, detail="body must be JSON")
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")

        init_images = payload.get("init_images") or []
        if not init_images:
            raise HTTPException(status_code=400, detail="init_images is required")
        try:
            init_filename = await asyncio.to_thread(_save_init_image, str(init_images[0]))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"init_images[0] decode failed: {exc}") from exc

        ckpt = _resolve_checkpoint_for_workflow(payload)
        workflow = _build_img2img_workflow(payload, ckpt, init_filename)

        # ComfyUI inference is the slow part — 20 steps of SDXL at ~6 s/it
        # = ~2 min. Run the entire submit + poll + read pipeline in a
        # worker thread so the FastAPI event loop stays free to service
        # other requests (Gradio's WebSocket keepalives, the action's
        # status polling, etc). Without this the UI's "Connection
        # errored out" fires before the inference finishes.
        t0 = time.time()
        b64, target = await asyncio.to_thread(_run_img2img_blocking, workflow)
        elapsed = time.time() - t0
        print(f"[comfy_shim] img2img done in {elapsed:.1f}s → {target}", flush=True)

        return {
            "images": [b64],
            "parameters": {
                "prompt": payload.get("prompt"),
                "negative_prompt": payload.get("negative_prompt"),
                "steps": payload.get("steps"),
                "cfg_scale": payload.get("cfg_scale"),
                "sampler_name": payload.get("sampler_name"),
                "denoising_strength": payload.get("denoising_strength"),
                "width": payload.get("width"),
                "height": payload.get("height"),
                "seed": payload.get("seed"),
            },
            "info": json.dumps({
                "prompt": payload.get("prompt"),
                "negative_prompt": payload.get("negative_prompt"),
                "seed": payload.get("seed"),
                "sampler_name": payload.get("sampler_name"),
                "steps": payload.get("steps"),
                "cfg_scale": payload.get("cfg_scale"),
                "denoising_strength": payload.get("denoising_strength"),
                "Model": Path(ckpt).stem,
                "Backend": "ComfyUI (Mac shim)",
            }),
        }

    print("[comfy_shim] mounted /sdapi/v1/* endpoints + /controlnet/model_list", flush=True)
