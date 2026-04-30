# Legacy Windows artifacts

These files are the **upstream Windows / Forge install chain** from
[`tori29umai0123/AI-Assistant`](https://github.com/tori29umai0123/AI-Assistant).
They are kept here as the AGPL-3.0 derivation record and for anyone who
wants to study how the original app installed itself on Windows + CUDA.

**They are not used by the Mac port.** Don't run them on Mac — see
[`MAC_SETUP.md`](../MAC_SETUP.md) for the Mac install path
(`mac/install.sh` etc.).

## What's here

| File | Purpose |
|---|---|
| `AI_Assistant_install.ps1` | PowerShell installer that clones Forge, runs setup.py, runs model downloader |
| `AI_Assistant_setup.py` | Python script that rewrites `__file__` paths in vendored Forge code, copies AI_Assistant overrides into `modules/` and `ldm_patched/`, edits `webui-user.bat`, appends to `requirements_versions.txt` |
| `AI_Assistant_model_DL.cmd` | Windows .cmd script that downloads ~12 GB of models with `certutil` SHA-256 verification (Mac equivalent: `mac/download_models.sh`) |
| `AI_Assistant_exUI.bat` | Windows launcher that runs the .exe with `--exui` |
| `venv.cmd` | One-line activator (`cmd /k venv\Scripts\activate`) |
| `AI_Assistant_ReadMe.txt` | Original Shift-JIS Japanese README |
| `repositories/` | Forge's vendored deps (BLIP, generative-models, k-diffusion, stable-diffusion-stability-ai, stable-diffusion-webui-assets) |

## Why keep them?

1. **AGPL-3.0 derivation record** — preserving the upstream artifacts
   we built on makes it explicit what this fork derives from.
2. **Possible re-merge** — if the Mac port ever needs to follow
   upstream behavior changes, having the original install chain
   side-by-side simplifies diffing.
3. **Reference implementation** — the Bash equivalents in `mac/`
   reference these files in comments (e.g. "mirrors
   `AI_Assistant_model_DL.cmd:180-202`").

## Why not delete?

Git history would still preserve them, but having them in a side
folder makes the "this is a port of that" relationship visible to
anyone browsing the repo without a `git log`.
