#!/usr/bin/env bash
# AI-Assistant — Mac model downloader
#
# Bash port of AI_Assistant_model_DL.cmd (Windows). Downloads the 23 files
# AI-Assistant ships with, verifying SHA-256 hashes (3-retry per file).
# Idempotent: re-running with all files already present and hash-valid is
# a no-op.
#
# Files land in $AI_ASSISTANT_MODELS_DIR/{StableDiffusion,Lora,ControlNet,Tagger}.
# These are CamelCase Stability Matrix conventions; ComfyUI sees the
# safetensors via the existing extra_model_paths.yaml, so AI-Assistant's
# inference path picks them up automatically.
#
# About the hashes: the values below are copied byte-for-byte from
# AI_Assistant_model_DL.cmd lines 180-202 — do not paraphrase or shorten.
# A drift here means a silent download integrity failure.
#
# Compatible with Apple's stock /bin/bash 3.2 (no associative arrays).

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# --- load mac/config.local.env (gitignored, optional) ----------------------

if [[ -f "$REPO_ROOT/mac/config.local.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/mac/config.local.env"
    set +a
fi

: "${AI_ASSISTANT_MODELS_DIR:=$REPO_ROOT/models}"

if [[ ! -d "$AI_ASSISTANT_MODELS_DIR" ]]; then
    echo "Creating $AI_ASSISTANT_MODELS_DIR ..."
    mkdir -p "$AI_ASSISTANT_MODELS_DIR"
fi

echo "AI_ASSISTANT_MODELS_DIR = $AI_ASSISTANT_MODELS_DIR"
echo

# --- hash table (mirror of AI_Assistant_model_DL.cmd lines 180-202) -------
#
# Plain whitespace-separated table (relative_path SHA256). Looked up via awk
# so we don't need bash 4 associative arrays.

HASH_TABLE='
ControlNet/CN-anytest_v3-50000_am_dim256.safetensors        9c022669c9225a926c9cbca9baaf40387f2a6d579ea004cd15b2d84b7d130052
ControlNet/CN-anytest_v4-marged_am_dim256.safetensors       62a63fb885caa1aff54cbceceb0a20968922f45b2d5a370e64b156a982132ffb
ControlNet/control-lora-canny-rank256.safetensors           21f79f7368eff07f57bcd507ca91c0fc89070d7da182960ff24ed1d58310c3a7
ControlNet/controlnet852AClone_v10.safetensors              58bae8a373d6a39b33a5d110c5b22894fc86b7b1e189b05b163e69446c7f48ee
ControlNet/Kataragi_lineartXL-lora128.safetensors           bdc33b12ff20900a7fdea0b239c8ee66180d53b9a13f6f87a9d89d2aee9eac91
ControlNet/CL_am31_pose3D_V7_marged_rank256.safetensors     a34b7efd90e9820e6c065c66665409f3ce2324eee98237f89a40f41a6218a3ad
Lora/sdxl-testlora-plainmaterial.safetensors                24df34c2c3abf62c7c1b7ee5432935861b10c1cd38b6997b37743701df6cfe71
Lora/anime01.safetensors                                    14fc521897c6298272d1ba62dbe9a41e2c2ea3464b23760c7a956d50dd2b0fd5
Lora/anime02.safetensors                                    a6cb70645577e8e5e757dbb511dc913046c492f1b46932d891a684e59108b038
Lora/anime03.safetensors                                    5a4c1dedb42b469243c1201213e6e59d9bd0f01edb3a99ce93705200886fb842
Lora/animenuri.safetensors                                  afe115b55d2141f3ff39bdad2ea656555568f659b6ab34a8db2dc22ed2410441
Lora/atunuri02.safetensors                                  da22a0ed520b03368d2403ed35db6c3c2213c04ab236535133bf7c830fe91b36
Lora/sdxl-testlora-normalmap_04b_dim32.safetensors          9432dee2c0b9e1636e7c6e9a544571991fc22a455d575ffc1e281a57efee727a
Lora/SDXL_baketu2.safetensors                               d3f935e50967dd7712afdccaa9cdbd115b47c1fb61950553f5b4a70f2d99b3c0
Lora/sdxl_BWLine.safetensors                                07c59708361b3e2e4f0b0c0f232183f5f39c32c31b6b6981b4392ea30d49dd57
Lora/sdxl_BW_bold_Line.safetensors                          eda02fe96a41c60fba6a885072837d24e51a83897eb5ca4ead24a5a248e840b7
Lora/suisai01.safetensors                                   f32045c2f8c824f783aebb86206e8dd004038ea9fef7b18b9f5aeff8c0b89d21
Lora/Fixhands_anime_bdsqlsz_V1.safetensors                  7fad91117c8205b11b7e7d37b2820ffc68ff526caabee546d54906907e373ed3
StableDiffusion/animagine-xl-3.1.safetensors                e3c47aedb06418c6c331443cd89f2b3b3b34b7ed2102a3d4c4408a8d35aad6b0
Tagger/config.json                                          ddcdd28facc40ee8d0ef4b16ee3e7c70e4d7b156aff7b0f2ccc180e617eda795
Tagger/model.onnx                                           e6774bff34d43bd49f75a47db4ef217dce701c9847b546523eb85ff6dbba1db1
Tagger/selected_tags.csv                                    298633d94d0031d2081c0893f29c82eab7f0df00b08483ba8f29d1e979441217
Tagger/sw_jax_cv_config.json                                4dda7ac5591de07f7444ca30f2f89971a21769f1db6279f92ca996d371b761c9
'

# get_hash <relative_path>
# Echoes the expected SHA-256 for <relative_path>, returns 1 if not found.
get_hash() {
    awk -v key="$1" '$1 == key { print $2; found=1; exit } END { exit !found }' <<< "$HASH_TABLE"
}

# --- helpers ---------------------------------------------------------------

# verify_hash <abs_path> <expected_sha256>
# Returns 0 if file present AND hash matches, 1 otherwise (silent on miss).
verify_hash() {
    local path="$1" expected="$2"
    [[ -f "$path" ]] || return 1
    local actual
    actual="$(shasum -a 256 "$path" | awk '{print $1}')"
    [[ "$actual" == "$expected" ]]
}

# download_one <relative_path> <url>
# Downloads if file missing or hash mismatch. 3-retry. Looks up the
# expected hash from HASH_TABLE by relative_path.
download_one() {
    local rel="$1" url="$2"
    local abs="$AI_ASSISTANT_MODELS_DIR/$rel"
    local expected
    if ! expected="$(get_hash "$rel")"; then
        echo "  ✗ $rel — no expected hash; refusing to download." >&2
        return 1
    fi

    if verify_hash "$abs" "$expected"; then
        echo "  ✓ $rel (cached)"
        return 0
    fi

    mkdir -p "$(dirname "$abs")"

    local attempt
    for attempt in 1 2 3; do
        echo "  ↓ $rel (attempt $attempt/3)"
        if curl --fail --location --show-error --silent \
                --connect-timeout 20 --retry 2 --retry-delay 5 \
                --output "$abs" "$url"; then
            if verify_hash "$abs" "$expected"; then
                echo "  ✓ $rel"
                return 0
            else
                echo "  ! $rel — hash mismatch, retrying ..." >&2
                rm -f "$abs"
            fi
        else
            echo "  ! $rel — curl failed, retrying ..." >&2
            rm -f "$abs"
        fi
    done

    echo "  ✗ $rel — gave up after 3 attempts." >&2
    return 1
}

# download_hf <subdir> <repo_id> <files-comma-separated> [<repo_subpath>]
# HuggingFace download — both default-path and custom-path variants.
# Mirrors :download_files_default and :download_files_custom in the .cmd.
download_hf() {
    local subdir="$1" repo_id="$2" files="$3" repo_subpath="${4:-}"
    local file url
    local arr=()
    IFS=',' read -ra arr <<< "$files"
    for file in "${arr[@]}"; do
        if [[ -n "$repo_subpath" ]]; then
            url="https://huggingface.co/${repo_id}/resolve/main/${repo_subpath}/${file}"
        else
            url="https://huggingface.co/${repo_id}/resolve/main/${file}"
        fi
        download_one "${subdir}/${file}" "$url" || return 1
    done
}

# --- main flow (mirrors AI_Assistant_model_DL.cmd :main, lines 176-275) ---

ANY_FAIL=0

run_step() {
    if ! "$@"; then
        ANY_FAIL=1
    fi
}

echo "Downloading Tagger model:"
run_step download_hf "Tagger" "SmilingWolf/wd-swinv2-tagger-v3" \
    "config.json,model.onnx,selected_tags.csv,sw_jax_cv_config.json"

echo
echo "Downloading Lora models:"
run_step download_hf "Lora" "tori29umai/lineart" \
    "sdxl_BW_bold_Line.safetensors,sdxl_BWLine.safetensors"
run_step download_hf "Lora" "tori29umai/SDXL_shadow" \
    "sdxl-testlora-normalmap_04b_dim32.safetensors,anime01.safetensors,anime02.safetensors,anime03.safetensors"
run_step download_hf "Lora" "tori29umai/flat_color" \
    "suisai01.safetensors,atunuri02.safetensors,animenuri.safetensors,SDXL_baketu2.safetensors"
run_step download_hf "Lora" "bdsqlsz/stable-diffusion-xl-anime-V5" \
    "Fixhands_anime_bdsqlsz_V1.safetensors"
run_step download_hf "Lora" "2vXpSwA7/iroiro-lora" \
    "sdxl-testlora-plainmaterial.safetensors" "test3"

echo
echo "Downloading ControlNet models:"
run_step download_hf "ControlNet" "2vXpSwA7/iroiro-lora" \
    "CN-anytest_v3-50000_am_dim256.safetensors,CN-anytest_v4-marged_am_dim256.safetensors" \
    "test_controlnet2"
run_step download_hf "ControlNet" "stabilityai/control-lora" \
    "control-lora-canny-rank256.safetensors" "control-LoRAs-rank256"
run_step download_hf "ControlNet" "kataragi/ControlNet-LineartXL" \
    "Kataragi_lineartXL-lora128.safetensors"
run_step download_hf "ControlNet" "tori29umai/CN_pose3D_V7" \
    "CL_am31_pose3D_V7_marged_rank256.safetensors"

echo
echo "Downloading Stable-Diffusion model:"
run_step download_hf "StableDiffusion" "cagliostrolab/animagine-xl-3.1" \
    "animagine-xl-3.1.safetensors"

echo
echo "Downloading from Civitai:"
# controlnet852AClone_v10 — Civitai download. The redirect pattern is
# different from HF (Civitai sometimes adds a token requirement). If this
# fails the user can download manually from
#   https://civitai.com/models/443821?modelVersionId=515749
# and place at $AI_ASSISTANT_MODELS_DIR/ControlNet/controlnet852AClone_v10.safetensors;
# the next run will verify and accept it.
CIVITAI_URL='https://civitai.com/api/download/models/515749?type=Model&format=SafeTensor'
run_step download_one \
    "ControlNet/controlnet852AClone_v10.safetensors" "$CIVITAI_URL"

# --- summary ---------------------------------------------------------------

echo

if [[ $ANY_FAIL -eq 0 ]]; then
    echo "All 23 files present and hash-verified."
    echo "Models tree: $AI_ASSISTANT_MODELS_DIR"
    echo "Next: ./mac/start.sh"
else
    cat <<EOF >&2

Some downloads failed. Re-run this script to retry only the missing files
(idempotent), or download manually and place at the printed paths. The
script will accept any file whose SHA-256 matches the expected hash.

If Civitai blocked controlnet852AClone_v10.safetensors (login required,
rate limited, etc.), grab it from:
  https://civitai.com/models/443821?modelVersionId=515749
and place it at:
  $AI_ASSISTANT_MODELS_DIR/ControlNet/controlnet852AClone_v10.safetensors
EOF
    exit 1
fi
