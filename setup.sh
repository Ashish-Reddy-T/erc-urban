#!/usr/bin/env bash
# One-shot setup after cloning. Safe to re-run: finished steps are skipped.
#
#   conda create -n rover_env python=3.10 -y && conda activate rover_env
#   ./setup.sh
#
# 1. installs requirements.txt into the ACTIVE python (erc_indoor / erc_urban)
# 2. creates earth-rovers-sdk/.venv for the SDK server (its own numpy pin)
# 3. downloads the checkpoints git cannot hold (~510 MB, HuggingFace)
# 4. pre-caches DINOv2 + SegFormer so nothing downloads at the venue
#
# SKIP_PIP=1 skips 1-2, SKIP_MODELS=1 skips 4.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python}
ERC3=ERC-3-earthrover-challenge
say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# fetch <url> <dest> <sha256>
fetch() {
  local url=$1 dest=$2 want=$3
  if [[ -f $dest && $(sha256 "$dest") == "$want" ]]; then
    echo "ok       $dest"; return
  fi
  echo "download $dest"
  mkdir -p "$(dirname "$dest")"
  curl -fL --retry 3 -C - -o "$dest.part" "$url"
  if [[ $(sha256 "$dest.part") != "$want" ]]; then
    rm -f "$dest.part"; echo "checksum mismatch for $dest" >&2; exit 1
  fi
  mv "$dest.part" "$dest"
}

# --- 1. stack deps -----------------------------------------------------------
if [[ ${SKIP_PIP:-0} != 1 ]]; then
  say "python deps -> $($PY -c 'import sys; print(sys.executable)')"
  if ! $PY -c 'import sys; sys.exit(sys.version_info[:2] != (3, 10))'; then
    echo "warning: tested on Python 3.10, this is $($PY --version)" >&2
  fi
  if [[ -z ${CONDA_DEFAULT_ENV:-}${VIRTUAL_ENV:-} || ${CONDA_DEFAULT_ENV:-} == base ]]; then
    echo "warning: no env active; installing into $($PY -c 'import sys; print(sys.prefix)')" >&2
  fi
  $PY -m pip install -r requirements.txt

  # --- 2. SDK server venv ----------------------------------------------------
  say "SDK venv -> earth-rovers-sdk/.venv"
  [[ -d earth-rovers-sdk/.venv ]] || python3 -m venv earth-rovers-sdk/.venv
  earth-rovers-sdk/.venv/bin/pip install -q -r earth-rovers-sdk/requirements.txt
fi

# --- 3. checkpoints ----------------------------------------------------------
say "checkpoints"
HF=https://huggingface.co
fetch "$HF/NHirose/MBRA_project_models/resolve/main/logonav.pth" \
  "$ERC3/mbra_repo/deployment/model_weights/logonav.pth" \
  9b61a5488e581ab71ceccf33d9db1e2c7ab2e7f09a1376c81c5ce39928518435
fetch "$HF/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth" \
  "$ERC3/third_party/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_hypersim_vits.pth" \
  b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545
fetch "$HF/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth" \
  "$ERC3/third_party/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_vkitti_vits.pth" \
  9203e538d35255c90dda4b7fedb47ff33fe725497bcca3b1e53b3a65ee63f0cb

# --- 4. hub caches -----------------------------------------------------------
if [[ ${SKIP_MODELS:-0} != 1 ]]; then
  say "DINOv2 (torch hub) + SegFormer (HuggingFace) caches"
  $PY -c 'from erc_indoor.place import PlaceEncoder; PlaceEncoder("cpu")'
  $PY -c 'from erc_indoor.freespace import FreeSpaceEstimator; FreeSpaceEstimator("cpu")'
fi

# --- checks ------------------------------------------------------------------
say "checks"
[[ -f erc_indoor/data/route_map/meta.json ]] && echo "ok       indoor route map" \
  || echo "MISSING  erc_indoor/data/route_map (pull again?)" >&2
[[ -f earth-rovers-sdk/.env ]] && echo "ok       earth-rovers-sdk/.env" \
  || echo "MISSING  earth-rovers-sdk/.env (copy .env.sample)" >&2
grep -q '^CHROME_EXECUTABLE_PATH=' earth-rovers-sdk/.env 2>/dev/null && {
  chrome=$(sed -n 's/^CHROME_EXECUTABLE_PATH="\{0,1\}\([^"]*\)"\{0,1\}$/\1/p' earth-rovers-sdk/.env)
  [[ -x $chrome ]] && echo "ok       Chrome at $chrome" \
    || echo "CHECK    CHROME_EXECUTABLE_PATH in earth-rovers-sdk/.env -> $chrome not found" >&2
}

say "done. Next: see README.md 'Run'"
