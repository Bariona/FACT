#!/usr/bin/env bash
# End-to-end setup: conda env, pinned deps, Wan2.2 checkpoint, RoboTwin dataset.
#
#   bash setup_env.sh
#
# Overrides: ENV_NAME, CONDA_PREFIX_DIR, WAN_MODEL_DIR, HF_TOKEN (private repos),
# SKIP_MODEL_DOWNLOAD=1, SKIP_DATA_DOWNLOAD=1, SKIP_DEEPSPEED_INSTALL=1 (plain DDP).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-fact}"
CONDA_PREFIX_DIR="${CONDA_PREFIX_DIR:-$HOME/miniconda3}"
WAN_MODEL_DIR="${WAN_MODEL_DIR:-${ROOT_DIR}/models/Wan2.2-TI2V-5B-Diffusers}"
DATASETS_DIR="${ROOT_DIR}/datasets"

log() { echo "[setup] $*"; }

# ── Conda env ─────────────────────────────────────────────────────────────────
if [ ! -d "${CONDA_PREFIX_DIR}" ]; then
  log "Installing Miniconda into ${CONDA_PREFIX_DIR} ..."
  curl -L -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p "${CONDA_PREFIX_DIR}"
fi
# shellcheck disable=SC1091
source "${CONDA_PREFIX_DIR}/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  log "Creating conda env ${ENV_NAME} ..."
  conda create -y -n "${ENV_NAME}" python=3.11
fi
conda activate "${ENV_NAME}"

# ── Pinned dependencies ───────────────────────────────────────────────────────
cd "${ROOT_DIR}"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0
# --no-deps so pip cannot move the Torch stack; requirements.txt pins what it needs.
python -m pip install --no-deps lerobot==0.3.2
python -m pip install -r requirements.txt
if [ "${SKIP_DEEPSPEED_INSTALL:-0}" != "1" ]; then
  python -m pip install deepspeed==0.18.9
fi
python -m pip install --no-deps -e "${ROOT_DIR}"

if [ -n "${HF_TOKEN:-}" ]; then
  huggingface-cli login --token "${HF_TOKEN}"
fi

# ── Wan2.2 checkpoint ─────────────────────────────────────────────────────────
if [ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
  if [ -f "${WAN_MODEL_DIR}/model_index.json" ]; then
    log "Wan model already present at ${WAN_MODEL_DIR}; skipping download."
  else
    log "Downloading Wan-AI/Wan2.2-TI2V-5B-Diffusers -> ${WAN_MODEL_DIR}"
    HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
      Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir "${WAN_MODEL_DIR}"
  fi
fi

# ── RoboTwin dataset ──────────────────────────────────────────────────────────
if [ "${SKIP_DATA_DOWNLOAD:-0}" != "1" ]; then
  if [ -d "${DATASETS_DIR}/RoboTwin/Clean" ] || [ -d "${DATASETS_DIR}/RoboTwin/Randomized" ]; then
    log "RoboTwin dataset already present at ${DATASETS_DIR}/RoboTwin; skipping download."
  else
    log "Downloading Bariona/robotwin-v2 -> ${DATASETS_DIR}"
    HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
      Bariona/robotwin-v2 robotwin-v2.tar --repo-type dataset --local-dir "${DATASETS_DIR}"
    log "Extracting robotwin-v2.tar ..."
    tar -xf "${DATASETS_DIR}/robotwin-v2.tar" -C "${DATASETS_DIR}"
  fi
fi

log "Done. Next: conda activate ${ENV_NAME}, then follow README.md (preprocess -> train -> evaluate)."
