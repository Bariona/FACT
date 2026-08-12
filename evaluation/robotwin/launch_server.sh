#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
LAUNCH_CONFIG_PATH=${ROBOTWIN_LAUNCH_CONFIG:-${SCRIPT_DIR}/launch_config.yml}

export SCRIPT_DIR REPO_ROOT

# shellcheck source=evaluation/robotwin/common.sh
source "${SCRIPT_DIR}/common.sh"

load_launch_config server

FACT_CONDA_ENV=${FACT_CONDA_ENV:-}
SERVER_PYTHON=${SERVER_PYTHON:-}
SERVER_HOST=${SERVER_HOST:-127.0.0.1}
PORT=${PORT:-8093}
MODEL_ID=${MODEL_ID:-${REPO_ROOT}/models/Wan2.2-TI2V-5B-Diffusers}
TRANSFORMER_PATH=${TRANSFORMER_PATH:-${REPO_ROOT}/models/fact-wam/transformer}
STATS_PATH=${STATS_PATH:-${REPO_ROOT}/models/fact-wam/norm_stats_delta.json}
T5_EMBEDDING_PKL=${T5_EMBEDDING_PKL:-}
DEVICE=${DEVICE:-cuda:0}
TEXT_ENCODER_DEVICE=${TEXT_ENCODER_DEVICE:-cpu}
DTYPE=${DTYPE:-bf16}
TEXT_LEN=${TEXT_LEN:-512}
T5_LEN=${T5_LEN:-64}
ACTION_CHUNK=${ACTION_CHUNK:-48}
NUM_FRAMES=${NUM_FRAMES:-5}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-30}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-0.0}
RETURN_IMAGES=${RETURN_IMAGES:-0}
ENABLE_PREFIX_CACHE=${ENABLE_PREFIX_CACHE:-0}
SKIP_FUTURE_STATE_VALUE=${SKIP_FUTURE_STATE_VALUE:-0}

if [[ -z "${SERVER_PYTHON}" ]]; then
  SERVER_PYTHON=$(resolve_env_python "${FACT_CONDA_ENV}")
fi

for path in "${MODEL_ID}" "${TRANSFORMER_PATH}" "${STATS_PATH}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Error: '${path}' does not exist. Set MODEL_ID / TRANSFORMER_PATH / STATS_PATH" >&2
    echo "       in ${LAUNCH_CONFIG_PATH} or as environment variables." >&2
    exit 1
  fi
done

cmd=(
  "${SERVER_PYTHON}" -m scripts.inference_server
  --host "${SERVER_HOST}"
  --port "${PORT}"
  --model_id "${MODEL_ID}"
  --transformer_path "${TRANSFORMER_PATH}"
  --stats_path "${STATS_PATH}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --text_len "${TEXT_LEN}"
  --t5_len "${T5_LEN}"
  --text_encoder_device "${TEXT_ENCODER_DEVICE}"
  --action_chunk "${ACTION_CHUNK}"
  --num_frames "${NUM_FRAMES}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --guidance_scale "${GUIDANCE_SCALE}"
)

if [[ -n "${T5_EMBEDDING_PKL}" ]]; then
  cmd+=(--t5_embedding_pkl "${T5_EMBEDDING_PKL}")
fi

if [[ "${RETURN_IMAGES}" == "1" ]]; then
  cmd+=(--return_images)
fi

if [[ "${ENABLE_PREFIX_CACHE}" == "1" ]]; then
  cmd+=(--enable_prefix_cache)
fi

if [[ "${SKIP_FUTURE_STATE_VALUE}" == "1" ]]; then
  cmd+=(--skip_future_state_value)
fi

cd "${REPO_ROOT}"
echo "Launching FACT inference server on ${SERVER_HOST}:${PORT}"
echo "Model: ${MODEL_ID}"
echo "Python: ${SERVER_PYTHON}"
echo "Launch config: ${LAUNCH_CONFIG_PATH}"
"${cmd[@]}"
