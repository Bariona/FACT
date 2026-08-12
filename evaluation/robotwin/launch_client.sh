#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
LAUNCH_CONFIG_PATH=${ROBOTWIN_LAUNCH_CONFIG:-${SCRIPT_DIR}/launch_config.yml}

export SCRIPT_DIR REPO_ROOT

# shellcheck source=evaluation/robotwin/common.sh
source "${SCRIPT_DIR}/common.sh"

load_launch_config client

ROBOTWIN_PATH=${ROBOTWIN_PATH:-${HOME}/RoboTwin}
DEPLOY_POLICY_PATH=${DEPLOY_POLICY_PATH:-${REPO_ROOT}/evaluation/robotwin/deploy_policy.yml}
ROBOTWIN_CONDA_ENV=${ROBOTWIN_CONDA_ENV:-}
CLIENT_PYTHON=${CLIENT_PYTHON:-}
# Machine-specific, off by default; set in launch_config.yml if needed.
if [[ -n "${EXTRA_LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${EXTRA_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
fi
if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  export TORCH_CUDA_ARCH_LIST
fi

task_name=${1:-${TASK_NAME:-beat_block_hammer}}
task_config=${2:-${TASK_CONFIG:-demo_clean}}
ckpt_setting=${3:-}
seed=${4:-${SEED:-0}}
test_num=${TEST_NUM:-1}
port=${PORT:-8093}
policy_name=${POLICY_NAME:-evaluation.robotwin.model2robotwin_interface}
execute_actions_per_plan=${EXECUTE_ACTIONS_PER_PLAN:-}
server_timeout_ms=${SERVER_TIMEOUT_MS:-}
server_wait_seconds=${SERVER_WAIT_SECONDS:-}
trace_root=${TRACE_ROOT:-}
print_action_stats=${PRINT_ACTION_STATS:-}
enable_sample=${ENABLE_SAMPLE:-}
best_of_n=${BEST_OF_N:-}
enable_value_vis=${ENABLE_VALUE_VIS:-}
trace_value_only=${TRACE_VALUE_ONLY:-}
vis_dir=${VIS_DIR:-}
low_frequency_rgb=${LOW_FREQUENCY_RGB:-}
skip_action_render_sync=${SKIP_ACTION_RENDER_SYNC:-}
eval_video_log=${EVAL_VIDEO_LOG:-}

if [[ "${policy_name}" != *.* ]]; then
  policy_name="evaluation.robotwin.${policy_name}"
fi

export PYTHONPATH="${REPO_ROOT}:${ROBOTWIN_PATH}:${PYTHONPATH:-}"

if [[ ! -f "${ROBOTWIN_PATH}/script/eval_policy.py" ]]; then
  echo "Error: no RoboTwin checkout at '${ROBOTWIN_PATH}'. Set ROBOTWIN_PATH in" >&2
  echo "       ${LAUNCH_CONFIG_PATH} or as an environment variable." >&2
  exit 1
fi

if [[ -z "${CLIENT_PYTHON}" ]]; then
  CLIENT_PYTHON=$(resolve_env_python "${ROBOTWIN_CONDA_ENV}")
fi

CLIENT_BIN_DIR=$(cd "$(dirname "${CLIENT_PYTHON}")" && pwd)
export PATH="${CLIENT_BIN_DIR}:${PATH}"

cmd=(
  "${CLIENT_PYTHON}" script/eval_policy.py
  --config "${DEPLOY_POLICY_PATH}"
  --overrides
  --task_name "${task_name}"
  --task_config "${task_config}"
  --seed "${seed}"
  --policy_name "${policy_name}"
  --port "${port}"
  --test_num "${test_num}"
)

if [[ -n "${ckpt_setting}" ]]; then
  cmd+=(--ckpt_setting "${ckpt_setting}")
fi

if [[ -n "${execute_actions_per_plan}" ]]; then
  cmd+=(--execute_actions_per_plan "${execute_actions_per_plan}")
fi
if [[ -n "${server_timeout_ms}" ]]; then
  cmd+=(--server_timeout_ms "${server_timeout_ms}")
fi
if [[ -n "${server_wait_seconds}" ]]; then
  cmd+=(--server_wait_seconds "${server_wait_seconds}")
fi
if [[ -n "${trace_root}" ]]; then
  cmd+=(--trace_root "${trace_root}")
fi
if [[ -n "${print_action_stats}" ]]; then
  cmd+=(--print_action_stats "${print_action_stats}")
fi
if [[ -n "${enable_sample}" ]]; then
  cmd+=(--enable_sample "${enable_sample}")
fi
if [[ -n "${best_of_n}" ]]; then
  cmd+=(--best_of_n "${best_of_n}")
fi
if [[ -n "${enable_value_vis}" ]]; then
  cmd+=(--enable_value_vis "${enable_value_vis}")
fi
if [[ -n "${trace_value_only}" ]]; then
  cmd+=(--trace_value_only "${trace_value_only}")
fi
if [[ -n "${vis_dir}" ]]; then
  cmd+=(--vis_dir "${vis_dir}")
fi
if [[ -n "${low_frequency_rgb}" ]]; then
  cmd+=(--low_frequency_rgb "${low_frequency_rgb}")
fi
if [[ -n "${skip_action_render_sync}" ]]; then
  cmd+=(--skip_action_render_sync "${skip_action_render_sync}")
fi

cd "${ROBOTWIN_PATH}"
echo "Running RoboTwin eval_policy.py for task=${task_name}, test_num=${test_num}, port=${port}"
echo "Python: ${CLIENT_PYTHON}"
echo "Launch config: ${LAUNCH_CONFIG_PATH}"
if [[ -n "${eval_video_log}" ]]; then
  export FACT_ROBOTWIN_EVAL_VIDEO_LOG="${eval_video_log}"
fi
env PYTHONPATH="${PYTHONPATH}" "${cmd[@]}"
