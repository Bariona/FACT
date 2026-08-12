#!/usr/bin/env bash
# Sweep every RoboTwin task and collect per-task + average success rate.
# Requires a running FACT server (see launch_server.sh).
#
#   bash evaluation/robotwin/eval_all_tasks.sh [task_config] [test_num]
#
# Overrides: SWEEP_OUT (output dir), ROBOTWIN_PATH, TASK_LIST="a b" (subset).
# Re-running with the same SWEEP_OUT skips tasks already in results.csv.
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
LAUNCH_CONFIG_PATH=${ROBOTWIN_LAUNCH_CONFIG:-${SCRIPT_DIR}/launch_config.yml}
export SCRIPT_DIR REPO_ROOT

# shellcheck source=evaluation/robotwin/common.sh
source "${SCRIPT_DIR}/common.sh"
load_launch_config client

task_config=${1:-${TASK_CONFIG:-demo_clean}}
test_num=${2:-${TEST_NUM:-50}}
ROBOTWIN_PATH=${ROBOTWIN_PATH:-${HOME}/RoboTwin}
policy_name=${POLICY_NAME:-evaluation.robotwin.model2robotwin_interface}
ckpt_setting=fact   # must match `ckpt_setting` in deploy_policy.yml

# Canonical evaluable-task list: the 50 tasks RoboTwin defines a step limit for.
# Resolved before SWEEP_OUT so a misconfigured run leaves no empty eval_runs/.
if [[ -n "${TASK_LIST:-}" ]]; then
  read -r -a tasks <<< "${TASK_LIST}"
else
  step_limit_yml="${ROBOTWIN_PATH}/task_config/_eval_step_limit.yml"
  if [[ ! -f "${step_limit_yml}" ]]; then
    echo "Error: no task list at '${step_limit_yml}'. Set ROBOTWIN_PATH in" >&2
    echo "       ${LAUNCH_CONFIG_PATH} (or as an env var), or pass TASK_LIST=\"task_a task_b\"." >&2
    exit 1
  fi
  mapfile -t tasks < <(grep -oE '^[a-z0-9_]+:' "${step_limit_yml}" | tr -d ':')
fi

# An empty list would sweep nothing and still report success.
if [[ ${#tasks[@]} -eq 0 ]]; then
  echo "Error: resolved an empty task list; nothing to evaluate." >&2
  exit 1
fi

SWEEP_OUT=${SWEEP_OUT:-${REPO_ROOT}/eval_runs/${task_config}_$(date +%Y%m%d_%H%M%S)}
mkdir -p "${SWEEP_OUT}/logs"
CSV="${SWEEP_OUT}/results.csv"
[[ -f "${CSV}" ]] || echo "task,success,total,success_rate" > "${CSV}"

echo "Sweeping ${#tasks[@]} tasks x ${test_num} episodes on ${task_config}"
echo "Output: ${SWEEP_OUT}"

for task in "${tasks[@]}"; do
  if grep -q "^${task}," "${CSV}"; then
    echo "[skip] ${task} (already in results.csv)"
    continue
  fi

  echo "[run ] ${task}  ($(date +%H:%M:%S))"
  task_start=$(date +%s)
  TEST_NUM="${test_num}" TASK_NAME="${task}" TASK_CONFIG="${task_config}" \
    bash "${SCRIPT_DIR}/launch_client.sh" "${task}" "${task_config}" \
    > "${SWEEP_OUT}/logs/${task}.log" 2>&1
  client_rc=$?

  # eval_policy.py writes eval_result/<task>/<policy>/<config>/<ckpt_setting>/<timestamp>/_result.txt
  # Only accept a _result.txt written by THIS run (mtime >= task_start) with a
  # zero client exit code — otherwise a failed task would silently pick up the
  # newest result from a previous sweep.
  result_dir="${ROBOTWIN_PATH}/eval_result/${task}/${policy_name}/${task_config}/${ckpt_setting}"
  latest=$(ls -1dt "${result_dir}"/*/ 2>/dev/null | head -1)
  rate=""
  if [[ ${client_rc} -eq 0 && -n "${latest}" && -f "${latest}/_result.txt" ]] \
     && [[ $(stat -c %Y "${latest}/_result.txt") -ge ${task_start} ]]; then
    rate=$(tail -n1 "${latest}/_result.txt")
  fi

  if [[ -z "${rate}" ]]; then
    echo "[fail] ${task}: client rc=${client_rc}, no fresh _result.txt — see ${SWEEP_OUT}/logs/${task}.log"
    echo "${task},,,ERROR" >> "${CSV}"
    continue
  fi

  succ=$(awk -v r="${rate}" -v n="${test_num}" 'BEGIN{printf "%d", r*n+0.5}')
  echo "${task},${succ},${test_num},${rate}" >> "${CSV}"
  echo "[done] ${task}: ${succ}/${test_num} = ${rate}"
done

echo
echo "================ SUMMARY (${task_config}) ================"
awk -F, 'NR>1 && $4!="ERROR" {printf "%-28s %6.1f%%  (%s/%s)\n", $1, $4*100, $2, $3; s+=$4; n++}
         NR>1 && $4=="ERROR" {printf "%-28s %6s\n", $1, "ERROR"; e++}
         END {printf "\n%-28s %6.1f%%   over %d tasks\n", "AVERAGE", (n?s/n*100:0), n;
              if (e) printf "%d task(s) errored\n", e}' "${CSV}" | tee "${SWEEP_OUT}/summary.txt"
echo
echo "Per-task CSV: ${CSV}"

# Non-zero exit when any task errored, so CI/callers notice.
! grep -q ',ERROR$' "${CSV}"
