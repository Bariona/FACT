# Shared helpers for launch_server.sh / launch_client.sh.
# Expects SCRIPT_DIR and LAUNCH_CONFIG_PATH to be set by the caller.

# Resolve a conda env name or env path to its bin/python.
resolve_env_python() {
  local env_spec="$1"
  local conda_base
  local prefix=""

  if [[ -z "${env_spec}" ]]; then
    command -v python
    return 0
  fi

  if [[ -x "${env_spec}/bin/python" ]]; then
    printf '%s\n' "${env_spec}/bin/python"
    return 0
  fi

  conda_base=$(conda info --base)

  prefix=$(conda env list | awk -v name="${env_spec}" 'NR > 2 && $1 == name {print $NF; exit}')
  if [[ -n "${prefix}" && -x "${prefix}/bin/python" ]]; then
    printf '%s\n' "${prefix}/bin/python"
    return 0
  fi

  if [[ -x "${conda_base}/envs/${env_spec}/bin/python" ]]; then
    printf '%s\n' "${conda_base}/envs/${env_spec}/bin/python"
    return 0
  fi

  echo "Error: env '${env_spec}' was not found as a Conda env name or path with bin/python." >&2
  return 1
}

# Export the given launch_config.yml section as shell variables (explicit
# environment variables keep precedence). Fails loudly rather than skipping the
# config, which would run the whole evaluation with default settings.
load_launch_config() {
  if [[ ! -f "${LAUNCH_CONFIG_PATH}" ]]; then
    return 0
  fi

  local python_bin exports
  python_bin=$(resolve_env_python "${FACT_CONDA_ENV:-}" 2>/dev/null) || python_bin=""
  if [[ -z "${python_bin}" ]]; then
    echo "Error: no python interpreter found to read ${LAUNCH_CONFIG_PATH}." >&2
    echo "       Activate the fact env, or export FACT_CONDA_ENV." >&2
    exit 1
  fi

  # `if !` so `set -e` cannot kill the shell before the error prints.
  if ! exports=$("${python_bin}" "${SCRIPT_DIR}/resolve_launch_config.py" \
      --config "${LAUNCH_CONFIG_PATH}" --section "$1"); then
    echo "Error: could not read ${LAUNCH_CONFIG_PATH} using ${python_bin}." >&2
    echo "       It needs PyYAML; export FACT_CONDA_ENV to pick another interpreter." >&2
    exit 1
  fi

  eval "${exports}"
}
