#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_RUN_DIR="${PROJECT_ROOT}/output_nht_max_private"
previous=""
for argument in "$@"; do
  if [[ "${argument}" == "--smoke-test" ]]; then
    DEFAULT_RUN_DIR="${PROJECT_ROOT}/output_nht_smoke"
  elif [[ "${previous}" == "--output-dir" ]]; then
    DEFAULT_RUN_DIR="${argument}"
  elif [[ "${argument}" == --output-dir=* ]]; then
    DEFAULT_RUN_DIR="${argument#--output-dir=}"
  fi
  previous="${argument}"
done
if [[ "${DEFAULT_RUN_DIR}" != /* ]]; then
  DEFAULT_RUN_DIR="${PROJECT_ROOT}/${DEFAULT_RUN_DIR}"
fi
RUN_DIR="${VTRACE_NHT_RUN_DIR:-${DEFAULT_RUN_DIR}}"
PID_FILE="${RUN_DIR}/orchestrator.pid"
LAUNCH_LOG="${RUN_DIR}/launcher.log"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

mkdir -p "${RUN_DIR}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}; run 'uv sync' first." >&2
  exit 2
fi

if [[ -f "${PID_FILE}" ]]; then
  EXISTING_PID="$(tr -d '[:space:]' < "${PID_FILE}")"
  if [[ -n "${EXISTING_PID}" ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
    echo "NHT orchestrator is already running with PID ${EXISTING_PID}."
    exit 0
  fi
fi

cd "${PROJECT_ROOT}"
nohup setsid "${PYTHON}" scripts/run_nht_max.py "$@" >> "${LAUNCH_LOG}" 2>&1 < /dev/null &
PID="$!"
printf '%s\n' "${PID}" > "${PID_FILE}"
sleep 2

if ! kill -0 "${PID}" 2>/dev/null; then
  echo "Orchestrator exited during startup. Inspect ${LAUNCH_LOG}." >&2
  exit 3
fi

echo "Started max-quality NHT pipeline (PID ${PID})."
echo "Monitor: tail -f ${LAUNCH_LOG}"
echo "A provider shutdown still requires relaunching this same command; checkpoints resume automatically."
