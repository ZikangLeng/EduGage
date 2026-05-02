#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

declare -a PYTHON_CMD=()
if command -v python.exe >/dev/null 2>&1; then
  PYTHON_CMD=("python.exe")
elif command -v py.exe >/dev/null 2>&1; then
  PYTHON_CMD=("py.exe" "-3")
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=("python")
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=("python3")
else
  echo "Could not find python.exe, py.exe, python, or python3 in PATH." >&2
  exit 1
fi

to_native_path() {
  local input_path="$1"
  if [[ "${PYTHON_CMD[0]}" == *.exe ]]; then
    if command -v wslpath >/dev/null 2>&1; then
      wslpath -w "$input_path"
      return
    fi
    if command -v cygpath >/dev/null 2>&1; then
      cygpath -w "$input_path"
      return
    fi
  fi
  printf '%s\n' "$input_path"
}

PARTICIPANT_ID=""
WINDOWS_PER_LABEL=""
PROVIDER=""
MODEL=""
EXAMPLES_PER_CLASS=""
WINDOW_SIZE_SEC=""
WINDOW_GENERATION_MODE=""
STRIDE_SEC=""
SEED="42"
SAMPLE_MODE="random"
TAG=""
LOG_LEVEL="INFO"

usage() {
  cat <<EOF
Usage:
  bash scripts/run_user_label_sweep.sh --participant-id <id> --windows-per-label <n> [options]

Options:
  --participant-id <id>         Participant ID to evaluate.
  --windows-per-label <n>       Number of windows to sample for each label.
  --provider <name>             Provider override, e.g. openai or heuristic.
  --model <name>                Model override, e.g. openai:local-qwen.
  --examples-per-class <n>      In-context examples per class.
  --window-size-sec <sec>       Window size override.
  --window-generation-mode <m>  interval_sliding or event_trailing.
  --stride-sec <sec>            Stride override.
  --seed <n>                    Sampling seed. Default: 42.
  --sample-mode <mode>          random or first. Default: random.
  --tag <text>                  Optional tag added to each run id.
  --log-level <level>           DEBUG, INFO, WARNING, ERROR. Default: INFO.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --participant-id)
      PARTICIPANT_ID="$2"
      shift 2
      ;;
    --windows-per-label)
      WINDOWS_PER_LABEL="$2"
      shift 2
      ;;
    --provider)
      PROVIDER="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --examples-per-class)
      EXAMPLES_PER_CLASS="$2"
      shift 2
      ;;
    --window-size-sec)
      WINDOW_SIZE_SEC="$2"
      shift 2
      ;;
    --window-generation-mode)
      WINDOW_GENERATION_MODE="$2"
      shift 2
      ;;
    --stride-sec)
      STRIDE_SEC="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --sample-mode)
      SAMPLE_MODE="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --log-level)
      LOG_LEVEL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$PARTICIPANT_ID" || -z "$WINDOWS_PER_LABEL" ]]; then
  usage
  exit 1
fi

RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)_p${PARTICIPANT_ID}_n${WINDOWS_PER_LABEL}"
if [[ -n "$TAG" ]]; then
  RUN_TAG="${RUN_TAG}_${TAG}"
fi
LOG_DIR="${ROOT_DIR}/artifacts/prompt_label_sweeps/${RUN_TAG}"
mkdir -p "$LOG_DIR"

declare -a PIDS=()

for LABEL in 1 2 3 4 5; do
  SCRIPT_PATH="$(to_native_path "${ROOT_DIR}/scripts/run_consensus_label_${LABEL}.py")"
  CMD=(
    "${PYTHON_CMD[@]}"
    "$SCRIPT_PATH"
    --participant-id "$PARTICIPANT_ID"
    --windows-per-label "$WINDOWS_PER_LABEL"
    --seed "$SEED"
    --sample-mode "$SAMPLE_MODE"
    --log-level "$LOG_LEVEL"
  )

  if [[ -n "$PROVIDER" ]]; then
    CMD+=(--provider "$PROVIDER")
  fi
  if [[ -n "$MODEL" ]]; then
    CMD+=(--model "$MODEL")
  fi
  if [[ -n "$EXAMPLES_PER_CLASS" ]]; then
    CMD+=(--examples-per-class "$EXAMPLES_PER_CLASS")
  fi
  if [[ -n "$WINDOW_SIZE_SEC" ]]; then
    CMD+=(--window-size-sec "$WINDOW_SIZE_SEC")
  fi
  if [[ -n "$WINDOW_GENERATION_MODE" ]]; then
    CMD+=(--window-generation-mode "$WINDOW_GENERATION_MODE")
  fi
  if [[ -n "$STRIDE_SEC" ]]; then
    CMD+=(--stride-sec "$STRIDE_SEC")
  fi
  if [[ -n "$TAG" ]]; then
    CMD+=(--tag "$TAG")
  fi

  LOG_PATH="${LOG_DIR}/label_${LABEL}.log"
  echo "Launching label ${LABEL}. Log: ${LOG_PATH}"
  "${CMD[@]}" >"$LOG_PATH" 2>&1 &
  PIDS+=("$!")
done

FAIL=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    FAIL=1
  fi
done

echo "Sweep complete. Logs written to: ${LOG_DIR}"
exit "$FAIL"
