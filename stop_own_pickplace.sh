#!/usr/bin/env bash
set -eo pipefail

USER_NAME="${USER:-$(id -un)}"
SELF_PID="$$"
PARENT_PID="$PPID"

mapfile -t PIDS < <(
  ps -u "$USER_NAME" -o pid= -o args= | awk -v self="$SELF_PID" -v parent="$PARENT_PID" '
    $1 == self || $1 == parent { next }
    /sim_main\.py/ && /Isaac-PickPlace-Cylinder-G129-Dex1-Joint/ { print $1; next }
    /run_pickplace_visual_debug\.py/ { print $1; next }
    /isaaclab\.sh -p sim_main\.py/ && /Isaac-PickPlace-Cylinder-G129-Dex1-Joint/ { print $1; next }
    /isaaclab\.sh -p run_pickplace_visual_debug\.py/ { print $1; next }
  '
)

if ((${#PIDS[@]} == 0)); then
  echo "[stop_own_pickplace] no previous $USER_NAME pick/place run found"
  exit 0
fi

echo "[stop_own_pickplace] stopping previous $USER_NAME pick/place run: ${PIDS[*]}"
kill "${PIDS[@]}" 2>/dev/null || true

for _ in {1..20}; do
  STILL_RUNNING=()
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      STILL_RUNNING+=("$pid")
    fi
  done
  if ((${#STILL_RUNNING[@]} == 0)); then
    echo "[stop_own_pickplace] stopped"
    exit 0
  fi
  sleep 0.5
done

echo "[stop_own_pickplace] force stopping: ${STILL_RUNNING[*]}"
kill -9 "${STILL_RUNNING[@]}" 2>/dev/null || true

for _ in {1..10}; do
  REMAINING=()
  for pid in "${STILL_RUNNING[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      REMAINING+=("$pid")
    fi
  done
  if ((${#REMAINING[@]} == 0)); then
    echo "[stop_own_pickplace] stopped"
    exit 0
  fi
  sleep 0.2
done

echo "[stop_own_pickplace] warning: still running after SIGKILL: ${REMAINING[*]}"
