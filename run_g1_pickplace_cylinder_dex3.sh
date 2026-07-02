#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASK="Isaac-PickPlace-Cylinder-G129-Dex3-Joint" \
HAND_DDS="dex3" \
ROBOT_TYPE="g129" \
exec "$SCRIPT_DIR/run_pickplace_isolated.sh" "$@"
