#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASK="Isaac-PickPlace-RedBlock-G129-Inspire-Joint" \
HAND_DDS="inspire" \
ROBOT_TYPE="g129" \
exec "$SCRIPT_DIR/run_pickplace_isolated.sh" "$@"
