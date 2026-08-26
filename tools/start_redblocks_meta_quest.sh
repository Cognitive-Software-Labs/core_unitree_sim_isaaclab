#!/usr/bin/env bash

set -Eeuo pipefail

# All paths below are derived from this script's location or from
# environment variables, rather than hardcoded to one contributor's home
# directory, so a fresh clone works on any machine/account.
readonly SIM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

# xr_teleoperate is a separate repository (see
# infodocs/hospital_teleoperation_integration.md for the companion branch).
# There is no universal default location for it, so it must be supplied.
readonly XR_REPO="${XR_TELEOPERATE_DIR:-}"

# Conda env names. Override if your local setup uses different names.
readonly SIM_CONDA_ENV="${SIM_CONDA_ENV:-unitree_sim_env}"
readonly XR_CONDA_ENV="${XR_CONDA_ENV:-tv}"

readonly DEFAULT_TASK="Isaac-PickPlace-MedicineBottle-Hospital-G129-Dex1-Joint"
readonly ROOM_USDA="$SIM_REPO/isaac-projects/new_base_room.usda"
# The in-repo scene carries the authored MedicalObjects scope, so this no
# longer needs to point outside the repository. Override MEDICAL_OBJECTS_USDA
# if you maintain a separate authored copy.
readonly MEDICAL_OBJECTS_USDA="${MEDICAL_OBJECTS_USDA:-$ROOM_USDA}"
readonly TABLE_OBJECT_SELECTOR="$SIM_REPO/tools/select_table_objects.py"
readonly VUER_CERT="${VUER_CERT:-$HOME/.config/xr_teleoperate/cert.pem}"
readonly VUER_KEY="${VUER_KEY:-$HOME/.config/xr_teleoperate/key.pem}"

usage() {
    cat <<'EOF'
Start the Isaac Lab hospital medicine-bottle task and Meta Quest bridge.

Usage:
  ./start_redblocks_meta_quest.sh [options]

Required environment:
  XR_TELEOPERATE_DIR    Path to xr_teleoperate's teleop/ directory (the
                         companion repo; see infodocs/hospital_teleoperation_integration.md)

Optional environment:
  SIM_CONDA_ENV          Conda env for the simulator (default: unitree_sim_env)
  XR_CONDA_ENV            Conda env for the Quest bridge (default: tv)
  CYCLONEDDS_HOME         If set, exported and prepended to LD_LIBRARY_PATH
                          for the simulator process (needed if your
                          cyclonedds Python package was built against a
                          local CycloneDDS install rather than a bundled one)
  MEDICAL_OBJECTS_USDA    Override the authored MedicalObjects scene file
  VUER_CERT / VUER_KEY    Override the Vuer TLS certificate/key paths

Options:
  --ip ADDRESS          Simulator/XR host LAN address (default: auto-detected
                         from the default route)
  --interface NAME      LAN interface used by CycloneDDS (default: the
                         default route's interface)
  --input-mode MODE     controller or hand (default: controller)
  --device DEVICE       Isaac Lab device (default: cuda:0)
  --sim-gui             Show the Isaac Sim window (headless is the default)
  -h, --help            Show this help

The Quest browser URL is printed after both terminals are opened.
Before launch, a GUI reads every asset in the authored MedicalObjects scope
and lets you exclude it or assign it an Important or Distractor role.
EOF
}

hold_after_command() {
    local status="$1"
    printf '\nProcess exited with status %s. Press Enter to close this terminal.\n' "$status"
    read -r _ || true
    return "$status"
}

# Find a Conda setup script without assuming a specific account's install
# location. CONDA_SETUP_SCRIPT overrides everything.
find_conda_setup() {
    if [[ -n "${CONDA_SETUP_SCRIPT:-}" ]]; then
        printf '%s' "$CONDA_SETUP_SCRIPT"
        return 0
    fi
    local candidate
    for candidate in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3"; do
        if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then
            printf '%s' "$candidate/etc/profile.d/conda.sh"
            return 0
        fi
    done
    if command -v conda >/dev/null 2>&1; then
        local base
        base="$(conda info --base 2>/dev/null)"
        if [[ -n "$base" && -f "$base/etc/profile.d/conda.sh" ]]; then
            printf '%s' "$base/etc/profile.d/conda.sh"
            return 0
        fi
    fi
    return 1
}

# Activate a named environment, trying Conda first and falling back to
# Micromamba. Some machines only install a given env under one of the two
# (e.g. an Isaac Lab env created by auto_setup_env.sh may only exist under
# Micromamba even when other envs on the same box are Conda envs).
activate_env() {
    local environment="$1"

    # A terminal launched from an active Isaac environment inherits its
    # PYTHONPATH. That path makes the base Python load Isaac's Python 3.11
    # packages and produces misleading pydantic_core/plugin errors. Isaac's
    # activation hook also reads optional variables without ${var:-} guards,
    # so nounset must be disabled while the hook runs.
    set +u
    unset PYTHONPATH

    local conda_setup
    if conda_setup="$(find_conda_setup)" && [[ -f "$conda_setup" ]]; then
        source "$conda_setup"
        if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$environment"; then
            conda activate "$environment"
            set -u
            return 0
        fi
    fi

    if command -v micromamba >/dev/null 2>&1; then
        eval "$(micromamba shell hook --shell bash 2>/dev/null)"
        if micromamba env list 2>/dev/null | awk '{print $1}' | grep -qx "$environment"; then
            micromamba activate "$environment"
            set -u
            return 0
        fi
    fi

    set -u
    printf 'Could not find environment "%s" under Conda or Micromamba.\n' "$environment" >&2
    printf 'Set SIM_CONDA_ENV / XR_CONDA_ENV to an env name that exists on this machine.\n' >&2
    exit 1
}

run_simulator() {
    local device="$1"
    local task="$2"
    local sim_gui="$3"
    local interface="$4"
    local object_roles="$5"
    # sim_conda_env and cyclonedds_home are passed explicitly rather than
    # read from this process's own environment: launch_terminal spawns this
    # via a terminal emulator's --execute, and at least xfce4-terminal runs
    # that command against its own (pre-existing) server process's
    # environment, not the environment of the shell that called
    # launch_terminal. Anything the child needs must travel as an argument.
    local sim_conda_env="$6"
    local cyclonedds_home="$7"
    local -a render_args=(--headless)

    if [[ "$sim_gui" == "true" ]]; then
        render_args=()
    fi

    activate_env "$sim_conda_env"
    cd "$SIM_REPO"
    export UNITREE_DDS_NETWORK_INTERFACE="$interface"
    export HOSPITAL_OBJECT_ROLES="$object_roles"
    export OMNI_KIT_ACCEPT_EULA=YES
    if [[ -n "$cyclonedds_home" ]]; then
        export LD_LIBRARY_PATH="$cyclonedds_home/lib:${LD_LIBRARY_PATH:-}"
    fi

    printf 'Starting Meta Quest medicine-bottle simulator...\n'
    printf 'Task: %s\nDevice: %s\nDDS interface: %s\n\n' "$task" "$device" "$interface"
    printf 'Medical-object roles: %s\n\n' "$object_roles"

    set +e
    python sim_main.py \
        --device "$device" \
        "${render_args[@]}" \
        --meta_quest \
        --task "$task"
    local status=$?
    set -e
    hold_after_command "$status"
}

run_xr_bridge() {
    local sim_ip="$1"
    local interface="$2"
    local input_mode="$3"
    # See the comment in run_simulator: these travel as arguments because
    # the spawned terminal does not reliably inherit this process's
    # environment.
    local xr_repo="$4"
    local xr_conda_env="$5"

    printf 'Waiting for the simulator image server on port 60000...\n'
    printf 'You can cancel this wait with Ctrl+C.\n\n'
    until ss -H -ltn 'sport = :60000' | grep -q .; do
        sleep 2
    done

    printf 'Simulator image server is ready. Waiting for rt/lowstate on %s...\n' "$interface"
    activate_env "$xr_conda_env"
    cd "$xr_repo"

    # Port 60000 becomes available before the simulator starts its controller
    # and DDS publishing threads. Probe the real arm-state topic so xr_teleoperate
    # cannot lose its own five-second startup race.
    until DDS_PROBE_INTERFACE="$interface" python -u -c '
import os
import threading

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

ChannelFactoryInitialize(1, networkInterface=os.environ["DDS_PROBE_INTERFACE"])
received = threading.Event()
subscriber = ChannelSubscriber("rt/lowstate", LowState_)
subscriber.Init(lambda _: received.set(), 1)
os._exit(0 if received.wait(timeout=3.0) else 1)
'; do
        printf 'Still waiting for simulator DDS...\n'
        sleep 2
    done

    printf 'Simulator DDS is ready. Starting the Quest bridge...\n\n'

    set +e
    python teleop_hand_and_arm.py \
        --input-mode="$input_mode" \
        --arm=G1_29 \
        --ee=dex1 \
        --sim \
        --img-server-ip="$sim_ip" \
        --network-interface="$interface"
    local status=$?
    set -e
    hold_after_command "$status"
}

launch_terminal() {
    local title="$1"
    shift

    if command -v xfce4-terminal >/dev/null 2>&1; then
        xfce4-terminal --title="$title" --hold --execute "$0" "$@" &
    elif command -v gnome-terminal >/dev/null 2>&1; then
        gnome-terminal --title="$title" -- "$0" "$@" &
    elif command -v xterm >/dev/null 2>&1; then
        xterm -T "$title" -hold -e "$0" "$@" &
    else
        printf 'Error: no supported graphical terminal was found.\n' >&2
        exit 1
    fi
}

validate_installation() {
    local interface="$1"
    local sim_ip="$2"

    [[ -f "$SIM_REPO/sim_main.py" ]] || { printf 'Missing simulator: %s\n' "$SIM_REPO" >&2; exit 1; }
    [[ -f "$ROOM_USDA" ]] || { printf 'Missing room layer: %s\n' "$ROOM_USDA" >&2; exit 1; }
    [[ -f "$MEDICAL_OBJECTS_USDA" ]] || { printf 'Missing MedicalObjects catalog layer: %s\n' "$MEDICAL_OBJECTS_USDA" >&2; exit 1; }
    [[ -f "$TABLE_OBJECT_SELECTOR" ]] || { printf 'Missing table-object selector: %s\n' "$TABLE_OBJECT_SELECTOR" >&2; exit 1; }
    if [[ -z "$XR_REPO" ]]; then
        printf 'XR_TELEOPERATE_DIR is not set. Point it at xr_teleoperate/teleop\n' >&2
        printf '(see infodocs/hospital_teleoperation_integration.md for the companion branch).\n' >&2
        exit 1
    fi
    [[ -f "$XR_REPO/teleop_hand_and_arm.py" ]] || { printf 'Missing XR launcher: %s\n' "$XR_REPO" >&2; exit 1; }
    [[ -d "$SIM_REPO/assets" ]] || { printf 'Missing simulator assets. Run `. fetch_assets.sh` in %s.\n' "$SIM_REPO" >&2; exit 1; }
    [[ -f "$VUER_CERT" ]] || { printf 'Missing Vuer TLS certificate: %s\n' "$VUER_CERT" >&2; exit 1; }
    [[ -f "$VUER_KEY" ]] || { printf 'Missing Vuer TLS private key: %s\n' "$VUER_KEY" >&2; exit 1; }
    ip link show dev "$interface" >/dev/null 2>&1 || { printf 'Network interface does not exist: %s\n' "$interface" >&2; exit 1; }
    ip -4 -o addr show dev "$interface" | grep -Fq " $sim_ip/" || {
        printf 'Address %s is not assigned to interface %s.\n' "$sim_ip" "$interface" >&2
        printf 'Run `ip -br addr` and pass the matching values with --ip and --interface.\n' >&2
        exit 1
    }
}

if [[ "${1:-}" == "__simulator" ]]; then
    shift
    run_simulator "$@"
    exit $?
fi

if [[ "${1:-}" == "__xr_bridge" ]]; then
    shift
    run_xr_bridge "$@"
    exit $?
fi

# Auto-detect this host's LAN address/interface from the default route
# instead of hardcoding one contributor's machine. --ip/--interface override.
# The "|| true" guards keep a host with no default route from tripping
# errexit here; the empty-value check below reports that case clearly.
default_route_interface="$(ip route show default 2>/dev/null | awk '{print $5; exit}')" || true
sim_ip=""
if [[ -n "$default_route_interface" ]]; then
    sim_ip="$(ip -4 -o addr show dev "$default_route_interface" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)" || true
fi
interface="$default_route_interface"
input_mode="controller"
device="cuda:0"
task="$DEFAULT_TASK"
sim_gui="false"

while (($#)); do
    case "$1" in
        --ip)
            [[ $# -ge 2 ]] || { printf 'Option --ip requires a value.\n' >&2; exit 2; }
            sim_ip="$2"
            shift 2
            ;;
        --interface)
            [[ $# -ge 2 ]] || { printf 'Option --interface requires a value.\n' >&2; exit 2; }
            interface="$2"
            shift 2
            ;;
        --input-mode)
            [[ $# -ge 2 ]] || { printf 'Option --input-mode requires a value.\n' >&2; exit 2; }
            input_mode="$2"
            shift 2
            ;;
        --device)
            [[ $# -ge 2 ]] || { printf 'Option --device requires a value.\n' >&2; exit 2; }
            device="$2"
            shift 2
            ;;
        --sim-gui)
            sim_gui="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$interface" || -z "$sim_ip" ]]; then
    printf 'Could not auto-detect a LAN interface/address from the default route.\n' >&2
    printf 'Pass --ip and --interface explicitly (see `ip -br addr`).\n' >&2
    exit 2
fi

if [[ "$input_mode" != "controller" && "$input_mode" != "hand" ]]; then
    printf 'Input mode must be controller or hand.\n' >&2
    exit 2
fi

if [[ ! "$sim_ip" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]]; then
    printf 'Invalid IPv4 address: %s\n' "$sim_ip" >&2
    exit 2
fi

validate_installation "$interface" "$sim_ip"

object_roles="$(python3 "$TABLE_OBJECT_SELECTOR" --usd "$MEDICAL_OBJECTS_USDA")"

launch_terminal "Medicine Bottle - Isaac Simulator" __simulator "$device" "$task" "$sim_gui" "$interface" "$object_roles" "$SIM_CONDA_ENV" "${CYCLONEDDS_HOME:-}"
launch_terminal "Medicine Bottle - Meta Quest Bridge" __xr_bridge "$sim_ip" "$interface" "$input_mode" "$XR_REPO" "$XR_CONDA_ENV"

printf 'Started the simulator and Quest bridge in two visible terminals.\n\n'
printf 'On the Meta Quest, open:\n'
printf 'https://%s:8012/?ws=wss://%s:8012\n\n' "$sim_ip" "$sim_ip"
printf 'After Vuer enters Virtual Reality, press r in the Quest Bridge terminal.\n'
printf 'Left/right index trigger: hold to close/grasp; release to open/relax the matching Dex1 gripper.\n'
printf 'Right stick: left/right turns the torso; forward leans toward a crate; back returns upright.\n'
printf 'Quest X starts/stops G1 camera recording; A moves the Ridgeback to its next arc point; Y/B reset the scene and recenter the torso.\n'
printf 'Quest Y/B scene resets also return the torso to center/upright.\n'
printf 'Press q there to stop teleoperation, then Ctrl+C in the simulator terminal.\n'
