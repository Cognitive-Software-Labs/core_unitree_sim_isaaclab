# Unitree Hospital Simulation for Isaac Lab

GPU-accelerated Unitree G1 and H1-2 manipulation simulation built on NVIDIA
Isaac Sim and Isaac Lab. The current project includes fixed-base and Wholebody
tasks, DDS control bridges, camera streaming for `xr_teleoperate`, and the
randomized hospital scenarios.

The primary hospital task is
`Isaac-PickPlace-MedicineBottle-Hospital-G129-Dex1-Joint`: use the G1 + Dex1
hand to place two pill bottles in the Ridgeback-mounted rear container while
the hospital layout and tabletop clutter are randomized.

[English](README.md) | [Chinese](README_zh-CN.md)

![Hospital simulation view](img/mainview.png)

## Important safety note

This project starts DDS publishers and subscribers whenever `sim_main.py` is
run. Its topics are intentionally compatible with Unitree control tooling.
Keep a simulator-only machine on an isolated network. Before using it on a
network that also contains a physical robot, explicitly select the intended
network interface:

```bash
export UNITREE_DDS_NETWORK_INTERFACE=enp3s0
```

Replace `enp3s0` with the simulator network interface. Do not run live DDS
teleoperation until you have verified that this interface and the DDS domain
are isolated from hardware you do not intend to control.

## What you need

The supported pip installation path is for **Linux x86_64 with an NVIDIA GPU**.
Use Ubuntu 22.04 or newer: the Isaac Sim pip distribution requires GLIBC 2.35
or later. Isaac Sim 5.1 requires Python 3.11.

Before starting, make sure that `nvidia-smi`, `python3.11`, and
`python3.11 -m venv` work. Install a current NVIDIA driver appropriate for the
GPU before continuing. RTX rendering, Isaac Sim extensions, and the project
assets make the first installation large and the first launch can take several
minutes.

This guide uses a Python virtual environment and **pip for every Python
package**. It does not use Conda or Docker.

## Install from zero

Run these commands in a terminal on the simulator machine.

### 1. Install operating-system prerequisites

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake git git-lfs unzip \
  libegl1 libgl1 libglib2.0-0 libglu1-mesa libsm6 libxext6 \
  python3.11 python3.11-venv
```

If your Ubuntu release does not provide `python3.11`, install Python 3.11 using
your organization's approved Python package source, then re-run the last two
checks above. Do not substitute Python 3.10 or 3.12: the pinned Isaac Sim 5.1
packages require 3.11.

### 2. Clone this repository and its camera-streaming submodule

```bash
git clone --recurse-submodules https://github.com/Cognitive-Software-Labs/core_unitree_sim_isaaclab.git
cd core_unitree_sim_isaaclab
git submodule update --init --recursive
```

### 3. Create and activate the Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Every new terminal session must activate this environment before running the
simulator:

```bash
cd /path/to/core_unitree_sim_isaaclab
source .venv/bin/activate
```

### 4. Install PyTorch, Isaac Sim, and Isaac Lab

The versions below are pinned to the current project baseline: Isaac Sim 5.1
and Isaac Lab 2.3.0.

```bash
python -m pip install \
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com

git clone --branch v2.3.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git ../IsaacLab
(cd ../IsaacLab && ./isaaclab.sh --install none)
```

`cu128` is the PyTorch wheel index currently recommended for Isaac Sim 5.1 on
Linux x86_64. If your NVIDIA driver cannot run CUDA 12.8 wheels, upgrade the
driver rather than mixing an arbitrary PyTorch build with this environment.

### 5. Install the Unitree SDK, project packages, and camera streamer

```bash
python -m pip install cyclonedds==0.10.2
python -m pip install "git+https://github.com/unitreerobotics/unitree_sdk2_python.git@master"
python -m pip install -r requirements.txt
python -m pip install -e teleimager
```

The first command uses the Cyclone DDS prebuilt package. If it cannot find a
wheel for your platform, follow the Unitree SDK's CycloneDDS build instructions,
set `CYCLONEDDS_HOME` to that installation, and repeat the Unitree SDK command.

### 6. Download the project assets

```bash
git lfs install
./fetch_assets.sh
```

The asset download is required. It creates `assets/` at the repository root,
including the G1 USDs, hospital room, Ridgeback, containers, and medical
objects. The script deliberately replaces an existing `assets/` directory, so
move any local asset edits elsewhere first.

### 7. Verify the installation

First verify that Isaac Sim itself launches. Accept NVIDIA's EULA when prompted
on its first use; extension downloads and shader caching can take more than ten
minutes.

```bash
isaacsim
```

Close the simulator window, then run the project's fast static checks:

```bash
python -m unittest \
  tests.test_meta_quest_redblocks \
  tests.test_hospital_tabletop_assets \
  tests.test_table_object_selector
```

Finally, launch the hospital task for 300 steps. This creates the scene, RTX
cameras, camera server, and DDS bridge, then exits cleanly.

```bash
UNITREE_DDS_NETWORK_INTERFACE=lo \
python sim_main.py \
  --headless \
  --device cuda:0 \
  --enable_cameras \
  --task Isaac-PickPlace-MedicineBottle-Hospital-G129-Dex1-Joint \
  --enable_dex1_dds \
  --robot_type g129 \
  --max_steps 300
```

`lo` confines DDS to the local host for this smoke test. Use the actual,
isolated simulator interface only when connecting a teleoperation client.

## Run the hospital task

For a local GUI session, omit `--headless` and let the process continue:

```bash
export UNITREE_DDS_NETWORK_INTERFACE=enp3s0

python sim_main.py \
  --device cuda:0 \
  --enable_cameras \
  --task Isaac-PickPlace-MedicineBottle-Hospital-G129-Dex1-Joint \
  --enable_dex1_dds \
  --robot_type g129
```

The task contains two scored pill bottles plus physical hospital/office
clutter. Place both bottles into the rear Ridgeback container to complete an
episode. The front and both wrist cameras are enabled for teleoperation.

For headless operation, retain camera rendering and use `--headless`. Do not
add `--no_render`: it stops the RTX camera frames required by teleoperation.

```bash
python sim_main.py \
  --headless \
  --device cuda:0 \
  --enable_cameras \
  --task Isaac-PickPlace-MedicineBottle-Hospital-G129-Dex1-Joint \
  --enable_dex1_dds \
  --robot_type g129
```

## Meta Quest teleoperation

Install and run the matching `xr_teleoperate` client separately, then launch a
supported task with `--meta_quest`. The flag selects the correct robot and hand
DDS bridge, enables the three required cameras, and configures the local ZMQ
camera stream.

```bash
export UNITREE_DDS_NETWORK_INTERFACE=enp3s0

python sim_main.py \
  --headless \
  --device cuda:0 \
  --task Isaac-PickPlace-MedicineBottle-Hospital-G129-Dex1-Joint \
  --meta_quest
```

The hospital task supports these controls through its DDS/Quest integration:

| Input | Effect |
| --- | --- |
| Index trigger | Close/open the corresponding Dex1 hand |
| A / reset category 4 | Move Ridgeback to the next validated arc position |
| X / reset category 5 | Toggle G1 camera recording to `~/Desktop/G1_Camera_Recordings` |
| Reset category 1 | Respawn the tabletop objects only |
| Reset category 2 | Randomize the complete table, robot, and Ridgeback layout |
| Reset category 3 | Restore the fixed table while scrambling the room and tabletop objects |

## Other tasks

Choose exactly one hand DDS flag that matches the task.

| Task | Robot and hand | Required flag |
| --- | --- | --- |
| `Isaac-PickPlace-Cylinder-G129-Dex1-Joint` | G1 + Dex1 | `--enable_dex1_dds` |
| `Isaac-PickPlace-RedBlock-G129-Dex3-Joint` | G1 + Dex3 | `--enable_dex3_dds` |
| `Isaac-PickPlace-RedBlock-G129-Inspire-Joint` | G1 + Inspire | `--enable_inspire_dds` |
| `Isaac-PickPlace-Hospital-G129-Dex1-Wholebody` | G1 + Dex1 Wholebody | `--enable_dex1_dds --enable_wholebody_dds` |
| `Isaac-PickPlace-MedicineBottle-Hospital-G129-Dex1-Joint` | G1 + Dex1 hospital | `--enable_dex1_dds` |

A Wholebody task automatically uses the Wholebody action path. DDS
teleoperation supports one simulated environment at a time.

## Troubleshooting

**`ModuleNotFoundError: isaaclab`**

Activate `.venv`, then ensure the Isaac Lab installation completed:

```bash
source .venv/bin/activate
(cd ../IsaacLab && ./isaaclab.sh --install none)
```

**`Could not locate cyclonedds` while installing `unitree_sdk2py`**

Install the matching `cyclonedds==0.10.2` package first. If your platform does
not provide its prebuilt wheel, build CycloneDDS 0.10.x and export both paths:

```bash
export CYCLONEDDS_HOME=/absolute/path/to/cyclonedds/install
export CMAKE_PREFIX_PATH="$CYCLONEDDS_HOME"
python -m pip install --force-reinstall \
  "git+https://github.com/unitreerobotics/unitree_sdk2_python.git@master"
```

**A hospital USD cannot be found**

The asset bundle is missing or incomplete. From the repository root, run
`./fetch_assets.sh` again. It replaces `assets/`.

**The scene starts but Quest cameras are black**

Use both `--headless` and `--enable_cameras`; do not use `--no_render`. Ensure
the `teleimager` submodule is present and installed with
`python -m pip install -e teleimager`.

**DDS is using the wrong network adapter**

Set `UNITREE_DDS_NETWORK_INTERFACE` before launching. The simulator prints the
selected interface during startup.

## Project layout

```text
action_provider/  DDS, replay, and policy action sources
assets/           Downloaded robot, room, and object USD assets
dds/              Unitree-compatible DDS publishers and subscribers
tasks/            Isaac Lab task definitions and hospital randomization
teleimager/       Camera-streaming submodule
tools/            Data, camera, Quest, and validation utilities
sim_main.py       Simulator entry point
```

## Related documentation

- [NVIDIA Isaac Lab pip installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)
- [Unitree SDK2 Python](https://github.com/unitreerobotics/unitree_sdk2_python)
- [Hospital teleoperation integration notes](infodocs/hospital_teleoperation_integration.md)
- [Hospital task details](tasks/g1_tasks/pickplace_medicine_bottle_hospital_g1_29dof_dex1/README.md)

## License

This repository is licensed under the [Apache License 2.0](LICENSE). Third-party
components and downloaded assets retain their own licenses.
