# Isaac Sim 6.1 / Isaac Lab 3.0 compatibility review

## Evidence and scope

Reviewed port head: `fe60f4a11683e1136bccdcd28dd6c0181da611cd`.
It is four commits ahead of `main` (`47283f5791385c52be156e5756af56155228e545`),
with 39 changed files. [PR #2](https://github.com/Cognitive-Software-Labs/core_unitree_sim_isaaclab/pull/2)
records the September 18, 2026 end-to-end hospital G1 Dex1 / Quest 3 test;
its head was the preceding `2fc186d` commit. The fourth commit adds portable
handoff instructions. The PR was closed for later review, without merging.

The record validates scene creation, all three camera streams, DDS command/state
initialization, reset categories and recording, and live Quest 3 teleoperation.
It does not establish clean-machine reproducibility, all-task compatibility,
a container build, or a 100-reset stress test. This documentation change has
not repeated the GPU or physical-headset test.

## Compatibility and dependency decisions

| Component | Retained legacy setup | Recorded 6.1 setup / decision |
| --- | --- | --- |
| Python | 3.11 | 3.12, separate venv |
| Isaac Sim | 5.1.0 | 6.1.0 |
| Isaac Lab | README baseline 2.3.0; Docker/helper clone a floating branch | 3.0.0, `release/3.0.0` in handoff; record resolved SHA locally |
| PyTorch | 2.7.0 / torchvision 0.22.0 / torchaudio 2.7.0 | Exact installed versions not recorded; follow the versioned NVIDIA setup, do not carry the old pins forward |
| CUDA wheels / image | README cu128, Docker/helper cu126, Docker CUDA runtime 12.2.0 | No 6.1 wheel-index/container baseline recorded; no replacement CUDA pin inferred |
| CycloneDDS | SDK metadata requires 0.10.2; scripts build floating `releases/0.10.x` | Bundled Python+C wheel 11.0.1 tested in both sim and XR bridge; intentional SDK metadata conflict |
| `teleimager` | Gitlink `9a79f851f3d410c6b79e1ad60567547a2dd5dd24` declares Python `<3.12` | Handoff locally widened to `<3.13` and installed editable with `--no-deps`; gitlink does not contain that edit |
| Project requirements | `requirements.txt` | Existing pins retained; no validated 6.1 lockfile or replacement dependency set is provided |
| Unitree SDK / XR bridge | Moving external revisions | Tested exact SHAs not recorded; use hospital button-mapping bridge, capture revisions before further validation |

Isaac Lab 3.0 requires the port's `SimulationCfg.physics` / `PhysxCfg` changes,
ProxyArray `.torch` views, and XYZW pose convention. Installing different
packages alone cannot make this code compatible with 2.x. Camera initialization
is enabled in `sim_main.py`; the old `--enable_cameras` and `--headless` CLI flags
are removed from the recorded 3.0 launcher. Use `HEADLESS=1`, or explicitly
request `--visualizer kit` for a GUI.

The [DyNAMO-Ridgeback `dev` README](https://github.com/Cognitive-Software-Labs/DyNAMO-Ridgeback/blob/dev/README.md)
also names Isaac Sim 6.1, but runs a ROS 2 Jazzy backend, not this Isaac Lab
teleoperation environment. Its [migration evidence](https://github.com/Cognitive-Software-Labs/DyNAMO-Ridgeback/blob/dev/archive/engineering/2026-09-13-isaac-6.1-migration.md)
is partial (tested revisions unknown). A shared simulator version does not
validate shared Python dependencies, DDS-library replacement in ROS, or a
combined G1/Ridgeback experiment. Keep ROS and the XR/sim environments separate
until that integration is tested. The recorded 5.1 RTX crash is specific to the
reported Blackwell/driver setup, not a universal claim that 5.1 is broken.

## Installation paths require porting and testing before use

The successful test record describes an existing, locally modified environment.
It does **not** validate a fresh installation through any of the paths below.
Before using one for this branch, attempt its port to Isaac Sim 6.1 / Isaac Lab
3.0 in a separate environment and test the result. Do not simply replace the
Isaac Sim version string and run the installer.

| Path | Current assumptions / required work before 6.1 use |
| --- | --- |
| Root README's retained installation steps | Python 3.11, Sim 5.1.0, Lab 2.3.0, torch 2.7.0/cu128 and DDS 0.10.2; port the complete environment and verify dependency resolution and runtime behavior |
| `Dockerfile` | Python 3.11, Sim 5.1.0, torch 2.7.0/cu126 on CUDA runtime 12.2.0; floating Lab clone and upstream Unitree checkout; needs a separate 6.1 port, build and GPU/camera/DDS testing |
| `auto_setup_env.sh` | Supports only 4.5/5.0/5.1, floating Lab checkout for 5.1 and source-built DDS 0.10.x; requires a tested 6.1 installation path |
| `tools/start_redblocks_meta_quest.sh` | Legacy Conda activation and removed `--headless` flag; needs a 3.0 launcher port and validation before use |
| Older English/Chinese installation and operator guides | Historical versions, dependency commands and launcher flags; reconcile and test before following them on this branch |

Existing installer pins and executable behavior remain unchanged. The old 5.1
setup is retained for a separate 5.1-compatible checkout, not advertised as a
working installation for this branch. No new requirements overlay, dependency
upgrade, Docker recipe, or automated metadata patch is provided here.

## Items a tested port must resolve

- Capture the exact PyTorch, CUDA wheel index, NumPy/OpenCV, Pinocchio, SDK and
  hospital XR bridge revisions from the working setup. These are not fully
  recorded; do not invent replacement pins. `sim_main.py` imports `pinocchio`,
  although the root requirements do not identify its distribution/version.
- Resolve the SDK's `cyclonedds==0.10.2` metadata requirement against the
  handoff's tested 11.0.1 bundled wheel. The handoff reports an intentional
  metadata conflict and a failure with mismatched Python/C library builds.
  Verify both simulator and XR bridge; do not alter system ROS 2 DDS packages
  or the separate working 5.1 environment on this evidence alone.
- The pinned teleimager submodule still declares Python `<3.12` and
  `numpy>=1.21,<2`. The handoff's local `<3.13` edit and editable `--no-deps`
  installation are historical workarounds, not a validated clean-install
  procedure. Check its dependencies against the actual Lab 3.0 environment
  before adopting a patch or allowing a resolver to downgrade packages.
- Check removed `--headless` / `--enable_cameras` CLI flags, `HEADLESS=1`,
  explicit GUI `--visualizer kit`, and this branch's camera initialization.
  Do not assume legacy wrappers use the same launch behavior.
- Keep the branch's XYZW pose conversion, PhysxCfg/physics API and ProxyArray
  assumptions aligned with Lab 3.0. Other tasks and remaining quaternion
  consumers require their own runtime coverage.

For any proposed setup port, build/install in isolation, run the existing
static checks, then verify hospital startup, all three camera streams, DDS
state/commands, resets, recording and live Quest control. Record resolved
package versions, all repository/submodule SHAs and local patches, GPU/driver
information and logs. A combined DyNAMO-Ridgeback/G1 experiment needs its own
integration test; matching the simulator version alone is insufficient.

The review here checks repository evidence and documentation only. No install,
container build, GPU run or physical-headset test was performed for this change.
