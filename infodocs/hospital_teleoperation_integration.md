# Hospital Teleoperation Integration Notes

## Summary

This update integrates the local Unitree G1 29-DoF + Dex1 teleoperation work
with the `hospital_env` branch. It preserves the upstream hospital-room
randomization improvements while adding the fixed-base and Wholebody
teleoperation behavior developed against Isaac Sim 4.5.

Integration commit: `6f020ad`  
Latest synchronized merge commit: `d370fcb`

## Included changes

- Registers `Isaac-PickPlace-Hospital-G129-Dex1-Wholebody` alongside the
  existing fixed-base Dex1 task.
- Adds a Ridgeback assistant and basket to the hospital scene, including
  stable-grasp detection, side approach, placement confirmation, and return.
- Adds hospital tabletop props and teleoperation-oriented contact settings.
- Keeps teleoperation sessions alive when an object falls and respawns the
  dropped bottle independently.
- Preserves manual object/full-scene reset handling while retaining the
  `hospital_env` room-randomization framework.
- Updates DDS and Wholebody action handling and avoids redundant simulation
  resets that can stall RTX-camera startup.
- Updates the `teleimager` submodule pointer to the available upstream `sim`
  commit `b81de44`; the previous target commit was no longer fetchable from the
  public submodule remote.

## Run examples

Fixed-base G1 + Dex1:

```bash
python sim_main.py \
  --headless \
  --device cuda:0 \
  --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Dex1-Joint \
  --enable_dex1_dds \
  --robot_type g129
```

Wholebody hospital task:

```bash
python sim_main.py \
  --headless \
  --device cuda:0 \
  --enable_cameras \
  --task Isaac-PickPlace-Hospital-G129-Dex1-Wholebody \
  --enable_dex1_dds \
  --enable_wholebody_dds \
  --robot_type g129
```

## Validation

- The integration was performed as a normal Git merge from the latest remote
  `hospital_env`; no force-push was used.
- Python syntax compilation passed for the modified simulator, task, and room
  randomization modules.
- Shell syntax checks passed for the modified setup scripts.
- A full Isaac Sim runtime smoke test is still recommended on the target GPU
  machine before using the branch for data collection.

## Collaboration notes

- Target branch: `Cognitive-Software-Labs/core_unitree_sim_isaaclab:hospital_env`
- Personal backup branch:
  `ShidanChen/unitree_sim_isaaclab:integration/hospital-env-20260821`
- Companion XR client branch:
  `ShidanChen/xr_teleoperate:integration/hospital-env-teleoperation-20260821`
  (Quest button mapping, motion-stick commands, reset publishing, and
  fixed-base waist-yaw control).
- Future updates should fetch and merge the latest `hospital_env` before
  pushing, because multiple contributors are working on the same branch.
