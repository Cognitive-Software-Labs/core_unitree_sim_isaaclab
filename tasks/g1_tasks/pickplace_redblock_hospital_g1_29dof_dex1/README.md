# Hospital Red-Block Pick-Place (G1 29-DoF + Dex1)

Teleoperation + evaluation task that runs the red-block pick-place in the
**hospital room** (`isaac-projects/new_base_room.usda`) instead of the default
warehouse. The G1 uses its **left** Dex1 gripper to pick a red block off the
tabletop and drop it into the grey mesh tray (`container_h20`, part of the
`PackingTable.usd` in the room). The table also carries a graspable hand
sanitizer and two prescription bottles; the bottles use the bundled
`assets/objects/textures/hospital_pharmacy_label.png` label texture.

Gym task id: **`Isaac-PickPlace-RedBlock-Hospital-G129-Dex1-Joint`**

> This is a **self-contained** task. It does not modify any existing warehouse
> task — it only *adds* a new package under `g1_tasks/` and registers it.

---

## Design: reuse the calibrated warehouse geometry, just translate it

The robot/table/object/camera relative poses in the warehouse red-block task are
already calibrated (IK, camera framing, grasp height). To avoid re-tuning all of
that for the hospital, the whole rig is **rigidly translated** into the open
hospital interior:

```
T = (-3.2, -3.3, 0)      # warehouse coords  ->  hospital interior
TABLE_POS  = (-7.5, -7.5, -0.2)
ROBOT_POS  = (-7.4, -7.0, 0.76)
OBJECT_POS = (-7.38, -7.25, 0.84)   # red block spawn centre, toward G1
```

The calibrated robot/table/object rig is the default Meta Quest layout, while
the inherited wall props are randomized around it. DDS reset category `2` (the
full-reset button) enables table-group randomization for the remainder of the
session; wall furniture, the red block, and tabletop distractors then scramble
together. Category `1` only moves the block and preserves whichever table pose
is active. Category `3` scrambles the room and tabletop objects while returning
the table to the calibrated fixed pose. On Quest controllers, **B** sends
category `2` and **Y** sends category `3`. Placement comes from
`tasks/utils/room_randomizer/constants.py`; the
matching furniture baked into the room shell is hidden to avoid duplicates.

**Anything that references a world coordinate had to be translated by `T` too** —
the success/out-of-range box and the reward's tray footprint. Forgetting this is
the classic failure mode: reward stays at `-1` forever because the success
region is still at the un-translated warehouse location. Current values:

```
SUCCESS_BOX (valid region):  x[-8.6, -6.1]  y[-8.35, -6.1]  z>0.5
container_h20 tray footprint: x[-7.22, -6.53]  y[-7.82, -7.37]
```

### Block spawn distribution (`pose_range`)

`reset_object` / `reset_object_self` randomize the block around `OBJECT_POS`.
The committed range is **left/centre-only**:

```python
pose_range = {"x": [-0.05, 0.10], "y": [-0.08, 0.06]}
```

This keeps the block in the region a **left-hand-only** policy can reach
(more-negative world-x pushes the block toward the right of the camera image,
which the left arm struggles to reach). Widen it back toward
`{"x": [-0.15, 0.0], "y": [-0.1, 0.1]}` if you want the full workspace.

> Editing this cfg or the shared room-randomizer constants requires a **sim
> restart** to take effect.

---

## Package layout

```
pickplace_redblock_hospital_g1_29dof_dex1/
├── __init__.py                                   gym.register(...)
├── pickplace_redblock_hospital_g1_29dof_dex1_joint_env_cfg.py
└── mdp/
    ├── observations.py
    ├── rewards.py          reward = block inside translated tray box
    └── terminations.py     success / reset-object estimate
```

Tunable numbers that may want a visual pass in Isaac Sim are tagged `# TUNE`.

---

## Running

Launch from the repo root. Requires `--enable_cameras`, the Dex1 DDS bridge and
`--robot_type g129`:

### Teleoperation recording
```bash
python sim_main.py --device cpu --enable_cameras \
  --task Isaac-PickPlace-RedBlock-Hospital-G129-Dex1-Joint \
  --enable_dex1_dds --robot_type g129
```
Drive it with `xr_teleoperate`. Controller mapping used for collection:
right-A = start tracking, right-B = start/stop recording, **left-X = save +
reset (randomize block)**, left-Y + right-B = quit. Press **left-X after every
episode** — otherwise the next episode starts with the block still in the tray,
and a missed press records one runaway multi-minute episode.

### Replay a recorded dataset
```bash
python sim_main.py --device cpu --enable_cameras \
  --task Isaac-PickPlace-RedBlock-Hospital-G129-Dex1-Joint \
  --enable_dex1_dds --robot_type g129 --replay --file_path <dataset_dir>
```

### Policy evaluation
Run the sim above (headless is fine), then drive it with the pi0/pi05 DDS client
bridge + eval harness (see the training-side worklog). Success is scored from
`rt/rewards_state` (`reward >= 1.0` held ~0.4 s). Reset command `"2,1"` = home the
robot then randomize the block.

---

## DDS / cameras

- CycloneDDS domain 1. Cameras over ZMQ: front (`cam_left_high`) `55555`,
  left wrist `55556`, right wrist `55557`.
- 16-dim state/action (7 left arm + 7 right arm + 1 left ee + 1 right ee),
  3 cameras. Language goal: `"Place the red block into the basket"`.

---

## Sim-side changes this task depends on

These live outside this package and were changed to support hospital teleop:

| File | Change | Why |
|------|--------|-----|
| `robots/unitree.py` | Dex1 finger `friction` `200 → 20` | 200 was sluggish, 10 too slippery, 40 too sticky; 20 = responsive close with slight damping, so gripper open/close reads clearly during teleop. |
| `dds/sim_state_dds.py` | sim-state shared-mem `4096 → 16384` bytes | Hospital `init_state` JSON (with the red-block world pose per frame, used for dataset cleaning/labeling) overflows the old 4 KB buffer. |
| `tasks/g1_tasks/__init__.py` | import + export this package | Register the task. |
