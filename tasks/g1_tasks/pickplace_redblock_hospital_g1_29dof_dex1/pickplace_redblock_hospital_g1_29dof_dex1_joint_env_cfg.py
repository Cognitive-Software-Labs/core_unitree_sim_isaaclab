# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Hospital-room red-block pick-place task for G1 (29 DoF) + Dex1 gripper, joint control.

This is an *independent* task: it does NOT modify any existing warehouse task.
The scene reuses the hospital RoomShell (new_base_room.usda) from
RandomizedRoomPickPlaceSceneCfg. The calibrated task group stays fixed while
the wall props use the shared room randomizer:

  * table / robot / object poses are the known-good warehouse red-block geometry
    translated by T = (-3.2, -3.3, 0) so the table sits in the open hospital
    interior at (-7.5, -7.5). That preserves the robot<->table<->object relative
    poses the IK, cameras and grasp are already calibrated for.
  * wall props are spawned separately and placed from
    tasks/utils/room_randomizer/constants.py on every full reset.

Tunable numbers that may need a visual pass in Isaac Sim are tagged  # TUNE.
"""

from __future__ import annotations

import os

import torch

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from tasks.common_config import CameraBaseCfg, CameraPresets, G1RobotPresets  # isort: skip
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_scene.base_scene_randomized_pickplace_cfg import (
    RandomizedRoomPickPlaceSceneCfg,
    tabletop_cube_cfg,
)
from tasks.common_scene.base_scene_pickplace_cylindercfg import (
    hospital_hand_sanitizer_cfg,
    hospital_medicine_bottle_cfg,
)
from tasks.utils.room_randomizer import (
    randomize_pickplace_room_layout,
    randomize_wall_props_layout,
)
from tasks.utils.room_randomizer.room_events import reset_target_on_current_table
from tasks.utils.room_randomizer.pickplace_config import (
    WALL_PROP_NAMES,
)

project_root = os.environ.get("PROJECT_ROOT")


# ----------------------------------------------------------------------------
# Fixed-layout world coordinates (hospital room interior).
# Derived from the warehouse red-block task translated by T = (-3.2, -3.3, 0).
# ----------------------------------------------------------------------------
TABLE_POS = (-7.5, -7.5, -0.2)          # TUNE: open interior of new_base_room.usda
ROBOT_POS = (-7.4, -7.0, 0.76)          # +Y of table by 0.5 m, same as warehouse
ROBOT_ROT = (0.7071, 0.0, 0.0, -0.7071)  # yaw -90deg, robot faces the table
OBJECT_POS = (-7.38, -7.25, 0.84)       # TUNE: red block closer to G1

# Success / out-of-range box (warehouse box translated by T). object inside => not done.
SUCCESS_BOX = dict(min_x=-8.6, max_x=-6.1, min_y=-8.35, max_y=-6.1, min_height=0.5)
REDBLOCK_TABLE_PROP_NAMES = [
    "hand_sanitizer",
    "medicine_bottle_a",
    "medicine_bottle_b",
    "blue_cube",
    "yellow_cube",
]
OBJECT_RESET_POSE_RANGE = {"x": [-0.05, 0.10], "y": [-0.08, 0.06]}


def _reset_fixed_table_target(env, env_ids: torch.Tensor) -> None:
    """Apply the calibrated spawn jitter used before full randomization."""
    base_mdp.reset_root_state_uniform(
        env,
        env_ids,
        pose_range=OBJECT_RESET_POSE_RANGE,
        velocity_range={},
        asset_cfg=SceneEntityCfg("object"),
    )


def reset_hospital_teleop_scene(
    env,
    env_ids: torch.Tensor | None,
    randomize_table_position: bool | None = None,
) -> None:
    """Reset the Quest scene and optionally change its persistent table switch."""
    if randomize_table_position is not None:
        env._teleop_randomize_table_position = bool(randomize_table_position)
    randomize_table_position = bool(
        getattr(env, "_teleop_randomize_table_position", False)
    )
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    base_mdp.reset_scene_to_default(env, env_ids)
    if randomize_table_position:
        randomize_pickplace_room_layout(
            env,
            env_ids,
            wall_prop_names=WALL_PROP_NAMES,
            table_prop_names=REDBLOCK_TABLE_PROP_NAMES,
            min_table_objects=len(REDBLOCK_TABLE_PROP_NAMES),
            randomize_table_position=True,
        )
    else:
        randomize_wall_props_layout(env, env_ids, wall_prop_names=WALL_PROP_NAMES)
        _reset_fixed_table_target(env, env_ids)

    mode = "full randomization" if randomize_table_position else "fixed table"
    print(f"[Meta Quest reset] hospital scene restored ({mode})", flush=True)


def reset_hospital_target(env) -> None:
    """Reset the block on the fixed or currently randomized packing table."""
    env_ids = torch.arange(env.num_envs, device=env.device)
    if bool(getattr(env, "_teleop_randomize_table_position", False)):
        reset_target_on_current_table(env, env_ids)
    else:
        _reset_fixed_table_target(env, env_ids)


def reset_hospital_room_fixed_table(env) -> None:
    """Scramble the hospital room while restoring the authored table anchor."""
    env_ids = torch.arange(env.num_envs, device=env.device)
    env._teleop_randomize_table_position = False
    base_mdp.reset_scene_to_default(env, env_ids)
    randomize_pickplace_room_layout(
        env,
        env_ids,
        wall_prop_names=WALL_PROP_NAMES,
        table_prop_names=REDBLOCK_TABLE_PROP_NAMES,
        min_table_objects=len(REDBLOCK_TABLE_PROP_NAMES),
        randomize_table_position=False,
    )
    print("[Meta Quest reset] hospital room scrambled (fixed table)", flush=True)


def terminate_outside_active_workspace(env) -> torch.Tensor:
    """Keep full-randomization teleoperation under explicit Quest reset control."""
    if bool(getattr(env, "_teleop_randomize_table_position", False)):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return mdp.reset_object_estimate(env, **SUCCESS_BOX)


##
# Scene definition
##
@configclass
class HospitalRedBlockSceneCfg(RandomizedRoomPickPlaceSceneCfg):
    """Fixed hospital-room layout: room shell + table + G1/Dex1 + red block."""

    # --- repurpose inherited entities -------------------------------------
    # table: reposition the inherited kinematic PackingTable into the interior
    packing_table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/PackingTable/PackingTable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=TABLE_POS),
    )

    # target object: red block (matches the G1_Dex1_Grasp_Red_Block data pipeline)
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_POS, rot=(1, 0, 0, 0)),
        spawn=tabletop_cube_cfg((1.0, 0.0, 0.0)),
    )

    # Graspable hospital props. Their authored poses form a clear row on the
    # open side of the fixed table; full Quest randomization scrambles them
    # collision-free with the red block and existing tabletop distractors.
    hand_sanitizer: RigidObjectCfg = hospital_hand_sanitizer_cfg(
        prim_path="/World/envs/env_.*/HandSanitizer",
        init_pos=(-7.72, -7.14, 0.875),
    )
    medicine_bottle_a: RigidObjectCfg = hospital_medicine_bottle_cfg(
        prim_path="/World/envs/env_.*/MedicineBottleA",
        init_pos=(-8.20, -7.14, 0.86),
    )
    medicine_bottle_b: RigidObjectCfg = hospital_medicine_bottle_cfg(
        prim_path="/World/envs/env_.*/MedicineBottleB",
        init_pos=(-7.98, -7.14, 0.86),
    )

    # --- robot + cameras ---------------------------------------------------
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex1_base_fix(
        init_pos=ROBOT_POS, init_rot=ROBOT_ROT
    )
    front_camera = CameraPresets.g1_front_camera()
    left_wrist_camera = CameraPresets.left_gripper_wrist_camera()
    right_wrist_camera = CameraPresets.right_gripper_wrist_camera()

    # spectator camera (GUI only): warehouse view translated to look at the table
    world_camera = CameraBaseCfg.get_camera_config(
        prim_path="/World/PerspectiveCamera",
        pos_offset=(-7.3, -8.2, 1.8),       # TUNE: GUI-only viewpoint
        rot_offset=(-0.3173, 0.94833, 0.0, 0.0),
    )

    # Keep the inherited wall-prop assets enabled. Their reset event reads the
    # shared WALL_PROP_META constants and hides the baked-in duplicates.
    coffee_cup = None
    desk_lamp = None
    box_portable = None


##
# MDP settings
##
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        robot_gipper_state = ObsTerm(func=mdp.get_robot_gipper_joint_states)
        camera_image = ObsTerm(func=mdp.get_camera_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    # object out of the (translated) working box => reset. thresholds passed explicitly.
    success = DoneTerm(func=terminate_outside_active_workspace)


@configclass
class RewardsCfg:
    # The success post-box must sit at the ACTUAL basket, which is the grey wire tray
    # `container_h20` inside PackingTable.usd (verified: prim local center
    # (0.625,-0.094,1.038), size 0.73x0.49x0.09). With the table placed unrotated at
    # TABLE_POS=(-7.5,-7.5,-0.2) the tray occupies world x[-7.24,-6.51], y[-7.84,-7.35],
    # z[0.79,0.88]. (The warehouse compute_reward defaults, even translated by T, point
    # at the *yellowbox* target of a different table asset (-7.40,-7.24) ~0.5 m away from
    # this basket -> genuine placements never scored, only reset-artifact hits when the
    # block randomly spawned in that patch. Confirmed by logging the block pos while the
    # model ran: it carries the block to this tray's near rim (x~-7.22, y~-7.35, z~0.89)
    # and stalls -- "卡在筐沿".)
    # post_min_x=-7.22 is kept 1 cm inside the rim so the block reset region
    # (x in [-7.48,-7.23]) can never overlap the post-box -> no false-positive successes.
    reward = RewTerm(
        func=mdp.compute_reward,
        weight=1.0,
        params={
            "min_x": -8.6, "max_x": -6.1, "min_y": -8.35, "max_y": -6.1,   # valid box (= SUCCESS_BOX)
            "post_min_x": -7.22, "post_max_x": -6.53,                      # container_h20 tray footprint
            "post_min_y": -7.82, "post_max_y": -7.37,
            "min_height": 0.5, "post_min_height": 0.79, "post_max_height": 0.87,
        },
    )


@configclass
class EventCfg:
    reset_teleop_scene = EventTermCfg(
        func=reset_hospital_teleop_scene,
        mode="reset",
    )


@configclass
class PickPlaceRedBlockHospitalG129DEX1EnvCfg(ManagerBasedRLEnvCfg):
    """G1 + Dex1 red-block pick-place inside the fixed hospital room."""

    scene: HospitalRedBlockSceneCfg = HospitalRedBlockSceneCfg(
        num_envs=1, env_spacing=16.0, replicate_physics=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    curriculum = None

    def __post_init__(self):
        """Post initialization (PhysX/sim settings copied from the warehouse task)."""
        self.decimation = 2
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024
        self.sim.physx.friction_correlation_distance = 0.003
        self.sim.physx.enable_ccd = True
        self.sim.physx.gpu_constraint_solver_heavy_spring_enabled = True
        self.sim.physx.num_substeps = 2
        self.sim.physx.contact_offset = 0.015
        self.sim.physx.rest_offset = 0.001
        self.sim.physx.num_position_iterations = 12
        self.sim.physx.num_velocity_iterations = 4

        # custom event manager (matches the warehouse red-block task)
        self.event_manager = SimpleEventManager()
        self.event_manager.register("reset_object_self", SimpleEvent(
            func=reset_hospital_target
        ))
        self.event_manager.register("reset_all_self", SimpleEvent(
            # Quest/xr_teleoperate's full-reset button sends DDS category 2.
            func=lambda env: reset_hospital_teleop_scene(
                env, None, randomize_table_position=True
            )
        ))
        self.event_manager.register(
            "reset_room_fixed_table_self",
            SimpleEvent(func=reset_hospital_room_fixed_table),
        )
