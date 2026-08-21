# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
import tempfile
import os
import time
import torch
from dataclasses import MISSING

from pink.tasks import FrameTask

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg
from . import mdp
# use Isaac Lab native event system

from tasks.common_config import  G1RobotPresets, CameraPresets  # isort: skip
from tasks.common_event.event_manager import SimpleEvent
# import public scene configuration
from tasks.common_scene.base_scene_randomized_pickplace_cfg import RandomizedRoomPickPlaceSceneCfg
from tasks.utils.room_randomizer import randomize_pickplace_room_layout
from tasks.utils.room_randomizer.constants import ROOM_X_MAX, ROOM_X_MIN, ROOM_Y_MAX, ROOM_Y_MIN
from tasks.utils.room_randomizer.pickplace_config import (
    TABLE_PROP_NAMES,
    WALL_PROP_NAMES,
    register_randomized_room_reset_events,
)
from tasks.common_scene.base_scene_pickplace_cylindercfg import project_root


RIDGEBACK_USD = (
    f"{project_root}/assets/robots/ridgeback_base_only.usda"
)


def reset_ridgeback_assistant(
    env, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg = SceneEntityCfg("ridgeback")
):
    """Return Ridgeback to its behind-G1 waiting pose and clear its state machine."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ridgeback = env.scene[asset_cfg.name]
    joint_pos = ridgeback.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    ridgeback.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    ridgeback.set_joint_position_target(joint_pos, env_ids=env_ids)
    env._ridgeback_assistant_state = "waiting"
    env._ridgeback_grasp_candidate = None
    env._ridgeback_grasp_since = None
    env._ridgeback_grasp_object_name = None
    env._ridgeback_placement_since = None
    env._ridgeback_demo_return_at = None
    env._ridgeback_demo_object_placed = False
    print("[ridgeback assistant] reset -> waiting behind G1", flush=True)


def update_ridgeback_assistant(env):
    """Detect a stable left/right grasp and drive Ridgeback to that side.

    This is intentionally a small deterministic test controller.  Ridgeback's
    fixed articulation root stays behind G1; its authored planar X/Y/yaw joints
    provide smooth physical motion to a side staging point and then the final
    delivery point.
    """
    ridgeback = env.scene["ridgeback"]
    robot = env.scene["robot"]
    graspable_names = ("object", "hand_sanitizer", "gauze_box", "specimen_cup")

    if not hasattr(env, "_ridgeback_assistant_state"):
        env._ridgeback_assistant_state = "waiting"
        env._ridgeback_grasp_candidate = None
        env._ridgeback_grasp_since = None
    if not hasattr(env, "_ridgeback_left_hand_id"):
        left_ids, _ = robot.find_bodies("left_hand_base_link")
        right_ids, _ = robot.find_bodies("right_hand_base_link")
        env._ridgeback_left_hand_id = int(left_ids[0])
        env._ridgeback_right_hand_id = int(right_ids[0])
        ridgeback_base_ids, _ = ridgeback.find_bodies("base_link")
        env._ridgeback_base_body_id = int(ridgeback_base_ids[0])
        demo_side = os.getenv("RIDGEBACK_ASSISTANT_DEMO", "").strip().lower()
        env._ridgeback_demo_side = demo_side if demo_side in ("left", "right") else None
        env._ridgeback_demo_enabled = bool(env._ridgeback_demo_side)
        env._ridgeback_demo_at = time.monotonic() + 3.0 if env._ridgeback_demo_side else None
        print("[ridgeback assistant] active; waiting for a stable grasp", flush=True)

    state = env._ridgeback_assistant_state
    if state == "waiting":
        left_pos = robot.data.body_pos_w[0, env._ridgeback_left_hand_id]
        right_pos = robot.data.body_pos_w[0, env._ridgeback_right_hand_id]
        candidate = None
        candidate_distance = float("inf")
        for object_name in graspable_names:
            graspable = env.scene[object_name]
            object_pos = graspable.data.root_pos_w[0]
            initial_z = float(graspable.data.default_root_state[0, 2])
            left_dist = float(torch.linalg.vector_norm(object_pos - left_pos))
            right_dist = float(torch.linalg.vector_norm(object_pos - right_pos))
            nearest = "left" if left_dist < right_dist else "right"
            nearest_dist = min(left_dist, right_dist)
            # Require a real, sustained pickup.  This rejects a hand merely
            # passing near an object and small collision/spawn disturbances.
            lifted = float(object_pos[2]) > initial_z + 0.080
            if lifted and nearest_dist < 0.18 and nearest_dist < candidate_distance:
                candidate = (nearest, object_name)
                candidate_distance = nearest_dist
        now = time.monotonic()
        if env._ridgeback_demo_side and now >= env._ridgeback_demo_at:
            candidate = (env._ridgeback_demo_side, "object")
            env._ridgeback_grasp_candidate = candidate
            env._ridgeback_grasp_since = now - 1.0
            env._ridgeback_demo_side = None
            print(f"[ridgeback assistant demo] simulating {candidate[0]}-hand grasp", flush=True)
        if candidate != env._ridgeback_grasp_candidate:
            env._ridgeback_grasp_candidate = candidate
            env._ridgeback_grasp_since = now if candidate else None
        elif candidate and env._ridgeback_grasp_since is not None and now - env._ridgeback_grasp_since >= 0.35:
            candidate_side, candidate_object = candidate
            # Calibrated against the visible fixed-base G1 orientation.
            env._ridgeback_assistant_side = -1.0 if candidate_side == "left" else 1.0
            env._ridgeback_grasp_object_name = candidate_object
            env._ridgeback_assistant_state = "staging"
            state = "staging"
            print(
                f"[ridgeback assistant] {candidate_object} in {candidate_side} hand confirmed; "
                f"approaching {candidate_side} side",
                flush=True,
            )

    if state in ("staging", "side"):
        side = env._ridgeback_assistant_side
        # The fixed root is at (-0.15, -1.80).  First move laterally behind G1,
        # then advance beside it.  Root orientation is identity, so planar joint
        # X/Y correspond directly to world X/Y offsets.
        if state == "staging":
            target = torch.tensor([[side * 0.58, 0.80, 1.5708]], device=env.device)
        else:
            # Restore the original reachable delivery pose.  The waiting root
            # moved farther back, so the larger Y joint travel preserves the
            # same world-space side position used by the first prototype.
            target = torch.tensor([[side * 0.62, 1.53, 1.5708]], device=env.device)
        ridgeback.set_joint_position_target(target)
        current = ridgeback.data.joint_pos[0]
        if float(torch.max(torch.abs(current - target[0]))) < 0.10:
            if state == "staging":
                env._ridgeback_assistant_state = "side"
                print("[ridgeback assistant] staging point reached; moving beside G1", flush=True)
            else:
                env._ridgeback_assistant_state = "holding"
                env._ridgeback_placement_since = None
                ridgeback.set_joint_position_target(target)
                ridgeback.set_joint_velocity_target(torch.zeros_like(target))
                print("[ridgeback assistant] side delivery pose reached; holding", flush=True)
    elif state == "holding":
        side = env._ridgeback_assistant_side
        target = torch.tensor([[side * 0.62, 1.53, 1.5708]], device=env.device)
        # Keep the chassis fully stationary beside G1.  Returning is unlocked
        # only after the real object has remained inside the basket.
        ridgeback.set_joint_position_target(target)
        ridgeback.set_joint_velocity_target(torch.zeros_like(target))
        now = time.monotonic()
        base_pos = ridgeback.data.body_pos_w[0, env._ridgeback_base_body_id]
        base_quat = ridgeback.data.body_quat_w[0, env._ridgeback_base_body_id]
        carried_object = env.scene[getattr(env, "_ridgeback_grasp_object_name", "object")]
        object_pos = carried_object.data.root_pos_w[0]
        delta_x = object_pos[0] - base_pos[0]
        delta_y = object_pos[1] - base_pos[1]
        # Convert the object position into Ridgeback's local frame.  The basket
        # is 0.47 x 0.31 m internally and follows base_link yaw.
        w, x, y, z = base_quat
        base_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        cos_yaw = torch.cos(base_yaw)
        sin_yaw = torch.sin(base_yaw)
        local_x = cos_yaw * delta_x + sin_yaw * delta_y
        local_y = -sin_yaw * delta_x + cos_yaw * delta_y
        relative_height = float(object_pos[2] - base_pos[2])
        object_speed = float(torch.linalg.vector_norm(carried_object.data.root_lin_vel_w[0]))
        left_pos = robot.data.body_pos_w[0, env._ridgeback_left_hand_id]
        right_pos = robot.data.body_pos_w[0, env._ridgeback_right_hand_id]
        hand_distance = min(
            float(torch.linalg.vector_norm(object_pos - left_pos)),
            float(torch.linalg.vector_norm(object_pos - right_pos)),
        )
        # Bottom top is z=0.335 and rim top is z=0.515 relative to base_link.
        # Margins reject objects resting on a wall/rim instead of inside it.
        in_basket = (
            abs(float(local_x)) < 0.205
            and abs(float(local_y)) < 0.125
            and 0.345 < relative_height < 0.500
            and object_speed < 0.12
            and hand_distance > 0.16
        )
        demo_placed = False
        if in_basket:
            if env._ridgeback_placement_since is None:
                # Demo has already waited three seconds; real placement still
                # needs a stability window to reject momentary fly-through.
                env._ridgeback_placement_since = now - (1.0 if demo_placed else 0.0)
            elif now - env._ridgeback_placement_since >= 0.80:
                env._ridgeback_assistant_state = "returning_side"
                print("[ridgeback assistant] placement confirmed; returning behind G1", flush=True)
        else:
            env._ridgeback_placement_since = None
    elif state in ("returning_side", "returning_home"):
        side = env._ridgeback_assistant_side
        if state == "returning_side":
            target = torch.tensor([[side * 0.58, 0.80, 1.5708]], device=env.device)
        else:
            target = torch.zeros((1, 3), device=env.device)
        ridgeback.set_joint_position_target(target)
        current = ridgeback.data.joint_pos[0]
        if float(torch.max(torch.abs(current - target[0]))) < 0.10:
            if state == "returning_side":
                env._ridgeback_assistant_state = "returning_home"
                print("[ridgeback assistant] cleared G1; returning to home pose", flush=True)
            else:
                # Complete the cycle only after Ridgeback is home: return the
                # object from the basket to its authored tabletop pose.
                object_name = getattr(env, "_ridgeback_grasp_object_name", None)
                if object_name:
                    carried_object = env.scene[object_name]
                    object_state = carried_object.data.default_root_state[[0]].clone()
                    object_state[:, :3] += env.scene.env_origins[[0]]
                    object_state[:, 7:13] = 0.0
                    carried_object.write_root_state_to_sim(object_state)
                    print(
                        f"[ridgeback assistant] {object_name} reset to tabletop for next cycle",
                        flush=True,
                    )
                env._ridgeback_assistant_state = "waiting"
                env._ridgeback_grasp_candidate = None
                env._ridgeback_grasp_since = None
                env._ridgeback_grasp_object_name = None
                env._ridgeback_placement_since = None
                ridgeback.set_joint_position_target(target)
                print("[ridgeback assistant] home behind G1; waiting", flush=True)


def respawn_dropped_object(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    floor_height: float = 0.32,
):
    """Respawn only the bottle after it falls to the floor.

    The Ridgeback basket is lower than the table, so the threshold deliberately
    sits below the basket instead of treating every off-table placement as a
    failure.
    """
    object_asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    root_pos = object_asset.data.root_pos_w[env_ids]
    ridgeback = env.scene["ridgeback"]
    base_ids, _ = ridgeback.find_bodies("base_link")
    ridgeback_pos = ridgeback.data.body_pos_w[env_ids, int(base_ids[0])]

    # A bottle is successful when it is in the Ridgeback basket, even though
    # the basket is lower than the table.  Everything near floor height is a
    # genuine drop.  The radial check also catches a bottle that comes to rest
    # on low hospital geometry instead of reaching the collision ground plane.
    basket_xy_distance = torch.linalg.vector_norm(root_pos[:, :2] - ridgeback_pos[:, :2], dim=1)
    in_basket = (basket_xy_distance < 0.42) & (root_pos[:, 2] > 0.30)
    low_or_lost = (root_pos[:, 2] < floor_height) | (
        (root_pos[:, 2] < 0.46) & ~in_basket
    )
    invalid = ~torch.isfinite(root_pos).all(dim=1)
    dropped = low_or_lost | invalid
    dropped_ids = env_ids[dropped]
    if len(dropped_ids) == 0:
        return

    root_state = object_asset.data.default_root_state[dropped_ids].clone()
    root_state[:, :3] += env.scene.env_origins[dropped_ids]
    root_state[:, 7:13] = 0.0
    before = root_pos[dropped].clone()
    object_asset.write_root_pose_to_sim(root_state[:, :7], env_ids=dropped_ids)
    object_asset.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=dropped_ids)
    object_asset.reset(dropped_ids)
    print(
        "[bottle respawn] dropped/lost: "
        f"{before[0].tolist()} -> {root_state[0, :3].tolist()}",
        flush=True,
    )


def never_terminate(env) -> torch.Tensor:
    """Keep teleoperation running; dropped bottles are handled independently."""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def reset_all_teleop_scene(env):
    """Reset the full teleop scene and explicitly restore the medicine bottle."""
    env_ids = torch.arange(env.num_envs, device=env.device)
    object_asset = env.scene["object"]
    before = object_asset.data.root_pos_w[env_ids].clone()

    base_mdp.reset_scene_to_default(env, env_ids)
    reset_ridgeback_assistant(env, env_ids)

    # Explicitly restore the task bottle instead of relying only on the generic
    # scene traversal. This also guarantees zero residual linear/angular speed.
    bottle_state = object_asset.data.default_root_state[env_ids].clone()
    bottle_state[:, 0] = env.scene.env_origins[env_ids, 0] - 0.68
    bottle_state[:, 1] = env.scene.env_origins[env_ids, 1] + 0.40
    bottle_state[:, 2] = env.scene.env_origins[env_ids, 2] + 0.86
    bottle_state[:, 7:13] = 0.0
    object_asset.write_root_state_to_sim(bottle_state, env_ids=env_ids)
    print(
        "[reset all] bottle position: "
        f"{before[0].tolist()} -> {bottle_state[0, :3].tolist()}"
    )
##
# Scene definition
##

@configclass
class ObjectTableSceneCfg(RandomizedRoomPickPlaceSceneCfg):
    """object table scene configuration class
    inherits from G1SingleObjectSceneCfg, gets the complete G1 robot scene configuration
    can add task-specific scene elements or override default configurations here
    """
    
    # Humanoid robot w/ arms higher
    # 5. humanoid robot configuration 
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex1_base_fix()
    # The stock fixed-base asset hard-locks every waist joint with zero velocity
    # and kp/kd=10000.  Release yaw for VR torso turning while keeping waist
    # roll/pitch and the complete lower body fixed.
    robot.actuators.pop("waist", None)
    robot.actuators["waist_yaw_teleop"] = ImplicitActuatorCfg(
        joint_names_expr=["waist_yaw_joint"],
        effort_limit_sim=350.0,
        velocity_limit_sim=2.5,
        stiffness=260.0,
        damping=18.0,
    )
    robot.actuators["waist_roll_pitch_lock"] = ImplicitActuatorCfg(
        joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
        effort_limit_sim=1000.0,
        velocity_limit_sim=0.1,
        stiffness=10000.0,
        damping=10000.0,
    )

    ridgeback: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Ridgeback",
        spawn=sim_utils.UsdFileCfg(
            usd_path=RIDGEBACK_USD,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True,
                enabled_self_collisions=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # Fixed articulation root waits behind G1.  Planar joints move the
            # chassis from here when the assistant state machine is triggered.
            # About 1.70 m behind G1 while no object is being carried.
            pos=(-0.15, -1.80, 0.0328),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "base_translation": ImplicitActuatorCfg(
                joint_names_expr=["dummy_base_prismatic_.*_joint"],
                effort_limit_sim=1600.0,
                velocity_limit_sim=0.55,
                stiffness=500.0,
                damping=90.0,
            ),
            "base_yaw": ImplicitActuatorCfg(
                joint_names_expr=["dummy_base_revolute_z_joint"],
                effort_limit_sim=900.0,
                velocity_limit_sim=0.75,
                stiffness=320.0,
                damping=55.0,
            ),
        },
    )

    # Open collision basket mounted directly to the Ridgeback chassis.  Since
    # it is a child of base_link it follows the randomized chassis pose.
    ridgeback_basket = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Ridgeback/base_link/Basket",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/ridgeback_basket.usda",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.31),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Additional hospital tabletop props.  They are separate rigid bodies, so
    # they provide meaningful visual clutter and can also be grasped or moved.
    hand_sanitizer = RigidObjectCfg(
        prim_path="/World/envs/env_.*/HandSanitizer",
        init_state=RigidObjectCfg.InitialStateCfg(
            # Rightmost item in the front 1x4 row.  Its geometry still ends
            # before the built-in container's x~=0.26 left boundary.
            pos=(0.07, 0.40, 0.875), rot=(1.0, 0.0, 0.0, 0.0)
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/hospital_hand_sanitizer.usda",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=1.5, angular_damping=3.0,
                max_linear_velocity=5.0, max_angular_velocity=10.0,
                max_depenetration_velocity=0.25,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.18),
        ),
    )
    gauze_box = RigidObjectCfg(
        prim_path="/World/envs/env_.*/GauzeBox",
        init_state=RigidObjectCfg.InitialStateCfg(
            # Third item in the front 1x4 row.
            pos=(-0.18, 0.40, 0.838), rot=(1.0, 0.0, 0.0, 0.0)
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/hospital_gauze_box.usda",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=1.5, angular_damping=3.0,
                max_linear_velocity=5.0, max_angular_velocity=10.0,
                max_depenetration_velocity=0.25,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
        ),
    )
    specimen_cup = RigidObjectCfg(
        prim_path="/World/envs/env_.*/SpecimenCup",
        init_state=RigidObjectCfg.InitialStateCfg(
            # Second item in the front 1x4 row.
            pos=(-0.43, 0.40, 0.845), rot=(1.0, 0.0, 0.0, 0.0)
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/hospital_specimen_cup.usda",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=1.5, angular_damping=3.0,
                max_linear_velocity=5.0, max_angular_velocity=10.0,
                max_depenetration_velocity=0.25,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
        ),
    )


    # 6. add camera configuration 
    front_camera = CameraPresets.g1_front_camera()
    left_wrist_camera = CameraPresets.left_gripper_wrist_camera()
    right_wrist_camera = CameraPresets.right_gripper_wrist_camera()

##
# MDP settings
##
@configclass
class ActionsCfg:
    """defines the action configuration related to robot control, using direct joint angle control
    """
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)



@configclass
class ObservationsCfg:
    """
    defines all available observation information
    """
    @configclass
    class PolicyCfg(ObsGroup):
        """policy group observation configuration class
        defines all state observation values for policy decision
        inherit from ObsGroup base class 
        """

        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        robot_gipper_state = ObsTerm(func=mdp.get_robot_gipper_joint_states)

        camera_image = ObsTerm(func=mdp.get_camera_image)

        def __post_init__(self):
            """post initialization function
            set the basic attributes of the observation group
            """
            self.enable_corruption = False  # disable observation value corruption
            self.concatenate_terms = False  # disable observation item connection

    # observation groups
    # create policy observation group instance
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    # Teleoperation must not reset G1/Ridgeback when only the bottle falls.
    success = DoneTerm(func=never_terminate)

@configclass
class RewardsCfg:
    reward = RewTerm(func=mdp.compute_reward,weight=1.0)

@configclass
class EventCfg:
    randomize_room_layout = EventTermCfg(
        func=randomize_pickplace_room_layout,
        mode="reset",
        params={
            "wall_prop_names": WALL_PROP_NAMES,
            "table_prop_names": TABLE_PROP_NAMES,
            "min_table_objects": 1,
        },
    )
    reset_ridgeback = EventTermCfg(
        func=reset_ridgeback_assistant,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("ridgeback")},
    )
    respawn_dropped_bottle = EventTermCfg(
        func=respawn_dropped_object,
        mode="interval",
        interval_range_s=(0.10, 0.10),
        is_global_time=True,
        params={"asset_cfg": SceneEntityCfg("object"), "floor_height": 0.32},
    )


@configclass
class PickPlaceG129DEX1BaseFixEnvCfg(ManagerBasedRLEnvCfg):
    """
    inherits from ManagerBasedRLEnvCfg, defines all configuration parameters for the entire environment
    """

    # 1. scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, # environment number: 1
                                                     env_spacing=16.0, # hospital room footprint needs wider spacing
                                                     replicate_physics=True # enable physics replication
                                                     )
    # basic settings
    observations: ObservationsCfg = ObservationsCfg()   # observation configuration
    actions: ActionsCfg = ActionsCfg()                  # action configuration
    # MDP settings

    terminations: TerminationsCfg = TerminationsCfg()    # termination configuration
    events = EventCfg()                                  # event configuration
    commands = None # command manager
    rewards: RewardsCfg = RewardsCfg()  # reward manager
    curriculum = None # curriculum manager
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 20.0
        self.viewer.origin_type = "world"
        self.viewer.eye = (-7.5, -3.2, 4.2)
        self.viewer.lookat = (-7.5, -7.6, 0.8)
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        # Hospital props must remain controllable in the Dex1 parallel jaws.
        # Use the stronger material in every contact pair and eliminate bounce.
        self.sim.physics_material.static_friction = 2.8
        self.sim.physics_material.dynamic_friction = 2.4
        self.sim.physics_material.restitution = 0.0
        self.sim.physics_material.friction_combine_mode = "max"
        self.sim.physics_material.restitution_combine_mode = "min"
        # create event manager
        register_randomized_room_reset_events(self)

        # Preserve the teleoperation-specific manual resets while retaining the
        # target branch's native room-randomization event above.
        self.event_manager.register("reset_object_self", SimpleEvent(
            func=lambda env: base_mdp.reset_root_state_uniform(
                env,
                torch.arange(env.num_envs, device=env.device),
                pose_range={"x": [-0.05, 0.05], "y": [0.0, 0.05]},
                velocity_range={},
                asset_cfg=SceneEntityCfg("object"),
            )
        ))

        self.event_manager.register("reset_all_self", SimpleEvent(
            func=reset_all_teleop_scene
        ))


@configclass
class ObjectTableWholebodySceneCfg(ObjectTableSceneCfg):
    """Hospital scene with the free-root G1 used by the locomotion policy."""

    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex1_wholebody(
        init_pos=(-0.15, -0.10, 0.8),
        # Preserve the task's calibrated +90-degree yaw toward the table.
        init_rot=(0.7071, 0.0, 0.0, 0.7071),
    )
    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=10,
        track_air_time=True,
        debug_vis=False,
    )


@configclass
class PickPlaceHospitalG129DEX1WholebodyEnvCfg(PickPlaceG129DEX1BaseFixEnvCfg):
    """Movable G1 variant retaining the hospital task and Ridgeback basket."""

    scene: ObjectTableWholebodySceneCfg = ObjectTableWholebodySceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
