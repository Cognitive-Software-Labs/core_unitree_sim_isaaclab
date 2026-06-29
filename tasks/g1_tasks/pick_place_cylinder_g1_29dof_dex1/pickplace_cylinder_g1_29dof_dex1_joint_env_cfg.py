# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
import tempfile
import torch
from dataclasses import MISSING

from pink.tasks import FrameTask

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from . import mdp
# use Isaac Lab native event system

from tasks.common_config import  G1RobotPresets, CameraPresets  # isort: skip
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager

# import public scene configuration
from tasks.common_scene.base_scene_randomized_pickplace_cfg import RandomizedRoomPickPlaceSceneCfg
from tasks.utils.room_randomizer import randomize_pickplace_room_layout
from tasks.utils.room_randomizer.constants import ROOM_X_MAX, ROOM_X_MIN, ROOM_Y_MAX, ROOM_Y_MIN

##
# Scene definition
##

WALL_PROP_NAMES = [
    "medical_cabinet",
    "shelf_set",
    "supply_cabinet",
    "supply_cart_a",
    "supply_cart_b",
    "trash_can",
    "plant_a",
    "plant_b",
]

# Keep coffee_cup and box_portable hidden for the first visual checks.
TABLE_PROP_NAMES = [
    "desk_lamp",
]


def _randomize_room_for_all_envs(env):
    randomize_pickplace_room_layout(
        env,
        torch.arange(env.num_envs, device=env.device),
        wall_prop_names=WALL_PROP_NAMES,
        table_prop_names=TABLE_PROP_NAMES,
        min_table_objects=1,
    )


def _reset_all_then_randomize_room(env):
    env_ids = torch.arange(env.num_envs, device=env.device)
    base_mdp.reset_scene_to_default(env, env_ids)
    randomize_pickplace_room_layout(
        env,
        env_ids,
        wall_prop_names=WALL_PROP_NAMES,
        table_prop_names=TABLE_PROP_NAMES,
        min_table_objects=1,
    )


@configclass
class ObjectTableSceneCfg(RandomizedRoomPickPlaceSceneCfg):
    """object table scene configuration class
    inherits from G1SingleObjectSceneCfg, gets the complete G1 robot scene configuration
    can add task-specific scene elements or override default configurations here
    """
    
    # Humanoid robot w/ arms higher
    # 5. humanoid robot configuration 
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex1_base_fix()


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
    # check if the object is out of the working range
    success = DoneTerm(
        func=mdp.reset_object_estimate,
        params={
            "min_x": ROOM_X_MIN,
            "max_x": ROOM_X_MAX,
            "min_y": ROOM_Y_MIN,
            "max_y": ROOM_Y_MAX,
            "min_height": 0.25,
        },
    )
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

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
        # create event manager
        self.event_manager = SimpleEventManager()

        self.event_manager.register("reset_object_self", SimpleEvent(
            func=_randomize_room_for_all_envs
        ))
        
        self.event_manager.register("reset_all_self", SimpleEvent(
            func=_reset_all_then_randomize_room
        ))
