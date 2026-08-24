# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Hospital-room medicine-bottle pick-place task for G1 + Dex1 joint control.

This is an *independent* task: it does NOT modify any existing warehouse task.
The scene reuses the hospital RoomShell (new_base_room.usda) from
RandomizedRoomPickPlaceSceneCfg. The calibrated task group stays fixed while
the wall props use the shared room randomizer:

  * table / robot / object poses reuse the known-good warehouse manipulation geometry
    translated by T = (-1.7, -3.3, 0) so the table sits in the open hospital
    interior at (-6.0, -7.5). That preserves the robot<->table<->object relative
    poses the IK, cameras and grasp are already calibrated for.
  * wall props are spawned separately and placed from
    tasks/utils/room_randomizer/constants.py on every full reset.

Tunable numbers that may need a visual pass in Isaac Sim are tagged  # TUNE.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import torch
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as base_mdp
from isaaclab.sim import schemas
from isaaclab.sim.spawners.from_files import from_files as file_spawners
from isaaclab.sim.utils import bind_physics_material, clone, get_current_stage
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from tasks.common_config import CameraBaseCfg, CameraPresets, G1RobotPresets  # isort: skip
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_scene.base_scene_randomized_pickplace_cfg import (
    RandomizedRoomPickPlaceSceneCfg,
)
from tasks.utils.room_randomizer import (
    StaticClusterMember,
    TabletopSpawnRegion,
    randomize_pickplace_room_layout,
)
from tasks.utils.room_randomizer.room_events import reset_target_on_current_table
from tasks.utils.room_randomizer.constants import BBox, TablePropMeta
from tasks.utils.room_randomizer.pickplace_config import (
    WALL_PROP_NAMES,
)
from tools.camera_optics import HOSPITAL_FRONT_CAMERA_OPTICS

project_root = os.environ.get("PROJECT_ROOT")

RIDGEBACK_OMNIVERSE_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/"
    "Isaac/Robots/Clearpath/RidgebackUr/ridgeback_ur5.usd"
)
CRATE_OMNIVERSE_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/"
    "Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_D_02.usd"
)
HOSPITAL_PROPS_OMNIVERSE_ROOT = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/"
    "Isaac/Environments/Hospital/Props"
)
OFFICE_PROPS_OMNIVERSE_ROOT = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/"
    "Isaac/Environments/Office/Props"
)
RIDGEBACK_DISABLED_PRIM_NAMES = {
    "ur_arm_shoulder_pan_joint",
    "ur_arm_shoulder_link",
    "ur_arm_upper_arm_link",
    "ur_arm_forearm_link",
    "ur_arm_wrist_1_link",
    "ur_arm_wrist_2_link",
    "ur_arm_wrist_3_link",
}

# Screenshot-derived yaw-only rotations. The small roll/pitch values shown by
# the live physics inspector are transient settling drift, so the task keeps
# the static logistics bases level and preserves their table/crate headings.
RIDGEBACK_LEFT_ROT = (0.2241687075, 0.0, 0.0, -0.9745503530)   # yaw -154.092°
RIDGEBACK_RIGHT_ROT = (0.9745503530, 0.0, 0.0, -0.2241687075)  # mirrored yaw -25.908°
RIDGEBACK_LEFT_YAW_OFFSET = -1.118616424188206
RIDGEBACK_RIGHT_YAW_OFFSET = 1.118616424188206
CRATE_LEFT_ROT = (0.7095584382, 0.0, 0.0, -0.7046465942)      # yaw -89.602°
CRATE_RIGHT_ROT = (0.7095584382, 0.0, 0.0, 0.7046465942)      # mirrored yaw +89.602°


# ----------------------------------------------------------------------------
# Task-local medical asset policy.
# ----------------------------------------------------------------------------
GRASP_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="max",
    restitution_combine_mode="min",
    static_friction=2.5,
    dynamic_friction=2.0,
    restitution=0.0,
)

FINGER_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="max",
    restitution_combine_mode="min",
    static_friction=3.0,
    dynamic_friction=2.0,
    restitution=0.0,
)
FINGER_COLLISION_PROPERTIES = sim_utils.CollisionPropertiesCfg(
    collision_enabled=True,
    contact_offset=0.001,
    rest_offset=0.0,
)
PILL_GRASP_PAD_RADII = {"left": 0.009, "right": 0.010}
DEX1_OPEN_JOINT_POSITION = -0.02
DISTAL_COLLIDER_LOCAL_CENTERS = {
    "Link1_3": (0.001997, 0.022747, -0.014200),
    "Link2_3": (-0.001997, 0.022747, -0.014200),
}


@clone
def _spawn_grasp_ready_dex1_usd(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn Dex1 and harden its exact finger hulls for manipulation."""
    root_prim = file_spawners._spawn_from_usd_file(
        prim_path, cfg.usd_path, cfg, translation, orientation
    )
    stage = get_current_stage()
    material_path = f"{prim_path}/GraspFingerPhysicsMaterial"
    FINGER_PHYSICS_MATERIAL.func(material_path, FINGER_PHYSICS_MATERIAL)

    # The Dex1 USD stores each exact finger hull in an instanced collision
    # branch. Make those branches editable so their contact properties and
    # materials can be overridden without changing the shared robot asset.
    instance_paths = [
        str(child.GetPath())
        for child in Usd.PrimRange(root_prim)
        if child.IsInstance()
        and (
            "/left_hand_Link" in str(child.GetPath())
            or "/right_hand_Link" in str(child.GetPath())
        )
    ]
    for child_path in instance_paths:
        stage.GetPrimAtPath(child_path).SetInstanceable(False)

    finger_branch_paths = [
        str(child.GetPath())
        for child in Usd.PrimRange(stage.GetPrimAtPath(prim_path))
        if child.GetName() == "collisions"
        and (
            "/left_hand_Link" in str(child.GetPath())
            or "/right_hand_Link" in str(child.GetPath())
        )
    ]

    finger_collider_paths: list[str] = []
    for child_path in finger_branch_paths:
        child = stage.GetPrimAtPath(child_path)

        branch_meshes = [
            descendant
            for descendant in Usd.PrimRange(child)
            if descendant.IsA(UsdGeom.Mesh)
        ]
        if len(branch_meshes) != 1:
            raise ValueError(
                f"Dex1 finger collision branch {child_path!r} must contain exactly one mesh; "
                f"found {[str(mesh.GetPath()) for mesh in branch_meshes]}"
            )

        mesh_prim = branch_meshes[0]
        if mesh_prim.IsInstanceProxy():
            raise ValueError(
                f"Dex1 finger collider remained instanced: {mesh_prim.GetPath()}"
            )

        UsdPhysics.CollisionAPI.Apply(child).CreateCollisionEnabledAttr().Set(True)
        UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr().Set(
            UsdPhysics.Tokens.convexHull
        )
        PhysxSchema.PhysxCollisionAPI.Apply(child)
        finger_collider_paths.append(child_path)

    per_hand_counts = {
        side: sum(f"/{side}_hand_Link" in path for path in finger_collider_paths)
        for side in ("left", "right")
    }
    if per_hand_counts != {"left": 6, "right": 6}:
        raise ValueError(
            f"Dex1 must expose six exact collision hulls per hand; found {per_hand_counts}"
        )

    # Natural-scale pill bottles are narrower than the stock Dex1 aperture at
    # its mapped full-close target. Add invisible spherical pads at the exact
    # distal-collider centers so the existing command can pinch them without
    # resizing either source asset or changing any controller mapping.
    grasp_pad_paths: list[str] = []
    for side in ("left", "right"):
        for link_name, local_center in DISTAL_COLLIDER_LOCAL_CENTERS.items():
            pad_path = f"{prim_path}/{side}_hand_{link_name}/PillGraspPad"
            pad = UsdGeom.Sphere.Define(stage, pad_path)
            pad.CreateRadiusAttr().Set(PILL_GRASP_PAD_RADII[side])
            UsdGeom.Xformable(pad.GetPrim()).AddTranslateOp().Set(
                Gf.Vec3d(*local_center)
            )
            UsdPhysics.CollisionAPI.Apply(pad.GetPrim()).CreateCollisionEnabledAttr().Set(
                True
            )
            PhysxSchema.PhysxCollisionAPI.Apply(pad.GetPrim())
            grasp_pad_paths.append(pad_path)

    for collider_path in finger_collider_paths + grasp_pad_paths:
        schemas.modify_collision_properties(
            collider_path, FINGER_COLLISION_PROPERTIES, stage=stage
        )
        bind_physics_material(collider_path, material_path, stage=stage)

    return root_prim


@clone
def _spawn_graspable_hospital_usd(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Turn one NVIDIA prop mesh into a convex-decomposed dynamic rigid body."""
    spawn_cfg = cfg.copy()
    spawn_cfg.rigid_props = None
    spawn_cfg.collision_props = None
    spawn_cfg.mass_props = None
    root_prim = file_spawners._spawn_from_usd_file(
        prim_path, cfg.usd_path, spawn_cfg, translation, orientation
    )
    stage = get_current_stage()

    collider_prims = [
        child for child in Usd.PrimRange(root_prim) if child.IsA(UsdGeom.Mesh)
    ]
    if len(collider_prims) != 1:
        raise ValueError(
            f"Hospital grasp prop {cfg.usd_path!r} must contain exactly one mesh; "
            f"found {[str(child.GetPath()) for child in collider_prims]}"
        )

    collider_prim = collider_prims[0]
    UsdPhysics.CollisionAPI.Apply(collider_prim).CreateCollisionEnabledAttr().Set(True)
    UsdPhysics.MeshCollisionAPI.Apply(collider_prim).CreateApproximationAttr().Set(
        UsdPhysics.Tokens.convexDecomposition
    )
    PhysxSchema.PhysxCollisionAPI.Apply(collider_prim)
    collider_path = str(collider_prim.GetPath())

    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)
    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props, stage=stage)

    rigid_roots = [
        child
        for child in Usd.PrimRange(root_prim)
        if child.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    authored_mass = UsdPhysics.MassAPI(root_prim).GetMassAttr().Get()
    if rigid_roots != [root_prim] or authored_mass is None or float(authored_mass) <= 0.0:
        raise ValueError(
            f"Hospital grasp prop {cfg.usd_path!r} requires one positive-mass root rigid body"
        )

    material_path = f"{prim_path}/GraspPhysicsMaterial"
    GRASP_PHYSICS_MATERIAL.func(material_path, GRASP_PHYSICS_MATERIAL)
    if cfg.collision_props is not None:
        schemas.modify_collision_properties(
            collider_path, cfg.collision_props, stage=stage
        )
    bind_physics_material(collider_path, material_path, stage=stage)

    return root_prim


@configclass
class GraspableHospitalUsdFileCfg(sim_utils.UsdFileCfg):
    """USD spawner configuration for convex-decomposed NVIDIA grasp props."""

    func: Callable = _spawn_graspable_hospital_usd


@clone
def _spawn_static_ridgeback_usd(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Import the Omniverse Ridgeback mesh as one base-only kinematic body."""
    spawn_cfg = cfg.copy()
    spawn_cfg.rigid_props = None
    spawn_cfg.collision_props = None
    root_prim = file_spawners._spawn_from_usd_file(
        prim_path, cfg.usd_path, spawn_cfg, translation, orientation
    )
    stage = get_current_stage()

    for child in list(Usd.PrimRange(root_prim)):
        if child.IsA(UsdPhysics.Joint):
            # Keep the referenced prim alive but make it a plain transform.
            # Deactivating/removing a joint leaves expired handles in Isaac's
            # stage parser; jointEnabled=false is still parsed as a joint.
            child.SetTypeName("Xform")
        if child.HasAPI(UsdPhysics.ArticulationRootAPI):
            child.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if child.HasAPI(PhysxSchema.PhysxArticulationAPI):
            child.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        if child != root_prim:
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                child.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if child.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                child.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)

        ancestor = child
        belongs_to_arm = False
        while ancestor.IsValid() and ancestor != root_prim:
            if ancestor.GetName() in RIDGEBACK_DISABLED_PRIM_NAMES:
                belongs_to_arm = True
                break
            ancestor = ancestor.GetParent()
        if belongs_to_arm:
            imageable = UsdGeom.Imageable(child)
            if imageable:
                imageable.MakeInvisible()
            if child.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr().Set(False)

    if not any(child.IsA(UsdGeom.Mesh) for child in Usd.PrimRange(root_prim)):
        raise ValueError(f"Ridgeback asset {cfg.usd_path!r} has no active mesh geometry")
    if not any(
        child.HasAPI(UsdPhysics.CollisionAPI) for child in Usd.PrimRange(root_prim)
    ):
        raise ValueError(f"Ridgeback asset {cfg.usd_path!r} has no collision geometry")

    schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)
    if cfg.collision_props is not None:
        schemas.modify_collision_properties(prim_path, cfg.collision_props, stage=stage)
    return root_prim


def _static_ridgeback_cfg(
    prim_path: str,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            func=_spawn_static_ridgeback_usd,
            usd_path=RIDGEBACK_OMNIVERSE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
                linear_damping=10.0,
                angular_damping=10.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=pos,
            rot=rot,
        ),
    )


@clone
def _spawn_static_crate_usd(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Import an Omniverse crate as mesh/collision children of its platform."""
    spawn_cfg = cfg.copy()
    spawn_cfg.rigid_props = None
    root_prim = file_spawners._spawn_from_usd_file(
        prim_path, cfg.usd_path, spawn_cfg, translation, orientation
    )

    # The crate is deliberately not a second physics body. Its mesh and
    # collision geometry inherit the Ridgeback base's kinematic transform.
    for child in list(Usd.PrimRange(root_prim)):
        if child.IsA(UsdPhysics.Joint):
            child.SetTypeName("Xform")
        if child.HasAPI(UsdPhysics.ArticulationRootAPI):
            child.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if child.HasAPI(PhysxSchema.PhysxArticulationAPI):
            child.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            child.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if child.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            child.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)

    if not any(child.IsA(UsdGeom.Mesh) for child in Usd.PrimRange(root_prim)):
        raise ValueError(f"Crate asset {cfg.usd_path!r} has no active mesh geometry")
    if not any(
        child.HasAPI(UsdPhysics.CollisionAPI) for child in Usd.PrimRange(root_prim)
    ):
        raise ValueError(f"Crate asset {cfg.usd_path!r} has no collision geometry")
    return root_prim


def _ridgeback_crate_cfg(
    prim_path: str,
    local_x: float,
    local_y: float,
    local_z: float,
    local_rot: tuple[float, float, float, float],
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            func=_spawn_static_crate_usd,
            usd_path=CRATE_OMNIVERSE_USD,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            # Screenshot-2026-08-23_14-03-34 records the crate centered on
            # RidgebackLeft's deck. Its local-Y offset is reflected for the
            # mirrored right platform, preserving a world-X mirror across G1
            # while keeping both crates deck-centered and table-aligned.
            pos=(local_x, local_y, local_z),
            rot=local_rot,
        ),
    )


@dataclass(frozen=True)
class HospitalPropSpec:
    filename: str
    asset_root: str
    scale: float
    mass: float
    min_z: float
    bbox: BBox
    grasp_width: float


TABLETOP_SURFACE_Z = 0.794
TABLETOP_SPAWN_CLEARANCE = 0.003
HOSPITAL_PROP_SPECS = {
    # Exact natural-scale selection from Screenshot-2026-08-23_16-05-21.png.
    # The two small pill bottles are the scored targets; the other four remain
    # fully dynamic, collision-enabled tabletop clutter.
    "pill_bottle_t": HospitalPropSpec(
        filename="SM_PillBottle_01t.usd",
        asset_root=HOSPITAL_PROPS_OMNIVERSE_ROOT,
        scale=1.0,
        mass=0.03,
        min_z=0.0,
        bbox=BBox(half_w=0.014611, half_d=0.014611),
        grasp_width=0.029222,
    ),
    "pill_bottle_v": HospitalPropSpec(
        filename="SM_PillBottle_01v.usd",
        asset_root=HOSPITAL_PROPS_OMNIVERSE_ROOT,
        scale=1.0,
        mass=0.03,
        min_z=0.0,
        bbox=BBox(half_w=0.013665, half_d=0.013665),
        grasp_width=0.027330,
    ),
    "medical_bottle_a": HospitalPropSpec(
        filename="SM_BottleA.usd",
        asset_root=HOSPITAL_PROPS_OMNIVERSE_ROOT,
        scale=1.0,
        mass=0.15,
        min_z=0.0,
        bbox=BBox(half_w=0.031000, half_d=0.031000),
        grasp_width=0.062000,
    ),
    "medical_bottle_f": HospitalPropSpec(
        filename="SM_BottleF.usd",
        asset_root=HOSPITAL_PROPS_OMNIVERSE_ROOT,
        scale=1.0,
        mass=0.15,
        min_z=0.0,
        bbox=BBox(half_w=0.030050, half_d=0.030050),
        grasp_width=0.060100,
    ),
    "marker_blue": HospitalPropSpec(
        filename="SM_MarkerBlue.usd",
        asset_root=OFFICE_PROPS_OMNIVERSE_ROOT,
        scale=1.0,
        mass=0.02,
        min_z=0.000130,
        bbox=BBox(half_w=0.058500, half_d=0.012140),
        grasp_width=0.024280,
    ),
    "marker_yellow": HospitalPropSpec(
        filename="SM_MarkerYellow.usd",
        asset_root=OFFICE_PROPS_OMNIVERSE_ROOT,
        scale=1.0,
        mass=0.02,
        min_z=0.000130,
        bbox=BBox(half_w=0.058500, half_d=0.012140),
        grasp_width=0.024280,
    ),
}

MEDICINE_BOTTLE_TABLE_PROP_META_OVERRIDES = {
    name: TablePropMeta(bbox=spec.bbox, dynamic=True, mandatory=True)
    for name, spec in HOSPITAL_PROP_SPECS.items()
}


def _hospital_prop_cfg(
    name: str,
    prim_path: str,
    init_xy: tuple[float, float],
) -> RigidObjectCfg:
    spec = HOSPITAL_PROP_SPECS[name]
    init_z = (
        TABLETOP_SURFACE_Z
        - spec.min_z * spec.scale
        + TABLETOP_SPAWN_CLEARANCE
    )
    return RigidObjectCfg(
        prim_path=prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(init_xy[0], init_xy[1], init_z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=GraspableHospitalUsdFileCfg(
            usd_path=f"{spec.asset_root}/{spec.filename}",
            scale=(spec.scale, spec.scale, spec.scale),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=1.5,
                angular_damping=3.0,
                max_linear_velocity=5.0,
                max_angular_velocity=10.0,
                max_depenetration_velocity=0.25,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.001,
                rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=spec.mass),
        ),
    )


# ----------------------------------------------------------------------------
# Fixed-layout world coordinates (hospital room interior).
# Derived from the warehouse task translated by T = (-1.7, -3.3, 0).
# ----------------------------------------------------------------------------
TABLE_POS = (-6.0, -7.5, -0.2)          # TUNE: open interior of new_base_room.usda
ROBOT_POS = (-5.9, -7.0, 0.76)          # +Y of table by 0.5 m, same as warehouse
ROBOT_ROT = (0.7071, 0.0, 0.0, -0.7071)  # yaw -90deg, robot faces the table

# Fixed-layout pose tuned in Isaac Sim. The left pose is the transform captured
# in Screenshot-2026-08-23_14-03-10.png; the right pose is its reflection across
# G1's x=-5.9 centerline. Keep the robot-local offsets below in sync so room
# resets preserve the same mirrored arrangement around G1.
RIDGEBACK_LEFT_POS = (-5.20520, -6.66770, 0.0328)
RIDGEBACK_RIGHT_POS = (-6.59480, -6.66770, 0.0328)
STATIC_LOGISTICS_CLUSTER = (
    StaticClusterMember(
        asset_name="ridgeback_left",
        # Screenshot-2026-08-23_14-03-10: mirrored, 90-degree yawed
        # Ridgebacks. The 1.38960 m center separation leaves positive physical
        # clearance after projecting their table-facing OBBs onto world X.
        robot_local_xy=(-0.33230, 0.69480),
        yaw_offset=RIDGEBACK_LEFT_YAW_OFFSET,
        bbox=BBox(half_w=0.50, half_d=0.42),
    ),
    StaticClusterMember(
        asset_name="ridgeback_right",
        robot_local_xy=(-0.33230, -0.69480),
        yaw_offset=RIDGEBACK_RIGHT_YAW_OFFSET,
        bbox=BBox(half_w=0.50, half_d=0.42),
    ),
)
HAND_REACHABLE_TABLETOP_REGION = TabletopSpawnRegion(
    # G1 is at table-local (0.10, 0.50), facing toward decreasing Y.
    # These bounds sit directly in front of the two default Dex1 jaws and keep
    # every footprint inside the 96-degree front-camera horizontal FOV.
    x_min=-0.32,
    x_max=0.18,
    y_min=-0.16,
    y_max=0.12,
)
COMPACT_TABLETOP_OBJECT_MARGIN = 0.015

MEDICINE_BOTTLE_TABLE_PROP_NAMES = [
    "pill_bottle_t",
    "pill_bottle_v",
    "medical_bottle_a",
    "medical_bottle_f",
    "marker_blue",
    "marker_yellow",
]


def _apply_hospital_front_camera_optics(env) -> None:
    """Keep the streamed USD camera at the optics saved from the Isaac viewport."""
    stage = get_current_stage()
    optics = HOSPITAL_FRONT_CAMERA_OPTICS
    vertical_aperture = optics.horizontal_aperture * optics.height / optics.width

    for env_id in range(env.num_envs):
        camera_path = f"/World/envs/env_{env_id}/Robot/d435_link/front_cam"
        camera_prim = stage.GetPrimAtPath(camera_path)
        if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
            raise RuntimeError(f"Front camera prim is missing or invalid: {camera_path}")

        usd_camera = UsdGeom.Camera(camera_prim)
        usd_camera.GetFocalLengthAttr().Set(optics.focal_length)
        usd_camera.GetFocusDistanceAttr().Set(optics.focus_distance)
        usd_camera.GetHorizontalApertureAttr().Set(optics.horizontal_aperture)
        usd_camera.GetVerticalApertureAttr().Set(vertical_aperture)

    if not getattr(env, "_hospital_front_camera_optics_reported", False):
        print(
            "[front_camera] enforced streamed POV: "
            f"{optics.width}x{optics.height}, focal={optics.focal_length}, "
            f"horizontal_aperture={optics.horizontal_aperture}, "
            f"h_fov={optics.horizontal_fov_degrees:.3f}, "
            f"v_fov={optics.vertical_fov_degrees:.3f}",
            flush=True,
        )
        env._hospital_front_camera_optics_reported = True


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
    _apply_hospital_front_camera_optics(env)
    randomize_pickplace_room_layout(
        env,
        env_ids,
        wall_prop_names=WALL_PROP_NAMES,
        table_prop_names=MEDICINE_BOTTLE_TABLE_PROP_NAMES,
        min_table_objects=len(MEDICINE_BOTTLE_TABLE_PROP_NAMES),
        randomize_table_position=randomize_table_position,
        table_prop_meta_overrides=MEDICINE_BOTTLE_TABLE_PROP_META_OVERRIDES,
        tabletop_spawn_region=HAND_REACHABLE_TABLETOP_REGION,
        tabletop_object_margin=COMPACT_TABLETOP_OBJECT_MARGIN,
        static_cluster_members=STATIC_LOGISTICS_CLUSTER,
    )

    mode = "full randomization" if randomize_table_position else "fixed table"
    print(f"[Meta Quest reset] hospital scene restored ({mode})", flush=True)


def reset_hospital_tabletop_props(env) -> None:
    """Respawn all six props without moving the table, robot, room, or bins."""
    env_ids = torch.arange(env.num_envs, device=env.device)
    for asset_name in MEDICINE_BOTTLE_TABLE_PROP_NAMES:
        reset_target_on_current_table(
            env,
            env_ids,
            asset_name=asset_name,
            table_prop_meta_overrides=MEDICINE_BOTTLE_TABLE_PROP_META_OVERRIDES,
            tabletop_spawn_region=HAND_REACHABLE_TABLETOP_REGION,
            tabletop_object_margin=COMPACT_TABLETOP_OBJECT_MARGIN,
        )
    print("[Meta Quest reset] all six tabletop props respawned", flush=True)


def reset_hospital_room_fixed_table(env) -> None:
    """Scramble the hospital room while restoring the authored table anchor."""
    env_ids = torch.arange(env.num_envs, device=env.device)
    env._teleop_randomize_table_position = False
    base_mdp.reset_scene_to_default(env, env_ids)
    _apply_hospital_front_camera_optics(env)
    randomize_pickplace_room_layout(
        env,
        env_ids,
        wall_prop_names=WALL_PROP_NAMES,
        table_prop_names=MEDICINE_BOTTLE_TABLE_PROP_NAMES,
        min_table_objects=len(MEDICINE_BOTTLE_TABLE_PROP_NAMES),
        randomize_table_position=False,
        table_prop_meta_overrides=MEDICINE_BOTTLE_TABLE_PROP_META_OVERRIDES,
        tabletop_spawn_region=HAND_REACHABLE_TABLETOP_REGION,
        tabletop_object_margin=COMPACT_TABLETOP_OBJECT_MARGIN,
        static_cluster_members=STATIC_LOGISTICS_CLUSTER,
    )
    print("[Meta Quest reset] hospital room scrambled (fixed table)", flush=True)


##
# Scene definition
##
@configclass
class HospitalMedicineBottleSceneCfg(RandomizedRoomPickPlaceSceneCfg):
    """Hospital table/G1 scene with a bottle and static logistics platforms."""

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

    # Exact six-prop selection from the latest hospital screenshot. Every
    # source mesh is a positive-mass rigid body with convex decomposition.
    object = None
    pill_bottle_t: RigidObjectCfg = _hospital_prop_cfg(
        "pill_bottle_t",
        prim_path="/World/envs/env_.*/PillBottleT",
        init_xy=(-6.02, -7.42),
    )
    pill_bottle_v: RigidObjectCfg = _hospital_prop_cfg(
        "pill_bottle_v",
        prim_path="/World/envs/env_.*/PillBottleV",
        init_xy=(-6.15, -7.42),
    )
    medical_bottle_a: RigidObjectCfg = _hospital_prop_cfg(
        "medical_bottle_a",
        prim_path="/World/envs/env_.*/MedicalBottleA",
        init_xy=(-6.27, -7.42),
    )
    medical_bottle_f: RigidObjectCfg = _hospital_prop_cfg(
        "medical_bottle_f",
        prim_path="/World/envs/env_.*/MedicalBottleF",
        init_xy=(-5.87, -7.42),
    )
    marker_blue: RigidObjectCfg = _hospital_prop_cfg(
        "marker_blue",
        prim_path="/World/envs/env_.*/MarkerBlue",
        init_xy=(-6.22, -7.58),
    )
    marker_yellow: RigidObjectCfg = _hospital_prop_cfg(
        "marker_yellow",
        prim_path="/World/envs/env_.*/MarkerYellow",
        init_xy=(-5.98, -7.58),
    )

    blue_cube = None
    yellow_cube = None

    # Two static Omniverse Ridgeback meshes flank G1.  Each crate is authored
    # below base_link, so it inherits its platform pose on every room reset.
    ridgeback_left: RigidObjectCfg = _static_ridgeback_cfg(
        "/World/envs/env_.*/RidgebackLeft",
        RIDGEBACK_LEFT_POS,
        RIDGEBACK_LEFT_ROT,
    )
    ridgeback_left_crate: AssetBaseCfg = _ridgeback_crate_cfg(
        "/World/envs/env_.*/RidgebackLeft/base_link/Crate",
        local_x=0.22877,
        local_y=0.00612,
        local_z=0.28576,
        local_rot=CRATE_LEFT_ROT,
    )
    ridgeback_right: RigidObjectCfg = _static_ridgeback_cfg(
        "/World/envs/env_.*/RidgebackRight",
        RIDGEBACK_RIGHT_POS,
        RIDGEBACK_RIGHT_ROT,
    )
    ridgeback_right_crate: AssetBaseCfg = _ridgeback_crate_cfg(
        "/World/envs/env_.*/RidgebackRight/base_link/Crate",
        local_x=0.22877,
        local_y=-0.00612,
        local_z=0.28576,
        local_rot=CRATE_RIGHT_ROT,
    )

    # --- robot + cameras ---------------------------------------------------
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex1_base_fix(
        init_pos=ROBOT_POS, init_rot=ROBOT_ROT
    )
    # Both fingers of both hands begin fully open. Scene resets restore this
    # pose; an index-trigger press then commands the matching pair closed.
    robot.init_state = robot.init_state.replace(
        joint_pos={
            **robot.init_state.joint_pos,
            "left_hand_Joint1_1": DEX1_OPEN_JOINT_POSITION,
            "left_hand_Joint2_1": DEX1_OPEN_JOINT_POSITION,
            "right_hand_Joint1_1": DEX1_OPEN_JOINT_POSITION,
            "right_hand_Joint2_1": DEX1_OPEN_JOINT_POSITION,
        }
    )
    robot.spawn = robot.spawn.replace(func=_spawn_grasp_ready_dex1_usd)
    # This task alone releases waist yaw and pitch for the Quest right stick:
    # yaw turns toward either rear crate and pitch lowers the hands over it.
    # Keep waist roll and the complete fixed lower body locked.
    robot.actuators = dict(robot.actuators)
    # The Quest index trigger is a binary hold/release command. The low drive
    # friction lets the authored jaw stroke finish visibly instead of stalling
    # halfway; speed and force remain capped for stable lightweight-prop contact.
    robot.actuators["hands"] = ImplicitActuatorCfg(
        joint_names_expr=[
            "left_hand_Joint1_1",
            "left_hand_Joint2_1",
            "right_hand_Joint1_1",
            "right_hand_Joint2_1",
        ],
        effort_limit_sim=12.0,
        velocity_limit_sim=0.5,
        stiffness=600.0,
        damping=8.0,
        friction=0.0,
    )
    robot.actuators.pop("waist", None)
    robot.actuators["waist_yaw_pitch_teleop"] = ImplicitActuatorCfg(
        joint_names_expr=["waist_yaw_joint", "waist_pitch_joint"],
        effort_limit_sim=350.0,
        velocity_limit_sim=2.5,
        stiffness=260.0,
        damping=18.0,
    )
    robot.actuators["waist_roll_lock"] = ImplicitActuatorCfg(
        joint_names_expr=["waist_roll_joint"],
        effort_limit_sim=1000.0,
        velocity_limit_sim=0.1,
        stiffness=10000.0,
        damping=10000.0,
    )
    # Optics tuned on /Robot/d435_link/front_cam in the Isaac Sim viewport
    # (Screenshot-2026-08-22_15-38-27.png). The default mount transform already
    # matches the captured translate=(0,0,0), Euler=(90,-90,0) pose.
    front_camera = CameraBaseCfg.get_camera_config(
        height=HOSPITAL_FRONT_CAMERA_OPTICS.height,
        width=HOSPITAL_FRONT_CAMERA_OPTICS.width,
        focal_length=HOSPITAL_FRONT_CAMERA_OPTICS.focal_length,
        focus_distance=HOSPITAL_FRONT_CAMERA_OPTICS.focus_distance,
        horizontal_aperture=HOSPITAL_FRONT_CAMERA_OPTICS.horizontal_aperture,
    )
    left_wrist_camera = CameraPresets.left_gripper_wrist_camera()
    right_wrist_camera = CameraPresets.right_gripper_wrist_camera()

    # spectator camera (GUI only): warehouse view translated to look at the table
    world_camera = CameraBaseCfg.get_camera_config(
        prim_path="/World/PerspectiveCamera",
        pos_offset=(-5.8, -8.2, 1.8),       # TUNE: GUI-only viewpoint
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
    # Both pill bottles may occupy either rear crate, including the same crate.
    success = DoneTerm(func=mdp.both_pill_bottles_contained)


@configclass
class RewardsCfg:
    reward = RewTerm(
        func=mdp.compute_pill_bottle_reward,
        weight=1.0,
    )


@configclass
class EventCfg:
    reset_teleop_scene = EventTermCfg(
        func=reset_hospital_teleop_scene,
        mode="reset",
    )


@configclass
class PickPlaceMedicineBottleHospitalG129DEX1EnvCfg(ManagerBasedRLEnvCfg):
    """G1 + Dex1 medicine-bottle pick-place in a randomized hospital room."""

    scene: HospitalMedicineBottleSceneCfg = HospitalMedicineBottleSceneCfg(
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
        self.sim.physx.friction_correlation_distance = 0.00625
        # GPU rigid-body CCD is unsupported in this Isaac Sim configuration;
        # bounded jaw velocity and convex-decomposed contacts prevent tunneling.
        self.sim.physx.enable_ccd = False
        self.sim.physx.gpu_constraint_solver_heavy_spring_enabled = True
        self.sim.physx.num_substeps = 2
        # The scored props are about 27--29 mm across. A 15 mm contact skin
        # made opposing jaw contacts appear far too early and injected large
        # depenetration impulses.  Keep the broad default and the task-local
        # finger/object offsets in the low-millimetre range.
        self.sim.physx.contact_offset = 0.003
        self.sim.physx.rest_offset = 0.0
        self.sim.physx.num_position_iterations = 16
        self.sim.physx.num_velocity_iterations = 4

        # Custom event manager (matches the warehouse manipulation task).
        self.event_manager = SimpleEventManager()
        self.event_manager.register("reset_object_self", SimpleEvent(
            func=reset_hospital_tabletop_props
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
