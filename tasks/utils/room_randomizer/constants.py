# constants.py
# Room placement constants for the hospital room environment.
# Uses oriented bounding boxes (OBB) and continuous wall zones.

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
ROOM_SHELL_USD = os.path.join(PROJECT_ROOT, "isaac-projects", "new_base_room.usda")

# ============================================================
# Room geometry
# ============================================================

ROOM_X_MIN = -13.0
ROOM_X_MAX = -0.5
ROOM_Y_MIN = -11.25
ROOM_Y_MAX = -5.0
FLOOR_Z = 0.0

# Default stand positions
ROBOT_Z = 0.76         # pelvis default height for base-fixed G1
TABLE_Z = -0.2         # packing table Z position
DESK_OBJECT_Z = 0.84   # object height on desk

# Wall surface positions (room-facing edge of each wall).
BACK_WALL_LINE_Y = -10.95
FRONT_WALL_LINE_Y = -5.47
RIGHT_WALL_LINE_X = -2.5

# ============================================================
# Bounding box primitives
# ============================================================

@dataclass(frozen=True)
class BBox:
    """2D oriented bounding box footprint (local frame).
    half_w: half-extent along the object's local X axis (width).
    half_d: half-extent along the object's local Y axis (depth).
    """
    half_w: float
    half_d: float


@dataclass(frozen=True)
class StaticRoomObstacle:
    """Static room-shell footprint that the table group must avoid."""
    name: str
    center: Tuple[float, float]
    bbox: BBox
    yaw: float = 0.0

# ============================================================
# Wall zones — continuous strips where wall props can be placed
# ============================================================

@dataclass(frozen=True)
class WallZone:
    """A continuous strip along a wall where props sample positions.
    For the back wall:  the free axis is X, fixed axis is Y.
    For the right wall: the free axis is Y, fixed axis is X.
    """
    wall: str           # "back" or "right"
    sample_min: float   # min of the free axis (X for back, Y for right)
    sample_max: float   # max of the free axis
    fixed_coord: float  # centre position on the constrained axis
    base_yaw: float     # yaw to face into the room (radians)

WALL_ZONES: List[WallZone] = [
    # Back wall: props slide along X, fixed near back wall Y.
    WallZone(
        wall="back",
        sample_min=-12.0,
        sample_max=-4.0,
        fixed_coord=-10.75,  # prop center Y
        base_yaw=0.0,        # face into room (+Y direction)
    ),
    # Right wall: props slide along Y, fixed near right wall X.
    WallZone(
        wall="right",
        sample_min=-10.0,
        sample_max=-7.0,
        fixed_coord=-3.0,    # prop center X
        base_yaw=math.pi / 2,  # face into room (-X direction)
    ),
]

# Static RoomShell wall footprints used by the OBB planner. These represent
# authored USD wall geometry that PhysX collides with but the randomizer would
# otherwise not know about.
STATIC_ROOM_OBSTACLES: List[StaticRoomObstacle] = [
    StaticRoomObstacle(
        name="static_front_wall",
        center=(-7.5, -5.47),
        bbox=BBox(half_w=5.35, half_d=0.12),
    ),
    StaticRoomObstacle(
        name="static_right_wall",
        center=(-2.5, -8.15),
        bbox=BBox(half_w=0.12, half_d=3.15),
    ),
    StaticRoomObstacle(
        name="static_back_wall",
        center=(-8.35, -10.95),
        bbox=BBox(half_w=4.35, half_d=0.12),
    ),
]

# ============================================================
# Room interior sampling zone (for table group)
# ============================================================

TABLE_SAMPLE_X_MIN = -10.0
TABLE_SAMPLE_X_MAX = -5.0
TABLE_SAMPLE_Y_MIN = -9.0
TABLE_SAMPLE_Y_MAX = -6.0
TABLE_FALLBACK_X = -7.50
TABLE_FALLBACK_Y = -7.50
TABLE_GROUP_MAX_TRIES = 300

# Stricter usable area for the table group. Unlike ROOM_* these bounds model
# where the robot/table may operate, including virtual limits on open sides.
TABLE_GROUP_X_MIN = -12.35
TABLE_GROUP_X_MAX = RIGHT_WALL_LINE_X - 0.35
TABLE_GROUP_Y_MIN = BACK_WALL_LINE_Y + 0.35
TABLE_GROUP_Y_MAX = FRONT_WALL_LINE_Y - 0.35

# Robot-facing targets are intentionally narrower than the full room bounds.
# The low-X side is the empty/open wall, so front/back target points stay away
# from that opening and the open side is never a selectable facing layout.
OPEN_WALL_NAMES = ("left_open",)
FRONT_BACK_WALL_TARGET_X_MIN = -10.0
FRONT_BACK_WALL_TARGET_X_MAX = -5.0
RIGHT_WALL_TARGET_Y_MIN = -9.0
RIGHT_WALL_TARGET_Y_MAX = -6.0
ROBOT_FACING_MAX_YAW_OFFSET_RAD = math.radians(15.0)


@dataclass(frozen=True)
class RobotFacingLayout:
    """Robot/table placement mode that keeps the robot facing a room wall.

    If target_axis is "y", wall points are sampled as (sample, fixed_coord).
    If target_axis is "x", wall points are sampled as (fixed_coord, sample).
    """
    name: str
    wall: str
    target_axis: str
    fixed_coord: float
    sample_min: float
    sample_max: float
    yaw_center: float


ROBOT_FACING_LAYOUTS: List[RobotFacingLayout] = [
    # Robot sees the front wall, with the back wall behind it.
    RobotFacingLayout(
        name="face_front_wall",
        wall="front",
        target_axis="y",
        fixed_coord=FRONT_WALL_LINE_Y,
        sample_min=FRONT_BACK_WALL_TARGET_X_MIN,
        sample_max=FRONT_BACK_WALL_TARGET_X_MAX,
        yaw_center=math.pi / 2,
    ),
    # Robot sees the back wall, with the front wall behind it.
    RobotFacingLayout(
        name="face_back_wall",
        wall="back",
        target_axis="y",
        fixed_coord=BACK_WALL_LINE_Y,
        sample_min=FRONT_BACK_WALL_TARGET_X_MIN,
        sample_max=FRONT_BACK_WALL_TARGET_X_MAX,
        yaw_center=-math.pi / 2,
    ),
    # Robot sees the side wall at the high-X edge of the room.
    RobotFacingLayout(
        name="face_right_wall",
        wall="right",
        target_axis="x",
        fixed_coord=RIGHT_WALL_LINE_X,
        sample_min=RIGHT_WALL_TARGET_Y_MIN,
        sample_max=RIGHT_WALL_TARGET_Y_MAX,
        yaw_center=0.0,
    ),
]


def _validate_robot_facing_layouts() -> None:
    """Guard against accidentally aiming the robot at the empty/open side."""
    for layout in ROBOT_FACING_LAYOUTS:
        if layout.wall in OPEN_WALL_NAMES:
            raise ValueError(f"robot facing layout targets open wall: {layout}")
        if layout.sample_min >= layout.sample_max:
            raise ValueError(f"invalid wall target span: {layout}")
        if layout.target_axis == "y":
            if not (ROOM_Y_MIN <= layout.fixed_coord <= ROOM_Y_MAX):
                raise ValueError(f"wall target y outside room: {layout}")
            if layout.sample_min < ROOM_X_MIN or layout.sample_max > ROOM_X_MAX:
                raise ValueError(f"wall target x span outside room: {layout}")
        elif layout.target_axis == "x":
            if not (ROOM_X_MIN <= layout.fixed_coord <= ROOM_X_MAX):
                raise ValueError(f"wall target x outside room: {layout}")
            if layout.sample_min < ROOM_Y_MIN or layout.sample_max > ROOM_Y_MAX:
                raise ValueError(f"wall target y span outside room: {layout}")
        else:
            raise ValueError(f"invalid wall target axis: {layout}")


_validate_robot_facing_layouts()

# Keep this at zero for strict wall-facing placement. Increase slightly only if
# training needs a controlled yaw perturbation around the selected wall layout.
ROBOT_FACING_YAW_JITTER_RAD = 0.0

# ============================================================
# Packing Table geometry (cloned desk)
# ============================================================

# Surface bounds relative to packing table center
DESK_LOCAL_X_MIN = -0.4
DESK_LOCAL_X_MAX = 0.4
DESK_LOCAL_Y_MIN = -0.2
DESK_LOCAL_Y_MAX = 0.2
DESK_OBJECT_MARGIN = 0.05   # margin between tabletop OBBs

# Regular pick/place local transforms relative to the packing table center.
# These match TableCylinderSceneCfg: robot=(-0.15, 0.0), table=(0.0, 0.55),
# object=(-0.35, 0.40) when the table yaw is zero.
ROBOT_ORBIT_OFFSET = (-0.15, -0.55)
OBJECT_TABLE_LOCAL_OFFSET = (-0.35, -0.15)

# Despawn height
DESPAWN_Z = -100.0

# Margin added around every OBB during placement checks (metres)
OBB_PLACEMENT_MARGIN = 0.15

# ============================================================
# Wall prop metadata
# ============================================================

@dataclass(frozen=True)
class WallPropMeta:
    """Placement metadata for a wall prop."""
    usd_name: str
    bbox: BBox              # footprint in the object's local frame
    bbox_center: Tuple[float, float] = (0.0, 0.0)
    tall: bool = False
    wall_offset: float = 0.0  # extra push away from wall surface (metres)
    yaw_offset: float = 0.0   # yaw adjustment relative to wall base yaw (radians)
    allowed_walls: Tuple[str, ...] = ("back", "right")

WALL_PROP_META: Dict[str, WallPropMeta] = {
    "medical_cabinet": WallPropMeta(
        "SM_MedicalCabinet_01a",
        bbox=BBox(half_w=0.436, half_d=0.328),
        bbox_center=(0.415679, 0.303706),
        tall=True,
        wall_offset=0.25,
        yaw_offset=math.pi,
        allowed_walls=("right",),
    ),
    "shelf_set": WallPropMeta(
        "SM_ShelfSet_01a",
        bbox=BBox(half_w=0.861, half_d=0.280),
        tall=True,
        wall_offset=-0.220,
        yaw_offset=math.pi,
        allowed_walls=("right",),
    ),
    "supply_cabinet": WallPropMeta(
        "SM_SupplyCabinet_01c",
        bbox=BBox(half_w=0.367, half_d=0.737),
        tall=True,
        wall_offset=0.167,
        yaw_offset=math.pi / 2,
        allowed_walls=("back",),
    ),
    "trash_can": WallPropMeta(
        "SM_TrashCan",
        bbox=BBox(half_w=0.150, half_d=0.150),
    ),
    "plant_a": WallPropMeta(
        "SM_Plant01",
        bbox=BBox(half_w=0.352, half_d=0.404),
    ),
    "plant_b": WallPropMeta(
        "SM_Plant02",
        bbox=BBox(half_w=0.252, half_d=0.3),
    ),
    "supply_cart_a": WallPropMeta(
        "SM_SupplyCart_02a",
        bbox=BBox(half_w=0.421, half_d=0.228),
    ),
    "supply_cart_b": WallPropMeta(
        "SM_SupplyCart_03a",
        bbox=BBox(half_w=0.298, half_d=0.556),
        yaw_offset=math.pi / 2,
    ),
}

# ============================================================
# Table group bounding boxes
# ============================================================

DESK_BBOX = BBox(half_w=1.10, half_d=0.65)
ROBOT_BBOX = BBox(half_w=0.25, half_d=0.25)
ROBOT_TABLE_MARGIN = 0.0

# ============================================================
# Tabletop object metadata
# ============================================================

@dataclass(frozen=True)
class TablePropMeta:
    """Placement metadata for a tabletop object."""
    bbox: BBox

TABLE_PROP_META: Dict[str, TablePropMeta] = {
    "coffee_cup":   TablePropMeta(bbox=BBox(half_w=0.043, half_d=0.043)),
    "desk_lamp":    TablePropMeta(bbox=BBox(half_w=0.241, half_d=0.134)),
    "box_portable": TablePropMeta(bbox=BBox(half_w=0.195, half_d=0.145)),
    "object":       TablePropMeta(bbox=BBox(half_w=0.05, half_d=0.05)),
}

# ============================================================
# Asset USD paths (Omniverse S3 CDN)
# ============================================================

_HOSPITAL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Hospital/Props"
_OFFICE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Office/Props"
_WAREHOUSE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props"

ASSET_PATHS: Dict[str, str] = {
    "SM_MedicalCabinet_01a": f"{_HOSPITAL}/SM_MedicalCabinet_01a.usd",
    "SM_ShelfSet_01a":       f"{_HOSPITAL}/SM_ShelfSet_01a.usd",
    "SM_SupplyCabinet_01c":  f"{_HOSPITAL}/SM_SupplyCabinet_01c.usd",
    "SM_TrashCan":           f"{_HOSPITAL}/SM_TrashCan.usd",
    "SM_SupplyCart_02a":     f"{_HOSPITAL}/SM_SupplyCart_02a.usd",
    "SM_SupplyCart_03a":     f"{_HOSPITAL}/SM_SupplyCart_03a.usd",
    "SM_Desk_04a":           f"{_HOSPITAL}/SM_Desk_04a.usd",
    "SM_Chair_04a":          f"{_HOSPITAL}/SM_Chair_04a.usd",
    "SM_Plant01":            f"{_OFFICE}/SM_Plant01.usd",
    "SM_Plant02":            f"{_OFFICE}/SM_Plant02.usd",
    "SM_CoffeeToGo":         f"{_OFFICE}/SM_CoffeeToGo.usd",
    "SM_Lamp02":             f"{_OFFICE}/SM_Lamp02.usd",
    "SM_BoxPortableC":       f"{_OFFICE}/SM_BoxPortableC.usd",
    "SM_CratePlastic_D_01":  f"{_WAREHOUSE}/SM_CratePlastic_D_01.usd",
}
