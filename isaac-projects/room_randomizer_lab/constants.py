# constants.py
# Room placement constants for the hospital room environment.
# Uses oriented bounding boxes (OBB) and continuous wall zones
# instead of circles and predefined slots.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ============================================================
# Room geometry
# ============================================================

ROOM_X_MIN = -13.0
# The floor mesh in new_base_room.usda spans x=[-13, -1], but the
# room-facing right wall is authored at about x=-2.5.
ROOM_X_MAX = -0.5
# Authored safe back-wall props sit around y=-10.75..-10.96 and
# the back wall transform is farther back than the original -11.0 limit.
ROOM_Y_MIN = -11.25
ROOM_Y_MAX = -5.0
FLOOR_Z = 0.0
ROBOT_Z = 0.0328

# Wall surface positions (room-facing edge of each wall).
BACK_WALL_LINE_Y = -10.95
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
        fixed_coord=-10.75,  # prop center Y (matches authored back-wall props)
        base_yaw=0.0,        # face into room (+Y direction)
    ),
    # Right wall: props slide along Y, fixed near right wall X.
    WallZone(
        wall="right",
        sample_min=-10.0,
        sample_max=-7.0,
        fixed_coord=-3.0,    # prop center X (slightly off the right wall surface)
        base_yaw=math.pi / 2,  # face into room (-X direction)
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

# ============================================================
# Desk geometry
# ============================================================

DESK_TOP_Z = 0.78
DESK_OBJECT_Z = DESK_TOP_Z + 0.04

# Desk surface sampling bounds (local to desk prim).
DESK_LOCAL_X_MIN = -0.38
DESK_LOCAL_X_MAX = 0.38
DESK_LOCAL_Y_MIN = -0.22
DESK_LOCAL_Y_MAX = 0.22
DESK_OBJECT_MARGIN = 0.03   # margin between tabletop OBBs

# ============================================================
# Orbital offsets (local-frame displacement from desk center)
# ============================================================

CHAIR_ORBIT_OFFSET = (0.0, -1.00)   # (local_x, local_y)
ROBOT_ORBIT_OFFSET = (-1.95, 1.10)

# ============================================================
# Source transforms (kept for reference)
# ============================================================

DESK_SOURCE_CENTER = (-4.40, -7.10, 0.0)
CHAIR_SOURCE_CENTER = (-5.80, -7.30, 0.0)
ROBOT_SOURCE_CENTER = (-6.1, -5.95, 0.0)

# Despawn height.
DESPAWN_Z = -100.0

# Margin added around every OBB during placement checks (metres).
OBB_PLACEMENT_MARGIN = 0.15

# ============================================================
# Wall prop metadata — bounding boxes replace spacing radii
# ============================================================


@dataclass(frozen=True)
class WallPropMeta:
    """Placement metadata for a wall prop."""
    usd_name: str
    bbox: BBox              # footprint in the object's local frame
    tall: bool = False
    wall_offset: float = 0.0  # extra push away from wall surface (metres)
    yaw_offset: float = 0.0   # yaw adjustment relative to wall base yaw (radians)
    allowed_walls: Tuple[str, ...] = ("back", "right")


WALL_PROP_META: Dict[str, WallPropMeta] = {
    "medical_cabinet": WallPropMeta(
        "SM_MedicalCabinet_01a",
        bbox=BBox(half_w=0.436, half_d=0.328),
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

DESK_BBOX = BBox(half_w=0.745, half_d=0.227)
CHAIR_BBOX = BBox(half_w=0.347, half_d=0.343)
ROBOT_BBOX = BBox(half_w=0.65, half_d=0.50)

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
}

# ============================================================
# Wall yaw lookup (kept for backward compat, but yaw_offset
# on WallPropMeta is the primary source now)
# ============================================================

WALL_PROP_YAW_BY_WALL: Dict[Tuple[str, str], float] = {
    ("SM_MedicalCabinet_01a", "back"):  180.0,
    ("SM_MedicalCabinet_01a", "right"): 180.0,
    ("SM_ShelfSet_01a",       "back"):  180.0,
    ("SM_ShelfSet_01a",       "right"): 180.0,
    ("SM_SupplyCabinet_01c",  "back"):  90.0,
    ("SM_SupplyCabinet_01c",  "right"): 90.0,
    ("SM_TrashCan",           "back"):  0.0,
    ("SM_TrashCan",           "right"): 0.0,
    ("SM_Plant01",            "back"):  0.0,
    ("SM_Plant01",            "right"): 0.0,
    ("SM_Plant02",            "back"):  0.0,
    ("SM_Plant02",            "right"): 0.0,
    ("SM_SupplyCart_02a",     "back"):  0.0,
    ("SM_SupplyCart_02a",     "right"): 0.0,
    ("SM_SupplyCart_03a",     "back"):  0.0,
    ("SM_SupplyCart_03a",     "right"): 0.0,
}

# ============================================================
# Asset USD paths (Omniverse S3 CDN) — kept for reference
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
    "RidgebackUr":           "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Clearpath/RidgebackUr/ridgeback_ur5.usd",
}

TEMPLATE_ROOM_USD = "/World/Environment"
