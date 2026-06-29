# room_events.py
# Event term functions for room randomization in pick & place tasks.

from __future__ import annotations

import math
import random
from typing import List, Optional

import torch
import omni.usd
from pxr import UsdGeom

from isaaclab.envs import ManagerBasedEnv

from .constants import (
    DESK_BBOX,
    DESK_LOCAL_X_MAX,
    DESK_LOCAL_X_MIN,
    DESK_LOCAL_Y_MAX,
    DESK_LOCAL_Y_MIN,
    DESK_OBJECT_MARGIN,
    DESPAWN_Z,
    FLOOR_Z,
    OBB_PLACEMENT_MARGIN,
    ROBOT_BBOX,
    ROBOT_ORBIT_OFFSET,
    TABLE_FALLBACK_X,
    TABLE_FALLBACK_Y,
    TABLE_GROUP_MAX_TRIES,
    TABLE_SAMPLE_X_MAX,
    TABLE_SAMPLE_X_MIN,
    TABLE_SAMPLE_Y_MAX,
    TABLE_SAMPLE_Y_MIN,
    TABLE_PROP_META,
    WALL_PROP_META,
    WALL_ZONES,
    WallZone,
)
from .placement_utils import (
    OBB,
    build_root_state,
    make_obb,
    obb_corners,
    obb_inside_room,
    obb_overlap,
    obb_overlap_any,
    offset_from_yaw,
    offset_from_yaw_batched,
)

_DUPLICATE_VISUAL_PROPS_TO_HIDE = [
    "RoomShell/Environment/props/room_props/SM_Desk_04a",
    "RoomShell/Environment/props/room_props/SM_Chair_04a",
    "RoomShell/Environment/props/wall_props/SM_MedicalCabinet_01a",
    "RoomShell/Environment/props/wall_props/SM_ShelfSet_01a",
    "RoomShell/Environment/props/wall_props/SM_SupplyCabinet_01c",
    "RoomShell/Environment/props/wall_props/SM_SupplyCart_02a",
    "RoomShell/Environment/props/wall_props/SM_SupplyCart_03a",
    "RoomShell/Environment/props/wall_props/SM_TrashCan",
    "RoomShell/Environment/props/wall_props/SM_Plant01",
    "RoomShell/Environment/props/wall_props/SM_Plant02",
]

_visual_props_hidden = False


def _hide_duplicate_visual_props(env_ids: torch.Tensor) -> None:
    """Hides the original duplicate meshes inside RoomShell."""
    global _visual_props_hidden
    if _visual_props_hidden:
        return

    stage = omni.usd.get_context().get_stage()
    for env_idx in range(len(env_ids)):
        env_id = int(env_ids[env_idx].item())
        for rel_path in _DUPLICATE_VISUAL_PROPS_TO_HIDE:
            prim_path = f"/World/envs/env_{env_id}/{rel_path}"
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.MakeInvisible()

    _visual_props_hidden = True


# ======================================================================
# Combined event term
# ======================================================================

def randomize_pickplace_room_layout(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    wall_prop_names: list[str],
    table_prop_names: list[str],
    min_table_objects: int = 1,
):
    """Randomize the full room layout for G1 pick and place environments.
    Uses OBB collision detection and continuous zone sampling.
    """
    M = len(env_ids)
    device = env.device

    # Hide duplicate meshes inside RoomShell
    _hide_duplicate_visual_props(env_ids)

    # Per-environment list of placed OBBs
    all_placed: List[List[OBB]] = [[] for _ in range(M)]
    all_placed_names: List[List[str]] = [[] for _ in range(M)]

    # Per-environment placement results for table group
    desk_positions = torch.zeros(M, 3, device=device)
    desk_yaws = torch.zeros(M, device=device)

    # --- Phase 1: Wall props ---
    _place_wall_props(env, env_ids, wall_prop_names, all_placed, all_placed_names)

    # --- Phase 2: Table group (desk/table + robot) ---
    _place_table_group(
        env, env_ids, all_placed, all_placed_names, desk_positions, desk_yaws
    )

    # --- Phase 3: Tabletop objects (target object + distractors) ---
    _place_desk_objects(
        env, env_ids, table_prop_names, desk_positions, desk_yaws, min_table_objects
    )


# ======================================================================
# Phase 1: Wall prop placement — continuous zone sampling
# ======================================================================

def _sample_wall_position(zone: WallZone, meta, rng: random.Random) -> tuple[float, float, float]:
    """Sample a random (cx, cy, yaw) along a wall zone strip."""
    pos_along_wall = rng.uniform(zone.sample_min, zone.sample_max)
    offset = meta.wall_offset

    if zone.wall == "back":
        cx = pos_along_wall
        cy = zone.fixed_coord + offset  # push into room
    else:  # "right"
        cx = zone.fixed_coord - offset  # push into room
        cy = pos_along_wall

    yaw = zone.base_yaw + meta.yaw_offset
    return cx, cy, yaw


def _place_wall_props(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    wall_prop_names: list[str],
    all_placed: List[List[OBB]],
    all_placed_names: List[List[str]],
):
    M = len(env_ids)
    device = env.device
    rng = random.Random()

    sorted_names = sorted(
        wall_prop_names,
        key=lambda n: not WALL_PROP_META[n].tall,
    )

    for name in sorted_names:
        if name not in env.scene.keys():
            continue
        meta = WALL_PROP_META[name]
        asset = env.scene[name]

        pos_local = torch.zeros(M, 3, device=device)
        yaw_rad = torch.zeros(M, device=device)

        for env_idx in range(M):
            success = False
            allowed_zones = [z for z in WALL_ZONES if z.wall in meta.allowed_walls]
            rng.shuffle(allowed_zones)

            for _ in range(100):
                zone = rng.choice(allowed_zones)
                cx, cy, yaw = _sample_wall_position(zone, meta, rng)
                candidate = make_obb(cx, cy, meta.bbox, yaw)

                if not obb_inside_room(candidate):
                    continue

                if obb_overlap_any(candidate, all_placed[env_idx], margin=OBB_PLACEMENT_MARGIN):
                    continue

                # Valid placement
                pos_local[env_idx] = torch.tensor([cx, cy, FLOOR_Z], device=device)
                yaw_rad[env_idx] = yaw
                all_placed[env_idx].append(candidate)
                all_placed_names[env_idx].append(name)
                success = True
                break

            if not success:
                pos_local[env_idx, 2] = DESPAWN_Z

        root_state = build_root_state(
            pos_local, yaw_rad,
            env.scene.env_origins, env_ids,
            asset.data.default_root_state,
        )
        asset.write_root_state_to_sim(root_state, env_ids=env_ids)


# ======================================================================
# Phase 2: Table group — continuous interior sampling
# ======================================================================

def _make_table_group(dx: float, dy: float, dyaw: float) -> tuple[list[tuple[str, OBB]], tuple[float, float]]:
    """Build desk/robot OBBs and robot XY positions."""
    rx, ry = offset_from_yaw(dx, dy, dyaw, ROBOT_ORBIT_OFFSET[0], ROBOT_ORBIT_OFFSET[1])
    robot_yaw = dyaw + math.pi / 2
    table_obbs = [
        ("packing_table", make_obb(dx, dy, DESK_BBOX, dyaw)),
        ("robot", make_obb(rx, ry, ROBOT_BBOX, robot_yaw)),
    ]
    return table_obbs, (rx, ry)


def _validate_table_group(
    table_obbs: list[tuple[str, OBB]],
    placed: List[OBB],
    placed_names: list[str],
) -> tuple[bool, list[str]]:
    """Validate table-group room bounds and overlaps."""
    issues: list[str] = []

    for name, box in table_obbs:
        if not obb_inside_room(box):
            issues.append(f"{name} outside_room")

    for name, box in table_obbs:
        for placed_name, placed_box in zip(placed_names, placed):
            if obb_overlap(box, placed_box, margin=OBB_PLACEMENT_MARGIN):
                issues.append(f"{name} overlaps {placed_name}")

    for i, (a_name, a_box) in enumerate(table_obbs):
        for b_name, b_box in table_obbs[i + 1:]:
            if obb_overlap(a_box, b_box, margin=OBB_PLACEMENT_MARGIN):
                issues.append(f"{a_name} overlaps {b_name}")

    return len(issues) == 0, issues


def _place_table_group(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    all_placed: List[List[OBB]],
    all_placed_names: List[List[str]],
    desk_positions: torch.Tensor,
    desk_yaws: torch.Tensor,
) -> list[list[tuple[str, OBB]]]:
    M = len(env_ids)
    device = env.device
    rng = random.Random()
    env_origins = env.scene.env_origins
    group_placed_mask = torch.zeros(M, dtype=torch.bool, device=device)
    final_table_obbs: list[list[tuple[str, OBB]]] = [[] for _ in range(M)]

    for env_idx in range(M):
        success = False
        selected_obbs: list[tuple[str, OBB]] = []

        for _ in range(TABLE_GROUP_MAX_TRIES):
            dx = rng.uniform(TABLE_SAMPLE_X_MIN, TABLE_SAMPLE_X_MAX)
            dy = rng.uniform(TABLE_SAMPLE_Y_MIN, TABLE_SAMPLE_Y_MAX)
            dyaw = rng.uniform(0, 2 * math.pi)

            table_obbs, _ = _make_table_group(dx, dy, dyaw)
            valid, _ = _validate_table_group(table_obbs, all_placed[env_idx], all_placed_names[env_idx])
            if not valid:
                continue

            # Valid!
            desk_positions[env_idx] = torch.tensor([dx, dy, FLOOR_Z], device=device)
            desk_yaws[env_idx] = dyaw
            selected_obbs = table_obbs
            success = True
            break

        if not success:
            # Fallback
            dx, dy = TABLE_FALLBACK_X, TABLE_FALLBACK_Y
            for _ in range(TABLE_GROUP_MAX_TRIES):
                dyaw = rng.uniform(0, 2 * math.pi)
                table_obbs, _ = _make_table_group(dx, dy, dyaw)
                valid, _ = _validate_table_group(table_obbs, all_placed[env_idx], all_placed_names[env_idx])
                if valid:
                    desk_positions[env_idx] = torch.tensor([dx, dy, FLOOR_Z], device=device)
                    desk_yaws[env_idx] = dyaw
                    selected_obbs = table_obbs
                    success = True
                    break

            if not success:
                desk_positions[env_idx] = torch.tensor([0.0, 0.0, DESPAWN_Z], device=device)
                desk_yaws[env_idx] = 0.0

        if success:
            final_table_obbs[env_idx] = selected_obbs
            all_placed[env_idx].extend([box for _, box in selected_obbs])
            all_placed_names[env_idx].extend([name for name, _ in selected_obbs])
            group_placed_mask[env_idx] = True

    # Get default Z coordinates dynamically from env assets
    desk_asset = env.scene["packing_table"]
    desk_default_z = (desk_asset.data.default_root_state[0, 2] - env_origins[0, 2]).item()
    desk_positions[:, 2] = desk_default_z

    robot_asset = env.scene["robot"]
    robot_default_z = (robot_asset.data.default_root_state[0, 2] - env_origins[0, 2]).item()

    # Build batched positions for robot
    robot_pos = offset_from_yaw_batched(
        desk_positions, desk_yaws,
        ROBOT_ORBIT_OFFSET[0], ROBOT_ORBIT_OFFSET[1], robot_default_z,
    )
    robot_yaw = desk_yaws + math.pi / 2

    # Handle failures by despawning
    invalid_mask = ~group_placed_mask
    if torch.any(invalid_mask):
        desk_positions[invalid_mask, 0:2] = 0.0
        desk_positions[invalid_mask, 2] = DESPAWN_Z
        robot_pos[invalid_mask, 0:2] = 0.0
        robot_pos[invalid_mask, 2] = DESPAWN_Z
        robot_yaw[invalid_mask] = 0.0

    desk_state = build_root_state(desk_positions, desk_yaws, env_origins, env_ids, desk_asset.data.default_root_state)
    desk_asset.write_root_state_to_sim(desk_state, env_ids=env_ids)

    robot_state = build_root_state(robot_pos, robot_yaw, env_origins, env_ids, robot_asset.data.default_root_state)
    robot_asset.write_root_state_to_sim(robot_state, env_ids=env_ids)

    return final_table_obbs


# ======================================================================
# Phase 3: Tabletop objects — OBB collision on desk surface
# ======================================================================

def _place_desk_objects(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    table_prop_names: list[str],
    desk_pos: torch.Tensor,
    desk_yaw_rad: torch.Tensor,
    min_table_objects: int = 1,
) -> list[list[tuple[str, OBB]]]:
    M = len(env_ids)
    debug_obbs: list[list[tuple[str, OBB]]] = [[] for _ in range(M)]
    device = env.device
    env_origins = env.scene.env_origins
    rng = random.Random()

    # Get target object Z height dynamically from default root state
    object_asset = env.scene["object"]
    object_default_z = (object_asset.data.default_root_state[0, 2] - env_origins[0, 2]).item()

    for env_idx in range(M):
        if desk_pos[env_idx, 2].item() <= DESPAWN_Z * 0.5:
            # Table is despawned, so despawn the object and distractors too
            for name in ["object"] + table_prop_names:
                if name in env.scene.keys():
                    asset = env.scene[name]
                    pos = torch.tensor([0.0, 0.0, DESPAWN_Z], device=device).unsqueeze(0)
                    yaw = torch.tensor([0.0], device=device)
                    eid = env_ids[env_idx:env_idx+1]
                    root_state = build_root_state(pos, yaw, env_origins, eid, asset.data.default_root_state)
                    asset.write_root_state_to_sim(root_state, env_ids=eid)
            continue

        desk_placed: List[OBB] = []

        # 1. Place target object first
        target_placed = False
        target_meta = TABLE_PROP_META["object"]
        for _ in range(100):
            # Sample local coordinates within G1's reaching boundaries
            lx = rng.uniform(-0.25, 0.25)
            ly = rng.uniform(0.0, 0.15)
            obj_yaw = rng.uniform(-math.pi/4, math.pi/4)

            candidate = make_obb(lx, ly, target_meta.bbox, obj_yaw)
            desk_placed.append(candidate)

            wx, wy = offset_from_yaw(
                desk_pos[env_idx, 0].item(),
                desk_pos[env_idx, 1].item(),
                desk_yaw_rad[env_idx].item(),
                lx, ly,
            )
            world_yaw = desk_yaw_rad[env_idx].item() + obj_yaw

            pos = torch.tensor([wx, wy, object_default_z], device=device).unsqueeze(0)
            yaw = torch.tensor([world_yaw], device=device)
            eid = env_ids[env_idx:env_idx+1]

            root_state = build_root_state(pos, yaw, env_origins, eid, object_asset.data.default_root_state)
            object_asset.write_root_state_to_sim(root_state, env_ids=eid)
            debug_obbs[env_idx].append(
                ("object", make_obb(wx, wy, target_meta.bbox, world_yaw))
            )
            target_placed = True
            break

        if not target_placed:
            wx, wy = offset_from_yaw(
                desk_pos[env_idx, 0].item(),
                desk_pos[env_idx, 1].item(),
                desk_yaw_rad[env_idx].item(),
                0.0, 0.1,
            )
            world_yaw = desk_yaw_rad[env_idx].item()
            pos = torch.tensor([wx, wy, object_default_z], device=device).unsqueeze(0)
            yaw = torch.tensor([world_yaw], device=device)
            eid = env_ids[env_idx:env_idx+1]
            root_state = build_root_state(pos, yaw, env_origins, eid, object_asset.data.default_root_state)
            object_asset.write_root_state_to_sim(root_state, env_ids=eid)

        # 2. Place optional tabletop distractors (0 to len(table_prop_names))
        count = rng.randint(0, len(table_prop_names))
        shuffled_props = list(table_prop_names)
        rng.shuffle(shuffled_props)

        for i, name in enumerate(shuffled_props):
            if name not in env.scene.keys():
                continue
            asset = env.scene[name]
            meta = TABLE_PROP_META[name]
            visible = i < count

            if visible:
                placed = False
                distractor_default_z = (asset.data.default_root_state[0, 2] - env_origins[0, 2]).item()
                for _ in range(100):
                    lx = rng.uniform(DESK_LOCAL_X_MIN, DESK_LOCAL_X_MAX)
                    ly = rng.uniform(DESK_LOCAL_Y_MIN, DESK_LOCAL_Y_MAX)
                    obj_yaw = rng.uniform(0, 2 * math.pi)

                    candidate = make_obb(lx, ly, meta.bbox, obj_yaw)
                    if not obb_overlap_any(candidate, desk_placed, margin=DESK_OBJECT_MARGIN):
                        desk_placed.append(candidate)

                        wx, wy = offset_from_yaw(
                            desk_pos[env_idx, 0].item(),
                            desk_pos[env_idx, 1].item(),
                            desk_yaw_rad[env_idx].item(),
                            lx, ly,
                        )
                        world_yaw = desk_yaw_rad[env_idx].item() + obj_yaw

                        pos = torch.tensor([wx, wy, distractor_default_z], device=device).unsqueeze(0)
                        yaw = torch.tensor([world_yaw], device=device)
                        eid = env_ids[env_idx:env_idx+1]

                        root_state = build_root_state(pos, yaw, env_origins, eid, asset.data.default_root_state)
                        asset.write_root_state_to_sim(root_state, env_ids=eid)
                        debug_obbs[env_idx].append(
                            (name, make_obb(wx, wy, meta.bbox, world_yaw))
                        )
                        placed = True
                        break

                if not placed:
                    pos = torch.tensor([0.0, 0.0, DESPAWN_Z], device=device).unsqueeze(0)
                    yaw = torch.tensor([0.0], device=device)
                    eid = env_ids[env_idx:env_idx+1]
                    root_state = build_root_state(pos, yaw, env_origins, eid, asset.data.default_root_state)
                    asset.write_root_state_to_sim(root_state, env_ids=eid)
            else:
                pos = torch.tensor([0.0, 0.0, DESPAWN_Z], device=device).unsqueeze(0)
                yaw = torch.tensor([0.0], device=device)
                eid = env_ids[env_idx:env_idx+1]
                root_state = build_root_state(pos, yaw, env_origins, eid, asset.data.default_root_state)
                asset.write_root_state_to_sim(root_state, env_ids=eid)

    return debug_obbs
