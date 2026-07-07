# room_events.py
# Event term functions for room randomization in pick & place tasks.

from __future__ import annotations

import math
import random
from typing import List, Optional

import torch
import omni.usd
from pxr import Usd, UsdGeom

from isaaclab.envs import ManagerBasedEnv

from .constants import (
    DESK_BBOX,
    DESK_LAMP_LOCAL_X_RANGE,
    DESK_LAMP_LOCAL_Y_RANGE,
    DESK_LAMP_LOCAL_YAW,
    DESK_LAMP_Z,
    DESK_LOCAL_X_MIN,
    DESK_LOCAL_X_MAX,
    DESK_LOCAL_Y_MIN,
    DESK_LOCAL_Y_MAX,
    DESK_OBJECT_MARGIN,
    DESK_OBJECT_Z,
    DESPAWN_Z,
    FLOOR_Z,
    OBJECT_TABLE_LOCAL_OFFSET,
    OBB_PLACEMENT_MARGIN,
    ROBOT_BBOX,
    ROBOT_FACING_LAYOUTS,
    ROBOT_FACING_MAX_YAW_OFFSET_RAD,
    ROBOT_FACING_YAW_JITTER_RAD,
    ROBOT_ORBIT_OFFSET,
    ROBOT_TABLE_MARGIN,
    STATIC_ROOM_OBSTACLES,
    TABLE_GROUP_X_MAX,
    TABLE_GROUP_X_MIN,
    TABLE_GROUP_Y_MAX,
    TABLE_GROUP_Y_MIN,
    TABLE_GROUP_MAX_TRIES,
    TABLE_RESERVED_AREAS,
    TABLETOP_CUBE_LOCAL_X_MAX,
    TABLETOP_CUBE_LOCAL_X_MIN,
    TABLETOP_CUBE_PROP_NAMES,
    TABLE_SAMPLE_X_MAX,
    TABLE_SAMPLE_X_MIN,
    TABLE_SAMPLE_Y_MAX,
    TABLE_SAMPLE_Y_MIN,
    TABLE_PROP_META,
    WALL_PROP_META,
    WALL_ZONES,
    WallZone,
    RobotFacingLayout,
)
from .placement_utils import (
    OBB,
    build_root_state,
    make_obb,
    obb_corners,
    obb_inside_room,
    obb_overlap,
    obb_overlap_any,
    obb_corners,
    offset_from_yaw,
    offset_from_yaw_batched,
)

_DUPLICATE_VISUAL_PROP_NAMES_TO_HIDE = {
    "SM_Desk_04a",
    "SM_Chair_04a",
    "SM_MedicalCabinet_01a",
    "SM_ShelfSet_01a",
    "SM_SupplyCabinet_01c",
    "SM_SupplyCart_02a",
    "SM_SupplyCart_03a",
    "SM_TrashCan",
    "SM_Plant01",
    "SM_Plant02",
    "SM_CoffeeToGo",
    "SM_Lamp02",
    "SM_BoxPortableC",
}

_visual_props_hidden = False
_DYNAMIC_TABLETOP_OBJECT_NAMES = {"object", "blue_cube", "yellow_cube"}
_TABLE_RESERVED_AREA_NAMES = {area.name for area in TABLE_RESERVED_AREAS}


def _write_root_pose_to_sim(asset, root_state: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write only root pose for kinematic props to avoid PhysX velocity errors."""
    asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)


def _write_tabletop_root_state(asset, name: str, root_state: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write tabletop object state, preserving kinematic handling for static props."""
    if name in _DYNAMIC_TABLETOP_OBJECT_NAMES:
        asset.write_root_state_to_sim(root_state, env_ids=env_ids)
    else:
        _write_root_pose_to_sim(asset, root_state, env_ids)


def _despawn_props(env: ManagerBasedEnv, env_ids: torch.Tensor, prop_names: list[str]) -> None:
    if not prop_names:
        return

    device = env.device
    env_origins = env.scene.env_origins
    pos = torch.zeros(len(env_ids), 3, device=device)
    pos[:, 2] = DESPAWN_Z
    yaw = torch.zeros(len(env_ids), device=device)

    for name in prop_names:
        if name == "object" or name not in env.scene.keys():
            continue
        asset = env.scene[name]
        root_state = build_root_state(pos, yaw, env_origins, env_ids, asset.data.default_root_state)
        if name in _DYNAMIC_TABLETOP_OBJECT_NAMES:
            asset.write_root_state_to_sim(root_state, env_ids=env_ids)
        else:
            _write_root_pose_to_sim(asset, root_state, env_ids)
        print(
            f"[PLACEMENT_DEBUG] object={name} disabled_table_or_wall_prop despawned=true",
            flush=True,
        )


def _hide_duplicate_visual_props(env_ids: torch.Tensor) -> None:
    """Hides the original duplicate meshes inside RoomShell."""
    global _visual_props_hidden
    if _visual_props_hidden:
        return

    stage = omni.usd.get_context().get_stage()
    for env_idx in range(len(env_ids)):
        env_id = int(env_ids[env_idx].item())
        room_shell = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/RoomShell")
        if not room_shell.IsValid():
            continue
        hidden_count = 0
        for prim in Usd.PrimRange(room_shell):
            if prim.GetName() not in _DUPLICATE_VISUAL_PROP_NAMES_TO_HIDE:
                continue
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                imageable.MakeInvisible()
                hidden_count += 1
        print(
            f"[PLACEMENT_DEBUG] env={env_id} hidden_room_shell_duplicates={hidden_count}",
            flush=True,
        )

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
    active_wall_props = set(wall_prop_names)
    active_table_props = set(table_prop_names)
    _despawn_props(
        env,
        env_ids,
        [name for name in WALL_PROP_META.keys() if name not in active_wall_props]
        + [name for name in TABLE_PROP_META.keys() if name != "object" and name not in active_table_props],
    )

    # Per-environment list of placed OBBs
    all_placed: List[List[OBB]] = [[] for _ in range(M)]
    all_placed_names: List[List[str]] = [[] for _ in range(M)]

    # Per-environment placement results for table group
    desk_positions = torch.zeros(M, 3, device=device)
    desk_yaws = torch.zeros(M, device=device)

    # --- Phase 1: Wall props ---
    wall_debug_obbs = _place_wall_props(env, env_ids, wall_prop_names, all_placed, all_placed_names)

    # --- Static RoomShell walls ---
    static_wall_debug_obbs = _add_static_room_obstacles(env_ids, all_placed, all_placed_names)

    # --- Phase 2: Table group (desk/table + robot) ---
    table_debug_obbs = _place_table_group(
        env, env_ids, all_placed, all_placed_names, desk_positions, desk_yaws
    )

    # --- Phase 3: Tabletop objects (target object + distractors) ---
    tabletop_debug_obbs = _place_desk_objects(
        env,
        env_ids,
        ["object"] + list(table_prop_names),
        desk_positions,
        desk_yaws,
        min_table_objects,
    )

    debug_obbs = getattr(env, "_room_randomizer_debug_obbs", {})
    for env_idx in range(M):
        env_id = _env_id_int(env_ids, env_idx)
        records = []
        records.extend(
            {
                "name": name,
                "category": "wall",
                "box": box,
                "z": FLOOR_Z + 0.04,
            }
            for name, box in wall_debug_obbs[env_idx]
            if box is not None
        )
        records.extend(
            {
                "name": name,
                "category": "static_wall",
                "box": box,
                "z": FLOOR_Z + 0.04,
            }
            for name, box in static_wall_debug_obbs[env_idx]
        )
        records.extend(
            {
                "name": name,
                "category": "robot" if name == "robot" else "table_group",
                "box": box,
                "z": FLOOR_Z + 0.04,
            }
            for name, box in table_debug_obbs[env_idx]
        )
        records.extend(
            {
                "name": name,
                "category": "table_reserved" if name in _TABLE_RESERVED_AREA_NAMES else "tabletop",
                "box": box,
                "z": DESK_OBJECT_Z + 0.04,
            }
            for name, box in tabletop_debug_obbs[env_idx]
        )
        debug_obbs[env_id] = records
    env._room_randomizer_debug_obbs = debug_obbs

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


def _env_id_int(env_ids: torch.Tensor, env_idx: int) -> int:
    """Return a printable env id from a tensor slice."""
    return int(env_ids[env_idx].item())


def _format_corners(box: OBB) -> str:
    """Compact corner formatting for placement diagnostics."""
    return "[" + ", ".join(f"({x:+.3f},{y:+.3f})" for x, y in obb_corners(*box)) + "]"


def _print_obb_debug(name: str, env_id: int, box: OBB, prefix: str = "[PLACEMENT_DEBUG]"):
    inside = obb_inside_room(box)
    print(
        f"{prefix} env={env_id} object={name} "
        f"pos=({box[0]:+.3f},{box[1]:+.3f}) yaw={box[4]:+.3f} "
        f"corners={_format_corners(box)} inside_room={inside}",
        flush=True,
    )
    if not inside:
        print(
            f"[PLACEMENT_ERROR] env={env_id} object={name} outside_room "
            f"pos=({box[0]:+.3f},{box[1]:+.3f}) yaw={box[4]:+.3f} "
            f"corners={_format_corners(box)}",
            flush=True,
        )


def _add_static_room_obstacles(
    env_ids: torch.Tensor,
    all_placed: List[List[OBB]],
    all_placed_names: List[List[str]],
) -> List[List[tuple[str, OBB]]]:
    """Add static RoomShell wall OBBs to the placement set."""
    debug_records: List[List[tuple[str, OBB]]] = [[] for _ in range(len(env_ids))]
    obstacle_obbs = [
        (obstacle.name, make_obb(obstacle.center[0], obstacle.center[1], obstacle.bbox, obstacle.yaw))
        for obstacle in STATIC_ROOM_OBSTACLES
    ]

    for env_idx in range(len(env_ids)):
        env_id = _env_id_int(env_ids, env_idx)
        for name, box in obstacle_obbs:
            all_placed[env_idx].append(box)
            all_placed_names[env_idx].append(name)
            debug_records[env_idx].append((name, box))
            _print_obb_debug(name, env_id, box)

    return debug_records


def _place_wall_props(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    wall_prop_names: list[str],
    all_placed: List[List[OBB]],
    all_placed_names: List[List[str]],
) -> List[List[tuple[str, Optional[OBB]]]]:
    """Place wall props using continuous zone sampling + OBB collision."""
    M = len(env_ids)
    device = env.device
    rng = random.Random()

    sorted_names = sorted(
        wall_prop_names,
        key=lambda n: not WALL_PROP_META[n].tall,
    )
    debug_records: List[List[tuple[str, Optional[OBB]]]] = [[] for _ in range(M)]

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
                debug_cx, debug_cy = offset_from_yaw(
                    cx, cy, yaw, meta.bbox_center[0], meta.bbox_center[1]
                )
                debug_records[env_idx].append(
                    (name, make_obb(debug_cx, debug_cy, meta.bbox, yaw))
                )
                success = True
                break

            if not success:
                pos_local[env_idx, 2] = DESPAWN_Z
                debug_records[env_idx].append((name, None))
                print(
                    f"[PLACEMENT_ERROR] env={_env_id_int(env_ids, env_idx)} "
                    f"object={name} wall_prop_placement_failed despawning=true",
                    flush=True,
                )

        root_state = build_root_state(
            pos_local, yaw_rad,
            env.scene.env_origins, env_ids,
            asset.data.default_root_state,
        )
        _write_root_pose_to_sim(asset, root_state, env_ids)

    for env_idx in range(M):
        env_id = _env_id_int(env_ids, env_idx)
        for name, box in debug_records[env_idx]:
            if box is None:
                print(f"[PLACEMENT_DEBUG] env={env_id} object={name} despawned=true", flush=True)
                continue
            _print_obb_debug(name, env_id, box)
        for i, a_box in enumerate(all_placed[env_idx]):
            for j, b_box in enumerate(all_placed[env_idx][i + 1:], start=i + 1):
                if obb_overlap(a_box, b_box, margin=OBB_PLACEMENT_MARGIN):
                    a_name = all_placed_names[env_idx][i]
                    b_name = all_placed_names[env_idx][j]
                    print(
                        f"[PLACEMENT_ERROR] env={env_id} a={a_name} b={b_name} overlap "
                        f"{a_name}_pos=({a_box[0]:+.3f},{a_box[1]:+.3f}) {a_name}_yaw={a_box[4]:+.3f} "
                        f"{a_name}_corners={_format_corners(a_box)} "
                        f"{b_name}_pos=({b_box[0]:+.3f},{b_box[1]:+.3f}) {b_name}_yaw={b_box[4]:+.3f} "
                        f"{b_name}_corners={_format_corners(b_box)}",
                        flush=True,
                    )

    return debug_records


# ======================================================================
# Phase 2: Table group — continuous interior sampling
# ======================================================================

def _make_table_group_from_robot(
    rx: float,
    ry: float,
    robot_yaw: float,
) -> tuple[list[tuple[str, OBB]], tuple[float, float], float]:
    """Build table/robot OBBs using the robot as the placement anchor."""
    table_yaw = robot_yaw - math.pi / 2
    dx, dy = offset_from_yaw(
        rx,
        ry,
        table_yaw,
        -ROBOT_ORBIT_OFFSET[0],
        -ROBOT_ORBIT_OFFSET[1],
    )
    table_obbs = [
        ("packing_table", make_obb(dx, dy, DESK_BBOX, table_yaw)),
        ("robot", make_obb(rx, ry, ROBOT_BBOX, robot_yaw)),
    ]
    return table_obbs, (dx, dy), table_yaw


def _normalize_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _sample_wall_facing_robot_pose(rng: random.Random) -> tuple[float, float, float, RobotFacingLayout]:
    """Sample robot placement and aim it at a random point on a wall."""
    layout = rng.choice(ROBOT_FACING_LAYOUTS)
    rx = rng.uniform(TABLE_SAMPLE_X_MIN, TABLE_SAMPLE_X_MAX)
    ry = rng.uniform(TABLE_SAMPLE_Y_MIN, TABLE_SAMPLE_Y_MAX)
    target = rng.uniform(layout.sample_min, layout.sample_max)
    if layout.target_axis == "y":
        target_x, target_y = target, layout.fixed_coord
    else:
        target_x, target_y = layout.fixed_coord, target
    raw_robot_yaw = math.atan2(target_y - ry, target_x - rx)
    yaw_delta = _normalize_angle(raw_robot_yaw - layout.yaw_center)
    yaw_delta = max(-ROBOT_FACING_MAX_YAW_OFFSET_RAD, min(ROBOT_FACING_MAX_YAW_OFFSET_RAD, yaw_delta))
    robot_yaw = layout.yaw_center + yaw_delta
    jitter = 0.0
    if ROBOT_FACING_YAW_JITTER_RAD > 0.0:
        jitter = rng.uniform(-ROBOT_FACING_YAW_JITTER_RAD, ROBOT_FACING_YAW_JITTER_RAD)
    return rx, ry, robot_yaw + jitter, layout


def _obb_inside_table_group_bounds(box: OBB) -> bool:
    """Check the stricter robot/table usable area, including virtual open-side limits."""
    for wx, wy in obb_corners(*box):
        if wx < TABLE_GROUP_X_MIN or wx > TABLE_GROUP_X_MAX:
            return False
        if wy < TABLE_GROUP_Y_MIN or wy > TABLE_GROUP_Y_MAX:
            return False
    return True


def _debug_table_group(env_id: int, table_obbs: list[tuple[str, OBB]], placed: List[OBB], placed_names: list[str]):
    """Print table group OBBs and overlap diagnostics."""
    for name, box in table_obbs:
        _print_obb_debug(name, env_id, box)

    for i, (a_name, a_box) in enumerate(table_obbs):
        for b_name, b_box in table_obbs[i + 1:]:
            if {a_name, b_name} == {"packing_table", "robot"}:
                print(
                    f"[PLACEMENT_DEBUG] env={env_id} overlap_check "
                    f"a={a_name} b={b_name} overlaps=skipped_regular_pickplace_transform",
                    flush=True,
                )
                continue
            margin = ROBOT_TABLE_MARGIN if {a_name, b_name} == {"packing_table", "robot"} else OBB_PLACEMENT_MARGIN
            overlaps = obb_overlap(a_box, b_box, margin=margin)
            print(
                f"[PLACEMENT_DEBUG] env={env_id} overlap_check "
                f"a={a_name} b={b_name} overlaps={overlaps}",
                flush=True,
            )
            if overlaps:
                print(
                    f"[PLACEMENT_ERROR] env={env_id} a={a_name} b={b_name} overlap "
                    f"{a_name}_pos=({a_box[0]:+.3f},{a_box[1]:+.3f}) {a_name}_yaw={a_box[4]:+.3f} "
                    f"{a_name}_corners={_format_corners(a_box)} "
                    f"{b_name}_pos=({b_box[0]:+.3f},{b_box[1]:+.3f}) {b_name}_yaw={b_box[4]:+.3f} "
                    f"{b_name}_corners={_format_corners(b_box)}",
                    flush=True,
                )

    for name, box in table_obbs:
        for placed_name, placed_box in zip(placed_names, placed):
            overlaps = obb_overlap(box, placed_box, margin=OBB_PLACEMENT_MARGIN)
            print(
                f"[PLACEMENT_DEBUG] env={env_id} overlap_check "
                f"a={name} b={placed_name} overlaps={overlaps}",
                flush=True,
            )
            if overlaps:
                print(
                    f"[PLACEMENT_ERROR] env={env_id} a={name} b={placed_name} overlap "
                    f"{name}_pos=({box[0]:+.3f},{box[1]:+.3f}) {name}_yaw={box[4]:+.3f} "
                    f"{name}_corners={_format_corners(box)} "
                    f"{placed_name}_pos=({placed_box[0]:+.3f},{placed_box[1]:+.3f}) "
                    f"{placed_name}_yaw={placed_box[4]:+.3f} "
                    f"{placed_name}_corners={_format_corners(placed_box)}",
                    flush=True,
                )


def _validate_table_group(
    table_obbs: list[tuple[str, OBB]],
    placed: List[OBB],
    placed_names: list[str],
) -> tuple[bool, list[str]]:
    """Validate table-group room bounds and overlaps."""
    issues: list[str] = []

    for name, box in table_obbs:
        if not obb_inside_room(box):
            issues.append(f"{name} outside_room corners={_format_corners(box)}")
        if not _obb_inside_table_group_bounds(box):
            issues.append(
                f"{name} outside_table_group_bounds "
                f"bounds=({TABLE_GROUP_X_MIN:+.3f},{TABLE_GROUP_X_MAX:+.3f},"
                f"{TABLE_GROUP_Y_MIN:+.3f},{TABLE_GROUP_Y_MAX:+.3f}) "
                f"corners={_format_corners(box)}"
            )

    for name, box in table_obbs:
        for placed_name, placed_box in zip(placed_names, placed):
            if obb_overlap(box, placed_box, margin=OBB_PLACEMENT_MARGIN):
                issues.append(
                    f"{name} overlaps {placed_name} "
                    f"{name}_corners={_format_corners(box)} "
                    f"{placed_name}_corners={_format_corners(placed_box)}"
                )

    for i, (a_name, a_box) in enumerate(table_obbs):
        for b_name, b_box in table_obbs[i + 1:]:
            if {a_name, b_name} == {"packing_table", "robot"}:
                continue
            margin = ROBOT_TABLE_MARGIN if {a_name, b_name} == {"packing_table", "robot"} else OBB_PLACEMENT_MARGIN
            if obb_overlap(a_box, b_box, margin=margin):
                issues.append(
                    f"{a_name} overlaps {b_name} "
                    f"{a_name}_corners={_format_corners(a_box)} "
                    f"{b_name}_corners={_format_corners(b_box)}"
                )

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
        selected_layout_name = "none"

        for _ in range(TABLE_GROUP_MAX_TRIES):
            rx, ry, robot_yaw, layout = _sample_wall_facing_robot_pose(rng)

            table_obbs, (dx, dy), dyaw = _make_table_group_from_robot(rx, ry, robot_yaw)
            valid, _ = _validate_table_group(table_obbs, all_placed[env_idx], all_placed_names[env_idx])
            if not valid:
                continue

            # Valid!
            desk_positions[env_idx] = torch.tensor([dx, dy, FLOOR_Z], device=device)
            desk_yaws[env_idx] = dyaw
            selected_obbs = table_obbs
            selected_layout_name = layout.name
            success = True
            break

        if not success:
            # Fallback
            fallback_issues: list[str] = []
            for _ in range(TABLE_GROUP_MAX_TRIES):
                rx, ry, robot_yaw, layout = _sample_wall_facing_robot_pose(rng)
                table_obbs, (dx, dy), dyaw = _make_table_group_from_robot(rx, ry, robot_yaw)
                valid, fallback_issues = _validate_table_group(table_obbs, all_placed[env_idx], all_placed_names[env_idx])
                if valid:
                    desk_positions[env_idx] = torch.tensor([dx, dy, FLOOR_Z], device=device)
                    desk_yaws[env_idx] = dyaw
                    selected_obbs = table_obbs
                    selected_layout_name = layout.name
                    success = True
                    print(
                        f"[PLACEMENT_DEBUG] env={_env_id_int(env_ids, env_idx)} "
                        f"table_group fallback_validated=true",
                        flush=True,
                    )
                    break

            if not success:
                env_id = _env_id_int(env_ids, env_idx)
                desk_positions[env_idx] = torch.tensor([0.0, 0.0, DESPAWN_Z], device=device)
                desk_yaws[env_idx] = 0.0
                print(
                    f"[PLACEMENT_ERROR] env={env_id} table_group placement_failed despawning=true "
                    f"last_issues={fallback_issues}",
                    flush=True,
                )

        if success:
            final_table_obbs[env_idx] = selected_obbs
            all_placed[env_idx].extend([box for _, box in selected_obbs])
            all_placed_names[env_idx].extend([name for name, _ in selected_obbs])
            group_placed_mask[env_idx] = True
            print(
                f"[PLACEMENT_DEBUG] env={_env_id_int(env_ids, env_idx)} "
                f"table_group_layout={selected_layout_name}",
                flush=True,
            )

    for env_idx in range(M):
        env_id = _env_id_int(env_ids, env_idx)
        if final_table_obbs[env_idx]:
            pre_group_count = len(all_placed[env_idx]) - len(final_table_obbs[env_idx])
            _debug_table_group(
                env_id,
                final_table_obbs[env_idx],
                all_placed[env_idx][:pre_group_count],
                all_placed_names[env_idx][:pre_group_count],
            )

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
    _write_root_pose_to_sim(desk_asset, desk_state, env_ids)
    actual_desk_pos = desk_asset.data.root_pos_w[env_ids] - env_origins[env_ids]
    for env_idx in range(M):
        if not final_table_obbs[env_idx]:
            continue
        pos = actual_desk_pos[env_idx]
        print(
            f"[PLACEMENT_DEBUG] env={_env_id_int(env_ids, env_idx)} "
            f"packing_table_actual_root_pos=({pos[0].item():+.3f},{pos[1].item():+.3f},{pos[2].item():+.3f})",
            flush=True,
        )

    robot_state = build_root_state(robot_pos, robot_yaw, env_origins, env_ids, robot_asset.data.default_root_state)
    robot_asset.write_root_state_to_sim(robot_state, env_ids=env_ids)
    robot_asset.write_joint_state_to_sim(
        robot_asset.data.default_joint_pos[env_ids],
        robot_asset.data.default_joint_vel[env_ids],
        env_ids=env_ids,
    )
    robot_asset.set_joint_position_target(robot_asset.data.default_joint_pos[env_ids], env_ids=env_ids)
    robot_asset.set_joint_velocity_target(robot_asset.data.default_joint_vel[env_ids], env_ids=env_ids)

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
    if not table_prop_names:
        return debug_obbs

    device = env.device
    env_origins = env.scene.env_origins
    rng = random.Random()

    num_total = len(table_prop_names)

    for env_idx in range(M):
        if desk_pos[env_idx, 2].item() <= DESPAWN_Z * 0.5:
            # Table is despawned, so despawn the object and distractors too
            for name in table_prop_names:
                if name in env.scene.keys():
                    asset = env.scene[name]
                    pos = torch.tensor([0.0, 0.0, DESPAWN_Z], device=device).unsqueeze(0)
                    yaw = torch.tensor([0.0], device=device)
                    eid = env_ids[env_idx:env_idx+1]
                    root_state = build_root_state(pos, yaw, env_origins, eid, asset.data.default_root_state)
                    _write_tabletop_root_state(asset, name, root_state, eid)
            print(
                f"[PLACEMENT_ERROR] env={_env_id_int(env_ids, env_idx)} "
                f"tabletop_objects skipped table_group_not_placed=true",
                flush=True,
            )
            continue

        desk_placed: List[OBB] = []
        for area in TABLE_RESERVED_AREAS:
            desk_placed.append(make_obb(area.center[0], area.center[1], area.bbox, area.yaw))
            wx, wy = offset_from_yaw(
                desk_pos[env_idx, 0].item(),
                desk_pos[env_idx, 1].item(),
                desk_yaw_rad[env_idx].item(),
                area.center[0],
                area.center[1],
            )
            debug_obbs[env_idx].append(
                (area.name, make_obb(wx, wy, area.bbox, desk_yaw_rad[env_idx].item() + area.yaw))
            )

        # Keep the target object at the same local table position as the regular task.
        if "object" in table_prop_names and "object" in env.scene.keys():
            asset = env.scene["object"]
            meta = TABLE_PROP_META["object"]
            wx, wy = offset_from_yaw(
                desk_pos[env_idx, 0].item(),
                desk_pos[env_idx, 1].item(),
                desk_yaw_rad[env_idx].item(),
                OBJECT_TABLE_LOCAL_OFFSET[0],
                OBJECT_TABLE_LOCAL_OFFSET[1],
            )
            world_yaw = desk_yaw_rad[env_idx].item()
            pos = torch.tensor([wx, wy, DESK_OBJECT_Z], device=device).unsqueeze(0)
            yaw = torch.tensor([world_yaw], device=device)
            eid = env_ids[env_idx:env_idx+1]
            root_state = build_root_state(pos, yaw, env_origins, eid, asset.data.default_root_state)
            asset.write_root_state_to_sim(root_state, env_ids=eid)
            desk_placed.append(make_obb(OBJECT_TABLE_LOCAL_OFFSET[0], OBJECT_TABLE_LOCAL_OFFSET[1], meta.bbox, 0.0))
            debug_obbs[env_idx].append(("object", make_obb(wx, wy, meta.bbox, world_yaw)))

        extra_names = [name for name in table_prop_names if name != "object"]
        if not extra_names:
            continue

        # How many extra tabletop props in this env.
        count = rng.randint(min_table_objects, len(extra_names))
        visible_extra_names = set(extra_names[:count])

        if "desk_lamp" in visible_extra_names and "desk_lamp" in env.scene.keys():
            asset = env.scene["desk_lamp"]
            meta = TABLE_PROP_META["desk_lamp"]
            placed = False
            for _ in range(100):
                lx = rng.uniform(DESK_LAMP_LOCAL_X_RANGE[0], DESK_LAMP_LOCAL_X_RANGE[1])
                ly = rng.uniform(DESK_LAMP_LOCAL_Y_RANGE[0], DESK_LAMP_LOCAL_Y_RANGE[1])
                obj_yaw = DESK_LAMP_LOCAL_YAW

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

                    pos = torch.tensor([wx, wy, DESK_LAMP_Z], device=device).unsqueeze(0)
                    yaw = torch.tensor([world_yaw], device=device)
                    eid = env_ids[env_idx:env_idx+1]

                    root_state = build_root_state(pos, yaw, env_origins, eid, asset.data.default_root_state)
                    _write_tabletop_root_state(asset, "desk_lamp", root_state, eid)
                    debug_obbs[env_idx].append(
                        ("desk_lamp", make_obb(wx, wy, meta.bbox, world_yaw))
                    )
                    placed = True
                    break

            if not placed:
                pos = torch.tensor([0.0, 0.0, DESPAWN_Z], device=device).unsqueeze(0)
                yaw = torch.tensor([0.0], device=device)
                eid = env_ids[env_idx:env_idx+1]
                root_state = build_root_state(pos, yaw, env_origins, eid, asset.data.default_root_state)
                _write_tabletop_root_state(asset, "desk_lamp", root_state, eid)
                print(
                    f"[PLACEMENT_ERROR] env={_env_id_int(env_ids, env_idx)} "
                    f"object=desk_lamp tabletop_placement_failed despawning=true",
                    flush=True,
                )
        elif "desk_lamp" in extra_names and "desk_lamp" in env.scene.keys():
            asset = env.scene["desk_lamp"]
            pos = torch.tensor([0.0, 0.0, DESPAWN_Z], device=device).unsqueeze(0)
            yaw = torch.tensor([0.0], device=device)
            eid = env_ids[env_idx:env_idx+1]
            root_state = build_root_state(pos, yaw, env_origins, eid, asset.data.default_root_state)
            _write_tabletop_root_state(asset, "desk_lamp", root_state, eid)

        for name in extra_names:
            if name == "desk_lamp":
                continue
            if name not in env.scene.keys():
                continue
            asset = env.scene[name]
            meta = TABLE_PROP_META[name]
            visible = name in visible_extra_names

            if visible:
                placed = False
                for _ in range(100):
                    local_x_min = TABLETOP_CUBE_LOCAL_X_MIN if name in TABLETOP_CUBE_PROP_NAMES else DESK_LOCAL_X_MIN
                    local_x_max = TABLETOP_CUBE_LOCAL_X_MAX if name in TABLETOP_CUBE_PROP_NAMES else DESK_LOCAL_X_MAX
                    lx = rng.uniform(local_x_min, local_x_max)
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

                        pos = torch.tensor([wx, wy, DESK_OBJECT_Z], device=device).unsqueeze(0)
                        yaw = torch.tensor([world_yaw], device=device)
                        eid = env_ids[env_idx:env_idx+1]

                        root_state = build_root_state(pos, yaw, env_origins, eid, asset.data.default_root_state)
                        _write_tabletop_root_state(asset, name, root_state, eid)
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
                    _write_tabletop_root_state(asset, name, root_state, eid)
                    print(
                        f"[PLACEMENT_ERROR] env={_env_id_int(env_ids, env_idx)} "
                        f"object={name} tabletop_placement_failed despawning=true",
                        flush=True,
                    )
            else:
                pos = torch.tensor([0.0, 0.0, DESPAWN_Z], device=device).unsqueeze(0)
                yaw = torch.tensor([0.0], device=device)
                eid = env_ids[env_idx:env_idx+1]
                root_state = build_root_state(pos, yaw, env_origins, eid, asset.data.default_root_state)
                _write_tabletop_root_state(asset, name, root_state, eid)

    return debug_obbs
