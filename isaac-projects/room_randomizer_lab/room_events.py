# room_events.py
# Event term functions for room randomization.
# Uses OBB collision + continuous wall-zone sampling.

from __future__ import annotations

import math
import random
from typing import List, Optional

import torch
import omni.usd
from pxr import Gf

from isaaclab.envs import ManagerBasedEnv

from .constants import (
    CHAIR_BBOX,
    CHAIR_ORBIT_OFFSET,
    DESK_BBOX,
    DESK_LOCAL_X_MAX,
    DESK_LOCAL_Y_MAX,
    DESK_OBJECT_MARGIN,
    DESK_OBJECT_Z,
    DESPAWN_Z,
    FLOOR_Z,
    OBB_PLACEMENT_MARGIN,
    ROBOT_BBOX,
    ROBOT_ORBIT_OFFSET,
    ROBOT_Z,
    TABLE_FALLBACK_X,
    TABLE_FALLBACK_Y,
    TABLE_GROUP_MAX_TRIES,
    TABLE_PROP_META,
    TABLE_SAMPLE_X_MAX,
    TABLE_SAMPLE_X_MIN,
    TABLE_SAMPLE_Y_MAX,
    TABLE_SAMPLE_Y_MIN,
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
        env_id = _env_id_int(env_ids, env_idx)
        for rel_path in _DUPLICATE_VISUAL_PROPS_TO_HIDE:
            prim_path = f"/World/envs/env_{env_id}/{rel_path}"
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                from pxr import UsdGeom
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.MakeInvisible()
    
    _visual_props_hidden = True


# ======================================================================
# Combined event term
# ======================================================================

def randomize_room_layout(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    wall_prop_names: list[str],
    table_prop_names: list[str],
    min_table_objects: int = 2,
):
    """Randomize the full room layout for the given environments.

    Uses OBB collision detection and continuous zone sampling.
    """
    M = len(env_ids)
    device = env.device

    # Hide original static meshes embedded in RoomShell to avoid duplicates
    _hide_duplicate_visual_props(env_ids)

    # Per-environment list of placed OBBs.
    all_placed: List[List[OBB]] = [[] for _ in range(M)]
    all_placed_names: List[List[str]] = [[] for _ in range(M)]

    # Per-environment placement results for later phases.
    desk_positions = torch.zeros(M, 3, device=device)
    desk_yaws = torch.zeros(M, device=device)

    # --- Phase 1: Wall props ------------------------------------------
    _place_wall_props(env, env_ids, wall_prop_names, all_placed, all_placed_names)

    # --- Phase 2: Table group (desk + chair + robot) ------------------
    _place_table_group(env, env_ids, all_placed, all_placed_names, desk_positions, desk_yaws)

    # --- Phase 3: Tabletop objects ------------------------------------
    _place_desk_objects(env, env_ids, table_prop_names, desk_positions, desk_yaws, min_table_objects)


# ======================================================================
# Phase 1: Wall prop placement — continuous zone sampling
# ======================================================================

def _sample_wall_position(zone: WallZone, meta, rng: random.Random) -> tuple[float, float, float]:
    """Sample a random (cx, cy, yaw) along a wall zone strip.

    Returns:
        (cx, cy, yaw_rad) for the prop centre.
    """
    pos_along_wall = rng.uniform(zone.sample_min, zone.sample_max)

    # Apply per-prop wall offset (push away from wall surface).
    offset = meta.wall_offset

    if zone.wall == "back":
        cx = pos_along_wall
        cy = zone.fixed_coord + offset  # push into room (+Y)
    else:  # "right"
        cx = zone.fixed_coord - offset  # push into room (-X)
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


def _validate_table_group(
    table_obbs: list[tuple[str, OBB]],
    placed: List[OBB],
    placed_names: list[str],
) -> tuple[bool, list[str]]:
    """Validate table-group room bounds and overlaps against placed objects."""
    issues: list[str] = []

    for name, box in table_obbs:
        if not obb_inside_room(box):
            issues.append(f"{name} outside_room corners={_format_corners(box)}")

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
            if obb_overlap(a_box, b_box, margin=OBB_PLACEMENT_MARGIN):
                issues.append(
                    f"{a_name} overlaps {b_name} "
                    f"{a_name}_corners={_format_corners(a_box)} "
                    f"{b_name}_corners={_format_corners(b_box)}"
                )

    return len(issues) == 0, issues


def _make_table_group(dx: float, dy: float, dyaw: float) -> tuple[list[tuple[str, OBB]], tuple[float, float], tuple[float, float]]:
    """Build desk/chair/robot OBBs and satellite XY positions."""
    cx, cy = offset_from_yaw(dx, dy, dyaw, CHAIR_ORBIT_OFFSET[0], CHAIR_ORBIT_OFFSET[1])
    rx, ry = offset_from_yaw(dx, dy, dyaw, ROBOT_ORBIT_OFFSET[0], ROBOT_ORBIT_OFFSET[1])

    chair_yaw = dyaw + math.pi
    robot_yaw = dyaw - math.pi / 2

    table_obbs = [
        ("desk", make_obb(dx, dy, DESK_BBOX, dyaw)),
        ("chair", make_obb(cx, cy, CHAIR_BBOX, chair_yaw)),
        ("ridgeback", make_obb(rx, ry, ROBOT_BBOX, robot_yaw)),
    ]
    return table_obbs, (cx, cy), (rx, ry)


def _debug_table_group(env_id: int, table_obbs: list[tuple[str, OBB]], placed: List[OBB], placed_names: list[str]):
    """Print table group OBBs and overlap diagnostics."""
    for name, box in table_obbs:
        _print_obb_debug(name, env_id, box)

    for i, (a_name, a_box) in enumerate(table_obbs):
        for b_name, b_box in table_obbs[i + 1:]:
            overlaps = obb_overlap(a_box, b_box, margin=OBB_PLACEMENT_MARGIN)
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


def _place_wall_props(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    wall_prop_names: list[str],
    all_placed: List[List[OBB]],
    all_placed_names: List[List[str]],
):
    """Place wall props using continuous zone sampling + OBB collision."""
    M = len(env_ids)
    device = env.device
    rng = random.Random()

    # Shuffle order: tall props first for best placement priority.
    sorted_names = sorted(
        wall_prop_names,
        key=lambda n: not WALL_PROP_META[n].tall,
    )
    debug_records: List[List[tuple[str, Optional[OBB]]]] = [[] for _ in range(M)]

    for name in sorted_names:
        meta = WALL_PROP_META[name]
        asset = env.scene[name]

        pos_local = torch.zeros(M, 3, device=device)
        yaw_rad = torch.zeros(M, device=device)

        for env_idx in range(M):
            # Try random positions across allowed walls.
            success = False

            # Shuffle which walls to try.
            allowed_zones = [z for z in WALL_ZONES if z.wall in meta.allowed_walls]
            rng.shuffle(allowed_zones)

            for _ in range(100):
                # Pick a random wall zone.
                zone = rng.choice(allowed_zones)
                cx, cy, yaw = _sample_wall_position(zone, meta, rng)

                candidate = make_obb(cx, cy, meta.bbox, yaw)

                # Check room bounds.
                if not obb_inside_room(candidate):
                    continue

                # Check overlap with already-placed props.
                if obb_overlap_any(candidate, all_placed[env_idx], margin=OBB_PLACEMENT_MARGIN):
                    continue

                # Valid placement!
                pos_local[env_idx] = torch.tensor([cx, cy, FLOOR_Z], device=device)
                yaw_rad[env_idx] = yaw
                all_placed[env_idx].append(candidate)
                all_placed_names[env_idx].append(name)
                debug_records[env_idx].append((name, candidate))
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

        # Write to simulation.
        root_state = build_root_state(
            pos_local, yaw_rad,
            env.scene.env_origins, env_ids,
            asset.data.default_root_state,
        )
        asset.write_root_state_to_sim(root_state, env_ids=env_ids)

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


# ======================================================================
# Phase 2: Table group — continuous interior sampling
# ======================================================================

def _place_table_group(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    all_placed: List[List[OBB]],
    all_placed_names: List[List[str]],
    desk_positions: torch.Tensor,
    desk_yaws: torch.Tensor,
):
    """Place desk + chair + robot with continuous sampling and OBB collision."""
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
            # Random position and yaw in room interior.
            dx = rng.uniform(TABLE_SAMPLE_X_MIN, TABLE_SAMPLE_X_MAX)
            dy = rng.uniform(TABLE_SAMPLE_Y_MIN, TABLE_SAMPLE_Y_MAX)
            dyaw = rng.uniform(0, 2 * math.pi)

            table_obbs, _, _ = _make_table_group(dx, dy, dyaw)
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
            # Fallback: keep fixed XY, but still require a fully valid yaw.
            dx, dy = TABLE_FALLBACK_X, TABLE_FALLBACK_Y
            fallback_issues: list[str] = []
            for _ in range(TABLE_GROUP_MAX_TRIES):
                dyaw = rng.uniform(0, 2 * math.pi)
                table_obbs, _, _ = _make_table_group(dx, dy, dyaw)
                valid, fallback_issues = _validate_table_group(
                    table_obbs, all_placed[env_idx], all_placed_names[env_idx]
                )
                if valid:
                    desk_positions[env_idx] = torch.tensor([dx, dy, FLOOR_Z], device=device)
                    desk_yaws[env_idx] = dyaw
                    selected_obbs = table_obbs
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

    # --- Write desk, chair, ridgeback to sim --------------------------
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

    # Build batched positions for satellites.
    chair_pos = offset_from_yaw_batched(
        desk_positions, desk_yaws,
        CHAIR_ORBIT_OFFSET[0], CHAIR_ORBIT_OFFSET[1], FLOOR_Z,
    )
    chair_yaw = desk_yaws + math.pi

    robot_pos = offset_from_yaw_batched(
        desk_positions, desk_yaws,
        ROBOT_ORBIT_OFFSET[0], ROBOT_ORBIT_OFFSET[1], ROBOT_Z,
    )
    robot_yaw = desk_yaws - math.pi / 2

    invalid_mask = ~group_placed_mask
    if torch.any(invalid_mask):
        desk_positions[invalid_mask, 0:2] = 0.0
        desk_positions[invalid_mask, 2] = DESPAWN_Z
        chair_pos[invalid_mask, 0:2] = 0.0
        chair_pos[invalid_mask, 2] = DESPAWN_Z
        robot_pos[invalid_mask, 0:2] = 0.0
        robot_pos[invalid_mask, 2] = DESPAWN_Z
        chair_yaw[invalid_mask] = 0.0
        robot_yaw[invalid_mask] = 0.0

    desk_asset = env.scene["desk"]
    desk_state = build_root_state(desk_positions, desk_yaws, env_origins, env_ids, desk_asset.data.default_root_state)
    desk_asset.write_root_state_to_sim(desk_state, env_ids=env_ids)

    chair_asset = env.scene["chair"]
    chair_state = build_root_state(chair_pos, chair_yaw, env_origins, env_ids, chair_asset.data.default_root_state)
    chair_asset.write_root_state_to_sim(chair_state, env_ids=env_ids)

    try:
        robot_asset = env.scene["ridgeback"]
    except KeyError:
        robot_asset = None
    if robot_asset is not None:
        robot_state = build_root_state(robot_pos, robot_yaw, env_origins, env_ids, robot_asset.data.default_root_state)
        robot_asset.write_root_state_to_sim(robot_state, env_ids=env_ids)


# ======================================================================
# Phase 3: Tabletop objects — OBB collision on desk surface
# ======================================================================

def _place_desk_objects(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    table_prop_names: list[str],
    desk_pos: torch.Tensor,
    desk_yaw_rad: torch.Tensor,
    min_table_objects: int = 2,
):
    """Place tabletop objects on the desk surface with rejection sampling.

    Randomly despawns 0–1 objects to achieve variable count (min_table_objects
    to len(table_prop_names)).
    """
    if not table_prop_names:
        return

    M = len(env_ids)
    device = env.device
    env_origins = env.scene.env_origins
    rng = random.Random()

    num_total = len(table_prop_names)

    for env_idx in range(M):
        if desk_pos[env_idx, 2].item() <= DESPAWN_Z * 0.5:
            for name in table_prop_names:
                asset = env.scene[name]
                pos = torch.tensor([0.0, 0.0, DESPAWN_Z], device=device).unsqueeze(0)
                yaw = torch.tensor([0.0], device=device)
                eid = env_ids[env_idx:env_idx+1]
                root_state = build_root_state(pos, yaw, env_origins, eid, asset.data.default_root_state)
                asset.write_root_state_to_sim(root_state, env_ids=eid)
            print(
                f"[PLACEMENT_ERROR] env={_env_id_int(env_ids, env_idx)} "
                f"tabletop_objects skipped table_group_not_placed=true",
                flush=True,
            )
            continue

        # How many objects in this env (2 or 3).
        count = rng.randint(min_table_objects, num_total)
        desk_placed: List[OBB] = []

        for i, name in enumerate(table_prop_names):
            asset = env.scene[name]
            meta = TABLE_PROP_META[name]
            visible = i < count

            if visible:
                # Rejection-sample local (x, y) on desk surface.
                placed = False
                for _ in range(100):
                    lx = rng.uniform(-DESK_LOCAL_X_MAX, DESK_LOCAL_X_MAX)
                    ly = rng.uniform(-DESK_LOCAL_Y_MAX, DESK_LOCAL_Y_MAX)
                    obj_yaw = rng.uniform(0, 2 * math.pi)

                    candidate = make_obb(lx, ly, meta.bbox, obj_yaw)
                    if not obb_overlap_any(candidate, desk_placed, margin=DESK_OBJECT_MARGIN):
                        desk_placed.append(candidate)

                        # Transform to world.
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
                        asset.write_root_state_to_sim(root_state, env_ids=eid)
                        placed = True
                        break

                if not placed:
                    # Couldn't fit safely; despawn instead of stacking objects.
                    pos = torch.tensor([0.0, 0.0, DESPAWN_Z], device=device).unsqueeze(0)
                    yaw = torch.tensor([0.0], device=device)
                    eid = env_ids[env_idx:env_idx+1]
                    root_state = build_root_state(pos, yaw, env_origins, eid, asset.data.default_root_state)
                    asset.write_root_state_to_sim(root_state, env_ids=eid)
                    print(
                        f"[PLACEMENT_ERROR] env={_env_id_int(env_ids, env_idx)} "
                        f"object={name} tabletop_placement_failed despawning=true",
                        flush=True,
                    )
            else:
                # Despawn.
                pos = torch.tensor([0.0, 0.0, DESPAWN_Z], device=device).unsqueeze(0)
                yaw = torch.tensor([0.0], device=device)
                eid = env_ids[env_idx:env_idx+1]
                root_state = build_root_state(pos, yaw, env_origins, eid, asset.data.default_root_state)
                asset.write_root_state_to_sim(root_state, env_ids=eid)
