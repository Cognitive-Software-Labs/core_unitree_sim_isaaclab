#!/usr/bin/env python3
"""Standalone visual test for OBB-based room placement.

Runs the placement algorithms WITHOUT Isaac Sim / Isaac Lab.
Produces a matplotlib figure showing oriented bounding boxes.

Usage:
    python test_placement.py
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constants import (
    BBox,
    CHAIR_BBOX,
    CHAIR_ORBIT_OFFSET,
    DESK_BBOX,
    DESK_LOCAL_X_MAX,
    DESK_LOCAL_Y_MAX,
    DESK_OBJECT_MARGIN,
    DESPAWN_Z,
    FALLBACK_NO_SPAWN_BOUNDARIES,
    FLOOR_Z,
    OBB_PLACEMENT_MARGIN,
    ROBOT_BBOX,
    ROBOT_FACING_LAYOUTS,
    ROBOT_FACING_MAX_YAW_OFFSET_RAD,
    ROBOT_FACING_YAW_JITTER_RAD,
    ROBOT_ORBIT_OFFSET,
    RIGHT_WALL_CONTACT_CLEARANCE,
    ROOM_X_MAX,
    ROOM_X_MIN,
    ROOM_Y_MAX,
    ROOM_Y_MIN,
    SPAWN_BOUNDARY_TOLERANCE,
    SPAWN_REGION_SEED,
    STATIC_ROOM_OBSTACLES,
    TABLE_GROUP_X_MAX,
    TABLE_GROUP_X_MIN,
    TABLE_GROUP_Y_MAX,
    TABLE_GROUP_Y_MIN,
    TABLE_GROUP_MAX_TRIES,
    TABLE_PROP_META,
    TABLE_SAMPLE_X_MAX,
    TABLE_SAMPLE_X_MIN,
    TABLE_SAMPLE_Y_MAX,
    TABLE_SAMPLE_Y_MIN,
    WALL_PROP_META,
    WALL_ZONES,
)
from placement_utils import (
    make_obb,
    obb_corners,
    obb_inside_room,
    obb_overlap,
    obb_overlap_any,
    offset_from_yaw,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Polygon
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not installed — will print text output only.")


# =====================================================================
# Pure-Python placement (mirrors the event term logic)
# =====================================================================

def _sample_wall_position(zone, meta, rng):
    pos = rng.uniform(zone.sample_min, zone.sample_max)
    offsets = getattr(meta, "wall_offsets", None)
    offset = offsets.get(zone.wall, meta.wall_offset) if offsets is not None else meta.wall_offset
    if zone.wall == "back":
        cx, cy = pos, zone.fixed_coord + offset
    else:
        cx, cy = zone.fixed_coord - offset, pos
    yaw = zone.base_yaw + meta.yaw_offset

    return cx, cy, yaw


def _wall_prop_footprint_obb(root_x, root_y, yaw, meta):
    footprint_x, footprint_y = offset_from_yaw(
        root_x, root_y, yaw, meta.bbox_center[0], meta.bbox_center[1]
    )
    return make_obb(footprint_x, footprint_y, meta.bbox, yaw)


def _right_wall_support_surfaces(candidate, static_wall_obbs):
    candidate_corners = obb_corners(*candidate)
    candidate_y_min = min(y for _, y in candidate_corners)
    candidate_y_max = max(y for _, y in candidate_corners)
    support_surfaces = []

    for wall_name, wall_box in static_wall_obbs:
        if _spawn_boundary_axis(wall_box) != 0 or SPAWN_REGION_SEED[0] >= wall_box[0]:
            continue
        wall_corners = obb_corners(*wall_box)
        wall_y_min = min(y for _, y in wall_corners)
        wall_y_max = max(y for _, y in wall_corners)
        if candidate_y_max < wall_y_min or candidate_y_min > wall_y_max:
            continue
        support_surfaces.append((wall_name, min(x for x, _ in wall_corners)))

    return support_surfaces


def _snap_right_wall_root_to_surface(root_x, root_y, yaw, meta, static_wall_obbs):
    candidate = _wall_prop_footprint_obb(root_x, root_y, yaw, meta)
    candidate_corners = obb_corners(*candidate)
    support_surfaces = _right_wall_support_surfaces(candidate, static_wall_obbs)
    if not support_surfaces:
        return None

    wall_face_x = min(face_x for _, face_x in support_surfaces)
    target_face_x = wall_face_x - RIGHT_WALL_CONTACT_CLEARANCE
    candidate_right_x = max(x for x, _ in candidate_corners)
    snapped_root_x = root_x + target_face_x - candidate_right_x
    snapped_candidate = _wall_prop_footprint_obb(snapped_root_x, root_y, yaw, meta)
    snapped_right_x = max(x for x, _ in obb_corners(*snapped_candidate))
    gap = wall_face_x - snapped_right_x
    support_names = tuple(name for name, _ in support_surfaces)
    return snapped_root_x, support_names, gap


def _spawn_boundary_axis(boundary_box):
    return 0 if boundary_box[2] <= boundary_box[3] else 1


def _outside_spawn_region_issue(name, box, spawn_boundaries):
    for boundary_name, boundary_box in spawn_boundaries:
        axis = _spawn_boundary_axis(boundary_box)
        center = boundary_box[axis]
        seed = SPAWN_REGION_SEED[axis]
        valid_sign = 1.0 if seed >= center else -1.0
        for corner in obb_corners(*box):
            coord = corner[axis]
            if valid_sign * (coord - center) < -SPAWN_BOUNDARY_TOLERANCE:
                axis_name = "x" if axis == 0 else "y"
                return f"{name} outside_spawn_region boundary={boundary_name} axis={axis_name} limit={center:+.3f}"
    return None


def _wall_zone_supports_boundary(zone, boundary_box):
    axis = _spawn_boundary_axis(boundary_box)
    center = boundary_box[axis]
    seed = SPAWN_REGION_SEED[axis]
    if zone.wall == "back":
        return axis == 1 and seed > center
    if zone.wall == "right":
        return axis == 0 and seed < center
    return False


def _blocking_static_wall_overlap(candidate, zone, static_wall_obbs):
    for wall_name, wall_box in static_wall_obbs:
        if _wall_zone_supports_boundary(zone, wall_box):
            continue
        if obb_overlap(candidate, wall_box, margin=OBB_PLACEMENT_MARGIN):
            return wall_name
    return None


def _make_table_group(dx, dy, dyaw):
    cx, cy = offset_from_yaw(dx, dy, dyaw, CHAIR_ORBIT_OFFSET[0], CHAIR_ORBIT_OFFSET[1])
    rx, ry = offset_from_yaw(dx, dy, dyaw, ROBOT_ORBIT_OFFSET[0], ROBOT_ORBIT_OFFSET[1])

    return {
        "desk": make_obb(dx, dy, DESK_BBOX, dyaw),
        "chair": make_obb(cx, cy, CHAIR_BBOX, dyaw + math.pi),
        "robot": make_obb(rx, ry, ROBOT_BBOX, dyaw - math.pi / 2),
    }


def _make_table_group_from_robot(rx, ry, robot_yaw):
    desk_yaw = robot_yaw + math.pi / 2
    dx, dy = offset_from_yaw(
        rx,
        ry,
        desk_yaw,
        -ROBOT_ORBIT_OFFSET[0],
        -ROBOT_ORBIT_OFFSET[1],
    )
    return _make_table_group(dx, dy, desk_yaw), (dx, dy), desk_yaw


def _normalize_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _sample_wall_facing_robot_pose(rng):
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
    if ROBOT_FACING_YAW_JITTER_RAD > 0.0:
        robot_yaw += rng.uniform(-ROBOT_FACING_YAW_JITTER_RAD, ROBOT_FACING_YAW_JITTER_RAD)
    return rx, ry, robot_yaw, layout


def _obb_inside_table_group_bounds(box):
    for wx, wy in obb_corners(*box):
        if wx < TABLE_GROUP_X_MIN or wx > TABLE_GROUP_X_MAX:
            return False
        if wy < TABLE_GROUP_Y_MIN or wy > TABLE_GROUP_Y_MAX:
            return False
    return True


def _validate_table_group(table_obbs, placed_obbs, spawn_boundaries):
    for name, box in table_obbs.items():
        if not obb_inside_room(box):
            return False
        if not _obb_inside_table_group_bounds(box):
            return False
        if _outside_spawn_region_issue(name, box, spawn_boundaries) is not None:
            return False

    for box in table_obbs.values():
        if obb_overlap_any(box, placed_obbs, margin=OBB_PLACEMENT_MARGIN):
            return False

    values = list(table_obbs.values())
    for i, a in enumerate(values):
        for b in values[i + 1:]:
            if obb_overlap(a, b, margin=OBB_PLACEMENT_MARGIN):
                return False

    return True


def _obb_to_result(box):
    return {"cx": box[0], "cy": box[1], "yaw": box[4], "hw": box[2], "hd": box[3]}


def randomize_one_room(rng=None):
    rng = rng or random.Random()
    placed_obbs = []
    static_wall_obbs = [
        (obstacle.name, make_obb(obstacle.center[0], obstacle.center[1], obstacle.bbox, obstacle.yaw))
        for obstacle in STATIC_ROOM_OBSTACLES
    ]
    spawn_boundaries = [
        (boundary.name, make_obb(boundary.center[0], boundary.center[1], boundary.bbox, boundary.yaw))
        for boundary in FALLBACK_NO_SPAWN_BOUNDARIES
    ]
    results = {
        "wall_props": [],
        "static_walls": [],
        "desk": None,
        "chair": None,
        "robot": None,
        "desk_objects": [],
    }

    # --- Phase 1: Wall props ---
    sorted_names = sorted(WALL_PROP_META.keys(), key=lambda n: not WALL_PROP_META[n].tall)

    for name in sorted_names:
        meta = WALL_PROP_META[name]
        allowed_zones = [z for z in WALL_ZONES if z.wall in meta.allowed_walls]

        success = False
        for _ in range(100):
            zone = rng.choice(allowed_zones)
            cx, cy, yaw = _sample_wall_position(zone, meta, rng)
            wall_contact = None
            if zone.wall == "right":
                wall_contact = _snap_right_wall_root_to_surface(
                    cx, cy, yaw, meta, static_wall_obbs
                )
                if wall_contact is None:
                    continue
                cx, _, _ = wall_contact
            candidate = _wall_prop_footprint_obb(cx, cy, yaw, meta)

            if not obb_inside_room(candidate):
                continue
            if _outside_spawn_region_issue(name, candidate, spawn_boundaries) is not None:
                continue
            if _blocking_static_wall_overlap(candidate, zone, static_wall_obbs) is not None:
                continue
            if obb_overlap_any(candidate, placed_obbs, margin=OBB_PLACEMENT_MARGIN):
                continue

            placed_obbs.append(candidate)
            record = {
                "name": name,
                "cx": candidate[0], "cy": candidate[1],
                "hw": meta.bbox.half_w, "hd": meta.bbox.half_d,
                "yaw": yaw,
                "tall": meta.tall,
                "wall": zone.wall,
            }
            if wall_contact is not None:
                _, support_names, wall_gap = wall_contact
                record["right_wall_support"] = support_names
                record["right_wall_gap"] = wall_gap
            results["wall_props"].append(record)
            success = True
            break

        if not success:
            results["wall_props"].append({
                "name": name, "cx": 0, "cy": 0, "hw": 0, "hd": 0,
                "yaw": 0, "tall": meta.tall, "wall": "despawned",
            })

    # --- Static RoomShell walls ---
    # These are planner obstacles from the authored room shell, including the
    # new front wall segment. The left/open side stays virtual-only via the
    # table-group bounds below.
    for obstacle_name, box in static_wall_obbs:
        placed_obbs.append(box)
        record = _obb_to_result(box)
        record["name"] = obstacle_name
        results["static_walls"].append(record)

    # --- Phase 2: Table group ---
    tg_success = False
    tg_source = "failed"
    tg_layout = "none"
    selected_table_obbs = None

    for _ in range(TABLE_GROUP_MAX_TRIES):
        rx, ry, robot_yaw, layout = _sample_wall_facing_robot_pose(rng)

        table_obbs, _, _ = _make_table_group_from_robot(rx, ry, robot_yaw)
        if not _validate_table_group(table_obbs, placed_obbs, spawn_boundaries):
            continue

        selected_table_obbs = table_obbs
        tg_success = True
        tg_source = "random"
        tg_layout = layout.name
        break

    if not tg_success:
        for _ in range(TABLE_GROUP_MAX_TRIES):
            rx, ry, robot_yaw, layout = _sample_wall_facing_robot_pose(rng)
            table_obbs, _, _ = _make_table_group_from_robot(rx, ry, robot_yaw)
            if not _validate_table_group(table_obbs, placed_obbs, spawn_boundaries):
                continue

            selected_table_obbs = table_obbs
            tg_success = True
            tg_source = "fallback"
            tg_layout = layout.name
            break

    if tg_success:
        placed_obbs.extend(selected_table_obbs.values())
        results["desk"] = _obb_to_result(selected_table_obbs["desk"])
        results["chair"] = _obb_to_result(selected_table_obbs["chair"])
        results["robot"] = _obb_to_result(selected_table_obbs["robot"])
        desk_x = results["desk"]["cx"]
        desk_y = results["desk"]["cy"]
        desk_yaw = results["desk"]["yaw"]
    else:
        results["desk"] = None
        results["chair"] = None
        results["robot"] = None

    results["tg_success"] = tg_success
    results["tg_source"] = tg_source
    results["tg_layout"] = tg_layout

    if not tg_success:
        return results

    # --- Phase 3: Desk objects ---
    desk_placed_obbs = []
    count = rng.randint(2, len(TABLE_PROP_META))
    for i, (name, meta) in enumerate(TABLE_PROP_META.items()):
        if i >= count:
            continue
        for _ in range(100):
            lx = rng.uniform(-DESK_LOCAL_X_MAX, DESK_LOCAL_X_MAX)
            ly = rng.uniform(-DESK_LOCAL_Y_MAX, DESK_LOCAL_Y_MAX)
            obj_yaw = rng.uniform(0, 2 * math.pi)
            candidate = make_obb(lx, ly, meta.bbox, obj_yaw)
            if not obb_overlap_any(candidate, desk_placed_obbs, margin=DESK_OBJECT_MARGIN):
                wx, wy = offset_from_yaw(desk_x, desk_y, desk_yaw, lx, ly)
                world_box = make_obb(wx, wy, meta.bbox, desk_yaw + obj_yaw)
                if _outside_spawn_region_issue(name, world_box, spawn_boundaries) is not None:
                    continue
                desk_placed_obbs.append(candidate)
                results["desk_objects"].append({
                    "name": name, "wx": wx, "wy": wy,
                    "lx": lx, "ly": ly,
                    "hw": meta.bbox.half_w, "hd": meta.bbox.half_d,
                    "yaw": desk_yaw + obj_yaw,
                })
                break

    return results


# =====================================================================
# Visualization
# =====================================================================

def _draw_obb(ax, cx, cy, hw, hd, yaw, color, alpha=0.3, label=None):
    """Draw a rotated rectangle on the axes."""
    corners = obb_corners(cx, cy, hw, hd, yaw)
    poly = Polygon(corners, closed=True, facecolor=color, edgecolor=color,
                   alpha=alpha, linewidth=1.5)
    ax.add_patch(poly)
    ax.plot(cx, cy, ".", color=color, markersize=3)
    # Draw yaw arrow.
    arrow_len = max(hw, hd) * 0.6
    dx = arrow_len * math.cos(yaw)
    dy = arrow_len * math.sin(yaw)
    ax.annotate("", xy=(cx + dx, cy + dy), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0))
    if label:
        ax.annotate(label, (cx, cy), fontsize=3.5, ha="center", va="center",
                    color=color, fontweight="bold")


def draw_room(ax, room, room_index):
    ax.set_xlim(ROOM_X_MIN - 1, ROOM_X_MAX + 1)
    ax.set_ylim(ROOM_Y_MIN - 1, ROOM_Y_MAX + 1)
    ax.set_aspect("equal")
    ax.set_title(f"Room {room_index + 1}", fontsize=10, fontweight="bold")

    # Room bounds.
    room_rect = patches.Rectangle(
        (ROOM_X_MIN, ROOM_Y_MIN),
        ROOM_X_MAX - ROOM_X_MIN,
        ROOM_Y_MAX - ROOM_Y_MIN,
        linewidth=2, edgecolor="black", facecolor="#f5f5f0",
    )
    ax.add_patch(room_rect)

    # Virtual operating bounds for table/chair/robot. The left/open side has no
    # wall mesh, but the planner still keeps the group inside this rectangle.
    table_bounds = patches.Rectangle(
        (TABLE_GROUP_X_MIN, TABLE_GROUP_Y_MIN),
        TABLE_GROUP_X_MAX - TABLE_GROUP_X_MIN,
        TABLE_GROUP_Y_MAX - TABLE_GROUP_Y_MIN,
        linewidth=1.2,
        edgecolor="#455a64",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(table_bounds)

    # Walls.
    ax.plot([ROOM_X_MIN, ROOM_X_MAX], [ROOM_Y_MIN, ROOM_Y_MIN], "k-", linewidth=3)
    ax.plot([ROOM_X_MAX, ROOM_X_MAX], [ROOM_Y_MIN, ROOM_Y_MAX], "k-", linewidth=3)

    # Wall zones (faint strips).
    for zone in WALL_ZONES:
        if zone.wall == "back":
            ax.axhline(y=zone.fixed_coord, color="#aaa", linestyle="--", linewidth=0.5)
            ax.plot([zone.sample_min, zone.sample_max], [zone.fixed_coord, zone.fixed_coord],
                    color="#ddd", linewidth=4, solid_capstyle="round")
        else:
            ax.axvline(x=zone.fixed_coord, color="#aaa", linestyle="--", linewidth=0.5)
            ax.plot([zone.fixed_coord, zone.fixed_coord], [zone.sample_min, zone.sample_max],
                    color="#ddd", linewidth=4, solid_capstyle="round")

    # Static wall OBBs used by the planner.
    for wall in room["static_walls"]:
        label = wall["name"].replace("static_", "").replace("_", "\n")
        _draw_obb(
            ax,
            wall["cx"],
            wall["cy"],
            wall["hw"],
            wall["hd"],
            wall["yaw"],
            "#b71c1c",
            alpha=0.18,
            label=label,
        )

    # Wall props.
    for wp in room["wall_props"]:
        if wp["wall"] == "despawned":
            continue
        color = "#d32f2f" if wp["tall"] else "#1976d2"
        label = wp["name"].replace("_", "\n")
        _draw_obb(ax, wp["cx"], wp["cy"], wp["hw"], wp["hd"], wp["yaw"], color, alpha=0.35, label=label)

    # Desk.
    d = room["desk"]
    if d is None:
        ax.annotate("TABLE GROUP SKIPPED", (-7.0, -7.5), fontsize=8, ha="center", color="#b00020")
        return
    _draw_obb(ax, d["cx"], d["cy"], d["hw"], d["hd"], d["yaw"], "#4caf50", alpha=0.25, label="DESK")

    # Chair.
    c = room["chair"]
    _draw_obb(ax, c["cx"], c["cy"], c["hw"], c["hd"], c["yaw"], "#ff9800", alpha=0.35, label="CHAIR")

    # Robot.
    r = room["robot"]
    _draw_obb(ax, r["cx"], r["cy"], r["hw"], r["hd"], r["yaw"], "#9c27b0", alpha=0.35, label="ROBOT")

    # Desk objects.
    for obj in room["desk_objects"]:
        _draw_obb(ax, obj["wx"], obj["wy"], obj["hw"], obj["hd"], obj["yaw"], "#ff5722", alpha=0.5)


def print_room_text(room, room_index):
    print(f"\n{'='*60}")
    print(f"  ROOM {room_index + 1}")
    print(f"{'='*60}")

    placed = [wp for wp in room["wall_props"] if wp["wall"] != "despawned"]
    despawned = [wp for wp in room["wall_props"] if wp["wall"] == "despawned"]
    print(f"\n  Wall Props: {len(placed)} placed, {len(despawned)} despawned")

    for wp in placed:
        tag = " [TALL]" if wp["tall"] else ""
        print(f"    {wp['name']:20s}  ({wp['cx']:+6.2f}, {wp['cy']:+6.2f})  wall={wp['wall']:5s}  yaw={math.degrees(wp['yaw']):+6.1f}°{tag}")
    for wp in despawned:
        print(f"    {wp['name']:20s}  DESPAWNED")

    print(f"\n  Static Walls: {len(room['static_walls'])} planner obstacles")
    for wall in room["static_walls"]:
        print(f"    {wall['name']:34s}  ({wall['cx']:+6.2f}, {wall['cy']:+6.2f})  yaw={math.degrees(wall['yaw']):+6.1f}°")

    d = room["desk"]
    if d is None:
        print(f"\n  Table Group: SKIPPED (no valid random or fallback placement)")
        print(f"\n  Desk Objects: skipped")
    else:
        status = room.get("tg_source", "unknown")
        layout = room.get("tg_layout", "none")
        print(
            f"\n  Desk:   ({d['cx']:+6.2f}, {d['cy']:+6.2f})  "
            f"yaw={math.degrees(d['yaw']):+6.1f}°  source={status} layout={layout}"
        )
        c = room["chair"]
        print(f"  Chair:  ({c['cx']:+6.2f}, {c['cy']:+6.2f})")
        r = room["robot"]
        print(f"  Robot:  ({r['cx']:+6.2f}, {r['cy']:+6.2f})")

        print(f"\n  Desk Objects ({len(room['desk_objects'])} placed):")
        for obj in room["desk_objects"]:
            print(f"    {obj['name']:20s}  world=({obj['wx']:+6.2f}, {obj['wy']:+6.2f})  local=({obj['lx']:+5.2f}, {obj['ly']:+5.2f})")

    # Validation: check all OBBs are inside room.
    issues = []
    named_boxes = []
    static_boxes = []
    spawn_boundaries = [
        (boundary.name, make_obb(boundary.center[0], boundary.center[1], boundary.bbox, boundary.yaw))
        for boundary in FALLBACK_NO_SPAWN_BOUNDARIES
    ]
    for wall in room["static_walls"]:
        box = make_obb(wall["cx"], wall["cy"], BBox(wall["hw"], wall["hd"]), wall["yaw"])
        static_boxes.append((wall["name"], box))
    for wp in placed:
        box = make_obb(wp["cx"], wp["cy"], BBox(wp["hw"], wp["hd"]), wp["yaw"])
        named_boxes.append((wp["name"], box))
        if not obb_inside_room(box):
            issues.append(f"  ⚠️  {wp['name']} OBB extends outside room!")
        spawn_issue = _outside_spawn_region_issue(wp["name"], box, spawn_boundaries)
        if spawn_issue is not None:
            issues.append(f"  ⚠️  {spawn_issue}")
        if wp["wall"] == "right":
            support_surfaces = _right_wall_support_surfaces(box, static_boxes)
            if not support_surfaces:
                issues.append(f"  ⚠️  {wp['name']} has no supporting right-wall segment!")
                continue
            right_wall_face_x = min(face_x for _, face_x in support_surfaces)
            gap = right_wall_face_x - max(x for x, _ in obb_corners(*box))
            if not math.isclose(gap, RIGHT_WALL_CONTACT_CLEARANCE, abs_tol=1.0e-6):
                issues.append(
                    f"  ⚠️  {wp['name']} right-wall gap is {gap:.6f} m; "
                    f"expected {RIGHT_WALL_CONTACT_CLEARANCE:.6f} m!"
                )
    if d is not None:
        for label, item in [("Desk", d), ("Chair", room["chair"]), ("Robot", room["robot"])]:
            box = make_obb(item["cx"], item["cy"], BBox(item["hw"], item["hd"]), item["yaw"])
            named_boxes.append((label, box))
            if not obb_inside_room(box):
                issues.append(f"  ⚠️  {label} OBB extends outside room!")
            if not _obb_inside_table_group_bounds(box):
                issues.append(f"  ⚠️  {label} OBB extends outside table-group bounds!")
            spawn_issue = _outside_spawn_region_issue(label, box, spawn_boundaries)
            if spawn_issue is not None:
                issues.append(f"  ⚠️  {spawn_issue}")
            for wall_name, wall_box in static_boxes:
                if obb_overlap(box, wall_box, margin=OBB_PLACEMENT_MARGIN):
                    issues.append(f"  ⚠️  {label} overlaps {wall_name}!")

        for obj in room["desk_objects"]:
            box = make_obb(obj["wx"], obj["wy"], BBox(obj["hw"], obj["hd"]), obj["yaw"])
            spawn_issue = _outside_spawn_region_issue(obj["name"], box, spawn_boundaries)
            if spawn_issue is not None:
                issues.append(f"  ⚠️  {spawn_issue}")

    for i, (a_name, a_box) in enumerate(named_boxes):
        for b_name, b_box in named_boxes[i + 1:]:
            if obb_overlap(a_box, b_box, margin=OBB_PLACEMENT_MARGIN):
                issues.append(f"  ⚠️  {a_name} overlaps {b_name}!")

    if issues:
        print(f"\n  VALIDATION ISSUES:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print(f"\n  ✅ All OBBs inside active bounds and non-overlapping")

    return issues


# =====================================================================
# Deterministic right-wall regression coverage
# =====================================================================

def _fallback_static_wall_obbs():
    return [
        (obstacle.name, make_obb(obstacle.center[0], obstacle.center[1], obstacle.bbox, obstacle.yaw))
        for obstacle in STATIC_ROOM_OBSTACLES
    ]


def _assert_right_wall_contact(name, root_x, root_y, yaw, meta, static_wall_obbs):
    contact = _snap_right_wall_root_to_surface(
        root_x, root_y, yaw, meta, static_wall_obbs
    )
    if contact is None:
        raise AssertionError(f"{name}: no right-wall support found")

    snapped_root_x, support_names, reported_gap = contact
    candidate = _wall_prop_footprint_obb(snapped_root_x, root_y, yaw, meta)
    support_surfaces = _right_wall_support_surfaces(candidate, static_wall_obbs)
    if not support_surfaces:
        raise AssertionError(f"{name}: snapped candidate lost right-wall support")

    wall_face_x = min(face_x for _, face_x in support_surfaces)
    measured_gap = wall_face_x - max(x for x, _ in obb_corners(*candidate))
    if not math.isclose(measured_gap, RIGHT_WALL_CONTACT_CLEARANCE, abs_tol=1.0e-9):
        raise AssertionError(
            f"{name}: measured gap {measured_gap:.9f} != "
            f"{RIGHT_WALL_CONTACT_CLEARANCE:.9f}"
        )
    if not math.isclose(reported_gap, measured_gap, abs_tol=1.0e-12):
        raise AssertionError(
            f"{name}: reported gap {reported_gap:.12f} != measured {measured_gap:.12f}"
        )
    if set(support_names) != {support_name for support_name, _ in support_surfaces}:
        raise AssertionError(f"{name}: reported support segments do not match geometry")

    support_by_name = dict(static_wall_obbs)
    for support_name in support_names:
        if obb_overlap(candidate, support_by_name[support_name], margin=0.0):
            raise AssertionError(f"{name}: overlaps supporting wall {support_name}")

    return candidate, support_names


def run_right_wall_geometry_tests():
    """Check every prop footprint, wall ends, seams, and missing-support behavior."""
    static_wall_obbs = _fallback_static_wall_obbs()
    right_zone = next(zone for zone in WALL_ZONES if zone.wall == "right")
    right_wall_box = next(box for name, box in static_wall_obbs if name == "static_right_wall")
    right_wall_corners = obb_corners(*right_wall_box)
    wall_y_min = min(y for _, y in right_wall_corners)
    wall_y_max = max(y for _, y in right_wall_corners)

    for name, meta in WALL_PROP_META.items():
        offset = getattr(meta, "wall_offsets", None)
        right_offset = offset.get("right", meta.wall_offset) if offset is not None else meta.wall_offset
        root_x = right_zone.fixed_coord - right_offset
        yaw = right_zone.base_yaw + meta.yaw_offset

        _assert_right_wall_contact(
            name, root_x, (wall_y_min + wall_y_max) * 0.5, yaw, meta, static_wall_obbs
        )

        # Put each prop as close as possible to both ends of the supporting
        # segment while keeping its projected footprint on that segment.
        origin_candidate = _wall_prop_footprint_obb(root_x, 0.0, yaw, meta)
        origin_corners = obb_corners(*origin_candidate)
        relative_y_min = min(y for _, y in origin_corners)
        relative_y_max = max(y for _, y in origin_corners)
        edge_root_y_values = (
            wall_y_min - relative_y_min + 1.0e-6,
            wall_y_max - relative_y_max - 1.0e-6,
        )
        for edge_index, root_y in enumerate(edge_root_y_values):
            _assert_right_wall_contact(
                f"{name}[edge={edge_index}]",
                root_x,
                root_y,
                yaw,
                meta,
                static_wall_obbs,
            )

    # A prop spanning adjacent wall segments must use the most restrictive
    # room-facing surface and remain clear of both segments.
    seam_walls = [
        ("right_segment_a", make_obb(-2.50, -8.75, BBox(0.05, 0.75), 0.0)),
        ("right_segment_b", make_obb(-2.48, -7.25, BBox(0.05, 0.75), 0.0)),
    ]
    seam_meta = WALL_PROP_META["trash_can"]
    seam_candidate, seam_supports = _assert_right_wall_contact(
        "trash_can[seam]",
        right_zone.fixed_coord,
        -8.0,
        right_zone.base_yaw + seam_meta.yaw_offset,
        seam_meta,
        seam_walls,
    )
    if set(seam_supports) != {"right_segment_a", "right_segment_b"}:
        raise AssertionError("seam candidate did not detect both supporting wall segments")
    if max(x for x, _ in obb_corners(*seam_candidate)) >= -2.55:
        raise AssertionError("seam candidate was not constrained by the innermost wall face")

    missing_support = _snap_right_wall_root_to_surface(
        right_zone.fixed_coord,
        ROOM_Y_MAX + 2.0,
        right_zone.base_yaw,
        seam_meta,
        static_wall_obbs,
    )
    if missing_support is not None:
        raise AssertionError("candidate outside the wall span unexpectedly found support")


def _wall_prop_validation_issues(room):
    issues = []
    static_boxes = [
        (
            wall["name"],
            make_obb(wall["cx"], wall["cy"], BBox(wall["hw"], wall["hd"]), wall["yaw"]),
        )
        for wall in room["static_walls"]
    ]
    spawn_boundaries = [
        (boundary.name, make_obb(boundary.center[0], boundary.center[1], boundary.bbox, boundary.yaw))
        for boundary in FALLBACK_NO_SPAWN_BOUNDARIES
    ]
    placed = []

    for prop in room["wall_props"]:
        if prop["wall"] == "despawned":
            continue
        box = make_obb(prop["cx"], prop["cy"], BBox(prop["hw"], prop["hd"]), prop["yaw"])
        placed.append((prop["name"], box))
        if not obb_inside_room(box):
            issues.append(f"{prop['name']} is outside the room")
        spawn_issue = _outside_spawn_region_issue(prop["name"], box, spawn_boundaries)
        if spawn_issue is not None:
            issues.append(spawn_issue)

        zone = next(zone for zone in WALL_ZONES if zone.wall == prop["wall"])
        wall_overlap = _blocking_static_wall_overlap(box, zone, static_boxes)
        if wall_overlap is not None:
            issues.append(f"{prop['name']} overlaps non-support wall {wall_overlap}")

        if prop["wall"] == "right":
            support_surfaces = _right_wall_support_surfaces(box, static_boxes)
            if not support_surfaces:
                issues.append(f"{prop['name']} has no right-wall support")
            else:
                wall_face_x = min(face_x for _, face_x in support_surfaces)
                gap = wall_face_x - max(x for x, _ in obb_corners(*box))
                if not math.isclose(gap, RIGHT_WALL_CONTACT_CLEARANCE, abs_tol=1.0e-9):
                    issues.append(f"{prop['name']} has right-wall gap {gap:.9f}")

    for index, (a_name, a_box) in enumerate(placed):
        for b_name, b_box in placed[index + 1:]:
            if obb_overlap(a_box, b_box, margin=OBB_PLACEMENT_MARGIN):
                issues.append(f"{a_name} overlaps {b_name}")

    return issues


def run_wall_prop_stress_test(room_count=250):
    right_wall_placements = 0
    for seed in range(room_count):
        room = randomize_one_room(random.Random(0xA11CE + seed))
        issues = _wall_prop_validation_issues(room)
        if issues:
            detail = "; ".join(issues)
            raise AssertionError(f"stress room seed={seed}: {detail}")
        right_wall_placements += sum(
            prop["wall"] == "right" for prop in room["wall_props"]
        )
    if right_wall_placements == 0:
        raise AssertionError("stress test did not generate any right-wall placements")


# =====================================================================
# Main
# =====================================================================

def main():
    NUM_ROOMS = 6
    STRESS_ROOMS = 250
    run_right_wall_geometry_tests()
    run_wall_prop_stress_test(STRESS_ROOMS)
    print(
        f"Right-wall regression checks passed for all {len(WALL_PROP_META)} prop shapes "
        f"and {STRESS_ROOMS} seeded layouts.\n"
    )
    print(f"Generating {NUM_ROOMS} randomized room layouts (OBB mode)...\n")

    rooms = [randomize_one_room() for _ in range(NUM_ROOMS)]

    total_placed = 0
    total_possible = 0
    all_issues = []
    for i, room in enumerate(rooms):
        issues = print_room_text(room, i)
        all_issues.extend((i + 1, issue.strip()) for issue in issues)
        placed = len([wp for wp in room["wall_props"] if wp["wall"] != "despawned"])
        total_placed += placed
        total_possible += len(room["wall_props"])

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {total_placed}/{total_possible} wall props placed across {NUM_ROOMS} rooms")
    print(f"  Average: {total_placed/NUM_ROOMS:.1f} / {total_possible/NUM_ROOMS:.0f} per room")
    print(f"{'='*60}")

    if HAS_MPL:
        cols = 3
        rows = (NUM_ROOMS + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 6))
        axes = axes.flatten()

        for i, room in enumerate(rooms):
            draw_room(axes[i], room, i)

        for j in range(NUM_ROOMS, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("OBB Room Placement Verification — 6 Random Layouts", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "placement_test.png")
        fig.savefig(output_path, dpi=150)
        print(f"\n📊 Plot saved to: {output_path}")

    if all_issues:
        detail = "\n".join(f"room {room_index}: {issue}" for room_index, issue in all_issues)
        raise AssertionError(f"Placement validation failed:\n{detail}")


if __name__ == "__main__":
    main()
