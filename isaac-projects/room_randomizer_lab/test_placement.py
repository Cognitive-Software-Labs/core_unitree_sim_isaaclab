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
    FLOOR_Z,
    OBB_PLACEMENT_MARGIN,
    ROBOT_BBOX,
    ROBOT_ORBIT_OFFSET,
    ROOM_X_MAX,
    ROOM_X_MIN,
    ROOM_Y_MAX,
    ROOM_Y_MIN,
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
    offset = meta.wall_offset
    if zone.wall == "back":
        cx, cy = pos, zone.fixed_coord + offset
    else:
        cx, cy = zone.fixed_coord - offset, pos
    yaw = zone.base_yaw + meta.yaw_offset
    return cx, cy, yaw


def _make_table_group(dx, dy, dyaw):
    cx, cy = offset_from_yaw(dx, dy, dyaw, CHAIR_ORBIT_OFFSET[0], CHAIR_ORBIT_OFFSET[1])
    rx, ry = offset_from_yaw(dx, dy, dyaw, ROBOT_ORBIT_OFFSET[0], ROBOT_ORBIT_OFFSET[1])

    return {
        "desk": make_obb(dx, dy, DESK_BBOX, dyaw),
        "chair": make_obb(cx, cy, CHAIR_BBOX, dyaw + math.pi),
        "robot": make_obb(rx, ry, ROBOT_BBOX, dyaw - math.pi / 2),
    }


def _validate_table_group(table_obbs, placed_obbs):
    for box in table_obbs.values():
        if not obb_inside_room(box):
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


def randomize_one_room():
    rng = random.Random()
    placed_obbs = []
    results = {"wall_props": [], "desk": None, "chair": None, "robot": None, "desk_objects": []}

    # --- Phase 1: Wall props ---
    sorted_names = sorted(WALL_PROP_META.keys(), key=lambda n: not WALL_PROP_META[n].tall)

    for name in sorted_names:
        meta = WALL_PROP_META[name]
        allowed_zones = [z for z in WALL_ZONES if z.wall in meta.allowed_walls]

        success = False
        for _ in range(100):
            zone = rng.choice(allowed_zones)
            cx, cy, yaw = _sample_wall_position(zone, meta, rng)
            candidate = make_obb(cx, cy, meta.bbox, yaw)

            if not obb_inside_room(candidate):
                continue
            if obb_overlap_any(candidate, placed_obbs, margin=OBB_PLACEMENT_MARGIN):
                continue

            placed_obbs.append(candidate)
            results["wall_props"].append({
                "name": name,
                "cx": cx, "cy": cy,
                "hw": meta.bbox.half_w, "hd": meta.bbox.half_d,
                "yaw": yaw,
                "tall": meta.tall,
                "wall": zone.wall,
            })
            success = True
            break

        if not success:
            results["wall_props"].append({
                "name": name, "cx": 0, "cy": 0, "hw": 0, "hd": 0,
                "yaw": 0, "tall": meta.tall, "wall": "despawned",
            })

    # --- Phase 2: Table group ---
    tg_success = False
    tg_source = "failed"
    selected_table_obbs = None

    for _ in range(TABLE_GROUP_MAX_TRIES):
        dx = rng.uniform(TABLE_SAMPLE_X_MIN, TABLE_SAMPLE_X_MAX)
        dy = rng.uniform(TABLE_SAMPLE_Y_MIN, TABLE_SAMPLE_Y_MAX)
        dyaw = rng.uniform(0, 2 * math.pi)

        table_obbs = _make_table_group(dx, dy, dyaw)
        if not _validate_table_group(table_obbs, placed_obbs):
            continue

        selected_table_obbs = table_obbs
        tg_success = True
        tg_source = "random"
        break

    if not tg_success:
        for _ in range(TABLE_GROUP_MAX_TRIES):
            dyaw = rng.uniform(0, 2 * math.pi)
            table_obbs = _make_table_group(TABLE_FALLBACK_X, TABLE_FALLBACK_Y, dyaw)
            if not _validate_table_group(table_obbs, placed_obbs):
                continue

            selected_table_obbs = table_obbs
            tg_success = True
            tg_source = "fallback"
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
                desk_placed_obbs.append(candidate)
                wx, wy = offset_from_yaw(desk_x, desk_y, desk_yaw, lx, ly)
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

    d = room["desk"]
    if d is None:
        print(f"\n  Table Group: SKIPPED (no valid random or fallback placement)")
        print(f"\n  Desk Objects: skipped")
    else:
        status = room.get("tg_source", "unknown")
        print(f"\n  Desk:   ({d['cx']:+6.2f}, {d['cy']:+6.2f})  yaw={math.degrees(d['yaw']):+6.1f}°  source={status}")
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
    for wp in placed:
        box = make_obb(wp["cx"], wp["cy"], BBox(wp["hw"], wp["hd"]), wp["yaw"])
        named_boxes.append((wp["name"], box))
        if not obb_inside_room(box):
            issues.append(f"  ⚠️  {wp['name']} OBB extends outside room!")
    if d is not None:
        for label, item in [("Desk", d), ("Chair", room["chair"]), ("Robot", room["robot"])]:
            box = make_obb(item["cx"], item["cy"], BBox(item["hw"], item["hd"]), item["yaw"])
            named_boxes.append((label, box))
            if not obb_inside_room(box):
                issues.append(f"  ⚠️  {label} OBB extends outside room!")

    for i, (a_name, a_box) in enumerate(named_boxes):
        for b_name, b_box in named_boxes[i + 1:]:
            if obb_overlap(a_box, b_box, margin=OBB_PLACEMENT_MARGIN):
                issues.append(f"  ⚠️  {a_name} overlaps {b_name}!")

    if issues:
        print(f"\n  VALIDATION ISSUES:")
        for issue in issues:
            print(f"    {issue}")
    else:
        print(f"\n  ✅ All OBBs inside room bounds and non-overlapping")


# =====================================================================
# Main
# =====================================================================

def main():
    NUM_ROOMS = 6
    print(f"Generating {NUM_ROOMS} randomized room layouts (OBB mode)...\n")

    rooms = [randomize_one_room() for _ in range(NUM_ROOMS)]

    total_placed = 0
    total_possible = 0
    for i, room in enumerate(rooms):
        print_room_text(room, i)
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


if __name__ == "__main__":
    main()
