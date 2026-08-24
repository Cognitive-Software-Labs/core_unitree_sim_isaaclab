# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Rear-crate containment checks for the hospital pill-bottle task."""

from __future__ import annotations

import torch


PILL_BOTTLE_NAMES = ("pill_bottle_t", "pill_bottle_v")
RIDGEBACK_NAMES = ("ridgeback_left", "ridgeback_right")

# The crates are collision children of the two kinematic Ridgeback roots.
# Values are the local transforms captured in the authored hospital layout.
CRATE_LOCAL_POSITIONS = (
    (0.22877, 0.00612, 0.28576),
    (0.22877, -0.00612, 0.28576),
)
CRATE_LOCAL_ORIENTATIONS = (
    (0.7095584382, 0.0, 0.0, -0.7046465942),
    (0.7095584382, 0.0, 0.0, 0.7046465942),
)

# Measured source-mesh bounds are x=+/-0.224775, y=+/-0.167839,
# z=[0, 0.188070]. These inner limits leave clearance for the crate walls and
# the roughly 14 mm-radius target bottles.
CRATE_INTERIOR_HALF_EXTENTS_XY = (0.205, 0.148)
CRATE_INTERIOR_Z_RANGE = (0.005, 0.183)

# Both pill-bottle assets use a base-origin pivot. Test their geometric center,
# not the base point, so tipped bottles are still classified correctly.
PILL_BOTTLE_LOCAL_CENTERS = (
    (0.0, 0.0, 0.0253035),
    (0.0, 0.0, 0.0209020),
)


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply scalar-first (w, x, y, z) quaternions."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quaternion_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by scalar-first unit quaternions."""
    xyz = quaternion[..., 1:]
    uv = torch.linalg.cross(xyz, vector, dim=-1)
    uuv = torch.linalg.cross(xyz, uv, dim=-1)
    return vector + 2.0 * (quaternion[..., :1] * uv + uuv)


def quaternion_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the inverse of a unit scalar-first quaternion."""
    result = quaternion.clone()
    result[..., 1:] = -result[..., 1:]
    return result


def points_in_oriented_crate(
    points_w: torch.Tensor,
    parent_positions_w: torch.Tensor,
    parent_orientations_w: torch.Tensor,
    crate_local_position: tuple[float, float, float],
    crate_local_orientation: tuple[float, float, float, float],
) -> torch.Tensor:
    """Return one containment flag per environment for one parented crate."""
    local_position = points_w.new_tensor(crate_local_position).expand_as(points_w)
    local_orientation = points_w.new_tensor(crate_local_orientation).expand(
        points_w.shape[0], 4
    )
    crate_position_w = parent_positions_w + quaternion_apply(
        parent_orientations_w, local_position
    )
    crate_orientation_w = quaternion_multiply(
        parent_orientations_w, local_orientation
    )
    points_crate = quaternion_apply(
        quaternion_conjugate(crate_orientation_w), points_w - crate_position_w
    )

    half_x, half_y = CRATE_INTERIOR_HALF_EXTENTS_XY
    min_z, max_z = CRATE_INTERIOR_Z_RANGE
    return (
        torch.isfinite(points_crate).all(dim=-1)
        & (points_crate[:, 0].abs() <= half_x)
        & (points_crate[:, 1].abs() <= half_y)
        & (points_crate[:, 2] >= min_z)
        & (points_crate[:, 2] <= max_z)
    )


def pill_bottles_contained(env) -> torch.Tensor:
    """Return shape ``(num_envs, 2)`` flags for bottles in either rear crate."""
    bottle_results = []
    for bottle_name, center_offset in zip(
        PILL_BOTTLE_NAMES, PILL_BOTTLE_LOCAL_CENTERS, strict=True
    ):
        bottle = env.scene[bottle_name]
        local_center = bottle.data.root_pos_w.new_tensor(center_offset).expand(
            env.num_envs, 3
        )
        center_w = bottle.data.root_pos_w + quaternion_apply(
            bottle.data.root_quat_w, local_center
        )

        bin_results = []
        for ridgeback_name, crate_position, crate_orientation in zip(
            RIDGEBACK_NAMES,
            CRATE_LOCAL_POSITIONS,
            CRATE_LOCAL_ORIENTATIONS,
            strict=True,
        ):
            ridgeback = env.scene[ridgeback_name]
            bin_results.append(
                points_in_oriented_crate(
                    center_w,
                    ridgeback.data.root_pos_w,
                    ridgeback.data.root_quat_w,
                    crate_position,
                    crate_orientation,
                )
            )
        bottle_results.append(torch.stack(bin_results, dim=-1).any(dim=-1))

    return torch.stack(bottle_results, dim=-1)


def both_pill_bottles_contained(env) -> torch.Tensor:
    """Terminate on success and request the same fixed-table reset as Quest Y."""
    success = pill_bottles_contained(env).all(dim=-1)
    if bool(success.any().item()):
        # ManagerBasedRLEnv resets successful environments inside env.step().
        # Select the Y-button reset mode before that reset event runs, then let
        # sim_main notify the XR bridge to clear its persistent torso target.
        env._teleop_randomize_table_position = False
        env._hospital_success_reset_pending = True
    return success


__all__ = [
    "CRATE_INTERIOR_HALF_EXTENTS_XY",
    "CRATE_INTERIOR_Z_RANGE",
    "CRATE_LOCAL_ORIENTATIONS",
    "CRATE_LOCAL_POSITIONS",
    "PILL_BOTTLE_LOCAL_CENTERS",
    "PILL_BOTTLE_NAMES",
    "RIDGEBACK_NAMES",
    "both_pill_bottles_contained",
    "pill_bottles_contained",
    "points_in_oriented_crate",
    "quaternion_apply",
    "quaternion_conjugate",
    "quaternion_multiply",
]
