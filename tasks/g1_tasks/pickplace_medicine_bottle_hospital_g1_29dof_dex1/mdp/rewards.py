# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Reward for placing both target pill bottles in the rear crates."""

from __future__ import annotations

import torch

from tasks.common_rewards.base_reward_pickplace_redblock import (
    _get_rewards_dds_instance,
)

from .container_goal import pill_bottles_contained


def compute_pill_bottle_reward(env) -> torch.Tensor:
    """Score 0.5 per target bottle contained in either rear crate."""
    reward = pill_bottles_contained(env).to(dtype=torch.float32).sum(dim=-1) * 0.5
    rewards_dds = _get_rewards_dds_instance()
    if rewards_dds:
        rewards_dds.write_rewards_data(reward)
    return reward


__all__ = ["compute_pill_bottle_reward"]
