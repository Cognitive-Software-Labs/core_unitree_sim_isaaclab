# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
# Reuse the red-block success/out-of-range check. The world-space thresholds are
# passed in via DoneTerm(params=...) from the env cfg, because this task places the
# table in the hospital room interior instead of the warehouse.
from tasks.common_termination.base_termination_pick_place_redblock import reset_object_estimate


__all__ = [
    "reset_object_estimate",
]
