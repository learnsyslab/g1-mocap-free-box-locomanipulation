# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Modifications copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich,
# licensed under Apache-2.0. See THIRD_PARTY_NOTICES.md.

"""Package containing task implementations for the extension."""

# Register Gym environments without importing the environment config modules.
from . import manager_based  # noqa: F401
