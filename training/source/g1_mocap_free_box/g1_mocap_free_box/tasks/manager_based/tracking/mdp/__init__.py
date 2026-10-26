# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""MDP terms for the SBTO motion-tracking task.

Exports are declared in __init__.pyi and resolved lazily. That is not just
style: several of these modules reach runtime classes that import USD, and
Hydra imports this package before SimulationApp starts -- a second USD
initialisation aborts the process inside TfEnum::_AddName.
"""

from isaaclab.utils.module import lazy_export

lazy_export()
