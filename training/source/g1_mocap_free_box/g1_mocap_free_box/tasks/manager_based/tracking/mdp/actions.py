# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""Residual joint-position action config, as in HDMI and the SBTO paper.

Config only. The term itself lives in actions_impl.py, because importing
JointPositionAction pulls in USD and Hydra loads this module before
SimulationApp starts.
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils.configclass import configclass


@configclass
class MotionResidualJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`MotionResidualJointPositionAction`."""

    class_type: type | str = "{DIR}.actions_impl:MotionResidualJointPositionAction"

    command_name: str = MISSING
    """Motion command term supplying the PD targets to offset against."""

    use_default_offset: bool = False
    """Unused -- the offset is resolved from the motion every step."""

    offset_source: str = "pd_target"
    """What the residual is added to: ``"pd_target"`` (the motion's SBTO PD
    target) or ``"ref_joint_pos"`` (the reference joint position)."""
