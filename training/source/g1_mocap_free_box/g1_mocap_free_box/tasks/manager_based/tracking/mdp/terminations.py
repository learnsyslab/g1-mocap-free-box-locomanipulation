# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
# Portions adapted from unitree_rl_lab's tasks/mimic (Apache-2.0), itself based on
# whole_body_tracking (MIT); modified. See THIRD_PARTY_NOTICES.md.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse


if TYPE_CHECKING:
    # Runtime asset/sensor classes pull in USD at import time. Hydra imports
    # this config before SimulationApp starts, and a second USD init aborts
    # the process inside TfEnum::_AddName -- so they stay type-only here.
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import TerminationTermCfg

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from g1_mocap_free_box.tasks.manager_based.tracking.mdp.commands import MotionCommand
from g1_mocap_free_box.tasks.manager_based.tracking.mdp.rewards import _get_body_indexes


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    # GRAVITY_VEC_W is a warp ProxyArray in IsaacLab 6.x; .torch is how
    # IsaacLab's own mdp terms reach the tensor behind it.
    gravity_w = asset.data.GRAVITY_VEC_W.torch
    motion_projected_gravity_b = quat_apply_inverse(command.anchor_quat_w, gravity_w)

    robot_projected_gravity_b = quat_apply_inverse(command.robot_anchor_quat_w, gravity_w)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)


