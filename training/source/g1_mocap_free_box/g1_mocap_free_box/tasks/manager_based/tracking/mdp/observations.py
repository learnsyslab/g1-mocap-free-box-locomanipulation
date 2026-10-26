# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
# Portions adapted from unitree_rl_lab's tasks/mimic (Apache-2.0), itself based on
# whole_body_tracking (MIT); modified. See THIRD_PARTY_NOTICES.md.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_conjugate, quat_mul, subtract_frame_transforms

from g1_mocap_free_box.tasks.manager_based.tracking.mdp.commands import MotionCommand, _as_tensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def _select_part(pos: torch.Tensor, rot6d: torch.Tensor, part: str) -> torch.Tensor:
    """pos (3) + rot6d (6) as one vector, or just one of them -- so position and
    rotation can carry different observation noise as separate terms."""
    if part == "pos":
        return pos
    if part == "rot":
        return rot6d
    if part != "all":
        raise ValueError(f"part must be 'all', 'pos' or 'rot', got {part!r}")
    return torch.cat([pos, rot6d], dim=-1)


##
# Object observations
##
# "In addition to the one-step desired robot trajectory, the policy observes the
# one-step object pose error with respect to the desired pose, as well as the
# object pose expressed in the robot frame." -- SBTO paper, section III-E.


def object_pose_b(env: ManagerBasedEnv, command_name: str, part: str = "all") -> torch.Tensor:
    """Where the object actually is, in the robot's anchor frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    pos_b, quat_b = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.robot_object_pos_w,
        command.robot_object_quat_w,
    )
    mat = matrix_from_quat(quat_b)
    return _select_part(pos_b, mat[..., :2].reshape(mat.shape[0], -1), part)


def object_pose_error_b(env: ManagerBasedEnv, command_name: str, part: str = "all") -> torch.Tensor:
    """How far the object is from where the reference wants it, robot-framed.

    Expressed in the robot's frame rather than the world's so the policy sees
    the error the way it would have to act on it.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    actual_pos_b, actual_quat_b = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.robot_object_pos_w,
        command.robot_object_quat_w,
    )
    desired_pos_b, desired_quat_b = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.object_pos_w,
        command.object_quat_w,
    )
    mat = matrix_from_quat(quat_mul(desired_quat_b, quat_conjugate(actual_quat_b)))
    return _select_part(desired_pos_b - actual_pos_b, mat[..., :2].reshape(mat.shape[0], -1), part)


def object_velocity_error(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Object linear and angular velocity error against the reference."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.cat(
        [
            command.object_lin_vel_w - command.robot_object_lin_vel_w,
            command.object_ang_vel_w - command.robot_object_ang_vel_w,
        ],
        dim=-1,
    )
