# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
# Portions adapted from unitree_rl_lab's tasks/mimic (Apache-2.0), itself based on
# whole_body_tracking (MIT); modified. See THIRD_PARTY_NOTICES.md.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude, quat_from_euler_xyz, quat_mul

from g1_mocap_free_box.tasks.manager_based.tracking.mdp.commands import MotionCommand, _as_tensor

if TYPE_CHECKING:
    # Runtime asset/sensor classes pull in USD at import time. Hydra imports
    # this config before SimulationApp starts, and a second USD init aborts
    # the process inside TfEnum::_AddName -- so they stay type-only here.
    from isaaclab.sensors import ContactSensor
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


##
# Object tracking and contact -- Table IV of the SBTO paper
##
# The paper writes its kernels as exp(-k * ||err||^2); this file's convention is
# exp(-err / std^2), so std = 1/sqrt(k). The mapping is recorded in the env cfg
# next to each weight.


def motion_joint_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Joint position tracking over every joint."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    # Mean, not sum: summing squared error over 53 joints saturates the kernel
    # (exp(-2.1/0.2) is ~2e-5), which zeroes both the reward and its gradient.
    # Every other exp-kernel term here averages over its elements.
    error = torch.mean(torch.square(command.joint_pos - command.robot_joint_pos), dim=-1)
    return torch.exp(-error / std**2)


def motion_object_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Object position tracking, on distance rather than squared distance.

    The paper's kernel is exp(-8||err||^2), which is fine while the object stays
    within ~0.5 m of its reference. Once it is lost -- which happens here because
    the reference does not transfer cleanly from MuJoCo to PhysX -- the squared
    form is numerically zero (1.7e-49 at 3.75 m) and so is its gradient, so
    nothing can pull the object back. HDMI, which this paper takes its residual
    action space from, uses exp(-err/sigma) instead. Sigma is chosen to keep the
    paper's half-reward distance (0.294 m), so the two agree near the reference
    and differ only in the tail.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.norm(command.object_pos_w - command.robot_object_pos_w, dim=-1)
    return torch.exp(-error / std)


def motion_object_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Object orientation tracking. Linear in the angle, for the same reason as
    the position term above; sigma preserves the paper's half-reward angle."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.object_quat_w, command.robot_object_quat_w)
    return torch.exp(-error / std)


def motion_object_orientation_error_symmetric_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Object orientation tracking that treats a 180-degree flip as equivalent.

    A rectangular box is unchanged by a half turn about its own vertical: the
    same faces end up presented to the hands, and the mass distribution is
    identical. The blind term therefore charges up to 180 degrees of error for a
    box that is, physically, already correct -- and demands the robot rotate it
    back through an angle it may not be able to cover without regrasping.

    This scores against whichever of the two equivalent reference orientations is
    closer, so the error never exceeds 90 degrees however far the box is turned:

        reduced = min(|theta|, 180 - |theta|)

    That has a useful consequence for curriculum design. Sampling yaw uniformly
    over +/-180 degrees gives a mean reduced error of 45 degrees -- exactly the
    mean of |theta| when sampling +/-90 degrees with the blind term. So the full
    placement range under this reward is no harder, in expectation, than half the
    range was before it.

    Deliberately a SEPARATE function rather than a flag on the existing one:
    motion_object_orientation_error_exp stays byte-identical, so tasks using it
    keep their reward and every measurement taken against it stays comparable.

    Note the flip is post-multiplied, which rotates about the box's OWN vertical
    rather than the world's -- the two coincide while the box is upright but not
    once it tilts.

    Only the 2-fold z symmetry is applied. The box is 205 x 125 x 280 mm, so a
    quarter turn is NOT a symmetry, and it stays upright throughout the motion,
    so the x/y flips of the full D2h group never apply.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref = command.object_quat_w
    actual = command.robot_object_quat_w

    zeros = torch.zeros(ref.shape[0], device=ref.device)
    half_turn = quat_from_euler_xyz(zeros, zeros, zeros + torch.pi)

    direct = quat_error_magnitude(ref, actual)
    flipped = quat_error_magnitude(quat_mul(ref, half_turn), actual)
    return torch.exp(-torch.minimum(direct, flipped) / std)


def motion_object_linear_velocity_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.object_lin_vel_w - command.robot_object_lin_vel_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_object_angular_velocity_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.object_ang_vel_w - command.robot_object_ang_vel_w), dim=-1)
    return torch.exp(-error / std**2)


def _contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Largest contact force on any of these bodies.

    This is the *net* force, so it cannot tell the box from the floor. PhysX's
    filtered contact views would draw that distinction, but they pair sensor
    shapes to filter shapes one-to-one, and a hand link can carry several
    collision shapes -- so a filter over the box cannot match them reliably. The reference has the hands touching essentially only the box during
    the grasp window, which is what makes the approximation tolerable; if
    hand-floor contact starts being rewarded as a grasp, this is the reason.
    Shape-level filtering exists on the Newton backend if it becomes a problem.

    The max over bodies reproduces MuJoCo's reduce="maxforce" over the hand.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return torch.norm(forces, dim=-1).max(dim=1).values


def motion_contact_match(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    contact_channel: str,
    force_limit: float = 10.0,
    force_decay: float = 0.1,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward matching the reference contact state, tapering off above 10 N.

    The paper writes this as I(c = c_ref) * exp(-0.1 * (||F|| - 10)). Taken
    literally that grows above 1 for light contacts, which reads as a
    transcription of "gradually penalizes forces that exceed 10N" -- so the
    penalty applies only above the limit, and a light touch scores a clean 1.

    One end effector per term, so the reward manager resolves the body ids; the
    paper's single 1.25 term is split evenly across the two hands.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    channel = command.motion.contact_names.index(contact_channel)

    force = _contact_force(env, sensor_cfg)
    matched = ((force > contact_threshold) == (command.contact_ref[:, channel] > 0)).float()
    return matched * torch.exp(-force_decay * torch.clamp(force - force_limit, min=0.0))
