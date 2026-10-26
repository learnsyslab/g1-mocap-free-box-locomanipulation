# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime half of the residual action term.

Separate from the config: importing JointPositionAction pulls in USD, and a
second USD initialisation aborts the process. The config names this class by
string, so it loads only once the simulation is up.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions import MotionResidualJointPositionActionCfg


class MotionResidualJointPositionAction(JointPositionAction):
    """Joint position targets expressed relative to the motion's own PD targets.

    The upstream term offsets against a fixed default pose, which makes the
    policy responsible for reproducing the whole trajectory. SBTO already
    optimized a PD target sequence that is dynamically feasible, so here the
    policy only supplies the correction to it and the reference does the rest.

    The motion stores its targets in SBTO's actuator order, which is not the
    articulation's order, so the columns are matched by joint name.
    """

    cfg: MotionResidualJointPositionActionCfg

    def __init__(self, cfg: MotionResidualJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._pd_columns: torch.Tensor | None = None

    def _resolve_pd_columns(self) -> torch.Tensor:
        """Map this term's joints onto the motion's PD-target columns."""
        command = self._env.command_manager.get_term(self.cfg.command_name)
        names = command.motion.pd_target_joint_names
        if not names:
            raise ValueError(
                f"Motion {command.cfg.motion_file} carries no dof_pd_target_joint_names, so a "
                "residual action space cannot be resolved. The reference must carry its optimised "
                "PD targets and their joint names."
            )
        missing = [n for n in self._joint_names if n not in names]
        if missing:
            raise ValueError(f"Joints absent from the motion's PD targets: {missing}")
        return torch.tensor([names.index(n) for n in self._joint_names], dtype=torch.long, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        command = self._env.command_manager.get_term(self.cfg.command_name)
        self._raw_actions[:] = actions
        if self.cfg.offset_source == "ref_joint_pos":
            # The motion's joint_pos is stored in the articulation's joint order.
            self._processed_actions = self._raw_actions * self._scale + command.joint_pos[:, self._joint_ids]
            return
        if self.cfg.offset_source != "pd_target":
            raise ValueError(f"Unknown offset_source {self.cfg.offset_source!r}")
        if self._pd_columns is None:
            self._pd_columns = self._resolve_pd_columns()
        # Index time_steps directly. In IsaacLab, CommandTerm.reset() calls only _resample()
        # (managers/command_manager.py) -- it does NOT call _update_command() -- and
        # command_manager.compute() runs in step() *after* process_actions. So after a
        # reset time_steps == k while the sim holds state[k], and joint_pd_target[time_steps]
        # is already pd_target[k]. The invariant that matters: a zero action applies
        # dof_pd_target[k] at state[k], the reference optimiser's own open-loop convention
        # (ctrl = dof_pd_target[t] at x_traj[t]).
        self._processed_actions = self._raw_actions * self._scale + command.joint_pd_target[:, self._pd_columns]

    def apply_actions(self):
        target = self.processed_actions
        self._asset.set_joint_position_target_index(target=target, joint_ids=self._joint_ids)
