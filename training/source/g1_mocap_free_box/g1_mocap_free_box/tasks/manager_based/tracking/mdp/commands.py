# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
# Portions adapted from unitree_rl_lab's tasks/mimic (Apache-2.0), itself based on
# whole_body_tracking (MIT); modified. See THIRD_PARTY_NOTICES.md.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    # Runtime asset/sensor classes pull in USD at import time. Hydra imports
    # this config before SimulationApp starts, and a second USD init aborts
    # the process inside TfEnum::_AddName -- so they stay type-only here.
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def _as_tensor(value):
    """Unwrap IsaacLab 6.x's warp-backed views into plain torch tensors.

    Several asset data properties now hand back a ProxyArray, which torch's
    math kernels refuse. IsaacLab's own mdp terms reach through ``.torch``.
    """
    return value.torch if hasattr(value, "torch") else value


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

        # Optional tracks of an optimised reference (SBTO). A motion converted from a
        # plain capture has none of them, and the object terms are simply not
        # configured in that case.
        def _optional(key):
            return torch.tensor(data[key], dtype=torch.float32, device=device) if key in data else None

        self.object_pos_w = _optional("object_pos")
        self.object_quat_w = _optional("object_rot")
        self.object_lin_vel_w = _optional("object_lin_vel")
        # Half-extents, needed to work out how high the box's centre sits when it
        # rests on a face other than the recorded one.
        self.object_geom_size = _optional("object_geom_size")
        self.object_ang_vel_w = _optional("object_ang_vel")
        self.joint_pd_target = _optional("dof_pd_target")
        # Articulated fixture (tracker_to_npz.py): its joint tracks, in the
        # file's own column order; MotionCommand maps them onto the asset by name.
        self.fixture_joint_pos = _optional("fixture_joint_pos")
        self.fixture_joint_vel = _optional("fixture_joint_vel")
        self.fixture_joint_names = (
            [str(n) for n in data["fixture_joint_names"]] if "fixture_joint_names" in data else []
        )
        # "contact_ref" (the PLANNED contact from the retargeted demo), NOT
        # "contact" (what SBTO's own sensors registered during its rollout).
        # Table IV's contact term is 1(c == c_ref) -- it matches the plan.
        #
        # In SBTO's own best_trajectory.npz the realized "contact" channel is
        # ALL ZERO for both hands (nonzero counts [252, 0, 0] vs contact_ref's
        # [253, 385, 385]; contact_force peaks at 0.000 N), evidently a
        # sensor-scoping artifact upstream. Reading "contact" hands
        # motion_contact_match a reference meaning "never touch the object",
        # so the reward pays full marks for keeping the hands OFF the box and
        # withdraws them on contact -- training the policy not to grasp.
        # Preference order:
        # "contact_computed" (measured from the reference geometry, so it sees
        # the wrist carry SBTO's hand-scoped sensor misses) > "contact_ref"
        # (the kinematic plan, Table IV's c_ref) > "contact" (SBTO's realized
        # readout, all-zero for the hands on the no-hands asset).
        # "contact_computed" is measured from the reference geometry in MuJoCo
        # (mj_geomDistance); the shipped motion carries it. A motion without it
        # silently falls through to "contact_ref" instead.
        self.contact_ref = _optional("contact_computed")
        contact_names_key = "contact_computed_names"
        if self.contact_ref is None:
            self.contact_ref = _optional("contact_ref")
            contact_names_key = "contact_names"
        if self.contact_ref is None:
            self.contact_ref = _optional("contact")
        if contact_names_key not in data:
            contact_names_key = "contact_names"
        self._contact_names_key = contact_names_key
        self.contact_force_ref = _optional("contact_force")
        # Keyed to whichever contact channel was actually selected above, so the
        # channel-name lookup in motion_contact_match stays aligned with it.
        self.contact_names = (
            [str(n) for n in data[self._contact_names_key]] if self._contact_names_key in data else []
        )
        self.pd_target_joint_names = (
            [str(n) for n in data["dof_pd_target_joint_names"]]
            if "dof_pd_target_joint_names" in data
            else []
        )
        self.has_object = self.object_pos_w is not None

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Re-anchoring: a per-env planar (x, y, yaw) offset applied to the WHOLE
        # reference -- robot bodies, anchor and object trajectory alike -- so the
        # task can be posed relative to wherever the box actually is instead of
        # where the recording happened to put it.
        #
        # Identity by default (zero offset, unit quaternion), so with
        # reanchor_pose_range unset every reference property returns exactly what
        # it did before and no existing task changes behaviour.
        #
        # The rotation pivot is the reference object's frame-0 position, so a yaw
        # offset turns the task about the box rather than about the world origin.
        # Only x/y/yaw are offered: the ground is a plane, so a planar transform
        # is the largest one that cannot disturb heights or gravity alignment.
        # NOTE: build the identity through quat_from_euler_xyz rather than
        # writing a 1.0 into a hard-coded slot. This IsaacLab build stores
        # quaternions xyzw, not wxyz -- quat_from_euler_xyz(0,0,0) returns
        # [0,0,0,1] -- so `q[:, 0] = 1.0` would be a 180 degree rotation about X,
        # not identity. Constructing it keeps this correct under either layout.
        _zero = torch.zeros(self.num_envs, device=self.device)
        _identity = quat_from_euler_xyz(_zero, _zero, _zero)

        self._reanchor_dxy = torch.zeros(self.num_envs, 2, device=self.device)
        self._reanchor_quat = _identity.clone()
        self._reanchor_active = bool(self.cfg.reanchor_pose_range)

        # Object-only offset: shifts and spins the BOX and its reference
        # trajectory while leaving the robot's reference alone, so the
        # robot-relative-to-box pose genuinely varies and the grasp has to adapt.
        #
        # Deliberately not the same thing as _reanchor_* above, which moves robot
        # and box together and so preserves their relative pose by construction.
        self._obj_off_dxy = torch.zeros(self.num_envs, 2, device=self.device)
        self._obj_off_quat = _identity.clone()   # see the layout note above
        self._obj_off_active = bool(self.cfg.object_offset_pose_range)
        # The xy that object_pose_range added at this env's last reset (zero if
        # none). Read by teacher_in_distribution to mask kickstarting to the box
        # placements the teacher was actually trained on.
        self._obj_pose_sample_xy = torch.zeros(self.num_envs, 2, device=self.device)

        # Resting-face randomization. `_rest_quat` pre-multiplies the reference
        # orientation and `_rest_dz` shifts the reference height, so the box and
        # the path it is asked to follow move TOGETHER -- otherwise the reward
        # demands a box lying 7.75 cm lower be carried along a path it is not on,
        # which is the same unsatisfiable ask object_pose_range makes in xy.
        self._rest_quat = _identity.clone()
        self._rest_dz = torch.zeros(self.num_envs, device=self.device)
        self._rest_active = bool(self.cfg.object_resting_states)
        # Per-env affine map of the object REFERENCE for a resized / moved fixture stack
        # (set by set_object_stack_transform; None = identity, every other task).
        self._stack = None
        if self._rest_active:
            self._rest_options = torch.tensor(
                self.cfg.object_resting_states, device=self.device, dtype=torch.float
            )
            self._rest_options = self._rest_options / self._rest_options.norm(
                dim=-1, keepdim=True
            )
            half = self.motion.object_geom_size
            if half is None:
                raise ValueError(
                    "object_resting_states needs object_geom_size in the motion "
                    "file: the resting height of a tipped box follows from its "
                    "half-extents and cannot be guessed"
                )
            self._rest_half = half.reshape(-1)[:3].to(self.device)
            # Support height of a box under each candidate rotation: the extent
            # along world +z is sum_i |R[2, i]| * half_i. Relative to the
            # recorded face, that difference is what the reference must shift by.
            #
            # Quaternions here are XYZW, per the note above -- the resting states
            # must be written in that layout too. Getting this wrong put four of
            # the six faces on an edge instead of a face, with support heights of
            # 0.155 m against a largest half-extent of 0.14, and the box up to
            # 8.7 cm through the floor.
            mats = matrix_from_quat(self._rest_options)
            heights = (mats[:, 2, :].abs() * self._rest_half).sum(dim=-1)
            self._rest_dz_options = heights - self._rest_half[2]

        # First frame the reference has BOTH hands on the box. The object offset
        # is only physically meaningful before this: after it the box is held,
        # and rotating it while the robot's reference hand pose is unchanged
        # places the box inside the hands. PhysX resolves that by ejecting it, so
        # an ungated offset would fill training with exploded states.
        self._grasp_onset = 0
        if self.motion.contact_ref is not None and self.motion.contact_ref.shape[1] >= 3:
            left, right = self.motion.contact_ref[:, 1] > 0, self.motion.contact_ref[:, 2] > 0
            # "both" for a two-handed carry (the box); "any" for a one-handed grasp,
            # where requiring both would put the onset nowhere and leave the gate open.
            held = (left & right) if self.cfg.grasp_onset_hands == "both" else (left | right)
            idx = torch.nonzero(held, as_tuple=False)
            if idx.numel():
                self._grasp_onset = int(idx[0].item())
        if self.motion.object_pos_w is not None:
            self._reanchor_pivot = self.motion.object_pos_w[0, :2].clone()
        else:
            self._reanchor_pivot = torch.zeros(2, device=self.device)

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._adaptive_reset_block = 0
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.object = None
        if self.cfg.object_name is not None:
            if not self.motion.has_object:
                raise ValueError(
                    f"{self.cfg.motion_file} carries no object track, but object_name="
                    f"{self.cfg.object_name!r} was configured."
                )
            self.object = env.scene[self.cfg.object_name]
            self.metrics["error_object_pos"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics["error_object_rot"] = torch.zeros(self.num_envs, device=self.device)

        self.fixture = None
        if self.cfg.fixture_name is not None:
            if self.motion.fixture_joint_pos is None:
                raise ValueError(
                    f"{self.cfg.motion_file} carries no fixture joint tracks, but fixture_name="
                    f"{self.cfg.fixture_name!r} was configured. Convert it with tracker_to_npz.py."
                )
            if self._reanchor_active:
                # The fixture is static in the world; re-anchoring rotates the
                # reference about the box, which would leave the two disagreeing.
                raise ValueError("reanchor_pose_range cannot be combined with a fixture")
            self.fixture = env.scene[self.cfg.fixture_name]
            missing = [j for j in self.fixture.joint_names if j not in self.motion.fixture_joint_names]
            if missing:
                raise ValueError(f"fixture joints with no track in the motion: {missing}")
            # Motion columns in the asset's joint order, for resets.
            self._fixture_cols = torch.tensor(
                [self.motion.fixture_joint_names.index(j) for j in self.fixture.joint_names], device=self.device
            )
            # Tracked subset (rewards/observations), in cfg order.
            self._tracked_asset_ids = torch.tensor(
                [self.fixture.joint_names.index(j) for j in self.cfg.fixture_tracked_joints], device=self.device
            )
            self._tracked_cols = torch.tensor(
                [self.motion.fixture_joint_names.index(j) for j in self.cfg.fixture_tracked_joints], device=self.device
            )
            self.metrics["error_fixture_joint_pos"] = torch.zeros(self.num_envs, device=self.device)

        self.metrics["sbto_pd_deviation"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        if self.cfg.command_frame_offsets:
            # Reference joint pos and vel at each
            # offset, (N, K, 2J) flattened, so the policy sees what is coming.
            idx = self._offset_steps(self.cfg.command_frame_offsets)
            return torch.cat([self.motion.joint_pos[idx], self.motion.joint_vel[idx]], dim=2).reshape(self.num_envs, -1)
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    ##
    # Re-anchoring transform
    ##
    # Applied to reference quantities in MOTION coordinates, before env_origins
    # is added, so the per-env clone offset stays independent of it.

    def _ra_pos(self, pos: torch.Tensor) -> torch.Tensor:
        """Rotate about the pivot, then translate. Shape (N, 3) or (N, B, 3)."""
        if not self._reanchor_active:
            return pos
        quat, dxy, pivot = self._reanchor_quat, self._reanchor_dxy, self._reanchor_pivot
        if pos.dim() == 3:
            quat = quat[:, None, :].expand(-1, pos.shape[1], -1)
            dxy = dxy[:, None, :]
        centred = pos.clone()
        centred[..., :2] -= pivot
        out = quat_apply(quat, centred)
        out[..., :2] += pivot + dxy
        return out

    def _ra_quat(self, quat: torch.Tensor) -> torch.Tensor:
        """Pre-multiply by the yaw offset. Shape (N, 4) or (N, B, 4)."""
        if not self._reanchor_active:
            return quat
        off = self._reanchor_quat
        if quat.dim() == 3:
            off = off[:, None, :].expand(-1, quat.shape[1], -1)
        return quat_mul(off, quat)

    def _ra_vec(self, vec: torch.Tensor) -> torch.Tensor:
        """Rotate a free vector (velocity): no pivot, no translation."""
        if not self._reanchor_active:
            return vec
        quat = self._reanchor_quat
        if vec.dim() == 3:
            quat = quat[:, None, :].expand(-1, vec.shape[1], -1)
        return quat_apply(quat, vec)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._ra_pos(self.motion.body_pos_w[self.time_steps]) + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._ra_quat(self.motion.body_quat_w[self.time_steps])

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._ra_vec(self.motion.body_lin_vel_w[self.time_steps])

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._ra_vec(self.motion.body_ang_vel_w[self.time_steps])

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return (
            self._ra_pos(self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index])
            + self._env.scene.env_origins
        )

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self._ra_quat(self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index])

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self._ra_vec(self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index])

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self._ra_vec(self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index])

    def set_object_stack_transform(self, pivot, shift, yaw, scale, object_scale, bottom_offset) -> None:
        """Map the object reference onto a per-env resized and moved fixture stack.

        The fixtures the object rests on (e.g. a table) are scaled by
        ``scale`` (N,) about ``pivot`` (N, 3, env frame), raised / moved by ``shift``
        (N, 3) and turned by ``yaw`` (N,) about the pivot; the object itself is scaled
        by ``object_scale`` (N,) about its origin, ``bottom_offset`` (3,) being its
        resting point in its own frame. The object's resting point follows the
        fixtures and its origin sits ``object_scale * bottom_offset`` above it:

            x' = P + d + R_yaw [ s (x - P) + (s - s_obj) R_ref o ],  q' = q_yaw q_ref,
            v' = s R_yaw v,  w' = R_yaw w
        """
        dev = self.device
        z = torch.zeros_like(torch.as_tensor(yaw, device=dev, dtype=torch.float32))
        half = 0.5 * torch.as_tensor(yaw, device=dev, dtype=torch.float32)
        self._stack = dict(
            pivot=torch.as_tensor(pivot, device=dev, dtype=torch.float32),
            shift=torch.as_tensor(shift, device=dev, dtype=torch.float32),
            q=torch.stack([z, z, torch.sin(half), torch.cos(half)], dim=-1),  # xyzw
            s=torch.as_tensor(scale, device=dev, dtype=torch.float32),
            s_obj=torch.as_tensor(object_scale, device=dev, dtype=torch.float32),
            o=torch.as_tensor(bottom_offset, device=dev, dtype=torch.float32),
        )

    def _stack_pose(self, pos: torch.Tensor, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the stack map to reference poses (env frame), pos (N, [K,] 3), quat (N, [K,] 4)."""
        st = self._stack
        if st is None:
            return pos, quat
        lead = pos.shape[:-1]
        ex = lambda t, d: t.reshape(t.shape[0], *([1] * (len(lead) - 1)), d).expand(*lead, d)  # noqa: E731
        q, s, ds = ex(st["q"], 4), ex(st["s"][:, None], 1), ex((st["s"] - st["s_obj"])[:, None], 1)
        P, d = ex(st["pivot"], 3), ex(st["shift"], 3)
        rest = quat_apply(quat.reshape(-1, 4), st["o"].expand(pos.reshape(-1, 3).shape[0], 3)).reshape(pos.shape)
        local = s * (pos - P) + ds * rest
        new_pos = P + d + quat_apply(q.reshape(-1, 4), local.reshape(-1, 3)).reshape(pos.shape)
        new_quat = quat_mul(q.reshape(-1, 4), quat.reshape(-1, 4)).reshape(quat.shape)
        return new_pos, new_quat

    @property
    def object_pos_w(self) -> torch.Tensor:
        pos = self.motion.object_pos_w[self.time_steps]
        if self._obj_off_active or self._rest_active:
            pos = pos.clone()
        if self._obj_off_active:
            pos[:, :2] += self._obj_off_dxy
        if self._rest_active:
            # The whole reference path rises or falls with the resting face, so
            # the reward asks for the box to be carried at the height it is
            # actually lying at rather than dragged to the recorded one.
            pos[:, 2] += self._rest_dz
        if self._stack is not None:
            pos, _ = self._stack_pose(pos, self.motion.object_quat_w[self.time_steps])
        return self._ra_pos(pos) + self._env.scene.env_origins

    @property
    def object_quat_w(self) -> torch.Tensor:
        quat = self.motion.object_quat_w[self.time_steps]
        if self._obj_off_active:
            # Pre-multiplying by a yaw rotation spins the box about the vertical
            # WITHOUT touching its position -- orientation and position are
            # independent here -- so the box is gripped at a different angle and
            # then carried along the very same path.
            quat = quat_mul(self._obj_off_quat, quat)
        if self._rest_active:
            # POST-multiplied, unlike the yaw offset above, so the rotation is
            # about the BOX's own axes rather than the world's. The reference
            # carries a ~-43 deg yaw at frame 0, and a roll about world X tips a
            # yawed box onto an EDGE, not onto a face: measured support heights
            # then reached 0.179 m, above the largest half-extent (0.14), with
            # the box up to 8.8 cm through the floor. A body-frame rotation
            # lands it flat on the intended face for any reference yaw.
            quat = quat_mul(quat, self._rest_quat)
        if self._stack is not None:
            quat = quat_mul(self._stack["q"], quat)
        return self._ra_quat(quat)

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        # A constant translation does not change a velocity, so the object-only
        # offset leaves this alone; only the whole-scene re-anchor rotates it.
        vel = self.motion.object_lin_vel_w[self.time_steps]
        if self._stack is not None:
            vel = self._stack["s"][:, None] * quat_apply(self._stack["q"], vel)
        return self._ra_vec(vel)

    @property
    def object_ang_vel_w(self) -> torch.Tensor:
        vel = self.motion.object_ang_vel_w[self.time_steps]
        if self._obj_off_active:
            # q'(t) = qz . q(t) implies omega' = R_z . omega.
            vel = quat_apply(self._obj_off_quat, vel)
        if self._stack is not None:
            vel = quat_apply(self._stack["q"], vel)
        return self._ra_vec(vel)

    def _offset_steps(self, offsets) -> torch.Tensor:
        """Reference frame indices time_steps + offset, clamped to the motion, (N, K)."""
        off = torch.as_tensor(offsets, device=self.device, dtype=torch.long)
        return torch.clamp(self.time_steps[:, None] + off[None, :], 0, self.motion.time_step_total - 1)

    def object_pose_w_at_offsets(self, offsets) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference object pose at frame offsets: pos (N, K, 3), quat (N, K, 4), world frame.

        Frame offsets are only supported without object offsets/resting-state
        randomisation, which would have to be re-applied per offset.
        """
        if self._obj_off_active or self._rest_active or self._reanchor_active:
            raise ValueError("object pose at frame offsets does not support object offsets/resting states/reanchoring")
        steps = self._offset_steps(offsets)
        pos, quat = self._stack_pose(self.motion.object_pos_w[steps], self.motion.object_quat_w[steps])
        return pos + self._env.scene.env_origins[:, None, :], quat

    def contact_ref_at_offsets(self, offsets) -> torch.Tensor:
        """Reference contact channels at frame offsets, (N, K, C)."""
        return self.motion.contact_ref[self._offset_steps(offsets)]

    @property
    def fixture_joint_pos(self) -> torch.Tensor:
        """Reference angle of each tracked fixture joint, (N, n_tracked)."""
        return self.motion.fixture_joint_pos[self.time_steps][:, self._tracked_cols]

    @property
    def robot_fixture_joint_pos(self) -> torch.Tensor:
        """Actual angle of each tracked fixture joint, (N, n_tracked)."""
        return _as_tensor(self.fixture.data.joint_pos)[:, self._tracked_asset_ids]

    @property
    def fixture_pos_w(self) -> torch.Tensor:
        return _as_tensor(self.fixture.data.root_pos_w)

    @property
    def fixture_quat_w(self) -> torch.Tensor:
        return _as_tensor(self.fixture.data.root_quat_w)

    @property
    def robot_object_pos_w(self) -> torch.Tensor:
        return _as_tensor(self.object.data.root_pos_w)

    @property
    def robot_object_quat_w(self) -> torch.Tensor:
        return _as_tensor(self.object.data.root_quat_w)

    @property
    def robot_object_lin_vel_w(self) -> torch.Tensor:
        return _as_tensor(self.object.data.root_lin_vel_w)

    @property
    def robot_object_ang_vel_w(self) -> torch.Tensor:
        return _as_tensor(self.object.data.root_ang_vel_w)

    @property
    def joint_pd_target(self) -> torch.Tensor:
        """The PD target SBTO optimized for this frame -- the residual baseline."""
        return self.motion.joint_pd_target[self.time_steps]

    @property
    def contact_ref(self) -> torch.Tensor:
        return self.motion.contact_ref[self.time_steps]

    @property
    def contact_force_ref(self) -> torch.Tensor:
        return self.motion.contact_force_ref[self.time_steps]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        # Refresh the relative-pose buffers, which _update_command would
        # otherwise not rewrite until the END of the step that follows the
        # reset. IsaacLab runs termination_manager.compute() BEFORE
        # command_manager.compute(), and CommandTerm.reset() only resamples, so
        # without this the first step of every episode compares the freshly
        # reset robot against the PREVIOUS episode's reference -- and on the
        # very first reset against the zero-initialised buffer, a ~1 m wrist
        # error that fires ee_body_pos on step 1 for essentially every env.
        # Updating the command here makes the first step compare against the new
        # episode's reference.
        #
        # The robot was just written onto the reference, so the reference
        # re-anchored to it is the reference itself. Reading the anchor back
        # from the sim instead would not work: write_root_state_to_sim does not
        # re-run forward kinematics, so robot.data.body_pos_w still holds the
        # pre-reset pose here. For the z-only gates this is exact anyway --
        # _update_command takes delta_pos_w[..., 2] from the reference anchor
        # and its delta rotation is yaw-only, so the relative z always equals
        # body_pos_w's z. Only the anchor's xy/yaw randomisation (<=5 cm,
        # <=0.2 rad) is dropped, and only for this one step.
        ids = slice(None) if env_ids is None else env_ids
        self.body_pos_relative_w[ids] = self.body_pos_w[ids]
        self.body_quat_relative_w[ids] = self.body_quat_w[ids]
        return extras

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        if self.fixture is not None:
            self.metrics["error_fixture_joint_pos"] = torch.abs(self.fixture_joint_pos - self.robot_fixture_joint_pos).mean(dim=-1)

        if self.object is not None:
            self.metrics["error_object_pos"] = torch.norm(self.object_pos_w - self.robot_object_pos_w, dim=-1)
            self.metrics["error_object_rot"] = quat_error_magnitude(self.object_quat_w, self.robot_object_quat_w)

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

        # His sbto_pd_deviation: applied PD target vs the motion's own PD target.
        action = self._env.action_manager._terms.get("JointPositionAction")
        if action is not None and self.motion.joint_pd_target is not None and hasattr(action, "_pd_columns"):
            if action._pd_columns is not None:
                ref = self.joint_pd_target[:, action._pd_columns]
                self.metrics["sbto_pd_deviation"] = torch.norm(ref - action.processed_actions, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        if self.cfg.start_frame is not None:
            # Pinned start: the sampler over-weights whichever part of the motion
            # the policy keeps failing at, which is right for training and
            # misleading for a demo or a transfer measurement.
            self.time_steps[env_ids] = self.cfg.start_frame
            return

        if self.cfg.sampling_mode == "uniform":
            # Every frame equally likely -- switched on by a curriculum term once
            # the policy has converged, so no part of the motion goes unpractised.
            lo = self.cfg.min_start_frame
            self.time_steps[env_ids] = (
                lo + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device) * (self.motion.time_step_total - 1 - lo)
            ).long()
            return
        if self.cfg.adaptive_reset_interval:
            # Forget the failure statistics every N learning iterations, so an old
            # weakness of a much worse policy does not keep steering the sampling.
            block = int(self._env.common_step_counter) // (self.cfg.steps_per_iteration * self.cfg.adaptive_reset_interval)
            if block > self._adaptive_reset_block:
                self._adaptive_reset_block = block
                self.bin_failed_count.zero_()

        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)
        if self.cfg.min_start_frame:
            # Bins lying wholly before the first allowed frame are never drawn.
            bin_end = (torch.arange(self.bin_count, device=self.device) + 1) * (self.motion.time_step_total - 1) / self.bin_count
            sampling_probabilities = sampling_probabilities * (bin_end > self.cfg.min_start_frame)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()
        if self.cfg.min_start_frame:
            self.time_steps[env_ids] = self.time_steps[env_ids].clamp(min=self.cfg.min_start_frame)

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)
        # Draw the re-anchoring offset BEFORE anything reads a reference
        # property: every read below (robot root pose, and object pose inside
        # _reset_object) must already see the transformed reference, or the robot
        # and box would be placed in two different frames.
        self._resample_reanchor(env_ids)
        self._resample_object_offset(env_ids)
        self._resample_resting_state(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

        self._reset_object(env_ids)
        self._reset_fixture(env_ids)

    def _reset_fixture(self, env_ids: Sequence[int]):
        """Every fixture joint to its reference at the episode's start frame (the root is fixed)."""
        if self.fixture is None:
            return
        steps = self.time_steps[env_ids]
        pos = self.motion.fixture_joint_pos[steps][:, self._fixture_cols]
        vel = self.motion.fixture_joint_vel[steps][:, self._fixture_cols]
        self.fixture.write_joint_state_to_sim(pos, vel, env_ids=env_ids)

    def _resample_reanchor(self, env_ids: Sequence[int]):
        """Draw a new planar offset for the whole reference, per env.

        With ``reanchor_pose_range`` unset this is a no-op and the offset stays
        identity, so existing tasks are bit-for-bit unaffected.

        Note this is NOT the same thing as ``object_pose_range``: that perturbs
        the box away from a fixed reference, which asks the robot to leave the
        recorded path, reach a box it has no arm length for, and then put it back
        on a path that no longer matches. This moves the box and the entire
        recording together, so the robot-box geometry at the grasp is always the
        demonstrated one, and what varies is only where in the world the task
        happens.
        """
        if not self._reanchor_active:
            return
        r = self.cfg.reanchor_pose_range
        ranges = torch.tensor(
            [r.get(k, (0.0, 0.0)) for k in ("x", "y", "yaw")], device=self.device, dtype=torch.float
        )
        s = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=self.device)
        self._reanchor_dxy[env_ids] = s[:, :2]
        zeros = torch.zeros_like(s[:, 2])
        self._reanchor_quat[env_ids] = quat_from_euler_xyz(zeros, zeros, s[:, 2])

    def _resample_object_offset(self, env_ids: Sequence[int]):
        """Draw a new object-only shift and spin, per env.

        Moves the box AND its reference trajectory together, leaving the robot's
        reference untouched -- so the box really is displaced and rotated
        relative to the robot, and the reward asks for it to be carried along a
        correspondingly moved path rather than wrestled back to the recorded
        orientation. The grasp is the only thing that has to change.

        Keep the translation small. Measured on this asset, the arms sit ~12 mm
        from their kinematic maximum at the grasp and the reference covers only
        0.10 m in the 2.16 s before it, so displacement past a few centimetres
        cannot be absorbed however long it trains. Yaw carries no such cost: the
        box stays where it is and only the faces presented to the hands change.
        """
        if not self._obj_off_active:
            return
        r = self.cfg.object_offset_pose_range
        ranges = torch.tensor(
            [r.get(k, (0.0, 0.0)) for k in ("x", "y", "yaw")], device=self.device, dtype=torch.float
        )
        s = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=self.device)

        # Zero the offset for any env starting at or after the grasp, where the
        # box is already in the robot's hands. Offsetting it there would place a
        # rotated box inside hands posed for an unrotated one -- an
        # interpenetrating state the simulator resolves by throwing the box out.
        # Those episodes get the plain reference instead, which is consistent.
        pre_grasp = (self.time_steps[env_ids] < self._grasp_onset).float().unsqueeze(-1)
        s = s * pre_grasp

        self._obj_off_dxy[env_ids] = s[:, :2]
        zeros = torch.zeros_like(s[:, 2])
        self._obj_off_quat[env_ids] = quat_from_euler_xyz(zeros, zeros, s[:, 2])

    def _resample_resting_state(self, env_ids: Sequence[int]):
        """Pick which face the box is lying on, and move its reference to match.

        Both halves matter. Rotating the box alone leaves the reward asking for
        it to be carried along the recorded path at the recorded height, which a
        box resting 7.75 cm lower cannot satisfy; shifting the reference with it
        turns an impossible correction into a different grasp.

        Gated to episodes starting before the grasp, for the reason
        _resample_object_offset is: a tipped box inside hands posed for an
        upright one is an interpenetrating state the simulator resolves by
        ejecting the box.
        """
        if not self._rest_active:
            return
        idx = torch.randint(
            len(self._rest_options), (len(env_ids),), device=self.device
        )
        pre_grasp = self.time_steps[env_ids] < self._grasp_onset

        quat = self._rest_options[idx]
        dz = self._rest_dz_options[idx]
        identity = torch.zeros_like(quat)
        identity[:, 0] = 1.0
        self._rest_quat[env_ids] = torch.where(pre_grasp.unsqueeze(-1), quat, identity)
        self._rest_dz[env_ids] = torch.where(pre_grasp, dz, torch.zeros_like(dz))

    def _reset_object(self, env_ids: Sequence[int]):
        """Place the object on its reference, randomized as the paper describes.

        Which quantity gets perturbed depends on where in the motion the episode
        starts: in the opening frames the object is still at rest, so its pose is
        what the policy has to cope with; once the motion is under way the object
        is already moving and its velocity is the uncertain part. Randomizing
        both everywhere would ask the policy to recover from states the reference
        never passes through.
        """
        if self.object is None:
            return

        pos = self.object_pos_w[env_ids].clone()
        quat = self.object_quat_w[env_ids].clone()
        lin_vel = self.object_lin_vel_w[env_ids].clone()
        ang_vel = self.object_ang_vel_w[env_ids].clone()

        early = self.time_steps[env_ids] < int(
            self.cfg.object_early_frame_fraction * self.motion.time_step_total
        )

        if self.cfg.object_pose_range and torch.any(early):
            ranges = torch.tensor(
                [self.cfg.object_pose_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]],
                device=self.device,
            )
            samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
            samples[~early] = 0.0
            pos += samples[:, :3]
            quat = quat_mul(quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5]), quat)
            self._obj_pose_sample_xy[env_ids] = samples[:, :2]
        else:
            self._obj_pose_sample_xy[env_ids] = 0.0

        if self.cfg.object_velocity_range and torch.any(~early):
            ranges = torch.tensor(
                [self.cfg.object_velocity_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]],
                device=self.device,
            )
            samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
            samples[early] = 0.0
            lin_vel += samples[:, :3]
            ang_vel += samples[:, 3:]

        self.object.write_root_state_to_sim(
            torch.cat([pos, quat, lin_vel, ang_vel], dim=-1), env_ids=env_ids
        )

    def _update_command(self):
        self.time_steps += 1
        if getattr(self._env.cfg.terminations, "motion_end", None) is not None:
            # His truncate-on-motion-end regime: the motion_ended termination
            # (time_out=True) ends the episode on the last frame. Clamp only
            # guards an env that a reset earlier in this step put on that frame.
            self.time_steps.clamp_(max=self.motion.time_step_total - 1)
        else:
            # Legacy: resample in place (teleport, no done signal).
            env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
            self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    object_name: str | None = None

    fixture_name: str | None = None
    """Scene entity of a fixed-base articulated fixture (e.g. a cabinet door) whose
    joints the motion tracks; reset to the reference at every episode start."""

    fixture_tracked_joints: list[str] = []
    """Fixture joints that the fixture observations and reward are about."""

    sampling_mode: str = "adaptive"
    """"adaptive" (failure-weighted start frames) or "uniform". A curriculum term
    can switch it during training."""

    adaptive_reset_interval: int | None = None
    """If set, zero the adaptive failure statistics every this many learning iterations."""

    steps_per_iteration: int = 24
    """Env steps per learning iteration (the runner's num_steps_per_env), to count iterations."""

    grasp_onset_hands: str = "both"
    """When the grasp counts as started, for gating object offsets: "both" hands
    in contact (two-handed carry) or "any" hand (one-handed grasp)."""
    """Scene entity holding the manipulated object, or None for motion-only tracking."""

    object_pose_range: dict[str, tuple[float, float]] = {}
    """Applied when an episode starts in the opening frames, where the object is
    still at rest and its pose is what the policy must cope with."""

    object_velocity_range: dict[str, tuple[float, float]] = {}
    """Applied when an episode starts mid-motion, where the object is already
    moving and its velocity is the uncertain quantity."""

    object_early_frame_fraction: float = 0.1
    """Boundary between those two cases, as a fraction of the motion."""

    object_resting_states: tuple[tuple[float, float, float, float], ...] = ()
    """Quaternions (w, x, y, z) for the faces the box may start resting on.

    One is drawn uniformly per reset and PRE-multiplied onto the reference
    orientation, while the reference height shifts by the change in the box's
    support height -- so the box and the path it must follow move together.
    Empty (the default) keeps the recorded face and reproduces current behaviour
    exactly.

    For the 0.205 x 0.125 x 0.28 m box, the six face-down options give three
    distinct geometries (opposite faces are identical for a plain cuboid):

        (1, 0, 0, 0)              recorded face, centre 0.140 m, 0.28 m tall
        (0, 1, 0, 0)              opposite face, same geometry
        (0.7071, 0.7071, 0, 0)    roll 90, centre 0.0625 m, 0.125 m tall
        (0.7071, -0.7071, 0, 0)   roll -90, same
        (0.7071, 0, 0.7071, 0)    pitch 90, centre 0.1025 m, 0.205 m tall
        (0.7071, 0, -0.7071, 0)   pitch -90, same

    Needs `object_geom_size` in the motion file; the support height of a tipped
    box follows from its half-extents.

    Unlike translation, this is not bounded by the arms' reach: the residual has
    enough authority in the legs and waist (hip/knee scales are 0.29-0.46 rad
    against the shoulder's 0.083) to squat to a lower box, which is observable
    on hardware -- the robot takes steps toward a box placed out of reach.
    """

    object_offset_pose_range: dict[str, tuple[float, float]] = {}
    """Planar (x, y, yaw) offset applied to the OBJECT and its reference only.

    The robot's reference is untouched, so the robot-relative-to-box pose really
    varies and the grasp has to adapt -- while the reward still asks for the box
    to be carried along a correspondingly moved path, rather than wrestled back
    to the recorded orientation. Empty (the default) means identity.

    The distinction from the two neighbouring fields matters:

    * :attr:`reanchor_pose_range` moves robot and box together, so their relative
      pose is preserved by construction. Good for placing the task anywhere in
      the world; teaches nothing about adapting.
    * :attr:`object_pose_range` perturbs the box away from a FIXED reference, so
      the reward then demands the box be returned to the recorded path. Combined
      with the reach limit, that is unsatisfiable; training on it measurably
      degraded the policy.
    * this field moves the box and its reference together, which asks only for a
      different grasp.

    Keep x/y small. Measured on this asset the arms are ~12 mm from their
    kinematic maximum at the grasp, and the reference covers just 0.10 m in the
    2.16 s before it, so displacement beyond a few centimetres cannot be absorbed
    no matter how long it trains. Yaw is not limited this way: the box stays put
    and only the faces presented to the hands change, which is shoulder and elbow
    work rather than reach.
    """

    reanchor_pose_range: dict[str, tuple[float, float]] = {}
    """Planar (x, y, yaw) offset applied to the WHOLE reference each reset.

    Robot bodies, anchor and object trajectory all move together, rotating about
    the reference object's frame-0 position, so the task is posed relative to
    wherever the box is rather than where the recording put it. Empty (the
    default) means identity: every reference property returns exactly what it did
    before this field existed.

    Distinct from :attr:`object_pose_range`, which moves the box away from a
    fixed reference. That asks the robot to leave the recorded path and fetch a
    box its arms cannot reach (measured: 7-9 mm of extension headroom at the
    grasp), then return it to a path that no longer matches -- a demand it cannot
    satisfy, and training on it measurably degrades the policy. Re-anchoring
    keeps the robot-box geometry at the grasp identical to the demonstration and
    varies only where in the world the task happens, which is a transform rather
    than a harder problem.

    Only x/y/yaw: the ground is a plane, so a planar transform is the largest one
    that cannot disturb body heights or gravity alignment.
    """

    command_frame_offsets: list[int] | None = None
    """Reference frames the ``command`` observation covers, e.g. [0, 1, 2, 4, 8]
    None keeps the current frame only."""

    min_start_frame: int = 0
    """Earliest frame an episode may start at, in both adaptive and uniform
    sampling. Skips a stretch of the reference the policy should not learn
    from, e.g. a lead-in that slides the feet."""

    start_frame: int | None = None
    """Pin every episode to this frame instead of adaptively sampling a start.

    For playback and for measuring transfer, where a sampler biased toward the
    hard parts of the motion would misrepresent what the policy does. Leave as
    None for training."""

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
