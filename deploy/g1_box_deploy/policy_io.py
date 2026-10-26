# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""Reference motion, observation and action of the box depth student.

Observation (154-D, every term unscaled, joint terms in Isaac order) + one depth image:

    slice      w  term
    0:58      58  motion command: reference joint_pos (29) + joint_vel (29) at frame k
    58:64      6  anchor orientation error: rot6d(conj(q_torso) * q_torso_ref[k])
    64:67      3  pelvis IMU gyro
    67:96     29  joint_pos - default_pos
    96:125    29  joint_vel
    125:154   29  last action (raw network output)
    depth  (1, 1, 48, 64)  metric depth, invalid -> far, clipped to [0.168, 2.5] m, / 2.5

Action (29, Isaac order): q_des(DDS) = dof_pd_target[k] + scale * action[ISAAC_TO_DDS],
sent with the training kp / kd. A residual on the reference PD target, not on a
standing pose: zero action replays the reference.

No position is observed anywhere -- only the torso ORIENTATION, from the pelvis IMU
plus the waist joints -- which is why the policy runs without motion capture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import layout
from .camera import CONTRACT
from .transforms import quat_conjugate, quat_from_axis_angle, quat_mul, rot6d, yaw_of

OBS_DIM = 154
DEPTH_SHAPE = (1, CONTRACT.height, CONTRACT.width)


class Motion:
    """The reference trajectory (npz written by the training repo, 50 fps)."""

    def __init__(self, path: str | Path, anchor_body: str = "torso_link"):
        d = np.load(Path(path).expanduser(), allow_pickle=True)
        names = [str(n).removesuffix("_joint") for n in d["joint_names"]]
        pd_names = [str(n).removesuffix("_joint") for n in d["dof_pd_target_joint_names"]]
        if names != layout.ISAAC_JOINT_ORDER or pd_names != layout.BODY_JOINTS:
            raise ValueError(f"{path}: unexpected joint order")
        self.fps = int(np.asarray(d["fps"]).reshape(-1)[0])
        self.command = np.concatenate([d["joint_pos"], d["joint_vel"]], axis=1).astype(float)  # (T, 58) Isaac
        self.pd_target = np.asarray(d["dof_pd_target"], dtype=float)  # (T, 29) DDS order
        self.n_frames = self.command.shape[0]
        bodies = [str(b) for b in d["body_names"]]
        q = np.asarray(d["body_quat_w"], dtype=float)[:, bodies.index(anchor_body)]
        self._anchor_quat0 = q[:, [3, 0, 1, 2]]  # npz is xyzw -> wxyz
        self.anchor_quat_w = self._anchor_quat0.copy()
        # Initial scene (sim2sim): robot root + joints and the box at frame 0.
        self.root_pos0 = np.asarray(d["body_pos_w"], dtype=float)[0, bodies.index("pelvis")]
        self.root_quat0 = np.asarray(d["body_quat_w"], dtype=float)[0, bodies.index("pelvis")][[3, 0, 1, 2]]
        self.joint_pos0_isaac = np.asarray(d["joint_pos"], dtype=float)[0]
        self.object_pos0 = np.asarray(d["object_pos"], dtype=float)[0]
        self.object_quat0 = np.asarray(d["object_rot"], dtype=float)[0][[3, 0, 1, 2]]

    def align_yaw(self, anchor_quat_w: np.ndarray) -> float:
        """Rotate the reference about z so its frame-0 torso yaw equals the robot's.
        The observation holds no world position, so yaw is the whole alignment."""
        yaw = yaw_of(anchor_quat_w) - yaw_of(self._anchor_quat0[0])
        qz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        self.anchor_quat_w = np.stack([quat_mul(qz, q) for q in self._anchor_quat0])
        return yaw


_WAIST = [layout.BODY_JOINTS.index(n) for n in ("waist_yaw", "waist_roll", "waist_pitch")]


def torso_quat(imu_quat_wxyz: np.ndarray, q_dds: np.ndarray) -> np.ndarray:
    """torso_link orientation = pelvis IMU orientation * waist (yaw, roll, pitch)."""
    yaw, roll, pitch = (float(q_dds[i]) for i in _WAIST)
    waist = quat_mul(quat_mul(quat_from_axis_angle((0, 0, 1), yaw), quat_from_axis_angle((1, 0, 0), roll)),
                     quat_from_axis_angle((0, 1, 0), pitch))
    return quat_mul(np.asarray(imu_quat_wxyz, dtype=float), waist)


def build_observation(motion: Motion, frame: int, imu_quat_wxyz, gyro, q_dds, dq_dds, last_action) -> np.ndarray:
    anchor_err = quat_mul(quat_conjugate(torso_quat(imu_quat_wxyz, q_dds)), motion.anchor_quat_w[frame])
    obs = np.concatenate([
        motion.command[frame],                                    # 58
        rot6d(anchor_err),                                        # 6
        np.asarray(gyro, dtype=float),                            # 3
        np.asarray(q_dds)[layout.DDS_TO_ISAAC] - layout.DEFAULT_POS_ISAAC,  # 29
        np.asarray(dq_dds)[layout.DDS_TO_ISAAC],                  # 29
        np.asarray(last_action, dtype=float),                     # 29
    ])
    assert obs.shape == (OBS_DIM,)
    return obs


def action_to_q_des(motion: Motion, frame: int, action: np.ndarray) -> np.ndarray:
    """Joint position targets (DDS order) for a network action (Isaac order)."""
    return motion.pd_target[frame] + layout.ACTION_SCALE_DDS * np.asarray(action, dtype=float)[layout.ISAAC_TO_DDS]


def preprocess_depth(metric: np.ndarray, cutoff_m: float = CONTRACT.cutoff_m,
                     min_depth_m: float = CONTRACT.min_depth_m) -> np.ndarray:
    """Metric depth (48, 64) in metres -> the policy image (1, 48, 64) in [0, 1].
    Invalid pixels (NaN / inf, e.g. stereo holes) read as FAR, as in training."""
    m = np.asarray(metric, dtype=np.float32)
    m = np.where(np.isfinite(m), m, cutoff_m)
    return (np.clip(m, min_depth_m, cutoff_m) / cutoff_m)[None].astype(np.float32)
