# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""The velocity-tracking locomotion policy that stands / walks the robot before and
after the box motion.

Observation (480-D): six terms, each a 5-frame history, oldest frame first, term by term
(all 5 frames of ang_vel, then all 5 of gravity, ...), as IsaacLab's CircularBuffer
lays them out. On the first frame after a reset every slot holds the current value.

    term         dim  scale
    ang_vel        3  0.2     pelvis IMU gyro
    gravity        3  1       gravity in the pelvis frame, from the IMU quaternion
    command        3  1       (vx, vy, wz) in m/s, m/s, rad/s
    joint_pos     29  1       joint position - default (Isaac order)
    joint_vel     29  0.05
    last_action   29  1

Action (29, Isaac order): q_des = default + 0.25 * action, sent with LOCO_KP / LOCO_KD.
"""

from __future__ import annotations

import collections

import numpy as np
import torch

from . import layout

HISTORY = 5
OBS_DIM = 480
ACTION_SCALE = 0.25
VX_RANGE = (-0.5, 1.0)
VY_MAX = 0.5
WZ_MAX = 1.0
TERMS = ("ang_vel", "gravity", "command", "joint_pos", "joint_vel", "last_action")

_GAINS = {  # joint name without left_/right_ -> (kp, kd), the locomotion policy's own gains
    "hip_pitch": (100.0, 2.0), "hip_roll": (100.0, 2.0), "hip_yaw": (100.0, 2.0), "knee": (150.0, 4.0),
    "ankle_pitch": (40.0, 2.0), "ankle_roll": (40.0, 2.0),
    "waist_yaw": (200.0, 5.0), "waist_roll": (40.0, 5.0), "waist_pitch": (40.0, 5.0),
    "shoulder_pitch": (40.0, 1.0), "shoulder_roll": (40.0, 1.0), "shoulder_yaw": (40.0, 1.0), "elbow": (40.0, 1.0),
    "wrist_roll": (40.0, 1.0), "wrist_pitch": (40.0, 1.0), "wrist_yaw": (40.0, 1.0),
}
LOCO_KP_DDS = layout._per_joint(_GAINS, 0)
LOCO_KD_DDS = layout._per_joint(_GAINS, 1)
# Same standing pose as the box student's default.
LOCO_DEFAULT_POS_DDS = layout.DEFAULT_POS_DDS
LOCO_DEFAULT_POS_ISAAC = layout.DEFAULT_POS_ISAAC


def projected_gravity(quat_wxyz) -> np.ndarray:
    """Unit gravity in the body frame for an IMU orientation (w, x, y, z)."""
    w, x, y, z = quat_wxyz
    return np.array([2.0 * (-z * x + w * y), -2.0 * (z * y + w * x), 1.0 - 2.0 * (w * w + z * z)])


class LocomotionPolicy:
    def __init__(self, path: str, device: str = "cpu"):
        self.device = device
        self.net = torch.jit.load(path, map_location=device).eval()
        with torch.inference_mode():
            out = self.net(torch.zeros(1, OBS_DIM, device=device))
        assert tuple(out.shape) == (1, 29), out.shape
        self.reset()

    def reset(self) -> None:
        self._hist: dict[str, collections.deque] | None = None
        self.last_action = np.zeros(29)

    def _observation(self, imu_quat_wxyz, gyro, q_dds, dq_dds, command) -> np.ndarray:
        terms = {
            "ang_vel": 0.2 * np.asarray(gyro, dtype=float),
            "gravity": projected_gravity(imu_quat_wxyz),
            "command": np.asarray(command, dtype=float),
            "joint_pos": np.asarray(q_dds)[layout.DDS_TO_ISAAC] - LOCO_DEFAULT_POS_ISAAC,
            "joint_vel": 0.05 * np.asarray(dq_dds)[layout.DDS_TO_ISAAC],
            "last_action": self.last_action.copy(),
        }
        if self._hist is None:
            self._hist = {n: collections.deque([terms[n]] * HISTORY, maxlen=HISTORY) for n in TERMS}
        else:
            for n in TERMS:
                self._hist[n].append(terms[n])
        obs = np.concatenate([np.concatenate(list(self._hist[n])) for n in TERMS])
        assert obs.shape == (OBS_DIM,)
        return obs

    def step(self, imu_quat_wxyz, gyro, q_dds, dq_dds, command) -> np.ndarray:
        """One 50 Hz tick -> joint position targets (DDS order)."""
        obs = self._observation(imu_quat_wxyz, gyro, q_dds, dq_dds, command)
        with torch.inference_mode():
            act = self.net(torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None])[0].cpu().numpy()
        self.last_action = act.astype(float)
        return (LOCO_DEFAULT_POS_ISAAC + ACTION_SCALE * self.last_action)[layout.ISAAC_TO_DDS]
