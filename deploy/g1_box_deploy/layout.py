# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""G1 (29 DoF) joint layout, gains and action scales of the box depth student.

Two joint orders meet here:

* DDS order (``BODY_JOINTS``): the index order of ``LowState_.motor_state`` /
  ``LowCmd_.motor_cmd`` on the real robot (Unitree's ``g1_joint_index_dds.md``).
* Isaac order (``ISAAC_JOINT_ORDER``): the articulation order IsaacLab trained the
  policy in. Every joint-indexed observation term and the action are in this order.

Defaults, gains and action scales are the training values (g1_mocap_free_box
``G1_29DOF_DUMMY_HANDS_MIMIC_CFG``).
"""

from __future__ import annotations

import numpy as np

BODY_JOINTS = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
assert len(BODY_JOINTS) == 29

ISAAC_JOINT_ORDER = [
    "left_hip_pitch", "right_hip_pitch", "waist_yaw", "left_hip_roll", "right_hip_roll", "waist_roll",
    "left_hip_yaw", "right_hip_yaw", "waist_pitch", "left_knee", "right_knee",
    "left_shoulder_pitch", "right_shoulder_pitch", "left_ankle_pitch", "right_ankle_pitch",
    "left_shoulder_roll", "right_shoulder_roll", "left_ankle_roll", "right_ankle_roll",
    "left_shoulder_yaw", "right_shoulder_yaw", "left_elbow", "right_elbow",
    "left_wrist_roll", "right_wrist_roll", "left_wrist_pitch", "right_wrist_pitch",
    "left_wrist_yaw", "right_wrist_yaw",
]
assert sorted(ISAAC_JOINT_ORDER) == sorted(BODY_JOINTS)

DDS_TO_ISAAC = np.array([BODY_JOINTS.index(n) for n in ISAAC_JOINT_ORDER])
ISAAC_TO_DDS = np.array([ISAAC_JOINT_ORDER.index(n) for n in BODY_JOINTS])

# Robot firmware constants (confirmed on the real G1): serial pitch/roll ankles.
MODE_PR = 0
MODE_MACHINE = 5

# Default joint pose (DDS order); only used as the zero of joint_pos_rel.
DEFAULT_POS_DDS = np.array([
    -0.10, 0.00, 0.00, 0.30, -0.20, 0.00,    # left leg
    -0.10, 0.00, 0.00, 0.30, -0.20, 0.00,    # right leg
    0.00, 0.00, 0.00,                        # waist yaw / roll / pitch
    0.30, 0.25, 0.00, 0.97, 0.15, 0.00, 0.00,     # left arm
    0.30, -0.25, 0.00, 0.97, -0.15, 0.00, 0.00,   # right arm
])
DEFAULT_POS_ISAAC = DEFAULT_POS_DDS[DDS_TO_ISAAC]

_GAINS = {  # joint name without left_/right_ -> (kp, kd), the training PD gains
    "hip_pitch": (75.0, 3.0), "hip_roll": (75.0, 3.0), "hip_yaw": (75.0, 3.0), "knee": (100.0, 4.0),
    "ankle_pitch": (30.0, 4.0), "ankle_roll": (30.0, 4.0),
    "waist_yaw": (75.0, 3.0), "waist_roll": (75.0, 3.0), "waist_pitch": (75.0, 3.0),
    "shoulder_pitch": (75.0, 3.0), "shoulder_roll": (75.0, 3.0), "shoulder_yaw": (75.0, 3.0), "elbow": (75.0, 3.0),
    "wrist_roll": (20.0, 3.0), "wrist_pitch": (20.0, 3.0), "wrist_yaw": (20.0, 3.0),
}
_ACTION_SCALE = {  # 0.25 * effort_limit / kp (BeyondMimic convention)
    "hip_pitch": 0.293333, "hip_roll": 0.463333, "hip_yaw": 0.293333, "knee": 0.347500,
    "ankle_pitch": 0.416667, "ankle_roll": 0.416667,
    "waist_yaw": 0.293333, "waist_roll": 0.166667, "waist_pitch": 0.166667,
    "shoulder_pitch": 0.083333, "shoulder_roll": 0.083333, "shoulder_yaw": 0.083333, "elbow": 0.083333,
    "wrist_roll": 0.312500, "wrist_pitch": 0.062500, "wrist_yaw": 0.062500,
}


def _per_joint(table, idx=None):
    vals = [table[n.removeprefix("left_").removeprefix("right_")] for n in BODY_JOINTS]
    return np.array([v if idx is None else v[idx] for v in vals], dtype=float)


KP_DDS = _per_joint(_GAINS, 0)
KD_DDS = _per_joint(_GAINS, 1)
ACTION_SCALE_DDS = _per_joint(_ACTION_SCALE)

# Damping used for faults (no position hold): what Unitree's own tools send.
DAMPING_KD = 8.0
