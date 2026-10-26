# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""Unitree G1 (29 DoF) with rigid box-shaped "dummy hands", as trained and deployed."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.configclass import configclass


@configclass
class UnitreeArticulationCfg(ArticulationCfg):
    """Configuration for Unitree articulations."""

    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.9


@configclass
class UnitreeUsdFileCfg(sim_utils.UsdFileCfg):
    activate_contact_sensors: bool = True
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    )


# repository root (this file is training/source/g1_mocap_free_box/g1_mocap_free_box/assets/g1.py)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]

# Unitree's G1 29-DoF USD with an override layer: primitive (capsule / box) colliders on
# every body, and the two dummy hands as static collision boxes on wrist_yaw_link.
_G1_29DOF_DUMMY_HANDS_PRIMITIVE_COLLISION_USD = (
    _REPOSITORY_ROOT / "assets/isaaclab/g1-29dof/g1_29dof_rev_1_0_primitive_collision.usda"
)

##
# Tracking configuration
##
# The PD gains are those of the MuJoCo model the reference trajectory was optimised
# through (SBTO). The residual tracking policy adds a correction on top of the
# reference's `dof_pd_target`, and that target was optimised under exactly these gains:
# changing them makes the same target produce different torques, and the reference stops
# being the dynamically feasible trajectory it was optimised to be.
#
# MuJoCo applies tau = kp*(q_des - q) - kv*qd from <position kp kv>, plus another
# -damping*qd from the joint itself. Both act on the same velocity, so IsaacLab's single
# damping term is kv + joint damping.
_JOINT_DAMPING = 1.0  # <default><joint damping="1"> on every body joint
_JOINT_FRICTION = 0.1  # frictionloss, same source

# Rotor inertia per motor type. These match unitree_rl_lab's constants exactly, because
# both descend from the same Unitree motor spec.
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

# The dummy hands add no joints of their own, so the actuated-joint set is exactly the 29
# body DoF.
G1_29DOF_DUMMY_HANDS_ACTUATED_JOINTS = [
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_roll_joint",
    ".*_wrist_pitch_joint",
    ".*_wrist_yaw_joint",
]

G1_29DOF_DUMMY_HANDS_MIMIC_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUsdFileCfg(
        usd_path=_G1_29DOF_DUMMY_HANDS_PRIMITIVE_COLLISION_USD.as_posix(),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # kp 75 / kv 2 -- the model's <default> for the body, minus the overrides below.
        "hips": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint"],
            effort_limit_sim={
                ".*_hip_pitch_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_yaw_joint": 88.0,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_yaw_joint": 32.0,
            },
            stiffness=75.0,
            damping=2.0 + _JOINT_DAMPING,
            armature={
                ".*_hip_pitch_joint": ARMATURE_7520_14,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
            },
            friction=_JOINT_FRICTION,
        ),
        "knees": ImplicitActuatorCfg(
            joint_names_expr=[".*_knee_joint"],
            effort_limit_sim=139.0,
            velocity_limit_sim=20.0,
            stiffness=100.0,
            damping=3.0 + _JOINT_DAMPING,
            armature=ARMATURE_7520_22,
            friction=_JOINT_FRICTION,
        ),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            # 50, not 25: the ankles and waist run doubled 5020s. unitree_rl_lab's tracking
            # config uses 50 here too, and the reference trajectory peaks at 30 N-m.
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=30.0,
            damping=3.0 + _JOINT_DAMPING,
            armature=2.0 * ARMATURE_5020,
            friction=_JOINT_FRICTION,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim={
                "waist_yaw_joint": 88.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 32.0,
                "waist_roll_joint": 37.0,
                "waist_pitch_joint": 37.0,
            },
            stiffness=75.0,
            damping=2.0 + _JOINT_DAMPING,
            armature={
                "waist_yaw_joint": ARMATURE_7520_14,
                "waist_roll_joint": 2.0 * ARMATURE_5020,
                "waist_pitch_joint": 2.0 * ARMATURE_5020,
            },
            friction=_JOINT_FRICTION,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim=25.0,
            velocity_limit_sim=37.0,
            stiffness=75.0,
            damping=2.0 + _JOINT_DAMPING,
            armature=ARMATURE_5020,
            friction=_JOINT_FRICTION,
        ),
        # All three wrist joints run at kp 20 in the model, including wrist_roll, even
        # though it is a 5020 motor like the arm joints.
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_roll_joint", ".*_wrist_pitch_joint", ".*_wrist_yaw_joint"],
            effort_limit_sim={
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness=20.0,
            damping=2.0 + _JOINT_DAMPING,
            armature={
                ".*_wrist_roll_joint": ARMATURE_5020,
                ".*_wrist_pitch_joint": ARMATURE_4010,
                ".*_wrist_yaw_joint": ARMATURE_4010,
            },
            friction=_JOINT_FRICTION,
        ),
    },
    # Motor order of the robot's low-level interface (rt/lowstate, rt/lowcmd).
    joint_sdk_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
)
"""29-DoF G1 + dummy box hands with the reference trajectory's PD gains, for motion tracking."""


def _build_action_scale(cfg: UnitreeArticulationCfg, joint_names: list[str]) -> dict[str, float]:
    """Per-joint action scale, 0.25 * effort / stiffness, as in BeyondMimic.

    Looks each joint pattern up in its actuator group rather than requiring the
    effort and stiffness dicts to be keyed identically to joint_names_expr --
    mismatched keys otherwise drop joints from the scale dict silently, leaving
    them unscaled.
    """

    def _lookup(value, name, names):
        if isinstance(value, dict):
            return value.get(name)
        return value

    scale = {}
    for actuator in cfg.actuators.values():
        names = actuator.joint_names_expr
        for name in names:
            if name not in joint_names:
                continue
            effort = _lookup(actuator.effort_limit_sim, name, names)
            stiffness = _lookup(actuator.stiffness, name, names)
            if effort is None or not stiffness:
                raise ValueError(f"No effort/stiffness resolved for {name!r}")
            scale[name] = 0.25 * effort / stiffness
    missing = [n for n in joint_names if n not in scale]
    if missing:
        raise ValueError(f"Joints without an action scale: {missing}")
    return scale


G1_29DOF_DUMMY_HANDS_MIMIC_ACTION_SCALE = _build_action_scale(
    G1_29DOF_DUMMY_HANDS_MIMIC_CFG, G1_29DOF_DUMMY_HANDS_ACTUATED_JOINTS
)
