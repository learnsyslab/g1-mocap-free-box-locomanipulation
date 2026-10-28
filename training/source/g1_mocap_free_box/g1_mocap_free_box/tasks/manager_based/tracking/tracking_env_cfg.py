# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
# Portions adapted from unitree_rl_lab's tasks/mimic (Apache-2.0), itself based on
# whole_body_tracking (MIT); modified. See THIRD_PARTY_NOTICES.md.
#
# SPDX-License-Identifier: Apache-2.0

"""Teacher environments: the G1 tracks the box pick-and-place reference with privileged state."""

from __future__ import annotations

import numpy as np
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise  # additive is the default operation

import g1_mocap_free_box.tasks.manager_based.tracking.mdp as mdp
from g1_mocap_free_box.assets.g1 import (
    G1_29DOF_DUMMY_HANDS_ACTUATED_JOINTS,
    G1_29DOF_DUMMY_HANDS_MIMIC_ACTION_SCALE,
    G1_29DOF_DUMMY_HANDS_MIMIC_CFG,
)

##
# Scene definition
##

# The dummy hands are two static collision boxes on wrist_yaw_link: there is no separate
# hand link, so wrist_yaw_link *is* the hand.
#
# Both wrist links count as hand contact, not wrist_yaw alone: the hand collider
# (r = 0.020 m, on wrist_yaw) sits 4.9 cm inside the wrist sphere (r = 0.035 m, on
# wrist_pitch), so it is geometrically shielded and can never make first contact. The
# reference carries the box on the wrist collider at -0.0023 m penetration, while the
# hand collider never gets closer than +0.0098 m. This matches the set the reference's
# computed contact channel measures (the wrist_pitch subtree).
LEFT_HAND_CONTACT_LINKS = ["left_wrist_pitch_link", "left_wrist_yaw_link"]
RIGHT_HAND_CONTACT_LINKS = ["right_wrist_pitch_link", "right_wrist_yaw_link"]

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robots
    robot: ArticulationCfg = G1_29DOF_DUMMY_HANDS_MIMIC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )

    # Size, mass and friction are overwritten in __post_init__ from the motion file, so
    # the box cannot drift from the one the reference trajectory was optimised against.
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.31, 0.31, 0.34),
            # Required for the hand/wrist sensors to filter against this prim.
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.6),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6, dynamic_friction=0.6, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.32, 0.30)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.2)),
    )


##
# MDP settings
##


def _motion_file() -> str:
    """The shipped reference, or $G1_BOX_MOTION. A relative path is taken from where the
    command was typed ($INIT_CWD under pixi, whose tasks run inside training/)."""
    path = os.environ.get("G1_BOX_MOTION")
    if not path:
        return f"{os.path.dirname(__file__)}/motions/sub3_realbox_dummy_hands.npz"
    path = os.path.expanduser(path)
    return path if os.path.isabs(path) else os.path.join(os.environ.get("INIT_CWD", os.getcwd()), path)


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        # The reference: the OmniRetarget sub3_largebox_005 human demonstration (from
        # OMOMO), refined with SBTO for the dummy hands and the real 205 x 125 x 280 mm
        # box. 50 fps.
        #
        # Another motion (docs/CUSTOM_MOTION.md): set $G1_BOX_MOTION, or `stage --motion`.
        # Not `env.commands.motion.motion_file=<new>.npz` on the CLI: IsaacLab's
        # hydra_task_config builds env_cfg first -- running __post_init__, and with it
        # _sync_object_to_motion() -- and only then applies `env.*` overrides, so the
        # reference would change while the box kept the OLD motion's extents, mass and
        # friction, silently. The environment variable is read before any of that.
        motion_file=_motion_file(),
        anchor_body_name="torso_link",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
        object_name="object",
        object_pose_range={
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (0.0, 0.0),
            "yaw": (-0.1, 0.1),
        },
        object_velocity_range={
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "z": (-0.1, 0.1),
            "roll": (-0.2, 0.2),
            "pitch": (-0.2, 0.2),
            "yaw": (-0.2, 0.2),
        },
        body_names=[
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ],
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Residual: the policy supplies a correction to the reference's optimised PD target,
    # rather than reproducing the whole trajectory itself.
    JointPositionAction = mdp.MotionResidualJointPositionActionCfg(
        asset_name="robot",
        command_name="motion",
        joint_names=G1_29DOF_DUMMY_HANDS_ACTUATED_JOINTS,
        scale=G1_29DOF_DUMMY_HANDS_MIMIC_ACTION_SCALE,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        object_pose_b = ObsTerm(
            func=mdp.object_pose_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        object_pose_error_b = ObsTerm(
            func=mdp.object_pose_error_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_pose_b = ObsTerm(func=mdp.object_pose_b, params={"command_name": "motion"})
        object_pose_error_b = ObsTerm(func=mdp.object_pose_error_b, params={"command_name": "motion"})
        object_velocity_error = ObsTerm(func=mdp.object_velocity_error, params={"command_name": "motion"})
        actions = ObsTerm(func=mdp.last_action)

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=G1_29DOF_DUMMY_HANDS_ACTUATED_JOINTS),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.4, 0.9),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 32,
        },
    )

    object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.8, 1.25),
            "operation": "scale",
        },
    )

    # interval
    push_object = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.1, 0.1)},
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- base
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    # -- tracking
    # exp(-5||.||^2)
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4472},
    )
    # exp(-3||.||^2)
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.5774},
    )
    # exp(-5||.||^2)
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4472},
    )
    # exp(-3||.||^2)
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.5774},
    )
    # exp(-0.5||.||^2)
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.4142},
    )
    # exp(-0.05||.||^2)
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 4.4721},
    )

    # -- joint tracking, exp(-5||u - u_ref||^2) at weight 2.0
    motion_joint_pos = RewTerm(
        func=mdp.motion_joint_position_error_exp,
        weight=2.0,
        params={"command_name": "motion", "std": 0.4472},
    )

    # -- object tracking
    # Linear kernel, sigma set so the half-reward distance matches the paper's
    # exp(-8||.||^2) at 0.294 m -- see the reward's docstring.
    motion_object_pos = RewTerm(
        func=mdp.motion_object_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4247},
    )
    # Likewise; matches the paper's exp(-5||.||^2) half-reward angle of 0.372 rad.
    motion_object_ori = RewTerm(
        func=mdp.motion_object_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.5372},
    )
    motion_object_lin_vel = RewTerm(  # exp(-2||.||^2)
        func=mdp.motion_object_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.7071},
    )
    motion_object_ang_vel = RewTerm(  # exp(-0.2||.||^2)
        func=mdp.motion_object_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 2.2361},
    )
    # The paper's single 1.25 contact term, split evenly across the two hands.
    contact_match_left = RewTerm(
        func=mdp.motion_contact_match,
        weight=0.625,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=LEFT_HAND_CONTACT_LINKS),
            "contact_channel": "left_hand_cnt",
            "force_limit": 10.0,
        },
    )
    contact_match_right = RewTerm(
        func=mdp.motion_contact_match,
        weight=0.625,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=RIGHT_HAND_CONTACT_LINKS),
            "contact_channel": "right_hand_cnt",
            "force_limit": 10.0,
        },
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                # Feet and hands are expected to make contact; everything else touching
                # anything is a penalty. wrist_yaw_link is the hand.
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                    r"(?!(left|right)_wrist_(yaw|pitch)_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP.

    No termination is tied to object tracking, following SBTO's recipe: "we do not
    introduce any additional termination conditions related to object tracking, as we
    found that such terminations consistently degrade performance and exacerbate the
    learning difficulty when combined with adaptive sampling during training from
    scratch" (Section IV-E).
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        },
    )


##
# Environment configuration
##


@configclass
class G1BoxTrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Teacher stage 0: track the box pick-and-place reference with privileged state."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 30.0
        self._sync_object_to_motion()

        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # Contact-rich: thousands of envs, each with a manipulated box. At the default
        # 5 * 2**15 patches PhysX silently overflows its patch buffer and drops contacts
        # -- the robot then falls through its own feet within a few steps and never
        # learns to stand. 8192 envs of this scene asked for 330982, so allow room.
        self.sim.physics = PhysxCfg(gpu_max_rigid_patch_count=20 * 2**15)

    def _sync_object_to_motion(self):
        """Take the object's shape, mass and friction from the motion file.

        The motion file stores the geometry the reference was optimised against, so the
        box here is whatever that optimisation used rather than a constant that has to be
        kept in step by hand. Silently skipped for motions without an object.
        """
        if not os.path.isfile(self.commands.motion.motion_file):
            return
        motion = np.load(self.commands.motion.motion_file)
        if "object_geom_size" not in motion:
            return

        spawn = self.scene.object.spawn
        # MuJoCo geom size is half-extents; CuboidCfg takes full extents. Cast to
        # plain floats -- numpy scalars leak into the config that Hydra has to
        # serialise, and OmegaConf does not accept them.
        spawn.size = tuple(float(2.0 * v) for v in np.asarray(motion["object_geom_size"], dtype=float))
        spawn.mass_props.mass = float(motion["object_mass"][0])
        friction = float(np.asarray(motion["object_friction"], dtype=float)[0])
        spawn.physics_material.static_friction = friction
        spawn.physics_material.dynamic_friction = friction


def zero_replay_randomization(cfg: ManagerBasedRLEnvCfg) -> None:
    """Zero every randomization source on an already-built env cfg, in place.

    For replaying or checking a reference exactly as recorded: no reset noise on
    the robot or the box, no startup randomization, no pushes.
    """
    cfg.commands.motion = cfg.commands.motion.replace(
        pose_range={},
        velocity_range={},
        joint_position_range=(0.0, 0.0),
        object_pose_range={},
        object_velocity_range={},
    )
    for name in ("physics_material", "add_joint_default_pos", "base_com", "object_physics_material",
                 "object_mass", "push_object", "push_robot"):
        setattr(cfg.events, name, None)


class G1BoxTrackingPlayEnvCfg(G1BoxTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9


##
# BoxYaw: teacher curriculum stage 1 -- box yaw +/-28.6 deg, every reset pre-grasp
##


@configclass
class CommandsCfgBoxYaw(CommandsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion = self.motion.replace(
            object_pose_range={
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (0.0, 0.0),
                "yaw": (-0.5, 0.5),
            },
            start_frame=0,
        )


@configclass
class G1DummyHandsRealBoxBoxYawEnvCfg(G1BoxTrackingEnvCfg):
    """Box turned up to +/-28.6 deg, recorded path unchanged; every reset pre-grasp."""

    commands: CommandsCfgBoxYaw = CommandsCfgBoxYaw()


class G1DummyHandsRealBoxBoxYawPlayEnvCfg(G1DummyHandsRealBoxBoxYawEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9


##
# BoxYaw Stage 2: yaw +/-60 deg, box xy +/-5 cm
##
# Both widenings are measured on the stage-1 policy rather than argued:
#
#   yaw +/-60 deg  38.7% with the rotation gate on, 84.3% with it off -- so the
#                  GRIP already handles 60 deg as well as it handles the trained
#                  29, and what fails is returning the box to the recorded
#                  orientation. That is the skill with headroom and the one
#                  wanted: catch the long face wherever it points, then rotate
#                  the whole assembly back.
#   xy +/-0.05 m   82.1% against 83.6% at the trained +/-0.02, i.e. free at the
#                  noise floor without ever having trained there. Including it
#                  adds little confounding risk, and training at +/-5 cm should
#                  push the free-generalisation edge toward the +/-8 cm where it
#                  currently costs 17 points.
#
# +/-90 deg is deliberately not this stage: there the grip itself degrades
# (56.0% even with the rotation gate off), so it is not purely an orientation
# problem. Decide that rung from this stage's numbers.


@configclass
class CommandsCfgBoxYawStage2(CommandsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion = self.motion.replace(
            object_pose_range={
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
                "yaw": (-1.047, 1.047),
            },
            start_frame=0,
        )


@configclass
class G1DummyHandsRealBoxBoxYawStage2EnvCfg(G1BoxTrackingEnvCfg):
    """Box turned up to +/-60 deg and shifted +/-5 cm; every reset pre-grasp."""

    commands: CommandsCfgBoxYawStage2 = CommandsCfgBoxYawStage2()


class G1DummyHandsRealBoxBoxYawStage2PlayEnvCfg(G1DummyHandsRealBoxBoxYawStage2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9


##
# BoxYaw Stage 3: yaw +/-90 deg, box xy +/-0.12 m
##
# Both from measurements on the stage-2 policy, read on the MEAN-error criterion
# -- the max-error one is ceiling-bound at these ranges, since it gates on the
# worst instant and the box starts at the applied offset, so trained policies sat
# at 92-97% of that ceiling across the yaw sweep.
#
#   yaw +/-90 deg  89.65% mean, up from 62.1% for the stage-1 policy -- and stage
#                  2 never trained past 60 deg, so that is generalisation.
#   xy +/-0.12 m   87.3% mean. +/-0.08 was already ~97% and would teach little;
#                  +/-0.16 / +/-0.20 measure 72.8% / 57.9% with box position error
#                  at 23 / 33 cm, a second large stretch competing with yaw.
#
# No 180-degree symmetry here on purpose: min(|t|, |180-t|) equals |t| for all
# |t| <= 90, so the flip is never the shorter path inside this range. It would
# also require changing the metric to match, breaking comparability with every
# number measured so far. That belongs with a move to +/-180.


@configclass
class CommandsCfgBoxYawStage3(CommandsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion = self.motion.replace(
            object_pose_range={
                "x": (-0.12, 0.12),
                "y": (-0.12, 0.12),
                "z": (0.0, 0.0),
                "yaw": (-1.571, 1.571),
            },
            start_frame=0,
        )


@configclass
class G1DummyHandsRealBoxBoxYawStage3EnvCfg(G1BoxTrackingEnvCfg):
    """Box turned up to +/-90 deg and shifted +/-12 cm; every reset pre-grasp."""

    commands: CommandsCfgBoxYawStage3 = CommandsCfgBoxYawStage3()


class G1DummyHandsRealBoxBoxYawStage3PlayEnvCfg(G1DummyHandsRealBoxBoxYawStage3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9


##
# BoxYaw Stage 4: yaw +/-180 deg with a symmetry-aware reward, box xy +/-0.30 m
##
# The box is 205 x 125 x 280 mm and stays upright, so a half turn about its own
# vertical leaves it physically unchanged -- same faces to the hands, same mass
# distribution. The blind orientation term charges up to 180 deg for a box that
# is already correct, and demands a rotation the robot may not manage without
# regrasping. The symmetric term scores against whichever equivalent reference
# is closer, capping the error at 90 deg.
#
# That is what makes the full placement range affordable: sampling yaw over
# +/-180 with the fold gives a mean reduced error of 45 deg -- measured 44.97 --
# which is exactly the mean of |theta| over +/-90 with the blind term (44.99).
# So stage 4 covers every physical box placement at the same expected difficulty
# stage 3 already handles at 98.9%.
#
# Yaw therefore adds coverage rather than difficulty, which leaves xy as the only
# axis with headroom, so the stretch goes there. Measured on the stage-3 policy:
# +/-0.20 is 95.2%, +/-0.25 is 87.9%, +/-0.30 is 71.2% with box position error at
# 18.8 cm. 71% is a comparable starting point to stage 2's 62.1% at +/-90 yaw,
# which reached 89.7%.
#
# Caveat worth remembering if this one disappoints: at +/-0.30 the box can start
# 42 cm away diagonally, and the hands span 34 cm around a 20.5 cm box -- about
# 7 cm of clearance -- so beyond that the robot must reposition its body, against
# an approach phase that covers only 10 cm in the recording. Since yaw
# contributes no new difficulty here, xy is the suspect for any regression.
#
# RewardsCfg and motion_object_orientation_error_exp are both left
# untouched; this swaps only the one term, on its own cfg.


@configclass
class RewardsCfgBoxYawSymmetric(RewardsCfg):
    """RewardsCfg with a 180-degree-symmetric object orientation term."""

    def __post_init__(self):
        super().__post_init__()
        self.motion_object_ori = self.motion_object_ori.replace(
            func=mdp.motion_object_orientation_error_symmetric_exp
        )


@configclass
class CommandsCfgBoxYawStage4(CommandsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion = self.motion.replace(
            object_pose_range={
                "x": (-0.30, 0.30),
                "y": (-0.30, 0.30),
                "z": (0.0, 0.0),
                "yaw": (-3.1416, 3.1416),
            },
            start_frame=0,
        )


@configclass
class G1DummyHandsRealBoxBoxYawStage4EnvCfg(G1BoxTrackingEnvCfg):
    """Box at any yaw and shifted +/-30 cm; orientation scored modulo a half turn."""

    commands: CommandsCfgBoxYawStage4 = CommandsCfgBoxYawStage4()
    rewards: RewardsCfgBoxYawSymmetric = RewardsCfgBoxYawSymmetric()


class G1DummyHandsRealBoxBoxYawStage4PlayEnvCfg(G1DummyHandsRealBoxBoxYawStage4EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
