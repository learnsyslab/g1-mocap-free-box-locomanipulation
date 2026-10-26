# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""Depth-student distillation env: yaw +/-180 deg, box xy +/-0.20 m.

The privileged teacher (Stage 4, model_24096) reads the box pose straight from
the simulator via ``object_pose_b`` / ``object_pose_error_b``. On hardware those
come from Vicon. The point of this env is to train a student that recovers the
same information from an egocentric depth image instead, so the mocap rig is not
needed at deployment.

Three observation groups, consumed through rsl-rl's ``obs_groups``:

  policy         the UNCHANGED teacher group, corruption OFF. The frozen teacher
                 must label from clean state to be the best available expert,
                 and the group has to stay byte-identical to what the Stage 4
                 checkpoint trained on or its normaliser and first layer will
                 not line up.
  student_state  the same group MINUS the two privileged object terms, still
                 noisy. This is realistic proprioception: what the real robot
                 knows without a camera or a mocap system.
  camera         the egocentric depth image, (N, 1, 48, 64). The student's ONLY
                 source of live information about the box.

Why the two dropped terms are exactly the right ones: an ablation earlier in
this project blanked them on a trained policy and success went from 61.4% to
0.0%. So there is no proprioceptive shortcut hiding in the rest of the group --
whatever the student achieves, it achieves by actually seeing the box.

Ranges, from the axis sweep on the Stage 4 policy (yaw +/-180 folded throughout):

    x-only  +/-0.20  81.6%      y-only  +/-0.20  77.7%      square +/-0.20  74.5%
    x-only  +/-0.30  72.9%      y-only  +/-0.30  66.0%      square +/-0.30  47.5%

+/-0.20 is the range the teacher actually handles. DAgger imitates the teacher
wherever you sample, so sampling the +/-0.30 corners -- where the teacher itself
fails, and where x-only * y-only = 48.1% against a measured 47.46%, i.e. the two
displacements fail almost independently -- would teach the student to copy
failure. Yaw stays at the full +/-180 with the symmetric reward, which the same
sweep shows costs nothing.

Camera pitch is 55 degrees down, NOT the 45 of the mjlab-distill config. Measured: at 45
the near corner (x=-0.20) projects to image row 46.1 of 48 and is clipped by the
bottom edge exactly when the box is closest. At 55 the near corner sits at row
39.0 and the far corner at 28.6, both clear. A box outside the frame is
unrecoverable for the student, so this is a correctness fix, not a preference.
"""

from __future__ import annotations

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import g1_mocap_free_box.tasks.manager_based.tracking.mdp as mdp

from .tracking_env_cfg import (
    CommandsCfg,
    G1BoxTrackingEnvCfg,
    RewardsCfgBoxYawSymmetric,
)

# --- camera geometry ---------------------------------------------------------
CAMERA_WIDTH = 64
CAMERA_HEIGHT = 48
CAMERA_FOVY_DEG = 80.0
CAMERA_PITCH_DOWN_DEG = 55.0
CAMERA_POS = (0.08, 0.0, 0.42)  # torso frame; head height on the G1
DEPTH_CUTOFF = 2.5


def _pitch_down_quat_xyzw(deg: float) -> tuple[float, float, float, float]:
    """Rotation about +Y that tilts the forward axis down by ``deg``.

    Specified in IsaacLab's "world" camera convention (forward +X, up +Z), which
    IS the torso frame, so this is a plain pitch. Deliberately not a port of the
    mjlab-distill quaternion, which encodes MuJoCo's convention (view along local -Z, +X
    right, +Y up) and would silently aim the camera somewhere else.
    """
    h = math.radians(deg) / 2.0
    return (0.0, math.sin(h), 0.0, math.cos(h))


def depth_camera_cfg() -> CameraCfg:
    focal = 24.0
    vertical_aperture = 2.0 * focal * math.tan(math.radians(CAMERA_FOVY_DEG) / 2.0)
    horizontal_aperture = vertical_aperture * CAMERA_WIDTH / CAMERA_HEIGHT
    spawn_kwargs = dict(
        focal_length=focal,
        horizontal_aperture=horizontal_aperture,
        clipping_range=(0.01, 20.0),
    )
    if "vertical_aperture" in PinholeCameraCfg.__annotations__:
        spawn_kwargs["vertical_aperture"] = vertical_aperture
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/depth_cam",
        update_period=0.0,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        data_types=["distance_to_image_plane"],
        spawn=PinholeCameraCfg(**spawn_kwargs),
        offset=CameraCfg.OffsetCfg(
            pos=CAMERA_POS,
            rot=_pitch_down_quat_xyzw(CAMERA_PITCH_DOWN_DEG),
            convention="world",
        ),
    )


@configclass
class CommandsCfgDistill(CommandsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion = self.motion.replace(
            object_pose_range={
                "x": (-0.20, 0.20),
                "y": (-0.20, 0.20),
                "z": (0.0, 0.0),
                "yaw": (-3.1416, 3.1416),
            },
            start_frame=0,
        )


@configclass
class DistillObservationsCfg:
    """Teacher group, student proprio group, and the depth image."""

    @configclass
    class TeacherCfg(ObsGroup):
        """Byte-identical to the Stage 4 policy group, with corruption OFF.

        Term order matters as much as term content: the teacher checkpoint's
        first layer is indexed by position, so reordering silently feeds it
        scrambled inputs.
        """

        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        object_pose_b = ObsTerm(func=mdp.object_pose_b, params={"command_name": "motion"})
        object_pose_error_b = ObsTerm(func=mdp.object_pose_error_b, params={"command_name": "motion"})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False  # clean labels
            self.concatenate_terms = True

    @configclass
    class StudentStateCfg(ObsGroup):
        """The teacher group minus the two privileged object terms, still noisy."""

        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CameraCfgGroup(ObsGroup):
        """Egocentric depth, (N, 1, H, W). Never concatenated -- the CNN needs the image."""

        depth = ObsTerm(
            func=mdp.camera_depth,
            params={
                "sensor_name": "depth_cam",
                "cutoff_distance": DEPTH_CUTOFF,
                "noise": True,
                "ramp_steps": 120_000,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False  # corruption lives inside camera_depth
            # True, not False: the manager concatenates with torch.cat and does
            # NOT flatten, so a single 4-D term passes through with its (N,C,H,W)
            # shape intact. False would hand rsl-rl a dict, which CNNModel
            # cannot index as an image group.
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

    @configclass
    class AuxTargetsCfg(ObsGroup):
        """Clean privileged object terms: regression targets for an optional aux head.

        Never a model input. Costs a few floats per env, so it is always present
        rather than gated behind the aux flag.
        """

        object_pose_b = ObsTerm(func=mdp.object_pose_b, params={"command_name": "motion"})
        object_pose_error_b = ObsTerm(func=mdp.object_pose_error_b, params={"command_name": "motion"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: TeacherCfg = TeacherCfg()
    student_state: StudentStateCfg = StudentStateCfg()
    camera: CameraCfgGroup = CameraCfgGroup()
    critic: PrivilegedCfg = PrivilegedCfg()
    aux_targets: AuxTargetsCfg = AuxTargetsCfg()


@configclass
class G1DummyHandsRealBoxDistillDepthEnvCfg(G1BoxTrackingEnvCfg):
    """Depth-student distillation env."""

    commands: CommandsCfgDistill = CommandsCfgDistill()
    rewards: RewardsCfgBoxYawSymmetric = RewardsCfgBoxYawSymmetric()
    observations: DistillObservationsCfg = DistillObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.depth_cam = depth_camera_cfg()


##
# RealCam: the camera the robot actually has
##
# Everything above uses an inherited ZED X approximation. These are the measured
# RealSense D435 depth optics, for depth produced by FastFoundationStereo from
# the IR stereo pair:
#
#   depth FOV     87 deg H x 58 deg V  (848x480, the D435's optimal mode)
#   baseline      50 mm
#   min depth     ~0.168 m at 848x480
#   frame         camera_infra1_optical_frame (LEFT IR, rectified)
#
# The vertical FOV is the consequential difference: 58 vs the 80 we trained on.
#
# DEPLOYMENT CONTRACT, which must match this env exactly:
#   1. take infra1 (LEFT) rectified depth -- the frame the node publishes in
#   2. center-crop 848x480 -> 640x480   (full 58 deg vertical, 73 deg horizontal)
#   3. NEAREST-neighbour downsample -> 64x48
#   4. NaN -> FAR (the cutoff), never near
#   5. clip to [min_depth, 2.5] and divide by 2.5
#
# Step 3 is nearest and not area-average on purpose: IsaacLab renders depth
# point-sampled with no anti-aliasing, so averaging would blur box edges and
# invent intermediate depths the policy never saw in training.
#
# Step 4 matters more than it looks. The node writes NaN for invalid disparity,
# including a left-edge band whose width equals the local disparity -- so it
# GROWS as the box gets closer. Filling those with zero would read as an
# obstacle at the robot's face, worst exactly when it is reaching for the box.
#
# Pitch is left MISSING until measured: the 55 deg used above was derived to keep
# the box inside an 80 deg frame and does not transfer to 58 deg. Run the pitch
# sweep against this FOV before training anything on it.

# --- CONFIRMED against the datasheet and the robot -------------------------
# d435_link is precisely the LEFT IR IMAGER, which is the frame
# FastFoundationStereo's depth is expressed in. Not an approximation: the asset
# is the hardware definition, so no extrinsic calibration is required.
#
# The quaternion is d435_link's own, in (x, y, z, w). It encodes a 47.6 deg
# rotation about +Y -- a pitch DOWN, because d435_link uses the BODY convention
# (+X forward), confirmed by rendering its axes. IsaacLab's "world" convention
# is also forward-+X / up-+Z, so this drops in unconverted. Reading it as an
# OPTICAL frame (+Z forward) would tilt the view up at the sky instead: same
# numbers, opposite camera.
D435_LINK_POS = (0.0576, 0.0175, 0.4299)
D435_LINK_ROT_XYZW = (0.0, 0.4035, 0.0, 0.9150)
D435_LINK_PITCH_DEG = 47.6  # equivalent to the quaternion above

# The physical mount MUST match this pitch: the policy has no other notion of
# where the camera looks. Check it on the robot before deploying, e.g. by fitting
# the floor plane in the depth image with the robot standing still. A mount that
# was ~14 deg steeper than this made the robot overshoot the box by ~23 cm and
# never grasp it.

REALCAM_FOVY_DEG = 58.0
REALCAM_WIDTH = 64
REALCAM_HEIGHT = 48
REALCAM_MIN_DEPTH = 0.168  # D435 minZ at 848x480


def realcam_depth_camera_cfg(
    pitch_down_deg: float = D435_LINK_PITCH_DEG, pos=D435_LINK_POS, rot=None
) -> CameraCfg:
    """Depth camera at the CONFIRMED d435_link pose with the real D435 FOV.

    Defaults are the hardware values, not tuned ones. The earlier 55 deg pitch
    was derived to keep the box inside an 80 deg frame and is now obsolete: the
    physical mount is 47.6 deg and is not a free parameter.

    Note the head occludes this viewpoint in sim -- the mesh models the visor as
    solid while the real sensor looks out through it. Hide head_link when using
    this camera, or the depth image is a wall of skull at ~1 cm.

    ``rot`` (x, y, z, w), when given, replaces the pure pitch -- used for the
    full calibrated rotation, which also carries a small yaw and roll.
    """
    focal = 24.0
    vertical_aperture = 2.0 * focal * math.tan(math.radians(REALCAM_FOVY_DEG) / 2.0)
    horizontal_aperture = vertical_aperture * REALCAM_WIDTH / REALCAM_HEIGHT
    spawn_kwargs = dict(
        focal_length=focal,
        horizontal_aperture=horizontal_aperture,
        clipping_range=(0.01, 20.0),
    )
    if "vertical_aperture" in PinholeCameraCfg.__annotations__:
        spawn_kwargs["vertical_aperture"] = vertical_aperture
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/depth_cam",
        update_period=0.0,
        width=REALCAM_WIDTH,
        height=REALCAM_HEIGHT,
        # Depth ONLY. The "rgb" that was here is dead weight in training -- the
        # policy reads distance_to_image_plane alone -- and across 4096 envs it
        # cost enough render memory to leave under 4 GB free, below what an
        # evaluation needs just to start. It exists solely for the video
        # workaround (a depth-only camera blacks out rgb on EVERY camera in the
        # scene), and scripts/eval/render_student_rollout.py adds it back for itself
        # when rendering. Observations are identical either way.
        data_types=["distance_to_image_plane"],
        spawn=PinholeCameraCfg(**spawn_kwargs),
        offset=CameraCfg.OffsetCfg(
            pos=pos,
            rot=rot if rot is not None else _pitch_down_quat_xyzw(pitch_down_deg),
            convention="world",
        ),
    )


@configclass
class G1DummyHandsRealBoxDistillRealCamEnvCfg(G1DummyHandsRealBoxDistillDepthEnvCfg):
    """Stage 2 with the CONFIRMED real camera: d435_link pose, 58 deg FOV.

    Differs from the 80-deg env in three ways that all matter:

    * the camera is at the real left-IR-imager pose (d435_link), so sim and the
      deployed FastFoundationStereo depth share an origin by construction;
    * 58 deg vertical FOV instead of 80 -- NARROWER, so the box actually gets
      ~1.4x MORE pixels (1.21 deg/row vs 1.67), which may help the ~15 cm
      position floor rather than hurt;
    * head_link is hidden, without which the depth image is a wall of skull.

    A policy trained on the 80 deg env does NOT transfer here -- measured, it
    drops the box. This needs training from the distillation stage.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.depth_cam = realcam_depth_camera_cfg()
        # Startup, not interval: it only needs doing once, before any render.
        self.events.hide_head = EventTerm(
            func=mdp.hide_prim_visual,
            mode="startup",
            params={"prim_suffix": "Robot/torso_link/head_link"},
        )


@configclass
class G1DummyHandsRealBoxDistillRealCamPlayEnvCfg(G1DummyHandsRealBoxDistillRealCamEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1e9
        self.observations.camera.depth.params["noise"] = False
        self.observations.student_state.enable_corruption = False


@configclass
class G1DummyHandsRealBoxRealCamFinetuneEnvCfg(G1DummyHandsRealBoxDistillRealCamEnvCfg):
    """Finetune env for the real camera: identical to the distillation env.

    Deliberately unchanged. Finetuning differs by ALGORITHM, not environment --
    PPO on the same task, same observation groups, same box range. Sharing the
    cfg guarantees the actor sees exactly the observation it was distilled on,
    which is what lets the distilled checkpoint load into the actor without a
    remap.

    The rewards were always present (inherited from the tracking env);
    distillation simply never used them, because DAgger optimises agreement with
    the teacher rather than reward. Finetuning is what finally reads them.
    """


@configclass
class G1DummyHandsRealBoxRealCamFinetunePlayEnvCfg(G1DummyHandsRealBoxDistillRealCamPlayEnvCfg):
    pass


# The six faces the box can rest on, as (x, y, z, w) -- XYZW, which is what this
# IsaacLab build stores (see the note in mdp/commands.py: quat_from_euler_xyz's
# identity is [0,0,0,1]). Writing these as wxyz put four of the six on an EDGE
# rather than a face, giving support heights above the largest half-extent and
# sinking the box 8.7 cm into the floor.
#
# Three distinct geometries: opposite faces are identical for a plain cuboid, and
# are listed separately only so a later asymmetry (texture, CoM offset,
# object_inertial) makes them differ without this list changing.
#
#   identity / roll 180        recorded face, centre 0.140 m, 0.28 m tall
#   roll  +/-90                centre 0.0625 m, 0.125 m tall
#   pitch +/-90                centre 0.1025 m, 0.205 m tall
#
# The reference height moves with the choice (see object_resting_states), so the
# reward asks for the box to be carried where it is lying rather than dragged to
# the recorded height.
_SQ2 = 0.70710678
BOX_RESTING_STATES = (
    (0.0, 0.0, 0.0, 1.0),          # identity
    (1.0, 0.0, 0.0, 0.0),          # roll 180
    (_SQ2, 0.0, 0.0, _SQ2),        # roll  +90
    (-_SQ2, 0.0, 0.0, _SQ2),       # roll  -90
    (0.0, _SQ2, 0.0, _SQ2),        # pitch +90
    (0.0, -_SQ2, 0.0, _SQ2),       # pitch -90
)


@configclass
class G1DummyHandsRealBoxRealCamFinetuneStage2EnvCfg(
    G1DummyHandsRealBoxRealCamFinetuneEnvCfg
):
    """Stage 3 continued against a harder box distribution.

    Two changes, both on the object:

    * xy widened to x[-0.40, 0.60], y[-0.50, 0.20] from +/-0.20, yaw unchanged
      at +/-180. Note these are WORLD axes -- commands.py adds the sample to the
      object's world position without rotating it -- and the reference heading is
      -96 deg at spawn, turning a further 52 deg by the grasp. So world +x is
      roughly the robot's left and +y its backward, and the range is not the
      forward-biased box it reads as.

      Measured with scripts/eval/show_spawn_region_bev.py: 79-93% of samples land
      beyond the reference hands' reach and 23% behind the robot at the grasp.
      Reach is survivable -- the residual has 0.29-0.46 rad of authority in the
      legs against 0.083 in the shoulders, and on hardware the robot steps
      toward a box placed out of reach. VISIBILITY is the real cost: only 1 of 9
      sampled positions was visible to the depth camera at both the spawn and the
      grasp pose, and the student's observation is single-frame with no memory,
      so a box that leaves the view before the grasp leaves it with nothing.

    * The box may start on any of its six faces rather than only the recorded
      one, which changes its height by up to 7.75 cm and swaps its footprint.

    Also changed, in the same run: object_inertia (+/-15% diagonal),
    object_size (+/-5%) and pixel_noise_std (0.02 -> 0.004). These were
    originally held back so a regression could be attributed to the box
    distribution alone; they are in now, which means a regression against Stage 1
    has five candidate causes rather than two.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion = self.commands.motion.replace(
            object_pose_range={
                "x": (-0.40, 0.60),
                "y": (-0.50, 0.20),
                "z": (0.0, 0.0),
                "yaw": (-3.1416, 3.1416),
            },
            object_resting_states=BOX_RESTING_STATES,
        )

        # --- the three randomisations of the mjlab-distill recipe ------------
        # CoM offset +/-1.5 cm, matching mjlab-distill's pseudo_inertia t_range. This is
        # the one that most affects whether a grasp holds, and which way a
        # tipped box topples.
        # CoM randomisation is DISABLED: randomize_rigid_body_com raises inside
        # IsaacLab for a RigidObject on the physx backend in this build --
        #
        #   isaaclab_physx/assets/rigid_object/rigid_object.py:774 set_coms_index
        #   wp.copy(self._cpu_body_coms, self.data._body_com_pose_b.data)
        #   RuntimeError: Incompatible array data types:
        #       destination dtype=float32, source dtype=transformf
        #
        # The internal CPU buffer is float32 while the data is warp's transformf,
        # so no choice of com_range avoids it. This is the half of mjlab-distill's
        # pseudo_inertia that most affects whether a grasp holds, so it is worth
        # returning to -- either by patching that copy, or by writing CoMs
        # through root_physx_view directly.
        #
        # self.events.object_com = EventTerm(
        #     func=mdp.randomize_rigid_body_com,
        #     mode="reset",
        #     params={
        #         "asset_cfg": SceneEntityCfg("object"),
        #         "com_range": {"x": (-0.015, 0.015),
        #                       "y": (-0.015, 0.015),
        #                       "z": (-0.015, 0.015)},
        #     },
        # )
        # Inertia +/-15% on the diagonal, matching mjlab-distill's alpha_range. Diagonal
        # only: off-diagonal terms are numerically harsher and mjlab-distill's
        # pseudo_inertia keeps the tensor physically consistent, which a blunt
        # per-element scale does not.
        self.events.object_inertia = EventTerm(
            func=mdp.randomize_rigid_body_inertia,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "inertia_distribution_params": (0.85, 1.15),
                "operation": "scale",
                "distribution": "uniform",
                "diagonal_only": True,
            },
        )
        # Box dimensions +/-5%, matching mjlab-distill's geom_size. Mode "usd", not
        # "reset": this rewrites xformOp:scale, which the physics engine parses
        # once at startup, so it is one fixed size per env for the whole run
        # rather than per-episode. mjlab-distill's is startup-mode for the same reason.
        #
        # NOTE this interacts with object_resting_states. The resting heights are
        # computed from the motion file's half-extents, which are unscaled, so a
        # box scaled by 1.05 spawns up to 7 mm off its true resting height
        # (0.14 * 0.05) and 3 mm on the flattest face. Physics settles that on
        # the first step, and it is the same order as the 4.5 mm offset already
        # present in the recorded pose -- but it is an approximation, not exact.
        # object_size is DISABLED, and was never actually active.
        #
        # It was written with mode="usd", which randomize_rigid_body_scale's
        # docstring names -- but this IsaacLab build renamed that mode.
        # manager_based_env applies prestartup/startup/interval/reset, and
        # EventManager.apply SILENTLY RETURNS on a mode it does not know. So the
        # term never ran, and the Stage 2 policy has no box-size robustness
        # despite the run being described as having it.
        #
        # Correcting the mode to "prestartup" makes it run and then raise:
        #
        #   RuntimeError: Scene replication is enabled, which may affect
        #   USD-level randomization ... set 'replicate_physics' to False
        #
        # which is a real cost at 4096 envs. An earlier probe of mine reported
        # that physics does honour the per-env scale under replication, but it
        # called the function through a shim and bypassed exactly this guard, so
        # it proved nothing about the trainer's path.
        #
        # Left off rather than silently broken. To enable: mode="prestartup" AND
        # scene.replicate_physics = False, then re-measure the startup cost.
        #
        # self.events.object_size = EventTerm(
        #     func=mdp.randomize_rigid_body_scale,
        #     mode="prestartup",
        #     params={"asset_cfg": SceneEntityCfg("object"),
        #             "scale_range": (0.95, 1.05)},
        # )

        # Depth noise down 5x. Measured on 1679 frames of FoundationStereo depth
        # from the robot: per-pixel temporal noise is 0.27 cm median, 0.94 cm
        # p95, against the 5 cm that pixel_noise_std=0.02 implies at a 2.5 m
        # cutoff -- 18x more corruption than the sensor produces. Learning
        # invariance to 5 cm of per-pixel depth error means declining to trust
        # exactly the structure needed to localise the box, which is the best
        # candidate yet for the ~15 cm position floor.
        #
        # Not taken all the way to 0.27 cm: one benign recording is not the whole
        # distribution, and some margin is wanted for glare, low texture and
        # glass. 0.004 is 1 cm at the cutoff.
        self.observations.camera.depth.params["pixel_noise_std"] = 0.004


@configclass
class G1DummyHandsRealBoxRealCamFinetuneStage2PlayEnvCfg(
    G1DummyHandsRealBoxRealCamFinetuneStage2EnvCfg
):
    """Play variant. Inherits the TRAIN cfg, not the Play one, because it needs
    the Stage 2 box distribution -- so everything the other Play cfgs set has to
    be repeated here by hand, and enable_corruption was initially missed.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.camera.depth.params["noise"] = False
        # Was left on, unlike every other Play cfg: evaluation would have run
        # with joint_vel corrupted by +/-0.5 rad/s and the anchor orientation by
        # +/-0.05, i.e. reported a tracking error the deployed policy never has.
        self.observations.student_state.enable_corruption = False
