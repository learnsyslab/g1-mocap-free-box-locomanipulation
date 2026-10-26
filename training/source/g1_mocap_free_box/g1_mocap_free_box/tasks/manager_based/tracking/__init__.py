# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""Box pick-and-place tracking for the Unitree G1 (29 DoF), privileged teacher
and depth-camera student.

The MDP terms are a port of unitree_rl_lab's BeyondMimic-style ``tasks/mimic``
(BSD-3), extended with the box: its tracking and contact rewards, box-pose
randomisation, and the head depth camera for the student.
"""

import gymnasium as gym

from . import agents


# Real-box pipeline (G1 29 DoF, dummy box hands, 205 x 125 x 280 mm box):
#
#   teacher  G1-DummyHands-RealBox-Tracking-No-Obj-Terminate     (stage 0, from scratch)
#         -> G1-DummyHands-RealBox-BoxYaw -> -Stage2 -> -Stage3 -> -Stage4
#            (privileged state, PPO; curriculum over box pose / yaw)
#   student  G1-DummyHands-RealBox-Distill-RealCam           (DAgger from the Stage4 teacher)
#         -> G1-DummyHands-RealBox-RealCam-Finetune           (PPO + teacher kickstart)
#         -> G1-DummyHands-RealBox-RealCam-Finetune-Stage2    (wider box range, six faces)
#
# Camera: the G1's d435_link at 47.6 deg down (set the head to this pitch on the robot).

##
# Teacher curriculum (privileged, Vicon box pose): Tracking-No-Obj-Terminate ->
# BoxYaw -> Stage2 -> Stage3 -> Stage4, each resuming from the previous one.
# Stage 4 is the expert the depth student is distilled from.
##

gym.register(
    id="G1-DummyHands-RealBox-Tracking-No-Obj-Terminate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:G1BoxTrackingEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:G1BoxTrackingPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DummyHandsRealBoxNoObjTerminatePPORunnerCfg",
    },
)

gym.register(
    id="G1-DummyHands-RealBox-Tracking-No-Obj-Terminate-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:G1BoxTrackingPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DummyHandsRealBoxNoObjTerminatePPORunnerCfg",
    },
)

gym.register(
    id="G1-DummyHands-RealBox-BoxYaw",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:G1DummyHandsRealBoxBoxYawEnvCfg",
        "play_env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:G1DummyHandsRealBoxBoxYawPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DummyHandsRealBoxBoxYawPPORunnerCfg"
        ),
    },
)

gym.register(
    id="G1-DummyHands-RealBox-BoxYaw-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:G1DummyHandsRealBoxBoxYawPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DummyHandsRealBoxBoxYawPPORunnerCfg"
        ),
    },
)

gym.register(
    id="G1-DummyHands-RealBox-BoxYaw-Stage2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:G1DummyHandsRealBoxBoxYawStage2EnvCfg",
        "play_env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:G1DummyHandsRealBoxBoxYawStage2PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DummyHandsRealBoxBoxYawPPORunnerCfg"
        ),
    },
)

gym.register(
    id="G1-DummyHands-RealBox-BoxYaw-Stage2-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:G1DummyHandsRealBoxBoxYawStage2PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DummyHandsRealBoxBoxYawPPORunnerCfg"
        ),
    },
)


gym.register(
    id="G1-DummyHands-RealBox-BoxYaw-Stage3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:G1DummyHandsRealBoxBoxYawStage3EnvCfg",
        "play_env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:G1DummyHandsRealBoxBoxYawStage3PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DummyHandsRealBoxBoxYawPPORunnerCfg"
        ),
    },
)

gym.register(
    id="G1-DummyHands-RealBox-BoxYaw-Stage3-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:G1DummyHandsRealBoxBoxYawStage3PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DummyHandsRealBoxBoxYawPPORunnerCfg"
        ),
    },
)

gym.register(
    id="G1-DummyHands-RealBox-BoxYaw-Stage4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:G1DummyHandsRealBoxBoxYawStage4EnvCfg",
        "play_env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:G1DummyHandsRealBoxBoxYawStage4PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DummyHandsRealBoxBoxYawPPORunnerCfg"
        ),
    },
)

gym.register(
    id="G1-DummyHands-RealBox-BoxYaw-Stage4-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:G1DummyHandsRealBoxBoxYawStage4PlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DummyHandsRealBoxBoxYawPPORunnerCfg"
        ),
    },
)


##
# Depth student (onboard depth + IMU + joints, no mocap), camera = the G1's CAD d435_link
# mount (torso_link + (0.0576, 0.0175, 0.4299) m, 47.6 deg down): DAgger distillation from
# the Stage 4 teacher -> PPO finetune with a teacher kickstart -> Stage 2 (harder box
# distribution: wider xy, any of six resting faces, kickstart off).
##

for _rc_id, _rc_env, _rc_play, _rc_agent in (
    (
        "G1-DummyHands-RealBox-Distill-RealCam",
        "G1DummyHandsRealBoxDistillRealCamEnvCfg",
        "G1DummyHandsRealBoxDistillRealCamPlayEnvCfg",
        "DummyHandsRealBoxRealCamDistillRunnerCfg",
    ),
    (
        "G1-DummyHands-RealBox-RealCam-Finetune",
        "G1DummyHandsRealBoxRealCamFinetuneEnvCfg",
        "G1DummyHandsRealBoxRealCamFinetunePlayEnvCfg",
        "DummyHandsRealBoxRealCamFinetuneRunnerCfg",
    ),
    # Its own runner: kickstarting is off because the teacher never saw this
    # distribution, and it logs to its own experiment_name.
    (
        "G1-DummyHands-RealBox-RealCam-Finetune-Stage2",
        "G1DummyHandsRealBoxRealCamFinetuneStage2EnvCfg",
        "G1DummyHandsRealBoxRealCamFinetuneStage2PlayEnvCfg",
        "DummyHandsRealBoxRealCamFinetuneStage2RunnerCfg",
    ),
):
    gym.register(
        id=_rc_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.distill_env_cfg:{_rc_env}",
            "play_env_cfg_entry_point": f"{__name__}.distill_env_cfg:{_rc_play}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{_rc_agent}",
        },
    )
    gym.register(
        id=f"{_rc_id}-Play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.distill_env_cfg:{_rc_play}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{_rc_agent}",
        },
    )
