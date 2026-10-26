# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Modifications copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich,
# licensed under Apache-2.0. See THIRD_PARTY_NOTICES.md.

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlPpoAlgorithmCfg as _RslRlPpoAlgorithmCfg,  # noqa: F401
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000  # the paper's budget
    save_interval = 100
    experiment_name = "G1_Box_Tracking"  # logs/rsl_rl/<experiment_name>
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# One experiment_name per task family, so each gets its own log directory; a task's
# -Play variant reuses its training counterpart's name on purpose, so play.py's "find the
# latest checkpoint" resolves against the run that actually produced it.
@configclass
class DummyHandsRealBoxNoObjTerminatePPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "G1_DummyHands_RealBox_Tracking_No_Obj_Terminate"


# Its own experiment_name so BoxYaw runs do not land in the base task's log
# directory, and so --resume resolves within this family.
@configclass
class DummyHandsRealBoxBoxYawPPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "G1_DummyHands_RealBox_BoxYaw"


##
# Depth-student distillation (DAgger from the Stage 4 privileged teacher)
##
# loss_type is "nll", NOT the library default "mse". Atarilab's mjlab-distill ablation
# (July 2026) found this single choice dominates every other knob: against mse, nll
# halved joint error (0.74 vs 1.36), cut body-rot error 56%, and all but
# eliminated ee-pos terminations (0.037 vs 2.65). mse was the WORST of the four,
# matching Rusu et al. 2016. Stock rsl-rl implements only mse/huber, so the
# algorithm below is our own subclass -- using the library class unchanged would
# silently select the worst variant.
#
# The teacher is a plain MLP over the "policy" group (the Stage 4 actor group,
# corruption off). The student is a spatial-softmax CNN over "student_state" +
# "camera". Both need a plain state-independent Gaussian: the stored teacher
# label is its deterministic forward output, which is the distribution's mean
# only under that assumption.


@configclass
class DummyHandsRealBoxDistillDepthRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 6000
    save_interval = 100
    experiment_name = "G1_DummyHands_RealBox_Distill_Depth"
    empirical_normalization = False
    clip_actions = None
    obs_groups = {
        "teacher": ["policy"],
        "student": ["student_state", "camera"],
    }
    student = RslRlCNNModelCfg(
        class_name=(
            "g1_mocap_free_box.tasks.manager_based.tracking.distill_models:SpatialSoftmaxCNNModel"
        ),
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=0.1,
        noise_std_type="scalar",
        state_dependent_std=False,
        cnn_cfg={
            # 48x64 in. Three strided convs shrink it before the spatial softmax
            # reduces each channel to an (x, y) keypoint.
            "output_channels": [16, 32, 32],
            "kernel_size": [4, 4, 3],
            "stride": [2, 2, 1],
            "padding": "zeros",
            "activation": "elu",
            "spatial_softmax_temperature": 1.0,
        },
    )
    teacher = RslRlMLPModelCfg(
        class_name="rsl_rl.models:MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        stochastic=True,
        init_noise_std=1.0,
        noise_std_type="scalar",
        state_dependent_std=False,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name=(
            "g1_mocap_free_box.tasks.manager_based.tracking.distill_algorithm:"
            "DistributionalDistillation"
        ),
        num_learning_epochs=1,
        learning_rate=1.0e-3,
        gradient_length=15,
        max_grad_norm=1.0,
        optimizer="adam",
        loss_type="nll",
    )


##
# Stage 3: kickstarted PPO fine-tune of the depth student
##
# Distillation caps the student AT the teacher, and our stage-2 run hit that
# ceiling: behaviour loss flat at full corruption, student 1.56% vs teacher
# 90.53%. This stage changes the objective to PPO on true task reward plus an
# annealed KL(teacher || student), so once the teacher term reaches zero the
# policy is free to exceed the teacher rather than chase it.
#
# Three models, three observation sets:
#   actor    the distilled depth student  (student_state + camera)
#   critic   PRIVILEGED, 310-D            (critic)  -- asymmetric on purpose
#   teacher  the frozen Stage 4 policy    (policy)
#
# The critic must keep the privileged group: that is what makes its value
# estimates usable despite the actor's partial observation. Remapping it to the
# actor's group would throw away the whole point.
#
# entropy_coef is lower than the base PPO cfg (0.005 -> 0.002): the actor starts
# from a distilled policy with an already-tight action distribution, and the
# usual exploration bonus would push it back apart before reward can guide it.


@configclass
class DummyHandsRealBoxDistillFinetuneRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "G1_DummyHands_RealBox_Distill_Finetune"
    empirical_normalization = False
    clip_actions = None
    obs_groups = {
        "actor": ["student_state", "camera"],
        "critic": ["critic"],
        "teacher": ["policy"],
    }
    actor = RslRlCNNModelCfg(
        class_name=(
            "g1_mocap_free_box.tasks.manager_based.tracking.distill_models:SpatialSoftmaxCNNModel"
        ),
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        stochastic=True,
        init_noise_std=0.1,
        noise_std_type="scalar",
        state_dependent_std=False,
        cnn_cfg={
            "output_channels": [16, 32, 32],
            "kernel_size": [4, 4, 3],
            "stride": [2, 2, 1],
            "padding": "zeros",
            "activation": "elu",
            "spatial_softmax_temperature": 1.0,
        },
    )
    critic = RslRlMLPModelCfg(
        class_name="rsl_rl.models:MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        stochastic=False,
        init_noise_std=1.0,
        noise_std_type="scalar",
        state_dependent_std=False,
    )
    teacher = RslRlMLPModelCfg(
        class_name="rsl_rl.models:MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        stochastic=True,
        init_noise_std=1.0,
        noise_std_type="scalar",
        state_dependent_std=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="g1_mocap_free_box.tasks.manager_based.tracking.distill_ppo:DistillPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        # Kickstarting knobs live on the algorithm dict; RslRlPpoAlgorithmCfg has
        # no fields for them, so they are attached here and flow through
        # construct_algorithm's **cfg["algorithm"] splat.
        for key, value in (
            ("kickstart_coef_initial", 0.5),
            ("kickstart_coef_final", 0.0),
            ("kickstart_anneal_iterations", 1000),
            ("kickstart_loss_type", "kl"),
        ):
            setattr(self.algorithm, key, value)


##
# RealCam runs: identical recipes, separate log directories
##
# Same algorithms and hyperparameters as the 80-deg runs -- only the ENVIRONMENT
# camera differs (d435_link pose, 58 deg FOV, head hidden). They get their own
# experiment_name so the two cannot be confused: checkpoints that differ solely
# by camera are indistinguishable by filename and completely non-interchangeable.


@configclass
class DummyHandsRealBoxRealCamDistillRunnerCfg(DummyHandsRealBoxDistillDepthRunnerCfg):
    experiment_name = "G1_DummyHands_RealBox_Distill_RealCam"


@configclass
class DummyHandsRealBoxRealCamFinetuneRunnerCfg(DummyHandsRealBoxDistillFinetuneRunnerCfg):
    experiment_name = "G1_DummyHands_RealBox_RealCam_Finetune"


@configclass
class DummyHandsRealBoxRealCamFinetuneStage2RunnerCfg(
    DummyHandsRealBoxRealCamFinetuneRunnerCfg
):
    """Stage 2 of the finetune: harder box distribution, and NO kickstarting.

    Two departures from the Stage 1 recipe.

    ``kickstart_coef_initial = 0.0``. Note this is belt-and-braces for the
    intended launch, not a fix: resuming Stage 1's model_2999.pt restores
    ``num_updates = 3000``, and the anneal is 1000 iterations, so the coefficient
    would evaluate to exactly 0.0 anyway. It is set explicitly so the intent does
    not depend on num_updates surviving a checkpoint, and so a FRESH Stage 2 run
    (num_updates = 0, coefficient 0.5) does not silently start kickstarting
    against a teacher that cannot do this task. That teacher is
    CommandsCfgBoxYawStage4:
    object xy +/-0.30 and the single recorded resting face. Stage 2 samples
    x[-0.40, 0.60], y[-0.50, 0.20] and any of six faces, so only 43% of xy
    samples and 2 of 6 faces fall inside what the teacher was trained on --
    about 14% of initial states, i.e. ~86% are outside it. The teacher still
    emits an action there, because it reads privileged object_pose_b, but it has
    never grasped a box lying on its side at 0.125 m instead of 0.28 m. A KL term
    pulling the student toward that is not a prior, it is noise with a
    confident-looking mean. PPO's guard against wrecking the initialisation is
    the initialisation itself: this run resumes from a student that already
    works on hardware.

    Retraining the teacher on the wider distribution first would let kickstarting
    back in, and is the better answer if PPO alone plateaus -- the teacher is
    state-based with no camera, so it is far cheaper than a distillation run.
    DistillPPO skips the teacher forward entirely when the coefficient is zero,
    so nothing is spent on it here.

    ``max_iterations = 6000``, double Stage 1. Two reasons:

    * The depth corruption ramp is 120_000 steps of env.common_step_counter,
      which advances by num_steps_per_env = 24 per iteration -- so full
      corruption arrives at iteration 5000, NOT at 3000. Stage 1 ran 3000 and
      therefore never trained above 60% corruption intensity. The counter is per
      env instance and resets on resume, so this run restarts the ramp from
      zero; 6000 completes it and leaves 1000 iterations at full strength.
      Fewer are needed than that suggests because pixel_noise_std is now 5x
      lower, so "full" here is much milder than Stage 1's full would have been.
    * The box distribution is far larger: 0.70 m^2 of xy against 0.16 m^2, a
      4.4x area, times three distinct resting geometries.

    At the 5.2 s/iteration the Stage 1 run held at 4096 envs, 6000 iterations is
    roughly 8.7 hours. save_interval is 100, so there are 60 checkpoints to stop
    at if it converges sooner.
    """

    experiment_name = "G1_DummyHands_RealBox_RealCam_Finetune_Stage2"
    max_iterations = 6000

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.kickstart_coef_initial = 0.0
        self.algorithm.kickstart_coef_final = 0.0
