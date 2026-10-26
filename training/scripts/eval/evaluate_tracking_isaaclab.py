# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""Table V-style evaluation of an IsaacLab object-tracking policy.

Metric definitions live in scripts/eval/tracking_eval_metrics.py -- see that module
for the DynaRetarget Section IV-E / Table V protocol and the threshold caveat.

There was no evaluation script on the IsaacLab side at all before this, so
there was previously no way to produce a Table V-comparable number there.

.. code-block:: bash

    $ISAAC_PYTHON scripts/eval/evaluate_tracking_isaaclab.py \
        --task G1-DummyHands-RealBox-BoxYaw-Stage4 \
        --checkpoint logs/rsl_rl/G1_DummyHands_RealBox_BoxYaw/<run>/model_<N>.pt \
        --num_envs 1024
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="G1-DummyHands-RealBox-BoxYaw-Stage4")
parser.add_argument("--checkpoint", type=str, required=True, help="Local .pt checkpoint.")
parser.add_argument("--num_envs", type=int, default=1024, help="Episodes; the paper's Table V uses 1024.")
parser.add_argument("--pos_threshold", type=float, default=None, help="Object position gate (m).")
parser.add_argument("--rot_threshold", type=float, default=None, help="Object orientation gate (rad).")
parser.add_argument(
    "--keep_pushes",
    action="store_true",
    help=(
        "Keep interval push_robot/push_object during evaluation. Off by default -- "
        "the paper's protocol is 'diverse initial configurations', not mid-episode "
        "disturbances; pushes are a training-time robustness technique."
    ),
)
parser.add_argument(
    "--no_randomization",
    action="store_true",
    help=(
        "Disable observation noise and all reset-time randomization (the set mjlab's "
        "play=True clears). Note this does NOT correspond to IsaacLab's own play "
        "path: the *PlayEnvCfg classes here only override num_envs and episode_length_s, "
        "and scripts/rsl_rl/play.py disables nothing, so an IsaacLab play session runs "
        "with full randomization -- the same condition as this evaluator's default. "
        "Neither is the paper's protocol, which averages over 'diverse initial "
        "configurations'."
    ),
)
parser.add_argument(
    "--disable",
    type=str,
    default="",
    help=(
        "Comma-separated subset of randomization sources to switch off, for attributing "
        "a success-rate change to a specific one: obs_noise, pose, velocity, object_pose, "
        "object_velocity. --no_randomization is equivalent to naming all five."
    ),
)
parser.add_argument(
    "--start_frame",
    type=int,
    default=None,
    help=(
        "Pin every episode to this motion frame instead of letting the adaptive "
        "sampler choose. Splits 'cannot approach the object' from 'cannot grasp "
        "it': pin just before the grasp phase and the robot is placed correctly "
        "next to the box, so only the grasp is under test."
    ),
)
parser.add_argument(
    "--object_size",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help=(
        "Override the box's full extents (m), e.g. to score an existing policy against a "
        "physical box that differs from the one in the motion file. x is the axis the "
        "forearms squeeze across, z is height. Mass and friction stay as in the motion."
    ),
)
parser.add_argument(
    "--object_yaw_range",
    type=float,
    default=None,
    metavar="RAD",
    help=(
        "Override how far the box may be turned, as a symmetric +/-RAD range, without "
        "needing a task per angle. Scores how far a policy's grasp adaptation extends "
        "past the range it trained on. Only yaw is touched; the box's x/y range is left "
        "alone. This script does not go through hydra, so `env.` overrides cannot reach "
        "it -- hence a flag."
    ),
)
parser.add_argument(
    "--object_xy_range",
    type=float,
    default=None,
    metavar="M",
    help=(
        "Override how far the box may be displaced, as a symmetric +/-M range on x and "
        "y. Only x/y are touched; the yaw range is left alone, so the two axes can be "
        "swept independently rather than confounded."
    ),
)
parser.add_argument(
    "--object_x_range",
    type=float,
    default=None,
    metavar="M",
    help=(
        "Override the box's x range alone, as +/-M, leaving y and yaw untouched. "
        "--object_xy_range moves both axes together, which cannot tell 'far forward' "
        "apart from 'off to the side' -- and the recording's approach phase travels "
        "forward, so there is no reason to expect those to cost the same. Applied "
        "after --object_xy_range, so the two compose: set the square, then override "
        "one axis."
    ),
)
parser.add_argument(
    "--object_y_range",
    type=float,
    default=None,
    metavar="M",
    help="Override the box's y (lateral) range alone, as +/-M. See --object_x_range.",
)
parser.add_argument(
    "--object_obs",
    type=str,
    default="true",
    choices=("true", "ref", "zero", "frozen"),
    help=(
        "What the POLICY is told about the box, to measure how much it actually uses "
        "that channel. 'true' (default) feeds the simulator's box pose, as in training. "
        "'ref' feeds the reference box pose instead, so object_pose_error_b is identically "
        "zero -- this is the hardware fallback of having no box tracking at all and "
        "assuming the box is wherever the motion says it is. 'zero' blanks both terms "
        "(total loss of box sensing). 'frozen' holds the first post-reset measurement for "
        "the whole episode (a mocap dropout at t=0). Only the actor group is patched; the "
        "critic is unused at inference."
    ),
)
parser.add_argument(
    "--object_obs_latency",
    type=int,
    default=0,
    help=(
        "Delay the policy's box observation by this many control steps, to price mocap "
        "plus network latency. At the 50 Hz control rate one step is 20 ms. Applies to "
        "--object_obs true; ignored by the other modes, which carry no live measurement."
    ),
)
parser.add_argument(
    "--object_symmetry",
    type=str,
    default="none",
    choices=("none", "2fold"),
    help=(
        "Whether the orientation metric treats a half turn of the box as equivalent. "
        "'none' (default) scores against the reference as recorded, keeping every number "
        "comparable with earlier runs. '2fold' scores against whichever of the two "
        "z-equivalent references is closer, capping the error at 90 degrees -- the right "
        "metric for a policy trained with the symmetric reward, and the only fair one "
        "once yaw is sampled past +/-90, where the recorded reference is often the "
        "farther of the two. Use it to READ a policy, not to flatter one: a run trained "
        "without the symmetric reward has no reason to exploit the flip."
    ),
)
parser.add_argument(
    "--depth_obs",
    type=str,
    default="true",
    choices=("true", "zero"),
    help=(
        "What the depth student is shown. 'true' (default) is the rendered image. "
        "'zero' blanks it, leaving the student only proprioception and the motion "
        "command. If blanking does not move the metrics, the CNN is decorative and the "
        "student is replaying the reference motion rather than seeing the box -- a "
        "failure that every training curve hides, because most of the teacher's action "
        "is predictable without the image."
    ),
)
parser.add_argument(
    "--depth_noise",
    type=str,
    default="on",
    choices=("on", "off"),
    help=(
        "Whether the depth image carries the simulated sensor corruption it was "
        "trained with. 'off' gives the student a clean render, which separates a "
        "noise-model problem from a resolution problem: if accuracy jumps, the "
        "corruption is what is hurting it; if not, the limit is the image itself."
    ),
)
parser.add_argument("--output_file", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
# A task with a rendering sensor produces no camera observation unless the app
# is launched with cameras enabled, and the env then dies with "A camera was
# spawned without the --enable_cameras flag".
#
# This cannot be decided by inspecting the env cfg: enable_cameras must be set
# before AppLauncher starts, and resolving the cfg requires isaaclab to be
# imported, which happens after. So it is a name match -- but a name match on
# ONE marker is fragile, and it already broke: matching only "Distill" silently
# skipped G1-DummyHands-RealBox-RealCam-Finetune and wasted three evaluations.
# Hence a marker LIST, plus an explicit --enable_cameras that callers can always
# pass to be certain.
_CAMERA_TASK_MARKERS = ("Distill", "Depth", "RealCam", "Camera", "-Cal")
if any(marker in args_cli.task for marker in _CAMERA_TASK_MARKERS):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json  # noqa: E402
import functools
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import g1_mocap_free_box.tasks  # noqa: F401, E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tracking_eval_metrics import (  # noqa: E402
    DEFAULT_POS_THRESHOLD_M,
    DEFAULT_ROT_THRESHOLD_RAD,
    TrackingEvalAccumulator,
    print_report,
    quat_error_magnitude_2fold_z_xyzw,
    quat_error_magnitude_xyzw,
    termination_causes,
)


def _stale(fn, latency: int, hold: bool):
    """Wrap an observation term so the policy sees an out-of-date value.

    ``hold`` freezes the first post-reset sample for the rest of the episode; the
    alternative delays the term by ``latency`` control steps. Kept as a closure
    rather than a callable class because IsaacLab's ObservationManager validates
    a term against ``inspect.signature(func)``, and a closure's signature is the
    ``(env, command_name)`` the manager expects.

    A reset is detected as the command's frame index decreasing. That covers both
    the episode reset and the command's end-of-motion wrap, and refills every
    buffered slot for the affected envs so no measurement leaks across episodes
    -- which would otherwise show up as a spurious start-of-episode error.
    """
    depth = 1 if hold else latency + 1
    state: dict = {"buf": None, "prev": None}

    def wrapped(env, command_name):
        command = env.command_manager.get_term(command_name)
        current = fn(env, command_name)
        steps = command.time_steps

        if state["buf"] is None or state["buf"].shape[1:] != current.shape:
            state["buf"] = current.unsqueeze(0).repeat(depth, 1, 1)
            state["prev"] = steps.clone()
            return state["buf"][0].clone()

        was_reset = steps < state["prev"]
        state["prev"] = steps.clone()
        if was_reset.any():
            state["buf"][:, was_reset] = current[was_reset]

        # Push before reading, so a depth of latency+1 yields exactly
        # value(t - latency). Reading first costs an extra step of delay.
        if not hold:
            state["buf"] = torch.roll(state["buf"], -1, dims=0)
            state["buf"][-1] = current
        return state["buf"][0].clone()

    return wrapped


_obj_rot_error = (
    quat_error_magnitude_2fold_z_xyzw
    if args_cli.object_symmetry == "2fold"
    else quat_error_magnitude_xyzw
)


def _patch_depth_noise(env_cfg) -> bool:
    """Turn the depth corruption off when --depth_noise off."""
    if args_cli.depth_noise != "off":
        return False
    group = getattr(env_cfg.observations, "camera", None)
    if group is None:
        raise SystemExit(f"--depth_noise: task {args_cli.task} has no 'camera' group")
    group.depth.params["noise"] = False
    return True


def _patch_depth_observation(env_cfg) -> bool:
    """Replace the depth image with zeros when --depth_obs zero."""
    if args_cli.depth_obs != "zero":
        return False
    group = getattr(env_cfg.observations, "camera", None)
    if group is None:
        raise SystemExit(f"--depth_obs: task {args_cli.task} has no 'camera' observation group")
    term = group.depth
    inner = term.func

    # functools.wraps so inspect.signature follows __wrapped__ back to the real
    # function: IsaacLab's manager validates term params against the signature,
    # and a bare *args/**kwargs wrapper reads as two mandatory params named
    # 'a' and 'kw'.
    @functools.wraps(inner)
    def _zero_depth(env, *a, **kw):
        return torch.zeros_like(inner(env, *a, **kw))

    group.depth = term.replace(func=_zero_depth)
    return True


def _patch_object_observations(env_cfg) -> bool:
    """Rewrite the actor group's two box-pose observations per --object_obs.

    Patched on the cfg before gym.make, so the ObservationManager builds with the
    wrappers already in place. Only the actor group is touched: the critic exists
    to shape training and is not evaluated at inference, so degrading it would
    measure nothing.

    Each term's ``noise`` is cleared alongside. Otherwise 'zero' would still emit
    corruption noise about zero, and 'frozen' would jitter despite being held --
    both of which would leak a live signal into a channel meant to be dead.
    """
    mode = args_cli.object_obs
    latency = max(0, args_cli.object_obs_latency)
    if mode == "true" and latency == 0:
        return False

    group = env_cfg.observations.policy
    missing = [n for n in ("object_pose_b", "object_pose_error_b") if getattr(group, n, None) is None]
    if missing:
        raise SystemExit(f"--object_obs: task {args_cli.task} has no policy term(s) {missing} to ablate")
    true_pose_fn = group.object_pose_b.func
    true_err_fn = group.object_pose_error_b.func

    if mode == "ref":
        # Feed the reference pose as though it had been measured, which makes the
        # error term identically zero rather than merely small.
        def pose_fn(env, command_name):
            command = env.command_manager.get_term(command_name)
            pos_b, quat_b = subtract_frame_transforms(
                command.robot_anchor_pos_w,
                command.robot_anchor_quat_w,
                command.object_pos_w,
                command.object_quat_w,
            )
            mat = matrix_from_quat(quat_b)
            return torch.cat([pos_b, mat[..., :2].reshape(mat.shape[0], -1)], dim=-1)

        def err_fn(env, command_name):
            return torch.zeros_like(true_err_fn(env, command_name))

    elif mode == "zero":

        def pose_fn(env, command_name):
            return torch.zeros_like(true_pose_fn(env, command_name))

        def err_fn(env, command_name):
            return torch.zeros_like(true_err_fn(env, command_name))

    else:  # "true" with latency, or "frozen"
        hold = mode == "frozen"
        pose_fn = _stale(true_pose_fn, latency, hold)
        err_fn = _stale(true_err_fn, latency, hold)

    group.object_pose_b.func = pose_fn
    group.object_pose_error_b.func = err_fn
    group.object_pose_b.noise = None
    group.object_pose_error_b.noise = None
    return True


def _agent_cfg_for(task: str):
    """Pull the task's registered rsl_rl runner cfg out of the gym registry.

    Then apply IsaacLab's own deprecation shim. This repo's PPO cfgs
    (agents/rsl_rl_ppo_cfg.py) still use the pre-rsl-rl-4.0 ``policy =
    RslRlPpoActorCriticCfg(...)`` field, while the installed rsl-rl expects
    separate ``actor``/``critic`` model configs -- its
    ``PPO.construct_algorithm`` does ``cfg["actor"].pop("class_name")`` and
    raises ``KeyError: 'class_name'`` without the translation.

    scripts/rsl_rl/play.py gets this for free because ``@hydra_task_config``
    runs the shim internally; loading the cfg straight out of the registry (as
    here) bypasses hydra entirely, so it has to be invoked explicitly.
    """
    import importlib.metadata as metadata

    from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import load_cfg_from_registry

    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    return handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))


def main() -> None:
    pos_thr = args_cli.pos_threshold if args_cli.pos_threshold is not None else DEFAULT_POS_THRESHOLD_M
    rot_thr = args_cli.rot_threshold if args_cli.rot_threshold is not None else DEFAULT_ROT_THRESHOLD_RAD

    # play=False equivalent: keep the training-time domain randomization, which
    # is what supplies the paper's "diverse initial configurations".
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = _agent_cfg_for(args_cli.task)

    if not args_cli.keep_pushes:
        env_cfg.events.push_robot = None
        env_cfg.events.push_object = None
    # The four ranges mjlab's play=True clears, plus observation noise.
    # joint_position_range is left alone, exactly as play=True leaves it.
    known = ("obs_noise", "pose", "velocity", "object_pose", "object_velocity")
    disabled = {s.strip() for s in args_cli.disable.split(",") if s.strip()}
    if unknown := disabled - set(known):
        raise SystemExit(f"--disable: unknown source(s) {sorted(unknown)}; choose from {list(known)}")
    if args_cli.no_randomization:
        disabled = set(known)
    if "obs_noise" in disabled:
        env_cfg.observations.policy.enable_corruption = False
    if "pose" in disabled:
        env_cfg.commands.motion.pose_range = {}
    if "velocity" in disabled:
        env_cfg.commands.motion.velocity_range = {}
    if "object_pose" in disabled:
        env_cfg.commands.motion.object_pose_range = {}
    if "object_velocity" in disabled:
        env_cfg.commands.motion.object_velocity_range = {}
    label = ", ".join(sorted(disabled)) if disabled else "none"
    print(f"[INFO] randomization disabled: {label}")
    if args_cli.start_frame is not None:
        print(f"[INFO] start frame pinned to {args_cli.start_frame}")
    if args_cli.object_size is not None:
        env_cfg.scene.object.spawn.size = tuple(args_cli.object_size)
        print(f"[INFO] object size overridden to {env_cfg.scene.object.spawn.size}")
    if args_cli.object_yaw_range is not None:
        v = abs(args_cli.object_yaw_range)
        # Copy before mutating: the range dict is shared with the cfg object, and
        # editing it in place would also rewrite what gets dumped to params/.
        rng = dict(env_cfg.commands.motion.object_pose_range)
        rng["yaw"] = (-v, v)
        env_cfg.commands.motion = env_cfg.commands.motion.replace(object_pose_range=rng)
        print(f"[INFO] box yaw range overridden to +/-{v:.4f} rad ({math.degrees(v):.1f} deg)")
    if args_cli.object_xy_range is not None:
        m = abs(args_cli.object_xy_range)
        rng = dict(env_cfg.commands.motion.object_pose_range)
        rng["x"] = (-m, m)
        rng["y"] = (-m, m)
        env_cfg.commands.motion = env_cfg.commands.motion.replace(object_pose_range=rng)
    for axis, val in (("x", args_cli.object_x_range), ("y", args_cli.object_y_range)):
        if val is None:
            continue
        m = abs(val)
        rng = dict(env_cfg.commands.motion.object_pose_range)
        rng[axis] = (-m, m)
        env_cfg.commands.motion = env_cfg.commands.motion.replace(object_pose_range=rng)
        print(f"[INFO] box {axis} range overridden to +/-{m:.3f} m")
        print(f"[INFO] box xy range overridden to +/-{m:.3f} m")
    if _patch_depth_noise(env_cfg):
        print("[INFO] depth sensor corruption DISABLED (--depth_noise off)")
    if _patch_depth_observation(env_cfg):
        print("[INFO] depth image BLANKED (--depth_obs zero)")
    if _patch_object_observations(env_cfg):
        detail = f"{args_cli.object_obs}"
        if args_cli.object_obs_latency:
            # 'frozen' holds one sample for the episode, so a delay on top of it
            # is meaningless; say so rather than printing a figure that is ignored.
            detail += (
                f", {args_cli.object_obs_latency} step(s) late"
                if args_cli.object_obs == "true"
                else f" (--object_obs_latency={args_cli.object_obs_latency} ignored in this mode)"
            )
        print(f"[INFO] policy's object observation: {detail} (critic untouched)")

    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base = env.unwrapped

    command = base.command_manager.get_term("motion")

    # Start every episode at frame 0 so "every timestep of the trajectory" is
    # well defined. Unlike mjlab, IsaacLab's MotionCommandCfg has no
    # `sampling_mode` field -- _resample_command calls _adaptive_sampling
    # unconditionally -- so the start frame is forced by replacing that method.
    # Reset-time pose
    # and velocity randomization is untouched, so the paper's "diverse initial
    # configurations" still applies.
    # --start_frame pins somewhere other than 0; the same override path is used
    # so the adaptive sampler can never win over the pin.
    pinned_start = args_cli.start_frame if args_cli.start_frame is not None else 0

    def _always_start_at_zero(env_ids):
        command.time_steps[env_ids] = pinned_start

    command._adaptive_sampling = _always_start_at_zero

    # End the episode exactly when the motion ends, so truncation means
    # "played the whole trajectory" -- the task default of 30 s would
    # otherwise wrap the ~6.4 s motion around several times.
    # (T - 1), not T. The command runs one frame ahead of the step index, so it
    # reaches time_step_total -- and _update_command wraps, teleporting robot and
    # object back to frame 0 -- at step T-1, one step BEFORE a T-step truncation
    # fires. That mid-episode teleport is what fabricated the 158cm "peak object
    # error" (= the box's whole 1.58m trajectory displacement). Ending at T-1
    # truncates before the wrap can happen.
    motion_seconds = (command.motion.time_step_total - 1 - pinned_start) * base.step_dt
    if base.max_episode_length_s > motion_seconds:
        base.cfg.episode_length_s = motion_seconds
        print(f"[INFO] episode_length_s -> {motion_seconds:.3f}s (one full motion, no wraparound)")

    # Re-reset so the FIRST episode starts at frame 0 as well. RslRlVecEnvWrapper
    # calls env.reset() in its constructor ("the RSL-RL runner does not call
    # reset"), which ran before _adaptive_sampling could be replaced above -- so
    # without this the opening episode starts at adaptive-sampled frames, each env
    # reaches the end of the motion at a different step, and _update_command wraps
    # it back to frame 0 mid-rollout. That teleport moves the reference object by
    # the box's whole trajectory displacement and shows up as a ~1.6 m "peak object
    # error" on an otherwise healthy run.
    base.reset()

    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=base.device)
    print(f"[INFO] loaded {args_cli.checkpoint}")

    acc = TrackingEvalAccumulator(args_cli.num_envs, base.device, pos_thr, rot_thr)
    obs, _ = wrapped.get_observations(), None
    print(f"[INFO] running {args_cli.num_envs} episodes ...")

    step = 0
    while not acc.all_done.all():
        # Snapshot the reference before stepping: the command advances its
        # motion frame after the reward is computed, so reading it afterwards
        # would pair the robot with the next frame.
        ref = SimpleNamespace(
            body_pos_w=command.body_pos_w.clone(),
            body_pos_relative_w=command.body_pos_relative_w.clone(),
            object_pos_w=command.object_pos_w.clone(),
            object_quat_w=command.object_quat_w.clone(),
        )
        with torch.no_grad():
            actions = policy(obs)
        obs, _, dones, _ = wrapped.step(actions)

        robot_body = command.robot_body_pos_w
        # Exclude envs that finished on THIS step. auto_reset makes env.step
        # return post-reset state, so pairing the pre-step reference (frame ~319)
        # with a just-reset object (frame 0) fabricates an error equal to the
        # box's whole trajectory displacement (~1.58m) -- which is exactly the
        # bogus 158cm "peak object error" this used to report.
        active = ~acc.all_done & ~dones.bool()
        acc.update(
            active,
            mpkpe=torch.norm(ref.body_pos_w - robot_body, dim=-1).mean(dim=-1),
            r_mpkpe=torch.norm(ref.body_pos_relative_w - robot_body, dim=-1).mean(dim=-1),
            obj_pos_error=torch.norm(ref.object_pos_w - command.robot_object_pos_w, dim=-1),
            obj_rot_error=_obj_rot_error(ref.object_quat_w, command.robot_object_quat_w),
        )
        acc.mark_done(
            dones.bool(),
            base.termination_manager.terminated,
            base.termination_manager.time_outs,
            termination_causes(base.termination_manager),
        )
        step += 1
        if step % 50 == 0:
            print(f"  step {step}: {int(acc.all_done.sum())}/{args_cli.num_envs} episodes done")

    metrics = acc.compute()
    print_report(
        metrics,
        f"IsaacLab  |  {args_cli.task}  |  {os.path.basename(args_cli.checkpoint)}  |  off: {label}",
    )
    if args_cli.output_file:
        Path(args_cli.output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args_cli.output_file).write_text(json.dumps(metrics, indent=2))
        print(f"[INFO] wrote {args_cli.output_file}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
