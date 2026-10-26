# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""Render the reference motion: the robot and the box replayed kinematically in IsaacLab.

No policy -- every frame writes the motion file's robot root pose, joint positions and box
pose into the simulator and takes one zero-residual control step, then renders. It shows what the policies
are asked to track, with the same third-person camera as render_student_rollout.py.

    pixi run -e train render-reference --out_dir videos/reference
    # or: python scripts/eval/render_reference_motion.py --out_dir videos/reference [--loops 2]
"""

from __future__ import annotations

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="G1-DummyHands-RealBox-Tracking-No-Obj-Terminate-Play",
                    help="any task of this repo; it only provides the robot, the box and the motion file")
parser.add_argument("--loops", type=int, default=1, help="how many times to play the motion")
parser.add_argument("--scene_eye", type=float, nargs=3, default=(1.6, -3.2, 1.7), help="third-person camera position")
parser.add_argument("--scene_lookat", type=float, nargs=3, default=(-0.2, -0.25, 0.45), help="third-person camera target")
parser.add_argument("--out_dir", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.sim import PinholeCameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import g1_mocap_free_box.tasks  # noqa: F401, E402


def look_at_quat_xyzw(eye, lookat):
    """Rotation (x, y, z, w) aiming a camera ("world" convention: forward +X, up +Z) from eye at lookat."""
    dx, dy, dz = (lookat[i] - eye[i] for i in range(3))
    yaw, pitch = math.atan2(dy, dx), math.atan2(-dz, math.hypot(dx, dy))
    sz, cz, sp, cp = math.sin(yaw / 2), math.cos(yaw / 2), math.sin(pitch / 2), math.cos(pitch / 2)
    return (-sz * sp, cz * sp, sz * cp, cz * cp)


def _as_tensor(x):
    return x.torch if hasattr(x, "torch") else x


def main() -> None:
    os.makedirs(args_cli.out_dir, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    # Show the motion exactly as recorded: frame 0, no reset randomisation, no pushes.
    m = env_cfg.commands.motion
    env_cfg.commands.motion = m.replace(start_frame=0, pose_range={}, velocity_range={}, joint_position_range=(0.0, 0.0),
                                        object_pose_range={}, object_velocity_range={}, debug_vis=False)
    for name in ("push_robot", "push_object"):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)

    # Third-person view from a scene camera aimed from eye at lookat.
    eye, lookat = tuple(args_cli.scene_eye), tuple(args_cli.scene_lookat)
    env_cfg.scene.scene_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/scene_cam", update_period=0.0, width=640, height=480, data_types=["rgb"],
        spawn=PinholeCameraCfg(clipping_range=(0.05, 50.0)),
        offset=CameraCfg.OffsetCfg(pos=eye, rot=look_at_quat_xyzw(eye, lookat), convention="world"),
    )
    # A depth-only camera (the student tasks) zeroes rgb on every camera; ask it for rgb too.
    depth_cam = getattr(env_cfg.scene, "depth_cam", None)
    if depth_cam is not None and "rgb" not in depth_cam.data_types:
        depth_cam.data_types = list(depth_cam.data_types) + ["rgb"]

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    command = env.command_manager.get_term("motion")
    robot, obj = env.scene["robot"], env.scene["object"]
    cam = env.scene["scene_cam"]
    env_ids = torch.arange(env.num_envs, device=env.device)
    n_frames = int(command.motion.time_step_total)
    print(f"[reference] {n_frames} frames at {1.0 / env.step_dt:.0f} fps, {args_cli.loops} loop(s)")

    zero_action = torch.zeros((env.num_envs, env.action_space.shape[1]), device=env.device)
    frames = []
    for _ in range(args_cli.loops):
        env.reset()   # start_frame = 0: the robot and the box are placed at frame 0
        for _ in range(n_frames - 1):
            # Overwrite the robot and the box with the reference at the current frame, then let the
            # environment take its normal step: the action is a residual on the reference's own PD
            # target, so zero holds the reference, the step renders the scene camera and advances
            # the motion one frame. What is drawn is the reference plus at most one control step
            # (20 ms) of physics under that target. (Writing state and rendering without a step
            # never reaches the renderer here; stepping the simulator by hand outside env.step
            # corrupts the PhysX GPU views.)
            robot.write_joint_state_to_sim(command.joint_pos, command.joint_vel, env_ids=env_ids)
            robot.write_root_state_to_sim(
                torch.cat([command.body_pos_w[:, 0], command.body_quat_w[:, 0],
                           command.body_lin_vel_w[:, 0], command.body_ang_vel_w[:, 0]], dim=-1),
                env_ids=env_ids)
            obj.write_root_state_to_sim(
                torch.cat([command.object_pos_w, command.object_quat_w,
                           command.object_lin_vel_w, command.object_ang_vel_w], dim=-1),
                env_ids=env_ids)
            env.step(zero_action)
            frames.append(_as_tensor(cam.data.output["rgb"])[0, ..., :3].detach().cpu().numpy().astype(np.uint8))

    if not frames or int(max(f.max() for f in frames[:50])) == 0:
        print("[reference] ERROR: the scene camera rendered nothing (all black) -- not writing a video")
    else:
        import imageio.v2 as imageio

        path = os.path.join(args_cli.out_dir, "reference_motion.mp4")
        imageio.mimsave(path, frames, fps=round(1.0 / env.step_dt))
        print(f"[reference] wrote {path} ({len(frames)} frames)")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
