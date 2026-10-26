# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""Render a depth-student rollout: third-person view beside the student's input.

The distillation curves say the student is learning; they cannot say WHY it
fails. Watching the scene alone shows the failure but not its cause, and
watching the depth alone shows the cause but not the consequence. Side by side,
a mis-grasp becomes readable: whether the box was even visible, whether it left
the frame, whether the sensor noise ate it.

The depth panel is rendered with corruption ON by default, because that is what
the student is actually handed at this point in training -- a clean panel would
show a picture the policy never sees. Pass --clean_depth for the uncorrupted
image when the question is framing rather than robustness.

Usage:
  python scripts/eval/render_student_rollout.py --checkpoint <student.pt> [--steps 700]
"""

from __future__ import annotations

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="G1-DummyHands-RealBox-RealCam-Finetune-Stage2-Play")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=495, help="control steps (the motion is 495)")
parser.add_argument("--box_x", type=float, default=None, help="pin box x offset (default: random)")
parser.add_argument("--box_y", type=float, default=None)
parser.add_argument("--box_yaw", type=float, default=None, help="pin box yaw in radians")
parser.add_argument("--box_rest", type=int, default=None,
                    help="pin the resting face: index into BOX_RESTING_STATES "
                         "(0 identity, 1 roll180, 2 roll+90, 3 roll-90, "
                         "4 pitch+90, 5 pitch-90). Default: whatever the task samples.")
parser.add_argument("--clean_depth", action="store_true", help="show depth without sensor corruption")
parser.add_argument("--cam_pos", type=float, nargs=3, default=None,
                    help="override the depth camera mount position in the torso frame")
parser.add_argument("--cam_pitch_deg", type=float, default=None,
                    help="override the depth camera pitch-down angle")
parser.add_argument("--cam_fovy_deg", type=float, default=None,
                    help="override the depth camera vertical FOV (real D435 depth is 58)")
parser.add_argument("--scene_eye", type=float, nargs=3, default=(1.6, -3.2, 1.7),
                    help="third-person camera position (default: beside the demo's walking path)")
parser.add_argument("--scene_lookat", type=float, nargs=3, default=(-0.2, -0.25, 0.45),
                    help="third-person camera target")
parser.add_argument("--hide_head", action="store_true",
                    help="hide head_link so it cannot occlude the depth camera; the real "
                         "sensor looks out through the visor and never sees its own housing")
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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

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

    m = env_cfg.commands.motion
    m.start_frame = 0
    rng = dict(m.object_pose_range)
    if args_cli.box_x is not None:
        rng["x"] = (args_cli.box_x, args_cli.box_x)
    if args_cli.box_y is not None:
        rng["y"] = (args_cli.box_y, args_cli.box_y)
    if args_cli.box_yaw is not None:
        rng["yaw"] = (args_cli.box_yaw, args_cli.box_yaw)
    # Pin the resting face. Passing a single-element tuple makes the command
    # term sample that face every reset, which is what makes a set of clips
    # comparable: with the full six-tuple the face would vary between rollouts
    # AND within one, and nothing could be attributed to it.
    rest = m.object_resting_states
    if args_cli.box_rest is not None:
        if not rest:
            raise SystemExit(
                f"--box_rest given but task {args_cli.task!r} has no "
                "object_resting_states; use a Stage2 task"
            )
        if not 0 <= args_cli.box_rest < len(rest):
            raise SystemExit(
                f"--box_rest {args_cli.box_rest} out of range for "
                f"{len(rest)} resting states"
            )
        rest = (rest[args_cli.box_rest],)
        print(f"[INFO] resting face pinned to index {args_cli.box_rest}: {rest[0]}")
    env_cfg.commands.motion = m.replace(object_pose_range=rng,
                                        object_resting_states=rest, debug_vis=False)

    # Camera overrides, so the robot's ACTUAL mount and the real sensor FOV can be
    # rendered without editing the task. The trained policy still expects what it
    # was trained on, so a clip under different optics shows what the camera
    # WOULD see -- not how a policy trained on those optics would behave.
    if any(v is not None for v in (args_cli.cam_pos, args_cli.cam_pitch_deg, args_cli.cam_fovy_deg)):
        import math as _m

        from isaaclab.sensors import CameraCfg as _CamCfg
        from isaaclab.sim import PinholeCameraCfg as _Pin

        old = env_cfg.scene.depth_cam
        pos = tuple(args_cli.cam_pos) if args_cli.cam_pos is not None else old.offset.pos
        # Defaults DERIVED from the task's own camera, not hardcoded. They used
        # to fall back to 55 deg / 80 deg -- the obsolete pre-d435_link camera --
        # so passing only --cam_pitch_deg silently widened the FOV from 58 to 80
        # and confounded any experiment that varied pitch alone.
        _r = old.offset.rot                       # (x, y, z, w), pitch about +Y
        _cur_pitch = _m.degrees(2.0 * _m.atan2(_r[1], _r[3]))
        _cur_fovy = _m.degrees(
            2.0 * _m.atan(old.spawn.vertical_aperture / (2.0 * old.spawn.focal_length))
        ) if getattr(old.spawn, "vertical_aperture", None) else 58.0
        pitch = (args_cli.cam_pitch_deg if args_cli.cam_pitch_deg is not None
                 else _cur_pitch)
        fovy = (args_cli.cam_fovy_deg if args_cli.cam_fovy_deg is not None
                else _cur_fovy)
        print(f"[INFO] camera: task pitch {_cur_pitch:.2f} deg fovy "
              f"{_cur_fovy:.2f} deg  ->  using pitch {pitch:.2f} fovy {fovy:.2f}")
        focal = 24.0
        va = 2.0 * focal * _m.tan(_m.radians(fovy) / 2.0)
        kw = dict(focal_length=focal, horizontal_aperture=va * old.width / old.height,
                  clipping_range=(0.01, 20.0))
        if "vertical_aperture" in _Pin.__annotations__:
            kw["vertical_aperture"] = va
        h = _m.radians(pitch) / 2.0
        env_cfg.scene.depth_cam = _CamCfg(
            prim_path=old.prim_path, update_period=0.0,
            width=old.width, height=old.height,
            data_types=list(old.data_types),
            spawn=_Pin(**kw),
            offset=_CamCfg.OffsetCfg(
                pos=pos, rot=(0.0, _m.sin(h), 0.0, _m.cos(h)), convention="world"
            ),
        )
        print(f"[INFO] camera override: pos={pos} pitch={pitch} deg fovy={fovy} deg")

    if not args_cli.clean_depth:
        # The play cfg turns corruption off; put it back so the panel shows the
        # image the policy is actually given.
        env_cfg.observations.camera.depth.params["noise"] = True

    # Third-person view from a scene CAMERA rather than the viewport: with a
    # Camera sensor present, viewport capture returns black frames (confirmed
    # against play.py --video on this same task). Same eye/lookat as every
    # earlier overlay so the clips stay comparable.
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import PinholeCameraCfg

    eye, lookat = tuple(args_cli.scene_eye), tuple(args_cli.scene_lookat)
    env_cfg.scene.scene_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/scene_cam",
        update_period=0.0,
        width=640,
        height=480,
        data_types=["rgb"],
        spawn=PinholeCameraCfg(clipping_range=(0.05, 50.0)),
        offset=CameraCfg.OffsetCfg(pos=eye, rot=look_at_quat_xyzw(eye, lookat), convention="world"),
    )
    env_cfg.viewer.eye = eye
    env_cfg.viewer.lookat = lookat

    # A DEPTH-ONLY camera in the scene puts the renderer in a mode with no
    # colour output, which zeroes rgb on EVERY camera -- including this one, and
    # including the viewport (play.py --video renders black on such a task too,
    # while it is fine on the camera-free Stage 2-4 tasks). Asking the robot's
    # depth camera for rgb as well keeps the colour pipeline alive: measured,
    # scene rgb max goes 0 -> 245. The student never reads this extra channel;
    # its observation is still the depth image alone.
    if "rgb" not in env_cfg.scene.depth_cam.data_types:
        env_cfg.scene.depth_cam.data_types = list(env_cfg.scene.depth_cam.data_types) + ["rgb"]
    for name in ("push_robot", "push_object"):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    base = env.unwrapped
    # RecordVideo, not per-step env.render(): the direct call returns black
    # frames here. This is the path play.py uses and that produced every earlier
    # overlay, and it must wrap the gym env BEFORE the rsl-rl wrapper.
    wrapped = RslRlVecEnvWrapper(env)

    from importlib import metadata

    from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import load_cfg_from_registry

    # The shim folds the flat cfg fields (stochastic / init_noise_std /
    # noise_std_type / state_dependent_std) into distribution_cfg. train.py runs
    # it internally; loading straight from the registry does not, and the model
    # constructor then rejects 'stochastic' as an unexpected kwarg.
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point"),
        metadata.version("rsl-rl-lib"),
    )
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=base.device)
    print(f"[INFO] loaded student {args_cli.checkpoint}")

    if args_cli.hide_head:
        from pxr import UsdGeom

        head = base.sim.stage.GetPrimAtPath(
            f"/World/envs/env_0/Robot/torso_link/head_link"
        )
        if head.IsValid():
            UsdGeom.Imageable(head).MakeInvisible()
            print("[INFO] head_link hidden (it cannot occlude the depth camera)")
        else:
            print("[WARN] head_link not found; nothing hidden")

    # Report what the box actually ended up as. The resting face is applied on
    # reset (which the wrapper does in its constructor), and a clip whose face
    # silently failed to apply looks exactly like a correct identity clip -- so
    # print the height rather than trusting the flag.
    _cmd = base.command_manager.get_term("motion")
    _obj = base.scene["object"]
    _st = _obj.data.root_state_w
    _st = _st.torch if hasattr(_st, "torch") else _st
    _org = base.scene.env_origins
    _org = _org.torch if hasattr(_org, "torch") else _org
    _pelvis = base.scene["robot"].data.root_state_w
    _pelvis = _pelvis.torch if hasattr(_pelvis, "torch") else _pelvis
    _bx, _by = (_st[0, 0] - _org[0, 0]).item(), (_st[0, 1] - _org[0, 1]).item()
    _px, _py = (_pelvis[0, 0] - _org[0, 0]).item(), (_pelvis[0, 1] - _org[0, 1]).item()
    _d = ((_bx - _px) ** 2 + (_by - _py) ** 2) ** 0.5
    print(f"[INFO] box spawn: xy ({_bx:+.3f}, {_by:+.3f}) m  centre z "
          f"{(_st[0, 2] - _org[0, 2]).item():.4f} m  "
          f"rest_quat {[round(v, 3) for v in _cmd._rest_quat[0].tolist()]}  "
          f"rest_dz {_cmd._rest_dz[0].item():+.4f} m  "
          f"options {_cmd._rest_options.shape[0] if hasattr(_cmd, '_rest_options') else 1}")
    print(f"[INFO] pelvis xy ({_px:+.3f}, {_py:+.3f}) m  -> horizontal box "
          f"distance {_d:.3f} m")

    obs, _ = wrapped.get_observations(), None
    scene_frames, depth_frames = [], []

    scene_cam = base.scene["scene_cam"]
    for step in range(args_cli.steps):
        rgb = _as_tensor(scene_cam.data.output["rgb"])[0, ..., :3]
        scene_frames.append(rgb.detach().cpu().numpy().astype(np.uint8))

        d = obs["camera"] if isinstance(obs, dict) or hasattr(obs, "keys") else None
        if d is not None:
            img = _as_tensor(d)[0, 0].detach().cpu().numpy()
            depth_frames.append((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8))

        with torch.no_grad():
            actions = policy(obs)
        obs, _, dones, _ = wrapped.step(actions)
        if bool(torch.as_tensor(dones).any()):
            print(f"[INFO] episode ended at step {step}")
            break

    print(f"[INFO] captured {len(scene_frames)} scene / {len(depth_frames)} depth frames")
    # An all-black panel means nothing rendered. Say so loudly rather than
    # shipping a half-blank video, which is exactly what happened once already.
    if scene_frames:
        mx = int(max(f.max() for f in scene_frames[: min(50, len(scene_frames))]))
        print(f"[INFO] scene frame max pixel value = {mx} (0 means NOTHING RENDERED)")
        if mx == 0:
            print("[ERROR] scene panel is entirely black -- not writing a broken video")
            env.close()
            return
    write_outputs(scene_frames, depth_frames, args_cli.out_dir)
    env.close()


def write_outputs(scene, depth, out_dir) -> None:
    import imageio.v2 as imageio

    if not scene or not depth:
        print(f"nothing to write (scene={len(scene)}, depth={len(depth)})")
        return
    n = min(len(scene), len(depth))
    scene, depth = scene[:n], depth[:n]

    sh, sw = scene[0].shape[:2]
    panel = []
    for s, d in zip(scene, depth):
        # Nearest-neighbour upscale: the depth is 64x48 and smoothing would hide
        # exactly the pixel-level dropouts worth seeing.
        rep_y = max(1, sh // d.shape[0])
        big = np.repeat(np.repeat(d, rep_y, axis=0), rep_y, axis=1)
        big = big[:sh, :]
        if big.shape[0] < sh:
            big = np.pad(big, ((0, sh - big.shape[0]), (0, 0)))
        rgb_d = np.stack([big] * 3, axis=-1)
        panel.append(np.concatenate([s[..., :3], rgb_d], axis=1))

    imageio.mimsave(os.path.join(out_dir, "scene_and_depth.mp4"), panel, fps=30)
    imageio.mimsave(os.path.join(out_dir, "scene.mp4"), scene, fps=30)
    print(f"wrote {out_dir}/scene_and_depth.mp4")

    from PIL import Image

    for i in (0, n // 3, 2 * n // 3, n - 1):
        Image.fromarray(panel[i]).save(os.path.join(out_dir, f"panel_{i:04d}.png"))
    print(f"wrote sample PNGs to {out_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
