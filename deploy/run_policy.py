# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the G1 box task over DDS: a locomotion policy stands / walks the robot, and on
the operator's signal the box depth student takes over for the 9.9 s pick-and-place.

    # sim2sim (start sim2sim.py first, in another terminal)
    python run_policy.py --target sim --input keyboard
    python run_policy.py --target sim --input script:3        # no operator: stand 3 s, then the box motion
    # real robot (see REAL_ROBOT.md first)
    python run_policy.py --target real --interface enp3s0 --input gamepad

Phases, at 50 Hz:
  STARTUP  ramp from the measured joints to the standing pose (2 s robot / 0.5 s sim), locomotion gains.
  LOCO     the locomotion policy tracks the operator's velocity command; zero = stand.
  BLEND    on START (gamepad Y / key '['): 0.1 s linear blend of q, kp and kd from the
           measured joints to the student's frame-0 target and gains.
  WAIT     hold frame 0 until a depth image younger than --max_depth_age arrives.
  BOX      align the reference's yaw to the robot, then each tick: observation -> student
           -> q_des for motion frame k, k += 1 (the motion is 50 fps).
  -> back to LOCO (blend) when the motion ends, on STOP (B / ']'), or if depth goes stale.
  DAMPING  operator request (LT+RT / '0'), lost robot state or a stalled loop: kp = 0,
           kd = 8 on every joint, then exit. Ctrl-C does the same.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from g1_box_deploy import layout  # noqa: E402
from g1_box_deploy import locomotion as loco  # noqa: E402
from g1_box_deploy.dds_io import RobotIO, init_dds  # noqa: E402
from g1_box_deploy.inputs import START, STOP, GamepadInput, KeyboardInput, ScriptInput  # noqa: E402
from g1_box_deploy.transforms import matrix_from_quat  # noqa: E402
from g1_box_deploy.policy_io import DEPTH_SHAPE, OBS_DIM, Motion, action_to_q_des, build_observation, preprocess_depth, torso_quat  # noqa: E402

DT = 0.02


def make_input(spec: str, pad_layout: str):
    limits = (loco.VX_RANGE, loco.VY_MAX, loco.WZ_MAX)
    if spec == "keyboard":
        return KeyboardInput(*limits)
    if spec == "gamepad":
        return GamepadInput(*limits, layout=pad_layout)
    if spec.startswith("script:"):
        return ScriptInput(float(spec.split(":")[1]))
    raise ValueError(f"--input {spec!r}: use keyboard, gamepad or script:SECONDS")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["sim", "real"], default="sim")
    ap.add_argument("--interface", default=None,
                    help="network interface wired to the G1 (default: $G1_NETWORK_INTERFACE)")
    ap.add_argument("--dry-run", action="store_true",
                    help="shadow mode: read the robot and run everything, but publish NO command; prints "
                         "what it would send once a second (safe to run on the real robot at any time)")
    ap.add_argument("--yes", action="store_true",
                    help="skip the type-'real' confirmation for the real robot (unattended runs only)")
    ap.add_argument("--input", default="keyboard", help="keyboard | gamepad | script:SECONDS (tests)")
    ap.add_argument("--pad_layout", default="auto", choices=["auto", "bluetooth", "xpad"],
                    help="gamepad axis/button numbering: auto (SDL database), bluetooth, or xpad (USB)")
    ap.add_argument("--policy", default=str(HERE / "policy/policy.pt"))
    ap.add_argument("--motion", default=str(HERE / "policy/motion.npz"))
    ap.add_argument("--loco_policy", default=str(HERE / "policy/locomotion.pt"))
    ap.add_argument("--startup", type=float, default=None,
                    help="startup ramp to the standing pose [s]; default 2.0 on the robot (in the gantry), 0.5 in sim, "
                         "where nothing holds the robot up and a slow ramp lets it tip over before locomotion starts")
    ap.add_argument("--blend", type=float, default=0.1, help="locomotion <-> box policy blend [s]")
    ap.add_argument("--max_depth_age", type=float, default=0.15,
                    help="older depth -> back to locomotion [s]; the real stereo pipeline delivers ~35 ms median, ~107 ms worst")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2,
                    help="CPU threads for the networks: the default pool (all cores) oversubscribes and takes ~75 ms/tick; 1-4 threads take ~0.2 ms")
    args = ap.parse_args()
    if args.startup is None:
        args.startup = 2.0 if args.target == "real" else 0.5

    torch.set_num_threads(args.threads)
    init_dds(args.target, args.interface, commands_robot=not args.dry_run, assume_yes=args.yes)
    io = RobotIO()
    shadow = {"q": None}
    if args.dry_run:
        # Nothing reaches rt/lowcmd: remember the would-be target instead, for the status line.
        def _shadow_send(q_des, kp, kd, mode_machine):
            shadow["q"] = np.asarray(q_des, dtype=float).copy()

        io.send = _shadow_send
        io.send_damping = lambda mode_machine: None
        print("[run] DRY RUN: no command is published; the robot is not moved")
    motion = Motion(args.motion)
    walker = loco.LocomotionPolicy(args.loco_policy, args.device)
    student = torch.jit.load(args.policy, map_location=args.device).eval()
    with torch.inference_mode():  # warm-up (TorchScript optimises during the first calls) + a shape check
        for _ in range(5):
            out = student(torch.zeros(1, OBS_DIM, device=args.device), [torch.zeros(1, *DEPTH_SHAPE, device=args.device)])
    assert tuple(out.shape) == (1, 29), out.shape
    print(f"[run] box policy {args.policy} | motion {motion.n_frames} frames @ {motion.fps} fps | locomotion {args.loco_policy}")

    t0 = time.time()
    while io.state() is None:
        if time.time() - t0 > 30.0:
            print("[run] no rt/lowstate after 30 s -- is the robot / sim2sim running on this bus?")
            return 1
        time.sleep(0.01)
    operator = make_input(args.input, args.pad_layout)
    s = io.state()
    mode_machine = s.mode_machine

    # A blend goes linearly in q, kp and kd from (where the robot is, the outgoing gains)
    # to (the incoming policy's start target, its gains), so its first command is a no-op.
    phase, since, blend_time, after_blend = "STARTUP", time.time(), args.startup, "LOCO"
    from_q, from_kp, from_kd = s.q.copy(), loco.LOCO_KP_DDS, loco.LOCO_KD_DDS
    to_q, to_kp, to_kd = loco.LOCO_DEFAULT_POS_DDS, loco.LOCO_KP_DDS, loco.LOCO_KD_DDS
    k, last_action = 0, np.zeros(29)

    def begin_blend(target: str, q_now: np.ndarray, kp_now: np.ndarray, kd_now: np.ndarray) -> None:
        nonlocal phase, since, blend_time, after_blend, from_q, from_kp, from_kd, to_q, to_kp, to_kd
        from_q, from_kp, from_kd = q_now.copy(), kp_now, kd_now
        if target == "BOX":
            to_q, to_kp, to_kd, after_blend = motion.pd_target[0], layout.KP_DDS, layout.KD_DDS, "WAIT"
        else:
            walker.reset()
            to_q, to_kp, to_kd, after_blend = loco.LOCO_DEFAULT_POS_DDS, loco.LOCO_KP_DDS, loco.LOCO_KD_DDS, "LOCO"
        phase, since, blend_time = "BLEND", time.time(), args.blend

    last_status = 0.0
    print("[run] STARTUP: ramping to the standing pose")
    next_tick = time.time()
    try:
        while True:
            now = time.time()
            if now - next_tick > 5 * DT:
                raise RuntimeError(f"control loop stalled ({now - next_tick:.3f} s late)")
            s = io.state()
            if s is None or now - s.stamp > 0.1:
                raise RuntimeError("robot state stale")
            cmd = operator.read()
            if cmd.damping:
                raise RuntimeError("operator requested DAMPING")
            d = io.depth()
            depth_fresh = d is not None and now - d[1] <= args.max_depth_age

            if phase == "LOCO" and cmd.signal == START:
                print("[run] START: blending to the box motion's first frame")
                begin_blend("BOX", s.q, loco.LOCO_KP_DDS, loco.LOCO_KD_DDS)
            elif phase in ("WAIT", "BOX") and cmd.signal == STOP:
                print("[run] STOP: back to locomotion")
                begin_blend("LOCO", s.q, layout.KP_DDS, layout.KD_DDS)

            if phase in ("STARTUP", "BLEND"):
                a = min(1.0, (now - since) / max(blend_time, 1e-6))
                io.send(from_q + a * (to_q - from_q), from_kp + a * (to_kp - from_kp),
                        from_kd + a * (to_kd - from_kd), mode_machine)
                if a >= 1.0:
                    phase = after_blend
                    print("[run] LOCO: standing / walking; start the box motion when ready" if phase == "LOCO"
                          else "[run] WAIT: holding frame 0 until a fresh depth image arrives")
            elif phase == "LOCO":
                q_des = walker.step(s.imu_quat_wxyz, s.gyro, s.q, s.dq, cmd.velocity)
                io.send(q_des, loco.LOCO_KP_DDS, loco.LOCO_KD_DDS, mode_machine)
            elif phase == "WAIT":
                io.send(motion.pd_target[0], layout.KP_DDS, layout.KD_DDS, mode_machine)
                if depth_fresh:
                    yaw = motion.align_yaw(torso_quat(s.imu_quat_wxyz, s.q))
                    print(f"[run] BOX: reference yaw aligned by {np.degrees(yaw):+.1f} deg")
                    phase, k, last_action = "BOX", 0, np.zeros(29)
            elif phase == "BOX":
                if not depth_fresh:
                    print(f"[run] depth stale ({now - d[1]:.3f} s)" if d else "[run] no depth", "-> back to locomotion")
                    begin_blend("LOCO", s.q, layout.KP_DDS, layout.KD_DDS)
                else:
                    obs = build_observation(motion, k, s.imu_quat_wxyz, s.gyro, s.q, s.dq, last_action)
                    img = preprocess_depth(d[0])
                    with torch.inference_mode():
                        act = student(torch.as_tensor(obs, dtype=torch.float32, device=args.device)[None],
                                      [torch.as_tensor(img, device=args.device)[None]])[0].cpu().numpy()
                    io.send(action_to_q_des(motion, k, act), layout.KP_DDS, layout.KD_DDS, mode_machine)
                    last_action = act
                    k += 1
                    if k >= motion.n_frames:
                        print("[run] box motion complete -> back to locomotion")
                        begin_blend("LOCO", s.q, layout.KP_DDS, layout.KD_DDS)

            if args.dry_run and now - last_status >= 1.0 and shadow["q"] is not None:
                last_status = now
                g = matrix_from_quat(s.imu_quat_wxyz)[2]  # world z in the pelvis frame's rows -> tilt
                tilt = np.degrees(np.arccos(np.clip(g[2], -1.0, 1.0)))
                depth_txt = (f"depth age {1e3 * (now - d[1]):5.0f} ms, {100 * np.isfinite(d[0]).mean():3.0f}% valid"
                             if d is not None else "no depth")
                gap = np.abs(shadow["q"] - s.q)
                print(f"[dry] {phase:7s} k={k:3d}  tilt {tilt:4.1f} deg  {depth_txt}  "
                      f"max |q_des - q| {gap.max():.3f} rad ({layout.BODY_JOINTS[int(gap.argmax())]})", flush=True)
            next_tick += DT
            time.sleep(max(0.0, next_tick - time.time()))
    except KeyboardInterrupt:
        print("\n[run] stopped -> DAMPING")
    except RuntimeError as e:
        print(f"[run] {e} -> DAMPING")
    finally:
        operator.close()
    for _ in range(25):
        io.send_damping(mode_machine)
        time.sleep(DT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
