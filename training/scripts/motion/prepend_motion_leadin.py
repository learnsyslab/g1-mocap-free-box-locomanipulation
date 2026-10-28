# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""Prepend a standing lead-in to an SBTO reference motion.

Why this exists: sub3_largebox_005 is a segment cut from the middle of a human
capture, so its frame 0 is mid-stride -- the peak joint velocity of the whole
clip (8.9 rad/s), the pelvis moving at 0.49 m/s, one foot 5 cm off the floor,
and the torso pitched 24 degrees over the box. Nothing downstream can start
there. A real robot cannot be placed into that state, and a policy handed over
from locomotion arrives upright, which is outside the tracker's basin: measured
in sim, the tracker needs the base pitched to within ~6 degrees of the
reference's 24, and a joint-space blend from standing cannot produce that lean
because the pose is not a static equilibrium.

Fixing it downstream is impossible; fixing it in SBTO's config is not enough
either, because tasks/g1/robot_object_ref.py sets the initial state from
``self.ref.x0`` unconditionally, overwriting any ``keyframe_x0``. So the fix is
to make ``ref.x0`` itself a standing pose: prepend frames that ease from a
standing keyframe into the demo's frame 0. The default is ``loco``, the pose the
deployed locomotion policy actually holds, so the eventual handover has nothing
to absorb. SBTO then optimises a dynamically feasible way to perform that lean,
and every stage after it -- the converted npz, the RL reset distribution, the
deployment handover -- inherits a start pose the robot can actually be in.

The lead-in does not need to be dynamically feasible. It is a tracking *cost*,
not a constraint.

Run from the repository root (the paths are relative to it)::

    pixi run -e sim motion-leadin /path/to/my_reference_original.npz -o /path/to/my_reference_leadin.npz

Then point ``cfg_ref.motion_path`` at the output and re-run SBTO. Note the
horizon grows by --seconds * fps frames (45 at the defaults, 193 -> 238, +23%),
so check the solver's T and t_knots budget before a long run.
"""

from __future__ import annotations

import argparse
import numpy as np

# qpos layout of an OmniRetarget/SBTO reference, verified against a real run:
# free joints are [quat, pos] (cfg_ref's flip_quat_pos) and quaternions are
# wxyz (quat_wxyz). The joint block is in the MJCF's own order, which is the
# same order as the keyframes below and as SBTO's dof_names.
BASE_QUAT, BASE_POS = slice(0, 4), slice(4, 7)
JOINTS = slice(7, 36)
OBJECT = slice(36, 43)
NQ = 43

# Mirrors sbto/models/unitree_g1/keyframes/g1_29dof_obj.xml. All three stand the
# base upright (identity quaternion) with zero velocity; only the leg and arm
# configuration differs. Kept as literals rather than parsed from the XML so
# this script does not need the sbto checkout on its path -- if you edit one,
# edit both.
KEYFRAMES = {
    "home": dict(
        height=0.783675,
        joints=[
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,          # left leg
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,          # right leg
            0.0, 0.0, 0.0,                            # waist yaw/roll/pitch
            0.2, 0.2, 0.0, 0.3, 0.0, 0.0, -1.57,      # left arm
            0.2, -0.2, 0.0, 0.3, 0.0, 0.0, 1.57,      # right arm
        ],
    ),
    "knees_bent": dict(
        height=0.755,
        joints=[
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
            0.0, 0.0, 0.0,
            0.2, 0.22, 0.0, 0.5, 0.0, 0.0, 0.0,
            0.2, -0.22, 0.0, 0.5, 0.0, 0.0, 0.0,
        ],
    ),
    # The pose the deployed locomotion policy actually settles into, standing
    # with a zero velocity command -- measured in MuJoCo, averaged over 2 s once
    # converged, then mirrored. Prefer this over `home` for anything that will
    # be handed over from locomotion: `home` matches in the legs (0.089 rad RMS)
    # but differs by 0.65 rad RMS in the arms, mostly its wrist_yaw of -+1.57
    # where the policy holds the wrists near zero. Starting the lead-in here
    # means the handover has no arm swing to absorb.
    "loco": dict(
        height=0.7854,
        joints=[
            -0.084, 0.098, -0.012, 0.158, -0.100, -0.103,
            -0.084, -0.098, 0.012, 0.158, -0.100, 0.103,
            0.0, 0.0, 0.019,
            0.369, 0.266, -0.033, 1.000, 0.171, 0.004, -0.022,
            0.369, -0.266, 0.033, 1.000, -0.171, 0.004, 0.022,
        ],
    ),
}


def yaw_of(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Shortest-arc interpolation between two wxyz quaternions."""
    q1 = q1 if float(np.dot(q0, q1)) >= 0.0 else -q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    theta = np.arccos(dot)
    if theta < 1e-8:
        out = (1.0 - alpha) * q0 + alpha * q1
    else:
        out = (np.sin((1.0 - alpha) * theta) * q0 + np.sin(alpha * theta) * q1) / np.sin(theta)
    return out / np.linalg.norm(out)


def standing_pose(first: np.ndarray, keyframe: dict) -> np.ndarray:
    """The demo's frame 0, but standing upright in the keyframe's configuration.

    Keeps the demo's xy and yaw so the lead-in is a pure lean -- the robot does
    not translate. That matters for deployment: it means the operator only has
    to get the robot to the right spot, and the transition into the crouch is
    the policy's job rather than the walk's.
    """
    pose = first.copy()
    pose[BASE_QUAT] = quat_from_yaw(yaw_of(first[BASE_QUAT]))
    pose[BASE_POS] = [first[BASE_POS][0], first[BASE_POS][1], keyframe["height"]]
    pose[JOINTS] = keyframe["joints"]
    # The object is untouched during the lead-in; nobody has reached it yet.
    return pose


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """3t^2 - 2t^3: zero slope at both ends, so no velocity step at the joins."""
    return t * t * (3.0 - 2.0 * t)


def build_leadin(first: np.ndarray, keyframe: dict, n_frames: int) -> np.ndarray:
    start = standing_pose(first, keyframe)
    t = np.linspace(0.0, 1.0, n_frames, endpoint=False)
    # Smoothstep, not a linear ramp: a linear ramp leaves the very first frame
    # with a nonzero velocity, which is the whole thing being fixed here.
    alphas = _smoothstep(t)

    frames = np.repeat(start[None, :], n_frames, axis=0)
    for i, alpha in enumerate(alphas):
        frames[i, BASE_QUAT] = slerp(start[BASE_QUAT], first[BASE_QUAT], alpha)
        frames[i, BASE_POS] = (1.0 - alpha) * start[BASE_POS] + alpha * first[BASE_POS]
        frames[i, JOINTS] = (1.0 - alpha) * start[JOINTS] + alpha * first[JOINTS]
        frames[i, OBJECT] = first[OBJECT]
    return frames


def build_leadout(qpos: np.ndarray, keyframe: dict, n_frames: int, fps: float) -> np.ndarray:
    """Ease from the demo's last frame back to a standing pose.

    Needed for the same reason as the lead-in, at the more dangerous end. The
    clip stops mid-stride -- 0.65 m/s of base velocity and 2.35 rad/s of joint
    velocity, still accelerating on the final frame -- so the tracker's DONE
    state holds an open-loop PD target on a leaning, moving robot, and the
    blend back to locomotion starts from a pose locomotion cannot catch. The
    task is already done by then: dropping the robot here costs a completed
    run, with the box on the floor by its feet.

    Unlike the lead-in, the base is NOT pinned in xy. The robot is walking when
    the reference runs out, so it has to be allowed to travel while it stops;
    holding xy fixed would ask for an instantaneous halt. The position follows
    the integral of the decaying speed profile, which keeps it C1-continuous
    with the demo's final velocity and brings it to rest at
    0.5 * v_end * duration downrange.
    """
    last = qpos[-1]
    velocity_xy = (last[BASE_POS][:2] - qpos[-2][BASE_POS][:2]) * fps
    duration = n_frames / fps

    end = last.copy()
    end[BASE_QUAT] = quat_from_yaw(yaw_of(last[BASE_QUAT]))
    end[BASE_POS] = [*(last[BASE_POS][:2] + 0.5 * velocity_xy * duration), keyframe["height"]]
    end[JOINTS] = keyframe["joints"]

    t = np.linspace(0.0, 1.0, n_frames + 1)[1:]
    alphas = _smoothstep(t)
    # Integral of the (1 - smoothstep) speed profile, so xy decelerates rather
    # than interpolating -- which would start by moving backwards.
    travel = t - (t**3 - 0.5 * t**4)

    frames = np.repeat(end[None, :], n_frames, axis=0)
    for i, (alpha, moved) in enumerate(zip(alphas, travel)):
        frames[i, BASE_QUAT] = slerp(last[BASE_QUAT], end[BASE_QUAT], alpha)
        frames[i, BASE_POS] = [
            *(last[BASE_POS][:2] + velocity_xy * duration * moved),
            (1.0 - alpha) * last[BASE_POS][2] + alpha * end[BASE_POS][2],
        ]
        frames[i, JOINTS] = (1.0 - alpha) * last[JOINTS] + alpha * end[JOINTS]
        frames[i, OBJECT] = last[OBJECT]
    return frames


def report(qpos: np.ndarray, fps: float, n_lead: int, n_tail: int) -> None:
    base_speed = np.linalg.norm(np.diff(qpos[:, BASE_POS], axis=0), axis=1) * fps
    joint_speed = np.abs(np.diff(qpos[:, JOINTS], axis=0)).max(axis=1) * fps
    demo = qpos.shape[0] - n_lead - n_tail
    print(f"[INFO] frames {qpos.shape[0]} ({n_lead} lead-in + {demo} demo + {n_tail} lead-out)"
          f"  = {qpos.shape[0] / fps:.2f} s @ {fps:g} Hz")
    print(f"[INFO] first frame: base {base_speed[0] * 100:6.2f} cm/s, joints "
          f"{joint_speed[0]:.3f} rad/s   (demo alone: 49.8 cm/s, 7.90 rad/s)")
    print(f"[INFO] last  frame: base {base_speed[-1] * 100:6.2f} cm/s, joints "
          f"{joint_speed[-1]:.3f} rad/s   (demo alone: 65.2 cm/s, 2.35 rad/s)")
    if n_lead:
        print(f"[INFO] lead-in  peak {joint_speed[:n_lead].max():.2f} rad/s")
    if n_tail:
        tail = joint_speed[-n_tail:]
        travel = np.linalg.norm(qpos[-1, BASE_POS][:2] - qpos[-n_tail - 1, BASE_POS][:2])
        print(f"[INFO] lead-out peak {tail.max():.2f} rad/s, base travels "
              f"{travel * 100:.0f} cm while stopping")
    quiet = int(np.sum((base_speed < 0.05) & (joint_speed < 0.5)))
    print(f"[INFO] near-rest frames: {quiet}")
    if quiet == 0:
        print("[WARN] no near-rest frame -- neither pad took effect")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="OmniRetarget reference npz, e.g. sub3_largebox_005_original.npz")
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--seconds", type=float, default=1.5,
                   help="Lead-in duration: how long the robot has to lean from standing "
                        "into the demo's first pose.")
    p.add_argument("--tail-seconds", type=float, default=2.0,
                   help="Lead-out duration: how long it has to stop. 0 disables it.")
    p.add_argument("--keyframe", choices=sorted(KEYFRAMES), default="loco")
    args = p.parse_args()

    src = dict(np.load(args.source, allow_pickle=True))
    qpos = np.asarray(src["qpos"], dtype=float)
    if qpos.shape[1] != NQ:
        raise SystemExit(
            f"{args.source}: qpos has {qpos.shape[1]} columns, expected {NQ} "
            "(29 joints + floating base + object). This script assumes the 29-DoF "
            "layout of the G1 with dummy hands."
        )
    fps = float(np.asarray(src["fps"]).reshape(-1)[0])
    n_lead = int(round(args.seconds * fps))
    n_tail = int(round(args.tail_seconds * fps))
    if n_lead < 2:
        raise SystemExit("--seconds is too short to interpolate over")
    if args.tail_seconds and n_tail < 2:
        raise SystemExit("--tail-seconds is too short to interpolate over")

    keyframe = KEYFRAMES[args.keyframe]
    parts = [build_leadin(qpos[0], keyframe, n_lead), qpos]
    if n_tail:
        parts.append(build_leadout(qpos, keyframe, n_tail, fps))
    out = dict(src)
    out["qpos"] = np.concatenate(parts, axis=0)
    # Every other per-frame track is held at its first value across the lead-in,
    # so array lengths stay consistent for anything that indexes them by frame.
    for key, value in src.items():
        value = np.asarray(value)
        if key != "qpos" and value.ndim >= 1 and value.shape[0] == qpos.shape[0]:
            out[key] = np.concatenate(
                [np.repeat(value[:1], n_lead, axis=0), value]
                + ([np.repeat(value[-1:], n_tail, axis=0)] if n_tail else []), axis=0)

    np.savez(args.output, **out)
    print(f"[INFO] wrote {args.output} (keyframe: {args.keyframe})")
    report(out["qpos"], fps, n_lead, n_tail)


if __name__ == "__main__":
    main()
