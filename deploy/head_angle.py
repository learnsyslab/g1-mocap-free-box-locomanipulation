# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""Head-camera level meter: how far the D435's pitch is from the student's trained mount.

The G1 head is articulated -- tilted by hand, it holds its angle by friction -- so the
camera angle changes whenever the robot is handled. The depth student only works with the
camera where it was trained: 47.6 deg below the torso's x axis. This measures the real angle
from the floor and says which way to move the head:

    pitch 53.4 deg  target 47.6  ->  tilt head UP 5.8 deg      roll +0.6  height 1.27 m  floor 2650 px

How: the depth image's floor plane (RANSAC) gives the floor normal in the camera; the pelvis
IMU plus the waist joints give "up" in torso_link (no motion capture). The d435_link pitch /
roll in torso_link that makes the two agree is the measurement.

Needs: the robot standing still on a flat floor (or in the gantry with the feet down), open
floor in the lower half of the view, and realsense_depth.py publishing rt/camera/depth.
Listen-only: it never sends a command, so it is safe to run next to anything.

    pixi run -e real head-angle --interface <nic>        # on the robot
    pixi run -e sim head-angle --target sim              # against sim2sim (should read ~47.6)
"""

from __future__ import annotations

import argparse
import collections
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from g1_box_deploy.camera import CONTRACT  # noqa: E402
from g1_box_deploy.policy_io import torso_quat  # noqa: E402
from g1_box_deploy.transforms import matrix_from_quat  # noqa: E402

R_LINK_OPT = np.array([[0.0, 0, 1], [-1, 0, 0], [0, -1, 0]])  # d435_link <- optical (x right, y down, z fwd)


def _Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def torso_up(imu_quat_wxyz, q_dds):
    """World up expressed in torso_link: pelvis IMU orientation + waist joints."""
    return matrix_from_quat(torso_quat(imu_quat_wxyz, q_dds)).T @ np.array([0.0, 0, 1])


def fit_floor(depth, contract, prior_pitch_deg, up_t, rng=np.random.default_rng(0)):
    """Floor plane in the optical frame: (unit normal pointing at the camera, camera height
    above it, inlier count) or None."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    pts = np.stack([(u - contract.cx) / contract.fx * depth, (v - contract.cy) / contract.fy * depth, depth], -1)
    ok = np.isfinite(depth) & (depth > 0.2)
    P = pts[ok]
    if len(P) < 200:
        return None
    # "up" in the optical frame under the prior mount, to pick the floor among planes
    up_opt = (_Ry(math.radians(prior_pitch_deg)) @ R_LINK_OPT).T @ up_t
    best = None
    for _ in range(300):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if n @ a > 0:
            n = -n  # point toward the camera (origin)
        if n @ up_opt < math.cos(math.radians(30)):
            continue  # not near-horizontal under the prior: a wall, the box side, ...
        inl = np.abs((P - a) @ n) < 0.02
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 150:
        return None
    Q = P[best]
    cen = Q.mean(0)
    n = np.linalg.svd(Q - cen)[2][-1]
    if n @ cen > 0:
        n = -n
    return n, float(-(n @ cen)), int(best.sum())


def solve_mount(n_opt, up_t, prior_pitch_deg):
    """d435_link pitch (about torso y) and roll (about link x) putting n_opt on up_t."""
    best = (np.inf, 0.0, 0.0)
    span, step, p0, r0 = 30.0, 0.5, prior_pitch_deg, 0.0
    for _ in range(3):
        ps = np.arange(p0 - span, p0 + span + 1e-9, step)
        rs = np.arange(r0 - min(span, 10), r0 + min(span, 10) + 1e-9, step)
        for p in ps:
            Rp = _Ry(math.radians(p))
            for r in rs:
                e = np.linalg.norm(Rp @ _Rx(math.radians(r)) @ R_LINK_OPT @ n_opt - up_t)
                if e < best[0]:
                    best = (e, p, r)
        _, p0, r0 = best
        span, step = step * 4, step / 5
    return best[1], best[2], best[0]


def measure(depth, imu_quat, q, contract, prior):
    up_t = torso_up(imu_quat, q)
    floor = fit_floor(depth, contract, prior, up_t)
    if floor is None:
        return None
    n, height, inliers = floor
    pitch, roll, _ = solve_mount(n, up_t, prior)
    return pitch, roll, height, inliers


def report(m, target, tol):
    if m is None:
        return "no floor found -- need open, flat floor in the lower part of the view"
    pitch, roll, height, inl = m
    d = target - pitch
    verdict = f"OK (within {tol:g} deg)" if abs(d) <= tol else f"tilt head {'DOWN' if d > 0 else 'UP'} {abs(d):.1f} deg"
    warn = "   (height looks wrong: floor found?)" if not 0.9 < height < 1.6 else ""
    return (f"pitch {pitch:5.1f} deg  target {target:5.1f}  ->  {verdict:24s}  roll {roll:+4.1f}  "
            f"height {height:4.2f} m  floor {inl:4d} px{warn}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["sim", "real"], default="real", help="DDS bus (real = domain 0 on --interface)")
    ap.add_argument("--interface", default=None, help="network interface wired to the G1 (default: $G1_NETWORK_INTERFACE)")
    ap.add_argument("--target-pitch", type=float, default=CONTRACT.pitch_deg,
                    help="deg below torso x (default: the trained mount)")
    ap.add_argument("--tol", type=float, default=0.5, help="deg counted as OK")
    args = ap.parse_args()

    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    from g1_box_deploy.dds_io import TOPIC_DEPTH, TOPIC_LOWSTATE, init_dds
    from g1_box_deploy.image_msg import Image_

    contract, target = CONTRACT, args.target_pitch
    print(f"[head_angle] target pitch {target:.2f} deg; camera contract {contract.width}x{contract.height}")
    init_dds(args.target, args.interface)
    lock = threading.Lock()
    frames = collections.deque(maxlen=10)
    latest = {}

    def on_depth(msg):
        if msg.encoding != "32FC1" or msg.height != contract.height or msg.width != contract.width:
            latest["shape"] = (msg.height, msg.width)
            return
        with lock:
            frames.append(np.frombuffer(bytes(msg.data), np.float32).reshape(msg.height, msg.width).copy())

    def on_low(msg):
        with lock:
            latest["imu"] = np.array(msg.imu_state.quaternion, float)
            latest["gyro"] = np.array(msg.imu_state.gyroscope, float)
            latest["q"] = np.array([msg.motor_state[i].q for i in range(29)], float)

    depth_sub, low_sub = ChannelSubscriber(TOPIC_DEPTH, Image_), ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
    depth_sub.Init(on_depth, 10)
    low_sub.Init(on_low, 10)
    print("[head_angle] live -- stand still on a flat floor; Ctrl+C to stop")
    try:
        while True:
            time.sleep(1.0)
            with lock:
                fs, imu, q, gyro = list(frames), latest.get("imu"), latest.get("q"), latest.get("gyro")
            if "shape" in latest and not fs:
                print(f"[head_angle] depth is {latest['shape']}, contract expects "
                      f"{(contract.height, contract.width)}")
                continue
            if not fs or imu is None:
                print(f"[head_angle] waiting for {TOPIC_DEPTH} and {TOPIC_LOWSTATE} ...")
                continue
            moving = np.linalg.norm(gyro) > 0.15
            m = measure(np.nanmedian(np.stack(fs), axis=0), imu, q, contract, target)
            print(report(m, target, args.tol) + ("   (robot moving: hold still)" if moving else ""), flush=True)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
