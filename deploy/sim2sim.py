# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo sim2sim for the box depth student, speaking the real G1's DDS interface.

Modelled on Unitree's unitree_mujoco (simulate_python, BSD-3): same unitree_sdk2py
topics and message types, DDS domain 1 on the loopback interface, so run_policy.py
talks to this exactly as it talks to the robot. Two additions it needs:

* the PD law is evaluated at EVERY physics step (1 ms) from the latest LowCmd_, as the
  G1's motor drivers do -- not once per 50 Hz command, which holds a stale torque for
  20 steps and destabilises the damping;
* the head camera (d435_depth, 47.6 deg down) is rendered as metric depth, 64 x 48,
  and published on rt/camera/depth at 30 Hz -- what FoundationStereo provides on the
  robot.

    python sim2sim.py                     # viewer (needs a display)
    MUJOCO_GL=egl python sim2sim.py --headless --duration 20

Start run_policy.py --target sim in a second terminal. Until the first LowCmd_ the
robot is frozen in its start pose, as if hanging in a gantry (state and depth are still
published); 0.5 s without commands after that -> damping. At exit the box
trajectory is written to --log (npz) and its lift / final position printed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from g1_box_deploy import layout  # noqa: E402
from g1_box_deploy.camera import CONTRACT  # noqa: E402
from g1_box_deploy.dds_io import TOPIC_DEPTH, TOPIC_LOWCMD, TOPIC_LOWSTATE, init_dds  # noqa: E402
from g1_box_deploy.image_msg import Image_  # noqa: E402
from g1_box_deploy.policy_io import Motion  # noqa: E402
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.default import std_msgs_msg_dds__Header_, unitree_hg_msg_dds__LowState_  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # noqa: E402

COMMAND_TIMEOUT = 0.5
STATE_EVERY = 5          # 1 ms steps -> LowState_ at 200 Hz
DEPTH_HZ = 30.0
DEPTH_W, DEPTH_H = CONTRACT.width, CONTRACT.height


class Sim:
    def __init__(self, scene: str, motion: Motion):
        self.m = mujoco.MjModel.from_xml_path(scene)
        self.d = mujoco.MjData(self.m)
        sid = lambda n: self.m.sensor_adr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, n)]  # noqa: E731
        self.act = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in layout.BODY_JOINTS]
        self.pos_adr = [sid(f"{n}_pos") for n in layout.BODY_JOINTS]
        self.vel_adr = [sid(f"{n}_vel") for n in layout.BODY_JOINTS]
        self.tau_adr = [sid(f"{n}_torque") for n in layout.BODY_JOINTS]
        self.quat_adr, self.gyro_adr, self.acc_adr = sid("imu_quat"), sid("imu_gyro"), sid("imu_acc")
        self._place(motion)
        self.cmd_q = self._q().copy()
        self.cmd_kp, self.cmd_kd = layout.KP_DDS.copy(), layout.KD_DDS.copy()
        self.cmd_dq, self.cmd_tau = np.zeros(29), np.zeros(29)
        self.last_cmd = None
        self.box = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.pelvis = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    def _place(self, motion: Motion) -> None:
        """Robot root + joints and the box at the reference's frame 0."""
        j = lambda n: self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)]  # noqa: E731
        r = j("floating_base_joint")
        self.d.qpos[r:r + 3], self.d.qpos[r + 3:r + 7] = motion.root_pos0, motion.root_quat0
        for i, n in enumerate(layout.ISAAC_JOINT_ORDER):
            self.d.qpos[j(f"{n}_joint")] = motion.joint_pos0_isaac[i]
        b = j("object_freejoint")
        self.d.qpos[b:b + 3], self.d.qpos[b + 3:b + 7] = motion.object_pos0, motion.object_quat0
        mujoco.mj_forward(self.m, self.d)

    def _q(self):
        return self.d.sensordata[self.pos_adr]

    def on_cmd(self, msg: LowCmd_) -> None:
        for i in range(29):
            mc = msg.motor_cmd[i]
            self.cmd_q[i], self.cmd_dq[i], self.cmd_kp[i], self.cmd_kd[i], self.cmd_tau[i] = mc.q, mc.dq, mc.kp, mc.kd, mc.tau
        self.last_cmd = time.time()

    def step(self) -> None:
        if self.last_cmd is None:
            # No controller yet: the robot is "in the gantry". A PD hold of the start
            # pose alone does not keep it standing for more than ~1-2 s, so instead of
            # letting it tip over while the controller starts up, time stands still.
            mujoco.mj_forward(self.m, self.d)
            return
        q, dq = self._q(), self.d.sensordata[self.vel_adr]
        if self.last_cmd is not None and time.time() - self.last_cmd > COMMAND_TIMEOUT:
            tau = -layout.DAMPING_KD * dq  # commands stopped -> damping
        else:
            tau = self.cmd_tau + self.cmd_kp * (self.cmd_q - q) + self.cmd_kd * (self.cmd_dq - dq)
        self.d.ctrl[self.act] = tau
        mujoco.mj_step(self.m, self.d)

    def fill_state(self, ls) -> None:
        sd = self.d.sensordata
        for i in range(29):
            ms = ls.motor_state[i]
            ms.mode, ms.q, ms.dq, ms.tau_est = 1, float(sd[self.pos_adr[i]]), float(sd[self.vel_adr[i]]), float(sd[self.tau_adr[i]])
        ls.imu_state.quaternion[:] = [float(v) for v in sd[self.quat_adr:self.quat_adr + 4]]
        ls.imu_state.gyroscope[:] = [float(v) for v in sd[self.gyro_adr:self.gyro_adr + 3]]
        ls.imu_state.accelerometer[:] = [float(v) for v in sd[self.acc_adr:self.acc_adr + 3]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=str(HERE.parent / "assets/mujoco/scene_box.xml"))
    ap.add_argument("--motion", default=str(HERE / "policy/motion.npz"))
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after this many seconds (0 = until closed)")
    ap.add_argument("--log", default="sim2sim_box_log.npz")
    args = ap.parse_args()

    init_dds("sim")
    motion = Motion(args.motion)
    sim = Sim(args.scene, motion)
    state_pub = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
    state_pub.Init()
    depth_pub = ChannelPublisher(TOPIC_DEPTH, Image_)
    depth_pub.Init()
    ChannelSubscriber(TOPIC_LOWCMD, LowCmd_).Init(sim.on_cmd, 10)
    ls = unitree_hg_msg_dds__LowState_()
    ls.mode_pr, ls.mode_machine = layout.MODE_PR, layout.MODE_MACHINE

    renderer = mujoco.Renderer(sim.m, height=DEPTH_H, width=DEPTH_W)
    renderer.enable_depth_rendering()  # metric depth along the camera axis
    depth_every = int(round(1.0 / (DEPTH_HZ * sim.m.opt.timestep)))
    viewer = None
    if not args.headless:
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(sim.m, sim.d)
    box_log, start, step = [], time.time(), 0
    print(f"[sim2sim] running: LowState_ {1 / (STATE_EVERY * sim.m.opt.timestep):.0f} Hz, depth {DEPTH_HZ:.0f} Hz on {TOPIC_DEPTH}")
    try:
        while args.duration <= 0 or time.time() - start < args.duration:
            sim.step()
            step += 1
            if step % STATE_EVERY == 0:
                sim.fill_state(ls)
                ls.tick = step
                state_pub.Write(ls)
                box_log.append([time.time(), *sim.d.xpos[sim.box], *sim.d.xpos[sim.pelvis], *sim.d.xquat[sim.pelvis]])
            if step % depth_every == 0:
                renderer.update_scene(sim.d, camera="d435_depth")
                img = np.asarray(renderer.render(), dtype=np.float32)
                t = time.time()
                hdr = std_msgs_msg_dds__Header_()
                hdr.stamp.sec, hdr.stamp.nanosec, hdr.frame_id = int(t), int((t % 1) * 1e9), CONTRACT.frame_id
                depth_pub.Write(Image_(header=hdr, height=DEPTH_H, width=DEPTH_W, encoding="32FC1", is_bigendian=0,
                                       step=4 * DEPTH_W, data=list(img.tobytes())))
            if viewer is not None and step % 20 == 0:
                if not viewer.is_running():
                    break
                viewer.sync()
            ahead = start + step * sim.m.opt.timestep - time.time()  # real time
            if ahead > 0:
                time.sleep(ahead)
    except KeyboardInterrupt:
        pass
    log = np.asarray(box_log)
    if len(log):
        np.savez(args.log, t=log[:, 0], box_pos=log[:, 1:4], pelvis_pos=log[:, 4:7], pelvis_quat_wxyz=log[:, 7:11])
        z0 = log[0, 3]
        print(f"[sim2sim] box: start z {z0:.3f} m, max lift {log[:, 3].max() - z0:.3f} m, final {np.round(log[-1, 1:4], 3)} -> {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
