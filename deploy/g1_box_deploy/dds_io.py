# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""DDS I/O with the robot (or the sim2sim simulator) through unitree_sdk2py.

Topics (identical in sim and on the real G1):
    rt/lowstate      unitree_hg LowState_   joint q / dq, IMU quaternion + gyro
    rt/lowcmd        unitree_hg LowCmd_     per-motor q, dq, kp, kd, tau (+ CRC)
    rt/camera/depth  sensor_msgs Image_     32FC1 metric depth, 64 x 48, header stamp = capture time

Bus: sim = DDS domain 1 on the loopback interface (unitree_mujoco's convention);
real robot = domain 0 on the ethernet interface wired to the G1.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from . import layout
from .image_msg import Image_

TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_DEPTH = "rt/camera/depth"


SIM_BUS = (1, "lo")
REAL_DOMAIN = 0


def init_dds(target: str, interface: str | None = None, *, commands_robot: bool = False,
             assume_yes: bool = False) -> None:
    """target "sim": domain 1 on lo; "real": domain 0 on the robot's interface.

    The real interface comes from --interface, else $G1_NETWORK_INTERFACE. commands_robot=True
    (a process that publishes actuator commands) makes the operator type 'real' before the
    real bus is touched: --target real is one word from --target sim and usually recalled
    from shell history. Read-only tools (depth driver, head-angle meter) only announce the bus.
    """
    if target == "sim":
        domain, interface = SIM_BUS[0], interface or SIM_BUS[1]
    elif target == "real":
        domain = REAL_DOMAIN
        interface = interface or os.environ.get("G1_NETWORK_INTERFACE")
        # A real robot is never on loopback; domain 0 on lo would reach nothing.
        if not interface or interface == SIM_BUS[1]:
            sys.exit(f"[dds] --target real needs the network interface wired to the G1, but "
                     f"--interface is not given and $G1_NETWORK_INTERFACE is {interface or 'unset'!r}")
    else:
        raise ValueError(target)
    is_sim = (domain, interface) == SIM_BUS
    # Printed before anything blocks waiting for traffic: a domain/interface mismatch is
    # otherwise indistinguishable from "the other end is not running".
    print(f"[dds] {'SIM' if is_sim else 'REAL HARDWARE'}: domain {domain} on interface {interface!r}")
    if commands_robot and not is_sim and not assume_yes:
        _confirm_real(domain, interface)
    ChannelFactoryInitialize(domain, interface)


def _confirm_real(domain: int, interface: str) -> None:
    """Make the operator say so before commands go to a real robot. A non-interactive run
    cannot answer, and defaulting to "proceed" there is exactly the accident this prevents."""
    print(f"[dds] ABOUT TO COMMAND REAL HARDWARE on domain {domain}, interface {interface!r}.\n"
          "[dds] Robot hanging in the gantry (or in damping), area clear, e-stop in hand?")
    if not sys.stdin.isatty():
        sys.exit("[dds] refusing to command real hardware from a non-interactive run -- "
                 "pass --yes if that is genuinely what you want")
    if input("[dds] type 'real' to continue: ").strip() != "real":
        sys.exit("[dds] aborted")


@dataclass
class RobotState:
    q: np.ndarray            # (29,) DDS order
    dq: np.ndarray           # (29,)
    imu_quat_wxyz: np.ndarray
    gyro: np.ndarray
    mode_machine: int
    stamp: float             # local receive time


class RobotIO:
    def __init__(self):
        self._lock = threading.Lock()
        self._state: RobotState | None = None
        self._depth: tuple[np.ndarray, float] | None = None
        self._crc = CRC()
        self._cmd = unitree_hg_msg_dds__LowCmd_()
        self._cmd_pub = ChannelPublisher(TOPIC_LOWCMD, LowCmd_)
        self._cmd_pub.Init()
        self._state_sub = ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
        self._state_sub.Init(self._on_state, 10)
        self._depth_sub = ChannelSubscriber(TOPIC_DEPTH, Image_)
        self._depth_sub.Init(self._on_depth, 2)

    def _on_state(self, msg: LowState_) -> None:
        s = RobotState(
            q=np.array([msg.motor_state[i].q for i in range(29)]),
            dq=np.array([msg.motor_state[i].dq for i in range(29)]),
            imu_quat_wxyz=np.array(msg.imu_state.quaternion, dtype=float),
            gyro=np.array(msg.imu_state.gyroscope, dtype=float),
            mode_machine=int(msg.mode_machine),
            stamp=time.time(),
        )
        with self._lock:
            self._state = s

    def _on_depth(self, msg: Image_) -> None:
        if msg.encoding != "32FC1":
            return
        img = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(msg.height, msg.width)
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        # Wall-clock stamp from the producer; fall back to arrival time if the clocks disagree.
        now = time.time()
        if not (now - 5.0 < stamp <= now + 0.05):
            stamp = now
        with self._lock:
            self._depth = (img.copy(), stamp)

    def state(self) -> RobotState | None:
        with self._lock:
            return self._state

    def depth(self) -> tuple[np.ndarray, float] | None:
        """(metric depth (H, W), capture time)."""
        with self._lock:
            return self._depth

    def send(self, q_des: np.ndarray, kp: np.ndarray, kd: np.ndarray, mode_machine: int) -> None:
        c = self._cmd
        c.mode_pr = layout.MODE_PR
        c.mode_machine = mode_machine
        for i in range(29):
            m = c.motor_cmd[i]
            m.mode = 1
            m.q = float(q_des[i])
            m.dq = 0.0
            m.tau = 0.0
            m.kp = float(kp[i])
            m.kd = float(kd[i])
        c.crc = self._crc.Crc(c)
        self._cmd_pub.Write(c)

    def send_damping(self, mode_machine: int) -> None:
        self.send(np.zeros(29), np.zeros(29), np.full(29, layout.DAMPING_KD), mode_machine)
