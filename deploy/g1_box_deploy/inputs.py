# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""Operator input: a velocity for the locomotion policy, a start / stop edge for the
box motion, and an emergency damping request.

    gamepad   (pygame; Bluetooth or USB Xbox-style pad)
              hold LT (deadman) + left stick: walk, right stick x: turn
              Y: start the box motion     B: back to locomotion
              LT + RT together: DAMPING (latches until restart)
    keyboard  (the terminal running run_policy.py)
              W/S vx, A/D vy, Q/E yaw rate (0.1 steps), Space: stop
              '[': start the box motion   ']': back to locomotion   '0': DAMPING
    script    no operator (tests): stand, start the box motion after N seconds
"""

from __future__ import annotations

import select
import sys
import termios
import time
import tty
from dataclasses import dataclass, field

import numpy as np

START, STOP = "start", "stop"


@dataclass(frozen=True)
class Command:
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    signal: str | None = None   # an edge: set only on the tick the button / key went down
    damping: bool = False


def _clip_velocity(v: np.ndarray, vx_range, vy_max, wz_max) -> np.ndarray:
    return np.array([np.clip(v[0], *vx_range), np.clip(v[1], -vy_max, vy_max), np.clip(v[2], -wz_max, wz_max)])


class KeyboardInput:
    STEP = 0.1

    def __init__(self, vx_range, vy_max, wz_max):
        if not sys.stdin.isatty():
            raise RuntimeError("--input keyboard needs an interactive terminal")
        self.fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.limits = (vx_range, vy_max, wz_max)
        self.velocity = np.zeros(3)
        print("[input] keyboard: W/S vx, A/D vy, Q/E yaw rate, Space stop")
        print("[input]           '[' start the box motion, ']' back to locomotion, '0' DAMPING")

    def read(self) -> Command:
        signal, damping = None, False
        keys = {"w": (0, 1), "s": (0, -1), "a": (1, 1), "d": (1, -1), "q": (2, 1), "e": (2, -1)}
        while select.select([sys.stdin], [], [], 0.0)[0]:
            k = sys.stdin.read(1).lower()
            if k == "[":
                signal = START
            elif k == "]":
                signal = STOP
            elif k == "0":
                damping = True
            elif k == " ":
                self.velocity[:] = 0.0
            elif k in keys:
                axis, sign = keys[k]
                self.velocity[axis] += sign * self.STEP
                self.velocity = _clip_velocity(self.velocity, *self.limits)
                print(f"[input] vx/vy/wz {self.velocity.round(2).tolist()}")
        return Command(self.velocity.copy(), signal, damping)

    def close(self) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)


# Raw pygame indices per connection. The same Xbox pad numbers its controls differently
# over USB (Linux xpad: triggers on axes 2/5) and Bluetooth (generic HID: right stick on
# 2/3, triggers after it, X/Y are buttons 3/4). "auto" asks SDL's game-controller database
# for the connected pad instead, which covers both.
LAYOUTS = {
    "xpad": {"leftx": 0, "lefty": 1, "rightx": 3, "lefttrigger": 2, "righttrigger": 5, "a": 0, "b": 1, "x": 2, "y": 3},
    "bluetooth": {"leftx": 0, "lefty": 1, "rightx": 2, "lefttrigger": 5, "righttrigger": 4, "a": 0, "b": 1, "x": 3, "y": 4},
}


def _resolve_layout(name: str) -> tuple[dict[str, int], str]:
    if name in LAYOUTS:
        return dict(LAYOUTS[name]), f"layout {name!r}"
    try:
        from pygame._sdl2 import controller

        controller.init()
        if not controller.is_controller(0):
            raise LookupError("pad not in SDL's controller database")
        pad = controller.Controller(0)
        mapping = pad.get_mapping()
        pad.quit()
        idx = {}
        for key, kind in (("leftx", "a"), ("lefty", "a"), ("rightx", "a"), ("lefttrigger", "a"),
                          ("righttrigger", "a"), ("a", "b"), ("b", "b"), ("x", "b"), ("y", "b")):
            raw = mapping.get(key, "")
            if not (raw[:1] == kind and raw[1:].isdigit()):
                raise LookupError(f"SDL maps {key} to {raw!r}")
            idx[key] = int(raw[1:])
        return idx, "SDL controller mapping"
    except Exception as exc:  # noqa: BLE001 -- any failure falls back loudly
        print(f"[input] gamepad: SDL mapping unavailable ({exc}); falling back to the 'bluetooth' layout")
        return dict(LAYOUTS["bluetooth"]), "layout 'bluetooth' (fallback)"


class GamepadInput:
    """LT is a deadman: released -> zero velocity (the policy keeps balancing; this is
    not an e-stop). It must be seen released once before it can engage, so starting
    with LT held, or an untouched trigger axis reading 0.0, cannot enable motion.
    Triggers rest at -1 and go to +1."""

    DEADZONE = 0.1
    STICK_REST_MAX = 0.3
    PRESS, RELEASE = 0.0, -0.2

    def __init__(self, vx_range, vy_max, wz_max, layout="auto"):
        import pygame

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("--input gamepad but no gamepad found (paired and connected?)")
        self.pg = pygame
        self.pad = pygame.joystick.Joystick(0)
        self.pad.init()
        self.idx, source = _resolve_layout(layout)
        self.vx_range, self.vy_max, self.wz_max = vx_range, vy_max, wz_max
        self._armed = False
        self._engaged = False
        self._held: dict[int, bool] = {}
        pygame.event.pump()
        self._check_rest()
        self._armed = True
        print(f"[input] gamepad: {self.pad.get_name()} ({source})")
        print("[input]   hold LT + left stick: walk, right stick: turn")
        print("[input]   Y: start the box motion, B: back to locomotion, LT + RT: DAMPING")

    def _check_rest(self) -> None:
        """Refuse to start unless the sticks rest centred and both triggers released:
        the layout of the other connection type (USB vs Bluetooth) swaps sticks and
        triggers, and this catches it. Start with the pad untouched."""
        for key in ("leftx", "lefty", "rightx"):
            v = self.pad.get_axis(self.idx[key])
            if abs(v) > self.STICK_REST_MAX:
                raise RuntimeError(f"gamepad: {key} (axis {self.idx[key]}) rests at {v:+.2f} -- wrong --pad_layout, "
                                   "or a stick is held")
        for key in ("lefttrigger", "righttrigger"):
            v = self.pad.get_axis(self.idx[key])
            if v > self.RELEASE:
                raise RuntimeError(f"gamepad: {key} (axis {self.idx[key]}) rests at {v:+.2f}, not released -- wrong "
                                   "--pad_layout, or the trigger is held")

    def _axis(self, key: str) -> float:
        v = self.pad.get_axis(self.idx[key])
        return 0.0 if abs(v) < self.DEADZONE else v

    def _edge(self, key: str) -> bool:
        b = self.idx[key]
        down = bool(self.pad.get_button(b))
        was, self._held[b] = self._held.get(b, False), down
        return down and not was

    def read(self) -> Command:
        self.pg.event.pump()
        try:
            lt = self.pad.get_axis(self.idx["lefttrigger"])
            rt = self.pad.get_axis(self.idx["righttrigger"])
            if lt <= self.RELEASE:
                self._armed = True
            held = self._armed and lt > self.PRESS
            signal = START if self._edge("y") else STOP if self._edge("b") else None
            damping = lt > self.PRESS and rt > self.PRESS
        except self.pg.error:  # pad disconnected: no motion, no signals
            self._armed, held, signal, damping = False, False, None, False
        if held != self._engaged:
            self._engaged = held
            print("[input] deadman engaged" if held else "[input] deadman released -> zero velocity")
        v = np.zeros(3)
        if held:
            ly, lx, rx = -self._axis("lefty"), self._axis("leftx"), self._axis("rightx")
            v = np.array([ly * (self.vx_range[1] if ly >= 0 else -self.vx_range[0]), -lx * self.vy_max, -rx * self.wz_max])
        return Command(v, signal, damping)

    def close(self) -> None:
        self.pg.quit()


class ScriptInput:
    """No operator: zero velocity, START once after ``start_after`` seconds."""

    def __init__(self, start_after: float):
        self.t_start = time.time() + start_after
        self.sent = False
        print(f"[input] script: stand, start the box motion after {start_after:.1f} s")

    def read(self) -> Command:
        if not self.sent and time.time() >= self.t_start:
            self.sent = True
            return Command(signal=START)
        return Command()

    def close(self) -> None:
        pass
