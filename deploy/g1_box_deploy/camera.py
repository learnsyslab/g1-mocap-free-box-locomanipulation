# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""The depth image the box student was trained on: one source of truth for the depth
driver (realsense_depth.py), the simulator (sim2sim.py) and the runtime (run_policy.py).

These are properties of the TRAINED POLICY -- the CNN's first layer is built for this
resolution and the normalisation divisor is the far plane it saw -- not of the sensor.
Drift between copies does not fail loudly: it surfaces as a plausible image with every
distance wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraContract:
    width: int = 64
    height: int = 48
    fovy_deg: float = 58.0       # the real D435's vertical FOV
    cutoff_m: float = 2.5        # far plane; also the normalisation divisor
    min_depth_m: float = 0.168   # D435 minZ at 848x480
    frame_id: str = "d435_link"  # the left IR imager
    # d435_link's pitch below torso_link's x axis that the student was trained with (the G1's
    # nominal head pose). The head is articulated: set it to this with head_angle.py.
    pitch_deg: float = 47.59

    @property
    def fy(self) -> float:
        """Focal length in pixels, from the vertical field of view."""
        return (self.height / 2.0) / math.tan(math.radians(self.fovy_deg) / 2.0)

    @property
    def fx(self) -> float:
        """Equal to fy: training scaled the horizontal aperture by W/H, which leaves the
        pixel square."""
        return self.fy

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def fovx_deg(self) -> float:
        """Follows from the aspect ratio and is NOT the sensor's own: 72.9 deg against the
        D435's 87, which is why the driver resamples through these intrinsics instead of
        resizing."""
        return math.degrees(2.0 * math.atan((self.width / 2.0) / self.fx))

    def describe(self) -> str:
        return (f"{self.width}x{self.height}  fx=fy={self.fx:.2f}  "
                f"fov {self.fovx_deg:.1f}x{self.fovy_deg:.1f} deg  "
                f"range {self.min_depth_m:.3f}..{self.cutoff_m:.2f} m  frame {self.frame_id}")


CONTRACT = CameraContract()
