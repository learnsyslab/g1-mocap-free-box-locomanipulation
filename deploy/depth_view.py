# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""Live view of rt/camera/depth: exactly the image the policy is about to see.

    pixi run -e real depth-view            # robot bus ($G1_NETWORK_INTERFACE or --interface)
    pixi run -e sim depth-view --target sim

The 64x48 metric depth, scaled up, coloured over the policy's range (near = bright,
2.5 m and beyond = dark); invalid pixels (NaN: stereo holes, the left-edge band, out of
range) in black -- the policy reads those as far. The title line shows the rate, the age of
the newest frame (by its capture stamp), the valid fraction and min / median / max.
Listen-only. Esc or q to quit.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from g1_box_deploy.camera import CONTRACT  # noqa: E402

SCALE = 10


def colourize(depth_m: np.ndarray) -> np.ndarray:
    """(H, W) metres -> (H, W, 3) uint8: near bright yellow, far dark blue, NaN black."""
    t = np.clip(depth_m / CONTRACT.cutoff_m, 0.0, 1.0)
    t = np.where(np.isfinite(t), 1.0 - t, np.nan)  # 1 = near
    stops = np.array([[20, 20, 80], [30, 110, 200], [60, 200, 120], [250, 230, 50]], dtype=float)
    x = np.nan_to_num(t, nan=0.0) * (len(stops) - 1)
    i = np.clip(x.astype(int), 0, len(stops) - 2)
    f = (x - i)[..., None]
    rgb = stops[i] * (1 - f) + stops[i + 1] * f
    rgb[~np.isfinite(t)] = 0
    return rgb.astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["sim", "real"], default="real")
    ap.add_argument("--interface", default=None, help="network interface wired to the G1 (default: $G1_NETWORK_INTERFACE)")
    args = ap.parse_args()

    import pygame
    from unitree_sdk2py.core.channel import ChannelSubscriber

    from g1_box_deploy.dds_io import TOPIC_DEPTH, init_dds
    from g1_box_deploy.image_msg import Image_

    init_dds(args.target, args.interface)
    lock = threading.Lock()
    latest = {"img": None, "stamp": 0.0, "n": 0}

    def on_depth(msg):
        if msg.encoding != "32FC1":
            return
        img = np.frombuffer(bytes(msg.data), np.float32).reshape(msg.height, msg.width).copy()
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        with lock:
            latest.update(img=img, stamp=stamp, n=latest["n"] + 1)

    sub = ChannelSubscriber(TOPIC_DEPTH, Image_)
    sub.Init(on_depth, 2)

    pygame.init()
    screen = pygame.display.set_mode((CONTRACT.width * SCALE, CONTRACT.height * SCALE))
    font = pygame.font.SysFont("monospace", 15)
    clock = pygame.time.Clock()
    rate, n_prev, t_prev = 0.0, 0, time.time()
    print(f"[depth_view] showing {TOPIC_DEPTH}; Esc or q to quit")
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_q)):
                pygame.quit()
                return 0
        with lock:
            img, stamp, n = latest["img"], latest["stamp"], latest["n"]
        now = time.time()
        if now - t_prev >= 1.0:
            rate, n_prev, t_prev = (n - n_prev) / (now - t_prev), n, now
        screen.fill((0, 0, 0))
        if img is None:
            lines = [f"waiting for {TOPIC_DEPTH} ..."]
        else:
            if img.shape != (CONTRACT.height, CONTRACT.width):
                lines = [f"WRONG SIZE {img.shape[1]}x{img.shape[0]}, the policy expects "
                         f"{CONTRACT.width}x{CONTRACT.height}"]
            else:
                surf = pygame.surfarray.make_surface(colourize(img).swapaxes(0, 1))
                screen.blit(pygame.transform.scale(surf, screen.get_size()), (0, 0))
                ok = np.isfinite(img)
                vals = img[ok]
                stats = (f"{vals.min():.2f} / {np.median(vals):.2f} / {vals.max():.2f} m" if vals.size else "no valid pixels")
                lines = [f"{rate:4.1f} Hz   age {1e3 * (now - stamp):5.0f} ms   {100 * ok.mean():3.0f}% valid",
                         f"min/med/max {stats}   (black = invalid -> far)"]
        for i, line in enumerate(lines):
            screen.blit(font.render(line, True, (255, 255, 255), (0, 0, 0)), (6, 6 + 18 * i))
        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    sys.exit(main())
