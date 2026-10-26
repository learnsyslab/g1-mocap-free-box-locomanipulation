# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""Render the final depth student at four extreme box setups, and tile them into one video.

Runs render_student_rollout.py once per setup (each a separate Isaac process: a fresh scene per
clip), then writes <out_dir>/showcase_2x2.mp4 -- the four third-person views, labelled -- next
to each setup's own scene / scene-and-depth clips.

    pixi run -e train render-showcase --out_dir videos/showcase
    # or: python scripts/eval/render_showcase.py --out_dir videos/showcase [--only far side]

The setups are offsets from the demonstrated box pose, inside the range the final student
(RealCam-Finetune-Stage2) trained on: x -0.40..+0.60 m, y -0.50..+0.20 m, any yaw, any of six
resting faces. Edit SETUPS to show others.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parents[1]

# name: (label, render_student_rollout.py arguments)
SETUPS = {
    "far": ("box 0.45 m further away", ["--box_x", "0.45", "--box_y", "0.0", "--box_yaw", "0.0", "--box_rest", "0"]),
    "side": ("box 0.40 m to the side", ["--box_x", "0.0", "--box_y", "-0.40", "--box_yaw", "0.0", "--box_rest", "0"]),
    "rotated": ("box rotated 90 deg", ["--box_x", "0.0", "--box_y", "0.0", "--box_yaw", f"{math.pi / 2:.4f}", "--box_rest", "0"]),
    "lying": ("box lying on its side", ["--box_x", "0.0", "--box_y", "0.0", "--box_yaw", "0.0", "--box_rest", "2"]),
}


def label_frames(frames, text):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.load_default(size=26)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    out = []
    for f in frames:
        im = Image.fromarray(f)
        d = ImageDraw.Draw(im)
        x0, y0, x1, y1 = d.textbbox((10, 8), text, font=font)
        d.rectangle([0, 0, x1 + 10, y1 + 8], fill=(0, 0, 0))
        d.text((10, 8), text, fill=(255, 255, 255), font=font)
        out.append(np.asarray(im))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(TRAINING.parent / "checkpoints/student_realcam_stage2_model_8998.pt"))
    ap.add_argument("--task", default="G1-DummyHands-RealBox-RealCam-Finetune-Stage2-Play")
    ap.add_argument("--steps", type=int, default=495, help="control steps per clip (the motion is 495)")
    ap.add_argument("--only", nargs="*", choices=list(SETUPS), help="render just these setups")
    ap.add_argument("--out_dir", required=True)
    args, extra = ap.parse_known_args()   # anything else goes to the renderer (e.g. --device)

    out_dir = Path(args.out_dir).resolve()
    names = args.only or list(SETUPS)
    for name in names:
        label, setup = SETUPS[name]
        cmd = [sys.executable, str(HERE / "render_student_rollout.py"), "--task", args.task,
               "--checkpoint", args.checkpoint, "--steps", str(args.steps), "--clean_depth",
               "--out_dir", str(out_dir / name), *setup, *extra]
        print(f"[showcase] {name}: {label}", flush=True)
        if subprocess.run(cmd, cwd=TRAINING).returncode != 0:
            print(f"[showcase] {name} failed; see its output above", flush=True)

    clips = {n: out_dir / n / "scene.mp4" for n in names if (out_dir / n / "scene.mp4").exists()}
    if len(clips) < 4:
        print(f"[showcase] {len(clips)} of 4 clips rendered; skipping the 2x2 tile")
        return 0 if clips else 1
    import imageio.v2 as imageio
    import numpy as np

    vids = {n: label_frames(imageio.mimread(p, memtest=False), SETUPS[n][0]) for n, p in clips.items()}
    n = min(len(v) for v in vids.values())
    a, b, c, d = (vids[k] for k in names[:4])
    tiled = [np.concatenate([np.concatenate([a[i], b[i]], axis=1), np.concatenate([c[i], d[i]], axis=1)], axis=0)
             for i in range(n)]
    path = out_dir / "showcase_2x2.mp4"
    imageio.mimsave(path, tiled, fps=30)
    print(f"[showcase] wrote {path} ({n} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
