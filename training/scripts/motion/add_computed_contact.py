# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""Inject a geometry-measured `contact_computed` channel into a motion npz.

The IsaacLab converter (sbto_to_npz.py) drives IsaacLab's Articulation
and has no mj_geomDistance available, so it cannot compute this channel. This
post-processor closes that gap:
the contact schedule depends only on the SBTO source trajectory plus the
robot's collision geometry -- both of which live in the MJCF -- so it can be
computed with raw MuJoCo and written into the converter's npz. Runs in the
``sim`` pixi env (needs mujoco, not isaaclab).

Why the channel is needed at all: SBTO's realized "contact" is all-zero for
the hands on the no-hands asset (its sensor watches *_hand_collision, which
sits 4.9cm inside the wrist sphere and is geometrically shielded), and
"contact_ref" is the kinematic plan for a hand grasp the dynamically-feasible
trajectory never performs. Validated on the dummy-hands run, whose SBTO hand
sensor DOES work: this computation reproduces SBTO's own achieved contact with
99.0% agreement at the default zero margin.

    pixi run -e sim add-contact <sbto_run_dir> --npz path/to/motion.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sbto_common import (  # noqa: E402
    SBTO_ROBOT_XML,
    add_floor_and_object,
    end_effector_geoms,
    load_sbto,
    resample,
    slerp,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--npz", required=True, help="Motion npz to modify in place.")
    p.add_argument("--contact-margin", type=float, default=0.0)
    args = p.parse_args()

    sbto = load_sbto(args.run_dir)
    target = dict(np.load(args.npz))
    fps = int(target["fps"][0])

    src_t = sbto["time"]
    dst_t = np.arange(src_t[0], src_t[-1], 1.0 / fps)
    n_target = target["joint_pos"].shape[0]
    if len(dst_t) != n_target:
        raise SystemExit(f"frame mismatch: npz has {n_target}, source resamples to {len(dst_t)} at {fps} fps")

    spec = mujoco.MjSpec.from_file(SBTO_ROBOT_XML)
    if not add_floor_and_object(spec, sbto):
        raise SystemExit("source run carries no object geometry; nothing to compute")
    model = spec.compile()
    data = mujoco.MjData(model)

    obj_g = model.geom("_probe_obj").id
    floor_g = model.geom("_probe_floor").id
    obj_q = model.joint("_probe_object_joint").qposadr[0]
    ee = {s: end_effector_geoms(model, s) for s in ("left", "right")}

    names = [str(n) for n in sbto["dof_names"]]
    qadr = [model.joint(n).qposadr[0] for n in names]
    root_q = model.joint("floating_base_joint").qposadr[0]

    dof = resample(sbto["dof_pos"], src_t, dst_t)
    rpos = resample(sbto["root_pos"], src_t, dst_t)
    rrot = slerp(sbto["root_rot"], src_t, dst_t)
    opos = resample(sbto["object_pos"], src_t, dst_t)
    orot = slerp(sbto["object_rot"], src_t, dst_t)

    ft = np.zeros(6)
    rows = np.zeros((len(dst_t), 3), dtype=np.int32)
    for f in range(len(dst_t)):
        data.qpos[root_q : root_q + 3] = rpos[f]
        data.qpos[root_q + 3 : root_q + 7] = rrot[f]
        for a, v in zip(qadr, dof[f]):
            data.qpos[a] = v
        data.qpos[obj_q : obj_q + 3] = opos[f]
        data.qpos[obj_q + 3 : obj_q + 7] = orot[f]
        mujoco.mj_forward(model, data)
        rows[f, 0] = int(mujoco.mj_geomDistance(model, data, obj_g, floor_g, 1.0, ft) <= args.contact_margin)
        for ch, side in ((1, "left"), (2, "right")):
            rows[f, ch] = int(
                any(mujoco.mj_geomDistance(model, data, g, obj_g, 1.0, ft) <= args.contact_margin for g in ee[side])
            )

    target["contact_computed"] = rows
    target["contact_computed_names"] = (
        sbto["contact_names"] if "contact_names" in sbto
        else np.array(["obj_static_cnt", "left_hand_cnt", "right_hand_cnt"])
    )
    np.savez(args.npz, **target)
    on = [int(rows[:, i].sum()) for i in range(3)]
    print(f"[INFO] wrote contact_computed {rows.shape} into {args.npz}: on-frames={on}")
    if "contact" in target and target["contact"][:, 1].any():
        agree = float(np.mean([(rows[:, i] == target["contact"][:, i]).mean() for i in range(3)]) * 100)
        print(f"[INFO] agreement with SBTO's achieved 'contact': {agree:.1f}%")


if __name__ == "__main__":
    main()
