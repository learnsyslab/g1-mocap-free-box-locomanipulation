# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""SBTO trajectory helpers for add_computed_contact.py.

Plain MuJoCo + numpy, no IsaacLab, so it runs in the ``sim`` pixi env.
"""

from __future__ import annotations

import os

import mujoco
import numpy as np


def slerp(quats: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    """Piecewise slerp of a (T, 4) wxyz quaternion track onto new sample times."""
    idx = np.clip(np.searchsorted(src_t, dst_t, side="right") - 1, 0, len(src_t) - 2)
    span = src_t[idx + 1] - src_t[idx]
    alpha = np.where(span > 0, (dst_t - src_t[idx]) / np.where(span > 0, span, 1.0), 0.0)

    q0, q1 = quats[idx], quats[idx + 1].copy()
    flip = np.sum(q0 * q1, axis=-1) < 0.0
    q1[flip] *= -1.0

    dot = np.clip(np.sum(q0 * q1, axis=-1), -1.0, 1.0)
    theta = np.arccos(dot)
    small = theta < 1e-6
    sin_theta = np.where(small, 1.0, np.sin(theta))
    w0 = np.where(small, 1.0 - alpha, np.sin((1.0 - alpha) * theta) / sin_theta)
    w1 = np.where(small, alpha, np.sin(alpha * theta) / sin_theta)

    out = w0[:, None] * q0 + w1[:, None] * q1
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def resample(track: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.interp(dst_t, src_t, track[:, i]) for i in range(track.shape[1])], axis=-1
    )


# SBTO's own G1 model for the dummy-hands robot (its collision geometry is what the
# contact schedule is measured against). Meshes are shared with the sim2sim model.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SBTO_ROBOT_XML = os.path.join(_REPO_ROOT, "assets", "mujoco", "sbto", "g1_mjx_dummy_hands.xml")


def add_floor_and_object(spec: mujoco.MjSpec, sbto: dict) -> bool:
    """Add a floor plane and the run's own object geom to the robot spec.

    Only used to *measure* geometry for the computed contact channel -- the
    converter still never steps physics. Appended after the robot's bodies, so
    existing qpos addresses are unchanged (and everything here is resolved by
    name anyway).
    """
    if "object_geom_size" not in sbto:
        return False
    st = spec.worldbody.add_body(name="_probe_static")
    st.add_geom(name="_probe_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.01])
    ob = spec.worldbody.add_body(name="_probe_object")
    ob.add_freejoint(name="_probe_object_joint")
    ob.add_geom(
        name="_probe_obj",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(np.asarray(sbto["object_geom_size"], dtype=float)),
        pos=list(np.asarray(sbto.get("object_geom_pos", [0.0, 0.0, 0.0]), dtype=float)),
        quat=list(np.asarray(sbto.get("object_geom_quat", [1.0, 0.0, 0.0, 0.0]), dtype=float)),
        mass=float(np.asarray(sbto["object_mass"], dtype=float).reshape(-1)[0]) if "object_mass" in sbto else 0.6,
    )
    return True


def end_effector_geoms(model: mujoco.MjModel, side: str) -> list[int]:
    """Collision geoms on the whole ``<side>_wrist_pitch_link`` subtree.

    Deliberately the *subtree*, not one named geom. SBTO's own
    hand_obj_contact.xml watches ``<side>_hand_collision`` alone, but on the
    no-hands G1 model that capsule (r=0.020, on wrist_yaw) sits 4.9cm INSIDE the
    wrist sphere (r=0.035, on wrist_pitch), so it is geometrically shielded and
    can never make first contact -- which is why SBTO's realized ``contact``
    channel is all zero for the hands even while the box is lifted 0.63m and
    carried 1.58m. Taking the subtree captures whichever geom actually does the
    carrying (the wrist here, the dummy hand), which is what "this
    end effector is touching the object" should mean.
    """
    root = f"{side}_wrist_pitch_link"
    try:
        root_id = model.body(root).id
    except KeyError:
        return []
    in_subtree = set()
    for b in range(model.nbody):
        p = b
        while p != 0:
            if p == root_id:
                in_subtree.add(b)
                break
            p = model.body_parentid[p]
    return [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] in in_subtree and model.geom_group[g] in (3, 4)
    ]


def load_sbto(run_dir: str) -> dict:
    path = run_dir if run_dir.endswith(".npz") else os.path.join(run_dir, "best_trajectory.npz")
    data = dict(np.load(path))
    if "dof_names" not in data:
        raise RuntimeError(
            f"{path} predates the dof_names key. Re-save it with sbto's "
            "scripts/add_contacts_to_run.py, which also backfills the contacts."
        )
    return data
