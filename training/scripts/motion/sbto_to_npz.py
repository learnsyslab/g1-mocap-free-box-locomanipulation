# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""Convert an SBTO trajectory into a motion npz the tracking task can load.

SBTO saves joint and root state in its own actuated-joint order; the motion
loader wants per-body world frames, which only exist once the pose is realized
on the articulation. So this replays the trajectory kinematically -- writing
state, never stepping physics -- and reads the body frames back out, the same
way ``unitree_rl_lab``'s ``csv_to_npz.py`` does for BVH captures.

Run from the repository root (the paths are relative to it)::

    pixi run -e train convert-sbto /path/to/sbto/run_dir -o motions/my_motion.npz

Then post-process the output with add_computed_contact.py, with the same run_dir::

    pixi run -e sim add-contact /path/to/sbto/run_dir --npz motions/my_motion.npz

That second pass is not optional. The motion loader (mdp/commands.py) prefers "contact_computed"
over "contact_ref" over "contact", and this converter cannot produce the first
of those: it drives IsaacLab's Articulation, so it has no mj_geomDistance to
measure the reference geometry with. Skipping the pass does not fail -- the
loader quietly falls back to "contact_ref", the kinematic plan for a grasp the
dynamically-feasible trajectory never performs, and the contact reward tracks
the wrong reference for the whole run.
"""

import argparse
import numpy as np
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_dir", type=str, help="SBTO run directory, or a best_trajectory.npz.")
parser.add_argument("--output", "-o", type=str, required=True, help="Output motion npz path.")
parser.add_argument("--output_fps", type=int, default=50, help="Output fps (default matches the task's 50 Hz).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from g1_mocap_free_box.assets.g1 import G1_29DOF_DUMMY_HANDS_MIMIC_CFG as ROBOT_CFG

# MuJoCo orders quaternions (w, x, y, z); IsaacLab 6.x orders them (x, y, z, w)
# throughout -- see write_root_pose_to_sim_index and isaaclab.utils.math.quat_mul.
# Everything below stays in MuJoCo's order until the last moment, because the
# helpers here operate on SBTO data; only what crosses into IsaacLab is
# reordered. Getting this wrong does not fail loudly: the robot simply stands
# in the wrong orientation and falls over.
def wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return quat[..., [1, 2, 3, 0]]


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def slerp(quats: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    """Piecewise slerp of a (T, 4) wxyz quaternion track onto new sample times."""
    idx = np.clip(np.searchsorted(src_t, dst_t, side="right") - 1, 0, len(src_t) - 2)
    span = src_t[idx + 1] - src_t[idx]
    alpha = np.where(span > 0, (dst_t - src_t[idx]) / np.where(span > 0, span, 1.0), 0.0)

    q0, q1 = quats[idx], quats[idx + 1].copy()
    # Take the short way round.
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


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of wxyz quaternions, broadcasting over leading dims."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v by wxyz quaternion q."""
    w, u = q[..., :1], q[..., 1:]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def compose_geom_frame(sbto: dict, pos, quat, lin_vel, ang_vel):
    """Move the object pose from MuJoCo's body frame onto the geom itself.

    SBTO's object_pos/object_rot track the freejoint body, but the box geom
    sits offset and rotated inside it. IsaacLab spawns the shape at the body
    origin, so the reference is expressed about the geom instead -- physically
    the same rigid body, just the frame the other simulator uses.
    """
    if "object_geom_pos" not in sbto:
        return pos, quat, lin_vel, ang_vel
    offset = np.asarray(sbto["object_geom_pos"], dtype=np.float64)
    geom_quat = np.asarray(sbto["object_geom_quat"], dtype=np.float64)

    lever = quat_rotate(quat, np.broadcast_to(offset, pos.shape))
    return (
        pos + lever,
        quat_mul(quat, np.broadcast_to(geom_quat, quat.shape)),
        lin_vel + np.cross(ang_vel, lever),
        ang_vel,
    )


def resample(track: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(dst_t, src_t, track[:, i]) for i in range(track.shape[1])], axis=-1)


def load_sbto(run_dir: str) -> dict:
    path = run_dir if run_dir.endswith(".npz") else os.path.join(run_dir, "best_trajectory.npz")
    data = dict(np.load(path))
    if "dof_names" not in data:
        raise RuntimeError(
            f"{path} predates the dof_names key. Re-save it with sbto's "
            "scripts/add_contacts_to_run.py, which also backfills the contacts."
        )
    return data


def main() -> None:
    sbto = load_sbto(args_cli.run_dir)

    src_t = sbto["time"]
    input_fps = 1.0 / float(np.diff(src_t).mean())
    dst_t = np.arange(src_t[0], src_t[-1], 1.0 / args_cli.output_fps)
    print(f"[INFO] {len(src_t)} frames @ {input_fps:.1f} Hz -> {len(dst_t)} frames @ {args_cli.output_fps} Hz")

    joint_names = [str(n) for n in sbto["dof_names"]]
    joint_pos = resample(sbto["dof_pos"], src_t, dst_t)
    joint_vel = resample(sbto["dof_vel"], src_t, dst_t)
    root_pos = resample(sbto["root_pos"], src_t, dst_t)
    root_rot = slerp(sbto["root_rot"], src_t, dst_t)
    root_lin_vel = resample(sbto["root_lin_vel"], src_t, dst_t)
    root_ang_vel = resample(sbto["root_ang_vel"], src_t, dst_t)

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ReplaySceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot = scene["robot"]
    driven_idx = robot.find_joints(joint_names, preserve_order=True)[0]

    log = {k: [] for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}

    for frame in range(len(dst_t)):
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] = _t(root_pos[frame], robot)
        root_state[:, :2] += scene.env_origins[:, :2]
        root_state[:, 3:7] = _t(wxyz_to_xyzw(root_rot[frame]), robot)
        root_state[:, 7:10] = _t(root_lin_vel[frame], robot)
        # SBTO's root_ang_vel is in MuJoCo's free-joint qvel convention, which is
        # BODY-local, not world. IsaacLab's root_state[:, 10:] is world-frame
        # (root_ang_vel_w), so it needs one body -> world rotation. Writing it
        # unconverted corrupts not just the reset velocity but every body
        # velocity read back below, since FK derives them from this root state.
        #
        # Confirmed empirically in MuJoCo: replaying SBTO's own
        # dof_pd_target through mj_step reproduces its saved trajectory to
        # 0.00000 summed joint error when root_ang_vel is treated as body-frame,
        # versus 0.148 rad at frame 1 when treated as world-frame.
        root_state[:, 10:] = _t(quat_rotate(root_rot[frame], root_ang_vel[frame]), robot)
        robot.write_root_state_to_sim(root_state)

        q = robot.data.default_joint_pos.clone()
        qd = robot.data.default_joint_vel.clone()
        q[:, driven_idx] = _t(joint_pos[frame], robot)
        qd[:, driven_idx] = _t(joint_vel[frame], robot)
        robot.write_joint_state_to_sim(q, qd)

        # Writing state does not re-run forward kinematics, so body_pos_w would
        # otherwise report whatever PhysX last computed -- stale poses that have
        # nothing to do with the joint values just written. forward() updates
        # articulation kinematics without stepping physics.
        sim.forward()
        scene.update(sim.get_physics_dt())

        log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].cpu().numpy().copy())

    out = {k: np.stack(v, axis=0) for k, v in log.items()}
    out["fps"] = np.array([args_cli.output_fps])
    out["body_names"] = np.array(robot.body_names)
    out["joint_names"] = np.array(robot.joint_names)

    # Carried through for the object-tracking and contact rewards. dof_pd_target
    # is one frame shorter than the state (it is the target applied at each
    # step), so it is padded by repeating the last target.
    pd_target = sbto["dof_pd_target"]
    pd_target = np.concatenate([pd_target, pd_target[-1:]], axis=0)
    out["dof_pd_target"] = resample(pd_target, src_t, dst_t)
    out["dof_pd_target_joint_names"] = np.array(joint_names)

    # Same body -> world rotation as the root above: SBTO's object_ang_vel is
    # body-local (MuJoCo qvel convention), while everything downstream treats
    # the npz's object_ang_vel as world-frame -- it is written via
    # write_root_state_to_sim (world-frame angular argument) and compared
    # against root_ang_vel_w. compose_geom_frame's lever-arm cross product
    # also needs world. Convert once, here, before either use.
    obj_body_quat = slerp(sbto["object_rot"], src_t, dst_t)
    obj_ang_vel_w = quat_rotate(obj_body_quat, resample(sbto["object_ang_vel"], src_t, dst_t))

    obj_pos, obj_rot, obj_v, obj_w = compose_geom_frame(
        sbto,
        resample(sbto["object_pos"], src_t, dst_t),
        obj_body_quat,
        resample(sbto["object_lin_vel"], src_t, dst_t),
        obj_ang_vel_w,
    )
    out["object_pos"] = obj_pos
    out["object_rot"] = wxyz_to_xyzw(obj_rot)
    out["object_lin_vel"] = obj_v
    out["object_ang_vel"] = obj_w
    for key in ("object_geom_type", "object_geom_size", "object_mass", "object_friction"):
        if key in sbto:
            out[key] = sbto[key]

    if "contact" in sbto:
        # Binary channels take the nearest source frame rather than an
        # interpolated value; the contact plan is likewise one frame short.
        contact_t = src_t[: len(sbto["contact"])]
        nearest = np.clip(np.searchsorted(contact_t, dst_t), 0, len(contact_t) - 1)
        out["contact"] = sbto["contact"][nearest]
        out["contact_ref"] = sbto["contact_ref"][nearest]
        out["contact_force"] = resample(sbto["contact_force"], contact_t, dst_t)
        out["contact_names"] = sbto["contact_names"]

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
    np.savez(args_cli.output, **out)
    print(f"[INFO] Wrote {args_cli.output}")
    for k, v in out.items():
        print(f"       {k:28s} {np.shape(v)}")


def _t(array, robot):
    """numpy row -> a torch row on the robot's device."""
    import torch

    return torch.tensor(array, dtype=torch.float32, device=robot.device).unsqueeze(0)


if __name__ == "__main__":
    main()
    simulation_app.close()
