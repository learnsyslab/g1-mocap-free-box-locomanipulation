# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic (NOT a gate): where does the reference pose touch the box in IsaacLab?

Kinematic replay with one physics step per frame: does the REFERENCE make
hand-box contact in IsaacLab's collision model? (contact_computed only says it
does in MuJoCo.) Reports per-link contact force on the arm links vs the npz's
contact_computed schedule. Agreement is NOT predictive of grasp transfer (large-box
95%, real-box v2 79%/73%, yet both grasp under zero residual -- see
check_reference_grasp_isaaclab.py, the actual gate). Useful for seeing which
link carries the contact.

    pixi run -e train check-contact --motion path/to/motion.npz
"""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__); p.add_argument("--task", default="G1-DummyHands-RealBox-Tracking-No-Obj-Terminate-Play")
p.add_argument("--motion", default=None, help="npz to test instead of the task's configured one")
AppLauncher.add_app_launcher_args(p); a = p.parse_args(); a.headless = True
app = AppLauncher(a).app

import gymnasium as gym, torch, numpy as np
import g1_mocap_free_box.tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry
from g1_mocap_free_box.tasks.manager_based.tracking.tracking_env_cfg import zero_replay_randomization

cfg = load_cfg_from_registry(a.task, "env_cfg_entry_point")
zero_replay_randomization(cfg)
if a.motion:
    cfg.commands.motion.motion_file = a.motion
    cfg._sync_object_to_motion()  # box size/mass/friction follow the npz, as in training
cfg.scene.num_envs = 1; cfg.commands.motion.start_frame = 0; cfg.episode_length_s = 1e9
env = gym.make(a.task, cfg=cfg); u = env.unwrapped; env.reset()
cmd = u.command_manager.get_term("motion")
robot, obj, sensor = u.scene["robot"], u.scene["object"], u.scene["contact_forces"]
ids = torch.arange(1, device=u.device)
T = int(cmd.motion.time_step_total)
names = list(sensor.body_names)
want = [n for n in names if ("wrist" in n or "elbow" in n)]
idx = [names.index(n) for n in want]
cref = cmd.motion.contact_ref.cpu().numpy()  # (T,3): static, left, right  (contact_computed)

F = np.zeros((T - 1, len(want))); minz = np.zeros(T - 1); dist_lw = np.zeros(T - 1); dist_rw = np.zeros(T - 1)
bn = list(robot.body_names); L = bn.index("left_wrist_yaw_link"); R = bn.index("right_wrist_yaw_link")
for t in range(T - 1):
    cmd.time_steps[:] = t
    robot.write_joint_state_to_sim(cmd.joint_pos, cmd.joint_vel, env_ids=ids)
    robot.write_root_state_to_sim(torch.cat([cmd.body_pos_w[:, 0], cmd.body_quat_w[:, 0],
                                             cmd.body_lin_vel_w[:, 0], cmd.body_ang_vel_w[:, 0]], -1), env_ids=ids)
    obj.write_root_state_to_sim(torch.cat([cmd.object_pos_w, cmd.object_quat_w,
                                           cmd.object_lin_vel_w, cmd.object_ang_vel_w], -1), env_ids=ids)
    u.sim.step(render=False); u.scene.update(u.sim.get_physics_dt())
    F[t] = sensor.data.net_forces_w[0, idx].norm(dim=-1).cpu().numpy()
    op = obj.data.root_pos_w[0]; bp = robot.data.body_pos_w[0]
    dist_lw[t] = torch.norm(bp[L] - op).item(); dist_rw[t] = torch.norm(bp[R] - op).item()
    minz[t] = op[2].item()

thr = 1.0
print("\n" + "=" * 70); print(f"KINEMATIC CONTACT  |  {a.task}  |  {T} frames"); print("=" * 70)
print(f"{'link':<26}{'frames |F|>1N':>14}{'first':>8}{'last':>7}{'maxF[N]':>10}")
for j, n in enumerate(want):
    on = np.where(F[:, j] > thr)[0]
    print(f"{n:<26}{len(on):>14}{(on[0] if len(on) else -1):>8}{(on[-1] if len(on) else -1):>7}{F[:, j].max():>10.1f}")
for side, ch in (("left", 1), ("right", 2)):
    cols = [j for j, n in enumerate(want) if n.startswith(side) and "wrist" in n]
    sim_on = (F[:, cols] > thr).any(1); ref_on = cref[: T - 1, ch] > 0
    both = sim_on & ref_on
    print(f"\n{side}: contact_computed frames {ref_on.sum()}  |  IsaacLab wrist-contact frames {sim_on.sum()}"
          f"  |  overlap {both.sum()}  |  agreement {(sim_on == ref_on).mean() * 100:.1f}%")
print(f"\nobject centre z during replay: min {minz.min():.3f} max {minz.max():.3f}")
print("=" * 70)
env.close(); app.close()
