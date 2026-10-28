# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""Gate an SBTO-derived motion BEFORE training: does the reference grasp transfer?

Zero-residual rollouts (= SBTO's own dof_pd_target under PhysX, terminations
off) pinned at several start frames around the grasp, one env per start. Per
start: peak box lift, frames the box tracked the reference while lifted, and
whether the box was dropped while the robot was still upright. The open-loop
robot itself falls after ~60-90 steps -- that is the gap the residual policy
covers -- so judge the grasp on the window before the fall.

Reference numbers (250 steps, starts ~105..140): the large-box motion that
trained to 60% held the box for 12/21/45/66 frames; sub3_realbox_v2 held
22/15/36/40. Kinematic pose penetration and PD-target penetration were tried
as gates first and DO NOT discriminate -- both are sub-millimetre for working
and failing references alike, because SBTO's pair contacts are stiff.

    pixi run -e train check-grasp --motion path/to/motion.npz --starts 108,118,128,138
"""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--task", default="G1-DummyHands-RealBox-Tracking-No-Obj-Terminate-Play"); p.add_argument("--motion", default=None)
p.add_argument("--starts", type=str, required=True); p.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(p); a = p.parse_args(); a.headless = True; app = AppLauncher(a).app
import gymnasium as gym, torch, numpy as np
import g1_mocap_free_box.tasks  # noqa
from isaaclab_tasks.utils import load_cfg_from_registry
from g1_mocap_free_box.tasks.manager_based.tracking.tracking_env_cfg import zero_replay_randomization
starts = [int(x) for x in a.starts.split(",")]; N = len(starts)
cfg = load_cfg_from_registry(a.task, "env_cfg_entry_point"); zero_replay_randomization(cfg)
if a.motion: cfg.commands.motion.motion_file = a.motion; cfg._sync_object_to_motion()
cfg.scene.num_envs = N; cfg.episode_length_s = 1e9
for k, v in list(vars(cfg.terminations).items()):
    if v is not None and k != "time_out" and not k.startswith("_"): setattr(cfg.terminations, k, None)
env = gym.make(a.task, cfg=cfg); u = env.unwrapped
cmd = u.command_manager.get_term("motion")
st = torch.tensor(starts, device=u.device)
cmd._adaptive_sampling = lambda env_ids: cmd.time_steps.__setitem__(env_ids, st[env_ids])
env.reset()
obj = u.scene["object"]; sensor = u.scene["contact_forces"]; robot = u.scene["robot"]
names = list(sensor.body_names); L = names.index("left_wrist_pitch_link"); R = names.index("right_wrist_pitch_link")
P = list(robot.body_names).index("pelvis")
act = torch.zeros((N, u.action_space.shape[1]), device=u.device)
z0 = obj.data.root_pos_w[:, 2].clone(); rz0 = cmd.object_pos_w[:, 2].clone()
peak = torch.zeros(N, device=u.device); held = torch.zeros(N, device=u.device)
lifted = torch.zeros(N, dtype=torch.bool, device=u.device)
dropped = torch.full((N,), -1, device=u.device); fell = torch.full((N,), -1, device=u.device)
for s in range(a.steps):
    env.step(act)
    z = obj.data.root_pos_w[:, 2]; rz = cmd.object_pos_w[:, 2]; lift = z - z0; rlift = rz - rz0
    pz = robot.data.body_pos_w[:, P, 2]
    peak = torch.maximum(peak, lift); lifted |= lift > 0.30
    held += ((rlift > 0.20) & ((z - rz).abs() < 0.10) & (fell < 0)).float()
    fell = torch.where((fell < 0) & (pz < 0.35), torch.full_like(fell, s), fell)
    dropped = torch.where((dropped < 0) & lifted & (lift < 0.15) & (pz > 0.50), torch.full_like(dropped, s), dropped)
print("\n" + "=" * 74); print(f"ZERO-RESIDUAL SWEEP  |  {a.task.split('-')[2] if 'RealBox' in a.task else 'largebox'}  |  {a.steps} steps")
print(f"{'start':>6}{'peak lift m':>12}{'held frames':>12}{'dropped upright @':>18}{'robot fell @':>13}")
for i, s0 in enumerate(starts):
    print(f"{s0:>6}{peak[i].item():>12.3f}{int(held[i].item()):>12}{int(dropped[i].item()):>18}{int(fell[i].item()):>13}")
print("held = steps with ref lift>0.2 and |box-ref| z <10cm, before any fall;  dropped = box fell back below 15cm lift while robot upright")
print("=" * 74); env.close(); app.close()
