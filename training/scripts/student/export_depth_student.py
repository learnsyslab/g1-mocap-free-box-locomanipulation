# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""Export the depth student to TorchScript, and prove the export is faithful.

deploy/run_policy.py loads the policy with torch.jit.load, so the student has to
survive JIT export. That is not obvious: it is a CNN with a spatial-softmax
head, and ONNX export is commonly skipped entirely for
image-input students. JIT is a different path and does work -- but "it exported"
is not the same as "it computes the same thing", so this checks the exported
module against the eager model on real observations and fails loudly if they
disagree.

The exported module takes TWO inputs:

    forward(obs_1d: Tensor[1, 154], obs_2d: list[Tensor[1, 1, 48, 64]]) -> Tensor[1, 29]

Copy the resulting policy.pt to deploy/policy/policy.pt.
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="G1-DummyHands-RealBox-RealCam-Finetune")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--tol", type=float, default=1e-4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from importlib import metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import g1_mocap_free_box.tasks  # noqa: F401, E402


def main() -> None:
    os.makedirs(args_cli.out_dir, exist_ok=True)
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=8)
    env = gym.make(args_cli.task, cfg=cfg)
    base = env.unwrapped
    wrapped = RslRlVecEnvWrapper(env)

    agent_cfg = handle_deprecated_rsl_rl_cfg(
        load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point"),
        metadata.version("rsl-rl-lib"),
    )
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.alg.get_policy()
    policy.eval()
    print(f"[INFO] loaded {args_cli.checkpoint}")
    print(f"[INFO] model {type(policy).__name__}")

    obs = wrapped.get_observations()
    groups = agent_cfg.obs_groups["actor" if "actor" in agent_cfg.obs_groups else "student"]
    print(f"[INFO] actor observation groups: {groups}")

    # --- export ---------------------------------------------------------------
    jit_model = policy.as_jit()
    jit_model.eval().to("cpu")
    scripted = torch.jit.script(jit_model)
    out_path = os.path.join(args_cli.out_dir, "policy.pt")
    scripted.save(out_path)
    # A second copy on the training device, purely so export fidelity can be
    # separated from cross-device float32 noise.
    scripted_gpu = torch.jit.load(out_path, map_location=base.device)
    scripted_gpu.eval()
    print(f"[INFO] wrote {out_path} ({os.path.getsize(out_path)} bytes)")

    # --- verify it computes the same thing ------------------------------------
    # Several steps, not one: a single sample can agree by luck, and the depth
    # image only varies once the scene moves.
    print("\n=== eager vs scripted ===")
    worst = 0.0
    act = torch.zeros((base.num_envs, base.action_manager.total_action_dim), device=base.device)
    for step in range(8):
        with torch.no_grad():
            eager = policy(obs).cpu()
            obs_1d = torch.cat(
                [obs[g].reshape(base.num_envs, -1) for g in groups if obs[g].ndim == 2], dim=-1
            )
            obs_2d = [obs[g] for g in groups if obs[g].ndim == 4]
            # SAME device as the eager model: comparing a CUDA eager model with a
            # CPU scripted one measures float32 kernel differences, not export
            # fidelity, and lands around 1e-4 no matter how correct the export is.
            same_dev = scripted_gpu(obs_1d, obs_2d).cpu()
            # And the deployment device, reported separately -- deploy/run_policy.py
            # loads with map_location="cpu".
            cpu_out = scripted(obs_1d.cpu(), [t.cpu() for t in obs_2d])
        diff = (eager - same_dev).abs().max().item()
        diff_cpu = (eager - cpu_out).abs().max().item()
        worst = max(worst, diff)
        print(f"  step {step}  1d{tuple(obs_1d.shape)} 2d{[tuple(t.shape) for t in obs_2d]}  "
              f"same-device {diff:.3e}   cpu-vs-cuda {diff_cpu:.3e}")
        obs, _, _, _ = wrapped.step(act)

    print(f"\nworst SAME-DEVICE disagreement {worst:.3e}  (tolerance {args_cli.tol})")
    print("cpu-vs-cuda differences of ~1e-4 are float32 kernel noise, not export error")
    if worst > args_cli.tol:
        raise SystemExit("EXPORT MISMATCH -- the scripted module does not match the eager model")

    print("\ninterface for deploy/run_policy.py:")
    print(f"  forward(obs_1d: Tensor[{obs_1d.shape[-1]}], "
          f"obs_2d: list[Tensor[{tuple(obs_2d[0].shape[1:])}]]) -> Tensor[{eager.shape[-1]}]")
    print("EXPORT OK")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
