# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""Run one stage of the teacher -> student training chain.

    pixi run -e train stage <stage> [--from CKPT_OR_RUN_DIR] [train.py args ...]

Each stage resumes from the previous one. train.py resumes from
logs/rsl_rl/<experiment>/<run>/<model_*.pt>, so this copies the starting checkpoint
there and launches train.py with the right --task / --resume flags. Without --from, a
stage starts from the checkpoint shipped in checkpoints/ (where there is one); pass
--from with your own previous run (a run directory -> its latest model_*.pt, or a .pt
file) to chain your own training. Anything after the options goes to train.py
(e.g. --num_envs 1024 --logger wandb).

    stage            task                                              starts from (default)
    stage0           G1-DummyHands-RealBox-Tracking-No-Obj-Terminate   scratch
    boxyaw           G1-DummyHands-RealBox-BoxYaw                      checkpoints/teacher_stage0_model_12100.pt
    stage2/3/4       G1-DummyHands-RealBox-BoxYaw-Stage{2,3,4}         --from the previous stage (required)
    distill          G1-DummyHands-RealBox-Distill-RealCam             checkpoints/teacher_stage4_model_24096.pt
    finetune         G1-DummyHands-RealBox-RealCam-Finetune            checkpoints/student_realcam_distill_model_5999.pt
    finetune-stage2  G1-DummyHands-RealBox-RealCam-Finetune-Stage2     checkpoints/student_realcam_finetune_model_2999.pt
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[2]
CHECKPOINTS = TRAINING.parent / "checkpoints"

# stage: (task id, experiment_name of its runner cfg, default starting checkpoint or None)
STAGES = {
    "stage0": ("G1-DummyHands-RealBox-Tracking-No-Obj-Terminate", "G1_DummyHands_RealBox_Tracking_No_Obj_Terminate", None),
    "boxyaw": ("G1-DummyHands-RealBox-BoxYaw", "G1_DummyHands_RealBox_BoxYaw", "teacher_stage0_model_12100.pt"),
    "stage2": ("G1-DummyHands-RealBox-BoxYaw-Stage2", "G1_DummyHands_RealBox_BoxYaw", None),
    "stage3": ("G1-DummyHands-RealBox-BoxYaw-Stage3", "G1_DummyHands_RealBox_BoxYaw", None),
    "stage4": ("G1-DummyHands-RealBox-BoxYaw-Stage4", "G1_DummyHands_RealBox_BoxYaw", None),
    "distill": ("G1-DummyHands-RealBox-Distill-RealCam", "G1_DummyHands_RealBox_Distill_RealCam", "teacher_stage4_model_24096.pt"),
    "finetune": ("G1-DummyHands-RealBox-RealCam-Finetune", "G1_DummyHands_RealBox_RealCam_Finetune", "student_realcam_distill_model_5999.pt"),
    "finetune-stage2": ("G1-DummyHands-RealBox-RealCam-Finetune-Stage2", "G1_DummyHands_RealBox_RealCam_Finetune_Stage2", "student_realcam_finetune_model_2999.pt"),
}


def latest_model(run_dir: Path) -> Path:
    models = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    if not models:
        raise SystemExit(f"[stage] no model_*.pt in {run_dir}")
    return models[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--from", dest="start", default=None,
                    help="starting checkpoint: a .pt file or a run directory (its latest model_*.pt)")
    args, train_args = ap.parse_known_args()

    task, experiment, default = STAGES[args.stage]
    cmd = [sys.executable, str(TRAINING / "scripts/rsl_rl/train.py"), "--headless", "--task", task]
    # pixi runs this task inside training/; resolve --from against where the user typed it.
    base = Path(os.environ.get("INIT_CWD", os.getcwd()))
    start = (base / Path(args.start).expanduser()).resolve() if args.start else (CHECKPOINTS / default if default else None)
    if args.stage != "stage0":
        if start is None:
            raise SystemExit(f"[stage] {args.stage} has no shipped starting checkpoint; pass --from "
                             "with your previous stage's run directory or model_*.pt")
        ckpt = latest_model(start) if start.is_dir() else start
        if not ckpt.is_file():
            raise SystemExit(f"[stage] starting checkpoint not found: {ckpt}")
        run = f"init_{args.stage}"
        dest = TRAINING / "logs/rsl_rl" / experiment / run
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ckpt, dest / ckpt.name)
        print(f"[stage] {args.stage}: starting from {ckpt} (copied to {dest})")
        cmd += ["--resume", "--load_run", run, "--checkpoint", ckpt.name]
    elif args.start:
        raise SystemExit("[stage] stage0 trains from scratch; --from does not apply")
    cmd += train_args
    print("[stage] " + " ".join(cmd))
    os.chdir(TRAINING)  # train.py writes logs/ relative to the working directory
    os.execv(cmd[0], cmd)


if __name__ == "__main__":
    sys.exit(main())
