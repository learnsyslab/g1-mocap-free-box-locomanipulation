# g1-mocap-free-box-locomanipulation

https://github.com/user-attachments/assets/b0faae6b-7b8b-404a-8bf3-21f9d36052a1

Motion-capture-free whole-body box loco-manipulation on the Unitree G1: depth camera + proprioception only.

A Unitree G1 (29 DoF) picks up a 205 × 125 × 280 mm, 0.6 kg box from the floor and
places it, guided only by its **head depth camera and proprioception**. At deployment
there is no motion capture, no object pose estimator and no robot position: the policy
sees a 64 × 48 depth image, the IMU and the joint encoders.

**The final depth student at four extreme box placements** (simulation, IsaacLab): box 0.45 m
further away, 0.40 m to the side, rotated 90°, and lying on its side.

https://github.com/user-attachments/assets/25a38868-5fd5-40d1-a547-4c8acb9204dc

**The reference motion** every policy tracks: one human demonstration, retargeted and refined
for the G1, replayed kinematically with the box.

https://github.com/user-attachments/assets/faa6211f-ae72-40c1-9a7d-478058e049b6

This repository contains everything for that one task:

| | |
|---|---|
| `training/` | IsaacLab tasks + scripts: privileged **teacher** (PPO) and depth **student** (DAgger, then PPO fine-tuning) |
| `checkpoints/` | the trained teacher and student checkpoints of every stage, to resume any stage or skip training |
| `deploy/` | a minimal runtime that runs the student on the G1 over Unitree DDS, and a MuJoCo **sim2sim** simulator that speaks the same DDS interface |
| `assets/` | G1 robot model: USD for IsaacLab, MJCF + meshes for MuJoCo |

## How it works

```
reference motion (one human demo from OmniRetarget/OMOMO, refined with SBTO, 50 fps, 9.9 s)
        │
        ▼
TEACHER  privileged PPO tracking policy: sees the box pose (in sim, as a mocap system would)
  stage 0  Tracking-No-Obj-Terminate   whole-body tracking of the reference
  stage 1  BoxYaw                      box pose randomised around the demo: yaw ±29°,  xy ±2 cm
  stage 2  BoxYaw-Stage2               yaw ±60°,  xy ±5 cm
  stage 3  BoxYaw-Stage3               yaw ±90°,  xy ±12 cm
  stage 4  BoxYaw-Stage4               yaw ±180°, xy ±30 cm
        │  distil (DAgger)
        ▼
STUDENT  same action space, NO privileged state
  Distill-RealCam            imitate the teacher from depth + proprioception (box xy ±20 cm, any yaw)
  RealCam-Finetune           PPO with a decaying teacher (kickstarting) term, same box range
  RealCam-Finetune-Stage2    PPO on a wider box range (x -0.40..0.60 m, y -0.50..0.20 m, any of six resting faces)
        │  TorchScript export
        ▼
DEPLOY   50 Hz loop over Unitree DDS: a locomotion policy stands / walks the robot (gamepad or
         keyboard); on "start" it blends into the student, which runs the 9.9 s motion from
         LowState + depth, then hands back to locomotion
```

What makes it run without motion capture:

* **No position anywhere in the observation.** The student observes the reference
  joint command, the torso *orientation* error to the reference (from the pelvis IMU and
  the waist joints), the gyro, joint positions/velocities, its last action, and depth.
  It never needs to know where it, or the box, is in the world.
* **The box is found in the depth image.** The CNN student (spatial-softmax head) learns
  where the box is relative to the head camera; randomising the box pose during training
  forces it to look rather than replay the demo.
* **Depth matches the real sensor.** The simulated camera is the G1's `d435_link`
  (left IR imager), 58° vertical FOV, 64 × 48, pitched **47.6°** down. Training corrupts
  depth with the real pipeline's artefacts: per-pixel noise, salt, rectangular dropout,
  held (repeated) frames, invalid → far, and 0–2 control steps (0–40 ms) of latency.

### Observation / action contract (student)

| slice | dim | term |
|---|---|---|
| 0:58 | 58 | reference joint position (29) + joint velocity (29) at frame k |
| 58:64 | 6 | torso orientation error, 6-D rotation: `rot6d(conj(q_torso) ⊗ q_torso_ref[k])` |
| 64:67 | 3 | pelvis IMU gyro |
| 67:96 | 29 | joint position − default |
| 96:125 | 29 | joint velocity |
| 125:154 | 29 | last action |
| depth | 1×48×64 | metric depth, invalid → 2.5 m, clipped to [0.168, 2.5] m, ÷ 2.5 |

Joint terms are in IsaacLab joint order. The action (29, IsaacLab order) is a residual on
the reference: `q_des = dof_pd_target[k] + action_scale ⊙ action`, sent with the training
PD gains. The exported policy is
`forward(obs: Tensor[1,154], depth: list[Tensor[1,1,48,64]]) -> Tensor[1,29]`.
`deploy/g1_box_deploy/policy_io.py` is the reference implementation of this contract.

## Results

Student `RealCam-Finetune-Stage2` model_8998 (`deploy/policy/policy.pt`):

* **IsaacLab**, 512 episodes from frame 0, box upright within ±0.2 m of the demo
  position: no falls. Mean peak box position error 24 cm; 21 % of episodes keep the box
  within 10 cm / 25° of the reference at every step (a strict tracking gate: the student
  places the box where it sees it, not where the demo put it).
* **MuJoCo sim2sim** through the deploy runtime and DDS: completes the motion and lifts
  the box 0.57 m (reference: 0.64 m).

## 0. Setup

Everything installs with [pixi](https://pixi.sh) >= 0.80 (`pixi self-update`), on Linux x86-64,
from the shipped `pixi.lock`. Three environments:

| env | for | needs |
|---|---|---|
| `sim` | deploy runtime + MuJoCo sim2sim | CPU only |
| `real` | deploy runtime on the robot + head-camera depth | NVIDIA GPU (CUDA 12) |
| `train` | IsaacLab teacher / student training | NVIDIA GPU, glibc >= 2.35 |

```bash
git clone --recurse-submodules <this repository>   # or, after a plain clone: git submodule update --init
cd g1-mocap-free-box-locomanipulation
pixi install -e sim                                 # minutes; CycloneDDS, unitree_sdk2py, MuJoCo, torch (CPU)
pixi install -e train && pixi run -e train setup    # once: torch, Isaac Sim 6.0.1, IsaacLab, this package (~20 GB)
```

Submodules: IsaacLab (`training/third_party/IsaacLab`, pinned to the release we trained
with), unitree_sdk2_python and Fast-FoundationStereo (`deploy/third_party/`).

Hardware we used, and what depends on it (yours will differ; see `deploy/REAL_ROBOT.md`):

* **GPU for the head-camera depth**: the Fast-FoundationStereo engine takes 14 ms per frame on
  an RTX 5090 and 37 ms on an RTX 3080. The policy expects ~50 Hz depth with little lag, so a
  3080-class GPU is the minimum (about 30 Hz); build the engine on the GPU that runs it.
* **D435 USB link**: through a USB hub, keep it at 5 Gb/s (10 Gb/s re-enumerated the camera in
  our setup); the depth driver prints the negotiated speed.
* **Network**: the PC on the robot's `192.168.123.x` network; set `G1_NETWORK_INTERFACE` to that
  interface's name.
* **Camera library**: pyrealsense2 built with the RSUSB backend (`pixi run -e real
  build-pyrealsense`), because the PyPI wheels' V4L2 backend failed on our kernel; try the wheel
  first if you like.
* **Disk**: ~30 GB for the `train` environment.

## 1. Training (IsaacLab)

Tested with Isaac Sim 6.0.1, IsaacLab v3.0.0-beta2.patch1 (PhysX backend), rsl-rl-lib 5.0.1,
torch 2.10, Python 3.12, on one RTX 5090. Logs go to `training/logs/rsl_rl/<experiment>/<run>`.

The chain runs one stage at a time with `pixi run -e train stage <stage>`. Each stage resumes
from the previous one: by default from the checkpoint shipped in `checkpoints/`, or from your
own previous run with `--from <run dir or model_*.pt>`. Anything else on the command line goes
to the trainer (e.g. `--num_envs 1024`, `--logger wandb`).

```bash
# ---- teacher (privileged; 8192 envs, ~2 h per 3000-iteration stage on an RTX 5090) ----
pixi run -e train stage stage0                                   # from scratch, ~12k iterations (shipped: model_12100)
pixi run -e train stage boxyaw                                   # from the shipped stage 0
pixi run -e train stage stage2 --from training/logs/rsl_rl/G1_DummyHands_RealBox_BoxYaw/<boxyaw run>
pixi run -e train stage stage3 --from training/logs/rsl_rl/G1_DummyHands_RealBox_BoxYaw/<stage2 run>
pixi run -e train stage stage4 --from training/logs/rsl_rl/G1_DummyHands_RealBox_BoxYaw/<stage3 run>   # shipped: model_24096

# ---- student (depth camera; 4096 envs) ----
pixi run -e train stage distill                                  # DAgger from the shipped Stage 4 teacher, 6000 it, ~8.5 h
pixi run -e train stage finetune                                 # PPO + teacher kickstart, 3000 it, ~5 h
pixi run -e train stage finetune-stage2                          # wider box range, six faces, 6000 it, ~9 h
```

`pixi run -e train stage --help` lists the stages and their tasks. Every task also has a
`-Play` variant (1 env, no noise) for `pixi run -e train play --task <task>-Play`.

### Evaluate, render, export

```bash
# these tasks run in training/, so the checkpoint paths are relative to it
# success-rate evaluation (512 episodes, every episode starts at frame 0 of the demo)
pixi run -e train evaluate --task G1-DummyHands-RealBox-RealCam-Finetune-Play \
    --checkpoint ../checkpoints/student_realcam_stage2_model_8998.pt --num_envs 512 --start_frame 0 \
    --object_symmetry 2fold
# video of one rollout: scene view + the student's depth input (--box_x/--box_y/--box_yaw/--box_rest
# pin the box placement)
pixi run -e train render --checkpoint ../checkpoints/student_realcam_stage2_model_8998.pt --out_dir /tmp/rollout
# the videos at the top of this README: the final student at four extreme box placements (2x2 tile),
# and the reference motion replayed kinematically
pixi run -e train render-showcase --out_dir /tmp/showcase
pixi run -e train render-reference --out_dir /tmp/reference
# TorchScript export, verified against the eager model; copy policy.pt to deploy/policy/
pixi run -e train export --task G1-DummyHands-RealBox-RealCam-Finetune-Stage2 \
    --checkpoint ../checkpoints/student_realcam_stage2_model_8998.pt --out_dir /tmp/export
```

### Your own reference motion or box

[docs/CUSTOM_MOTION.md](docs/CUSTOM_MOTION.md): where SBTO fits in, converting its output
(`motion-leadin`, `convert-sbto`, `add-contact`), checking the reference before training,
and training and deploying on it (`stage --motion`, `$G1_BOX_MOTION`).

## 2. Sim2sim (MuJoCo, no robot needed)

```bash
pixi run -e sim sim2sim                      # terminal 1: MuJoCo viewer
pixi run -e sim run-sim --input keyboard     # terminal 2 (or --input gamepad)
# headless, no operator: stand 3 s, then the box motion
#   MUJOCO_GL=egl pixi run -e sim sim2sim --headless --duration 28 &
#   pixi run -e sim run-sim --input script:3
```

Controls (`--input` of `run-sim` / `run-real`):

| | gamepad (Bluetooth or USB) | keyboard (the run_policy terminal) |
|---|---|---|
| walk / turn | hold **LT** (deadman) + left stick, right stick to turn | W/S, A/D, Q/E (0.1 steps), Space = stop |
| start the box motion | **Y** | `[` |
| back to locomotion | **B** | `]` |
| damping (limp, then exit) | **LT + RT** together | `0` |

The gamepad layout is read from SDL's controller database (`--pad_layout auto`); if your pad
is not in it, pass `--pad_layout bluetooth` or `xpad` (USB). The program refuses to start if
the sticks or triggers do not read "at rest", which is what a wrong layout looks like, so
start it with the pad untouched.

`sim2sim.py` is modelled on Unitree's `unitree_mujoco`: it publishes `rt/lowstate` at
200 Hz and the head-camera depth on `rt/camera/depth` at 30 Hz, and applies `rt/lowcmd`
as a 1 kHz PD law, on DDS domain 1 over the loopback interface. `run_policy.py` cannot
tell it from the robot. At exit it prints the box's lift and final position.

The locomotion policy (`deploy/policy/locomotion.pt`, a 50 Hz velocity tracker trained in
IsaacLab with 5 frames of IMU + joint history) is our first G1 locomotion policy; it is
included so the robot can stand and walk up to the box, not as a contribution of this
repository.

## 3. Real robot

See [deploy/REAL_ROBOT.md](deploy/REAL_ROBOT.md): the head-camera depth pipeline
(`deploy/realsense_depth.py`, Fast-FoundationStereo as a TensorRT engine), setting the
articulated head to the trained 47.6 deg with the level meter (`pixi run -e real head-angle`),
network setup and the bring-up procedure with the gamepad.

## Repository layout

```
pixi.toml                        environments (sim / real / train) and tasks
assets/isaaclab/g1-29dof/        G1 USD (primitive collision shapes) used for training
assets/mujoco/                   G1 MJCF (camera at 47.6°), meshes, box, scene_box.xml
assets/mujoco/sbto/              SBTO's G1 model, for the contact schedule of a new reference
training/source/g1_mocap_free_box/  IsaacLab extension: tasks, MDP terms, student model + algorithm
  .../tracking/__init__.py         task registry (the pipeline above)
  .../tracking/tracking_env_cfg.py teacher environments
  .../tracking/distill_env_cfg.py  student environments, depth camera, depth corruption
  .../tracking/motions/            the reference motion
training/scripts/                train / play / eval / render / export
training/scripts/motion/         SBTO run -> motion npz, lead-in, reference checks
docs/CUSTOM_MOTION.md            training on your own reference motion or box
training/third_party/IsaacLab/   submodule
checkpoints/                     teacher stage 0 + 4, student stages 1-3
deploy/run_policy.py             the 50 Hz runtime (sim or robot): locomotion <-> box student
deploy/sim2sim.py                MuJoCo simulator with the robot's DDS interface
deploy/realsense_depth.py        head D435 -> Fast-FoundationStereo (TensorRT) -> rt/camera/depth
deploy/camera_d435.toml          camera / engine / publish settings for realsense_depth.py
deploy/head_angle.py             head-camera level meter: measured pitch vs the trained 47.6 deg
deploy/g1_box_deploy/            joint layout + gains, observation/action, camera contract,
                                 locomotion, gamepad/keyboard, DDS I/O
deploy/policy/                   policy.pt (student) + motion.npz (the reference) + locomotion.pt
deploy/third_party/              submodules: unitree_sdk2_python, Fast-FoundationStereo
```

## Acknowledgements

* The reference motion is `sub3_largebox_005` from the
  [OmniRetarget dataset](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset)
  (retargeted from [OMOMO](https://github.com/lijiaman/omomo_release)), refined with
  Atarilab's SBTO for the dummy hands and the real box.
* The tracking MDP is adapted from the BeyondMimic-style `tasks/mimic` task of Unitree's
  [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab), itself based on
  [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking).
* The depth student's distillation (spatial-softmax CNN, NLL distillation, PPO with teacher
  kickstarting) is adapted from Atarilab's mjlab-distill (Majid Khadiv, Shafeef Omar), on top
  of [rsl_rl](https://github.com/leggedrobotics/rsl_rl).
* The robot model is Unitree's G1 description; `sim2sim.py` follows the DDS conventions of
  [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco).

## License

Apache-2.0, see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Third-party code, models and
data keep their own licenses, listed with what was used in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Author: Jim Yun-Jin Li, Learning Systems and Robotics Lab (LSY), Technical University of Munich.
