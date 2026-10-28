# Agent notes for g1-mocap-free-box-locomanipulation

A Unitree G1 walks to a box, picks it up and carries it with no motion capture:
a privileged teacher trained in IsaacLab, distilled into a depth + IMU student, and
a minimal runtime that runs it on the robot (or in MuJoCo) over Unitree DDS.
README.md has the pipeline and commands, `deploy/REAL_ROBOT.md` the bring-up
procedure. This file is what an agent needs on top of them.

## Real robot safety

- Nothing that commands hardware (`pixi run -e real run-real`, anything with
  `--target real`) without the user's explicit go for that run. Read-only steps are
  fine: `realsense`, `depth-view`, `head-angle`, `engine-check`, and
  `run-real --dry-run` (reads the robot, builds every observation, sends nothing).
- `--target real` asks the operator to type `real` before the first command
  (`deploy/g1_box_deploy/dds_io.py`). Never script past it: no `--yes`, no piped input.
- Bus: sim = DDS domain 1 on `lo`, real = domain 0 on `$G1_NETWORK_INTERFACE`
  (or `--interface`). Before starting `sim2sim` or `run-sim`, check `ps` for another
  sim or controller on domain 1.
- Stop: LT + RT on the gamepad (keyboard `0`) or Ctrl-C, which means 0.5 s damping, then exit.

## Fixed contracts (change only together with a retrained policy)

- **Student observation and depth**: README "Observation / action contract";
  `deploy/g1_box_deploy/policy_io.py` is the reference implementation and
  `deploy/g1_box_deploy/camera.py` (`CONTRACT`) the depth spec: 64 x 48, 58 deg
  vertical FOV, invalid -> far (2.5 m), clipped to [0.168, 2.5] m, / 2.5.
- **Gains and action scales**: `deploy/g1_box_deploy/layout.py` (`KP_DDS`,
  `KD_DDS`, `ACTION_SCALE_DDS`) are the training actuator model.
- **Head camera**: d435_link (left IR imager) 47.6 deg down, the stock upright
  head. The G1 head is articulated, so before every real run the head is set to
  that pitch with `pixi run -e real head-angle` (OK within 0.5 deg); training
  randomizes pitch by only +-0.5 deg.
- **Joint order**: policy terms are in IsaacLab order and DDS in Unitree order;
  `layout.py` holds the maps. Never index one with the other.
- **Reference motion and box**: one npz carries both; the task sizes the box from it
  (`_sync_object_to_motion`). Switch motions with `$G1_BOX_MOTION` or `stage --motion`,
  never with `env.commands.motion.motion_file=` (the box would keep the old size).
  New motions: `docs/CUSTOM_MOTION.md`.
- The shipped `deploy/policy/policy.pt` is `RealCam-Finetune-Stage2` model_8998
  (`checkpoints/student_realcam_stage2_model_8998.pt`); `locomotion.pt` is the
  walking policy used before and after the box motion.

## Repo conventions

- Environments are pixi only (pixi >= 0.80): `sim` (MuJoCo, CPU torch), `real`
  (CUDA torch, pyrealsense2, TensorRT depth), `train` (Isaac Sim 6.0.1, IsaacLab
  submodule, rsl-rl 5.0.1). Add dependencies to `pixi.toml`, never with ad-hoc pip.
  The train env installs through `training/constraints.txt` (`PIP_CONSTRAINT`);
  keep new pins there.
- Third-party code is a git submodule (IsaacLab, Fast-FoundationStereo,
  unitree_sdk2_python) and is not edited in place. Assets are copied in.
- Licence: Apache-2.0. New source files start with the SPDX header used in the
  existing files; third-party attributions go in `THIRD_PARTY_NOTICES.md`.
- This repo is public: no person names, home directories, host names or
  internal run links in code or docs. Paths are arguments, env vars (`FFS_ENGINE`,
  `G1_NETWORK_INTERFACE`) or repo-relative.
- Commit messages: no Co-Authored-By / session attribution lines. Commit and push
  only when asked.

## Checks after a change

- Deploy side (no robot): headless sim2sim end to end,
  `MUJOCO_GL=egl pixi run -e sim sim2sim --headless --duration 28 &` then
  `pixi run -e sim run-sim --input script:3` (stands 3 s, runs the box motion).
  `--dry-run` on either target runs everything but publishes no command.
  To watch the depth image the student gets in sim: `pixi run -e sim depth-view --target sim`
  (listen-only; without `--target sim` it listens on the robot bus).
- Depth pipeline: `pixi run -e real engine-check` (no camera needed).
- Training side: `pixi run -e train check`, then one iteration of the touched stage
  (`pixi run -e train stage <stage> --max_iterations 1`, see `stage --help`).
- Ask before full training runs; the student stages take 5-9 h each.

## Traps

- **pyrealsense2 from PyPI** is built with the V4L2 backend and cannot talk to the
  G1's D435; build it with `pixi run -e real build-pyrealsense` (RSUSB backend).
- The TensorRT engine is built per GPU and TensorRT version (`build-engine`); point
  at an existing one with `--engine` or `FFS_ENGINE`.
- IsaacLab rendering over SSH with X11 forwarding fails with X `BadValue` errors:
  `unset DISPLAY XAUTHORITY` before render scripts.
- Kinematic replay (`render_reference_motion.py`) must write the states and then
  `env.step` a zero action; writing state and only rendering leaves the robot frozen.
- Videos for the README: H.264 (`libx264 -pix_fmt yuv420p -movflags +faststart`),
  not H.265, which browsers show as a black picture with sound.
