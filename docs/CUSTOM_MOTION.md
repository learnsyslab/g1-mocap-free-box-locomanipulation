# Your own reference motion or box

The shipped policies track one reference: a human pick-up-and-carry demonstration
(OmniRetarget `sub3_largebox_005`), refined with [SBTO](https://github.com/Atarilab/sbto)
for the G1 with dummy hands and a 205 x 125 x 280 mm, 0.6 kg box. To use another motion,
or the same motion with another box, you produce a new reference with SBTO, convert it,
and train the whole chain again on it. The shipped checkpoints only fit the shipped
reference.

```
your reference (OmniRetarget-format npz: qpos = base quat + pos, 29 joints, object pose)
   │
   ├─ 1. motion-leadin    start from standing, end at rest           (this repo)
   ├─ 2. SBTO             dynamically feasible robot + box trajectory (SBTO's repo)
   ├─ 3. convert-sbto     SBTO run -> motion npz for IsaacLab          (this repo)
   ├─ 4. add-contact      hand / box / floor contact schedule          (this repo, required)
   ├─ 5. check            replay, contact, zero-residual grasp         (this repo)
   ├─ 6. train            teacher stages, then the depth student       (this repo)
   └─ 7. deploy           sim2sim, then the robot                      (this repo)
```

Run the commands below from the repository root.

## 1. Lead-in (before SBTO)

```bash
pixi run -e sim motion-leadin /path/to/my_reference.npz -o /path/to/my_reference_leadin.npz
```

Prepends 1.5 s that ease from the standing pose the locomotion policy holds into the
reference's first frame, and appends 2 s that bring it to rest (`--seconds`,
`--tail-seconds`). A clip cut from a longer capture usually starts mid-stride; the robot
cannot be placed in that state, and the policy takes over from a standing robot. The
lead-in is a tracking cost for SBTO, not a constraint, so it need not be feasible itself.
Skip it only if your reference already starts and ends standing still.

## 2. SBTO

Run SBTO on the lead-in reference; see [its README](https://github.com/Atarilab/sbto) for
how. What this repo needs from that run:

* **Robot**: SBTO's 29-DoF G1 with dummy hands (`g1_mjx_dummy_hands`, the model vendored in
  `assets/mujoco/sbto/`).
* **Object**: your box as a **box primitive** in SBTO's rollout scene, with its real size,
  mass and friction. The training task spawns a cuboid from those numbers; meshes are not
  supported.
* **Output**: the run directory with `best_trajectory.npz`, including `dof_names` (runs
  saved without it: re-save with SBTO's `scripts/add_contacts_to_run.py`). Fields read:
  `time`, `dof_names`, `dof_pos`, `dof_vel`, `dof_pd_target`, `root_pos`, `root_rot`,
  `root_lin_vel`, `root_ang_vel`, `object_pos`, `object_rot`, `object_lin_vel`,
  `object_ang_vel`, `object_geom_size`, `object_geom_pos`, `object_geom_quat`,
  `object_mass`, `object_friction`, `contact`, `contact_ref`, `contact_force`,
  `contact_names`.

Same motion with another box: run steps 1-2 on the original reference
(`robot-object/sub3_largebox_005` of the OmniRetarget dataset) with your box. The grasp
depends on the box size, so SBTO has to re-optimise it.

## 3-4. Convert

```bash
pixi run -e train convert-sbto /path/to/sbto_run -o $PWD/motions/my_motion.npz
pixi run -e sim add-contact /path/to/sbto_run --npz $PWD/motions/my_motion.npz
```

`convert-sbto` replays the trajectory kinematically in IsaacLab (50 fps) and writes per-body
frames, the PD targets, the box track and the box's size, mass and friction. `add-contact`
then measures when each hand touches the box, and the box the floor, on SBTO's geometry.
Do not skip it: without it the contact reward silently follows SBTO's kinematic plan
instead of the grasp the trajectory actually performs.

## 5. Check (before training)

```bash
G1_BOX_MOTION=$PWD/motions/my_motion.npz pixi run -e train render-reference --out_dir /tmp/my_reference
pixi run -e train check-contact --motion $PWD/motions/my_motion.npz
pixi run -e train check-grasp --motion $PWD/motions/my_motion.npz --starts 108,118,128,138
```

* `render-reference`: a video of the reference replayed kinematically; the robot and box
  should move as in SBTO. The camera is placed for the shipped motion; move it with
  `--scene_eye x y z --scene_lookat x y z` if yours walks elsewhere.
* `check-grasp` is the gate: zero-residual rollouts (SBTO's PD targets under PhysX) started
  around the grasp (`--starts`: frames shortly before the lift). The box should be lifted
  and held for some frames before the open-loop robot falls. If it is never lifted, the
  grasp does not transfer and training will not fix it; go back to SBTO.
* `check-contact` is a diagnostic: which links carry the contact in IsaacLab.

## 6. Train

`--motion` on `stage` sets `$G1_BOX_MOTION` for that run: the task loads your reference
and sizes the box from it. Start at stage 0 and chain every stage from your previous one
with `--from` (the shipped checkpoints do not apply):

```bash
M=$PWD/motions/my_motion.npz
pixi run -e train stage stage0 --motion $M
pixi run -e train stage boxyaw --motion $M --from training/logs/rsl_rl/G1_DummyHands_RealBox_Tracking_No_Obj_Terminate/<stage0 run>
pixi run -e train stage stage2 --motion $M --from training/logs/rsl_rl/G1_DummyHands_RealBox_BoxYaw/<boxyaw run>
# ... stage3, stage4, distill, finetune, finetune-stage2, each --from the previous run
```

For every other command (`play`, `evaluate`, `render`, `export`) set the variable instead:
`G1_BOX_MOTION=$M pixi run -e train evaluate ...`. Do not override
`env.commands.motion.motion_file=` on the command line: the box would keep the shipped
motion's size, mass and friction.

The box randomization of each stage (README, "How it works") is an offset from the
reference's own box pose, so it carries over; check it still makes sense for your scene in
`tracking_env_cfg.py` / `distill_env_cfg.py`. Expect the full chain to take on the order of
30-40 GPU hours (teacher ~2 h per 3000 iterations, student stages 5-9 h each, RTX 5090).

## 7. Deploy

```bash
pixi run -e train export --task G1-DummyHands-RealBox-RealCam-Finetune-Stage2 \
    --checkpoint <your finetune-stage2 model_N.pt> --out_dir /tmp/export
cp /tmp/export/policy.pt deploy/policy/policy.pt
cp motions/my_motion.npz deploy/policy/motion.npz
```

sim2sim (README, section 2) sizes its box from `deploy/policy/motion.npz`, so it needs no
change. On the robot, the real box must match the size and mass SBTO used, and be placed
roughly where the reference starts it.

## Limits

* **The camera must see the box.** The student finds the box only in the 64 x 48 depth
  image from the head (47.6 deg down, 58 deg vertical FOV, 2.5 m range). A box that leaves
  the view before the grasp, or sits beyond 2.5 m, cannot be tracked.
* **Small boxes** give a fragile grasp: the dummy hands squeeze the box between the wrists.
* **Episode start**: the policy takes over from the locomotion policy standing in front of
  the box, so the reference should start standing (step 1).
