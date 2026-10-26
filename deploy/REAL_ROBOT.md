# Running the box student on a real G1

`run_policy.py --target real` speaks the same DDS interface as `sim2sim.py`, so the only
differences from sim2sim are the network, the depth source, the camera mount and a 2 s
(instead of 0.5 s) startup ramp, done in the gantry.
Run sim2sim first and make sure it lifts the box on your machine.

## What you need

* Unitree G1, 29 DoF (`mode_machine` 5), firmware with the `unitree_hg` low-level interface.
  Hands are not used: the policy was trained with rigid box-shaped "dummy hands".
* The head Intel RealSense D435(i) (`d435_link`), with the **head set to 47.6° pitch down** (see below).
* A PC on the robot's network (`192.168.123.x`) running `run_policy.py` (CPU is enough:
  ~0.2 ms per inference with 2 threads) and the depth pipeline `realsense_depth.py` (NVIDIA
  GPU, see below); one PC with a GPU can run both.
* The box: 205 × 125 × 280 mm, ~0.6 kg, standing on its 205 × 125 face. In the demo it
  starts 0.32 m in front of the pelvis and 7 cm to the right (`object_pos[0]` in
  `policy/motion.npz`); the student was trained with the box anywhere within ±0.2 m of
  that (wider in Stage 2) and at any yaw, so place it roughly there.
* A gantry or harness, and someone holding the e-stop / remote.

## 1. Camera pitch: 47.6° down

The policy has no other idea of where the camera looks, so the camera pitch has to match
training. The camera frame is `d435_link` = the **left IR imager**, at
`(0.0576, 0.0175, 0.4299)` m in `torso_link`, pitched 47.6° about +Y (quaternion
`wxyz = (0.9150, 0, 0.4035, 0)`), the G1's nominal head pose. A head left at another
angle (we once ran at ~62°) made the robot overshoot the box by ~23 cm and never grasp it.

The head is tilted **by hand** (it holds its angle by friction), so set and check it before
every session, without motion capture, with the level meter:

```bash
pixi run -e real realsense                      # terminal 1: depth (see section 2)
pixi run -e real head-angle                     # terminal 2: prints once a second
# pitch  53.4 deg  target  47.6  ->  tilt head UP 5.8 deg   roll +0.6  height 1.27 m  floor 2650 px
```

Robot standing still (or in the gantry with the feet on the floor), camera seeing open, flat
floor. Tilt the head until it prints `OK (within 0.5 deg)`, let go, and check it again after
2 s. It fits the floor plane in the depth image (RANSAC) and compares it with "up" from the
pelvis IMU and the waist joints, so it needs no motion capture; `height` (camera above the
floor, ~1.27 m standing) and `floor` (pixels on the plane) tell you whether it really found the
floor. It only listens, so it is safe to run next to everything else. Checked in sim2sim, where
the true pitch is known: it reads 47.6 deg for the trained mount and 55.0 deg (-> "tilt head UP
7.4 deg") for a camera tilted to 55 deg.

## 2. Depth stream contract

The runtime subscribes to **`rt/camera/depth`**, type `sensor_msgs.msg.dds_.Image_`
(ROS 2 layout, defined in `g1_box_deploy/image_msg.py`), on the same DDS domain as the robot:

| field | value |
|---|---|
| `encoding` | `32FC1`, metric depth in **metres** along the camera axis (z-depth), little-endian, `step = 4·width` |
| size | **64 × 48** |
| intrinsics | pinhole **fx = fy = 43.30, cx = 32, cy = 24** (58.0° vertical, 72.9° horizontal FOV), undistorted |
| frame | left IR imager (`d435_link`) |
| invalid pixels | `NaN` (or inf). The runtime maps them to far (2.5 m), as in training |
| `header.stamp` | **capture time** (wall clock of the publisher). Frames older than `--max_depth_age` (0.15 s) fault the run |
| rate | ≥ 30 Hz (the policy runs at 50 Hz and uses the newest frame) |

### Producing it: `realsense_depth.py`

`deploy/realsense_depth.py` is the pipeline we run on the robot: IR1 + IR2 at 848 × 480,
60 fps, **projector off** → Fast-FoundationStereo as a TensorRT engine (fp16, 640 × 480
input, 8 refinement iterations, ~13 ms on an RTX 5090) → disparity to metric depth
`Z = fx·baseline / d` in the left-IR frame → one nearest-neighbour ray resample from the
sensor intrinsics onto the 64 × 48 pinhole above → `rt/camera/depth` at 50 Hz, stamped with
the sensor's capture time. Measured age at publish: 22 ms median, 30 ms p95. Settings are in
`deploy/camera_d435.toml`; the GPU can be on the PC that runs `run_policy.py`.

One-time setup, on the machine the camera is plugged into (`real` pixi environment):

1. **pyrealsense2 with the RSUSB backend**: `pixi run -e real build-pyrealsense`. Every PyPI
   `pyrealsense2` wheel uses the V4L2 backend, which on our kernel fails with
   `xioctl(UVCIOC_CTRL_QUERY) failed ... Protocol error`; the script builds librealsense
   v2.58.3 with `-DFORCE_RSUSB_BACKEND=ON` (libusb) and puts it first on the path. If the
   wheel works on your machine,
   `pixi run -e real python -m pip install pyrealsense2==2.58.3` is enough.
2. **Weights**: download NVIDIA's research checkpoint `23-36-37` (link in
   `deploy/third_party/Fast-FoundationStereo/README_ORIG.md`) into
   `deploy/third_party/Fast-FoundationStereo/weights/23-36-37/`.
3. **Engine**: `pixi run -e real build-engine`. Exports the ONNX (640 × 480, 8 iterations,
   max disparity 192) and builds an fp16 TensorRT 10.11 engine in the submodule's own pixi
   environment. An engine only runs on the GPU model and TensorRT version that built it,
   so build it on the deployment machine.
4. **Check**: `pixi run -e real engine-check` (no camera). Runs the reference stereo pair at
   two resolutions and fails unless metric depth is resolution-invariant -- the guard against
   a disparity-rescale error, which produces a plausible image with every distance wrong.

**Already have an engine** on this machine (built with TensorRT 10.11.0.33 for this GPU)? Skip
steps 2-3 and point the driver at it, per command or once per shell:

```bash
export FFS_ENGINE=/path/to/fast_foundationstereo.engine    # or: --engine /path/to/...
pixi run -e real engine-check                              # still worth running once
```

Bring-up, camera plugged in:

```bash
pixi run -e real realsense --backend rs --dry-run                 # geometry, one frame, no publish
pixi run -e real realsense --backend rs                           # D435's own depth, projector on
pixi run -e real realsense                                        # Fast-FoundationStereo
```

`--backend rs` exercises the whole chain (topic, format, resample, `run_policy.py`'s depth
freshness) with no network in the loop, so plumbing problems and model problems stay
separate. Every 5 s the driver prints its rate, compute time, age at publish and the
fraction of valid pixels; it also prints the USB link speed (see below).

The training noise model matches stereo depth like this (thin, near-object holes; little
temporal noise). The D435's own ASIC depth also fits the contract (enable the projector and
resample the same way) and is fine for bring-up, but its noise and holes differ from what
the policy was trained on.

Practical notes from our setup:

* The camera is wired to the workstation over USB. Through a USB hub, make sure it
  negotiates 5 Gb/s: our VL822 hub at 10 Gb/s re-enumerated the camera every few seconds.
  Running the camera from the robot's onboard computer and streaming the IR pair over
  Ethernet was not reliable for us (the camera dropped off its USB within minutes).
* Fast-FoundationStereo is NVIDIA's (https://github.com/NVlabs/Fast-FoundationStereo), under
  NVIDIA's own source-code license: **non-commercial / research use only**. It is included as
  a git submodule (our fork with the pixi / TensorRT build tasks); no weights or engines are
  included.
* Measure the depth rate as the policy sees it (fraction of 50 Hz ticks with a new image;
  ours is 96-99 %), not with a DDS recorder, which can drop large image messages.

Check the stream before the robot moves: after START, `run_policy.py` prints
`WAIT: holding frame 0 until a fresh depth image arrives` and stays there until a frame
younger than `--max_depth_age` arrives.

## 3. Network

```bash
ip -br addr | grep 192.168.123        # the interface on the robot's network, e.g. enp3s0
ping 192.168.123.161                  # the robot's locomotion computer
export G1_NETWORK_INTERFACE=enp3s0    # once per shell (or put it in your shell profile)
```

The real-robot commands (`run-real`, `realsense`, `head-angle`) use DDS domain 0 on
`$G1_NETWORK_INTERFACE`; `--interface <name>` overrides it per command. They refuse to start
without one, and never accept the loopback `lo`. Every command prints the bus it chose
(`[dds] REAL HARDWARE: domain 0 on interface 'enp3s0'`) before it waits for traffic.

`run-real` is the only one that sends commands, so it asks first: it prints `ABOUT TO COMMAND
REAL HARDWARE` and continues only if you type `real` (anything else aborts; a
non-interactive run refuses unless you pass `--yes`). `realsense` and `head-angle` only
listen and publish depth, so they start without asking.

## 4. Bring-up procedure

1. Robot hanging in the gantry, feet just touching the ground. Put the robot in
   **debug (low-level) mode (L2 + B and then L2 + R2 to enter dev mode)** with the remote as described in Unitree's G1 developer
   guide (the built-in locomotion controller must release the motors, otherwise it fights
   `rt/lowcmd`).
2. `export G1_NETWORK_INTERFACE=<nic>` (section 3) in each terminal, then start the depth
   pipeline: `pixi run -e real realsense`; its 5 s status
   line should show ~50 Hz and an age at publish well under 100 ms.
   Check the image with `pixi run -e real depth-view`: a live window of exactly what the
   policy will see (floor as a smooth near-to-far gradient, the box as a bright blob; black =
   invalid), with the rate, age and valid fraction.
3. Set the head: `pixi run -e real head-angle` until it reads `OK`
   (section 1). The gantry pose with the feet on the floor is fine for this.
4. Place the box upright on the floor in front of the robot; you will walk the robot up to it.
   Before the first live run of a session, `pixi run -e real run-real --dry-run`: it reads
   the real robot and depth and runs every phase (`[` / Y to start the box motion), but
   publishes nothing; once a second it prints the phase, torso tilt, depth age and the
   largest gap between the target it would send and the joints. It never moves the robot,
   so it does not ask for `real`.
5. Pair the Bluetooth gamepad with the PC (it must show up as a joystick, e.g. in
   `jstest /dev/input/js0`), leave it untouched, and start
   `pixi run -e real run-real`
   (`--input keyboard` works too, from the same terminal; see the controls table in the README).
   * type `real` to confirm the robot is hung, the area is clear and the e-stop is in hand;
   * **STARTUP**: ramps all joints in 2 s from where they are to the standing pose,
     then the **locomotion** policy takes over at zero velocity;
   * lower the gantry until the robot stands on its own, then slacken it.
     Hold **LT** and use the sticks to walk up to the box until it is roughly at the
     demonstrated spot (0.32 m in front, 7 cm to the right of the pelvis);
   * press **Y**: a 0.1 s blend into the motion's first frame, then (once a fresh depth image
     is there) the student aligns the reference heading to the robot's current yaw and plays
     the 9.9 s motion: squat, grasp, lift, turn, place;
   * when the motion ends the robot blends back to locomotion and stands. **B** returns to
     locomotion at any time during the motion.
6. **LT + RT** together (keyboard `0`) or Ctrl-C: damping (kp = 0, kd = 8) for 0.5 s, then exit.
   The same happens on its own if `rt/lowstate` stops or a control tick is > 100 ms late.
   Stale depth during the motion (> `--max_depth_age`) hands back to locomotion.
   Keep the e-stop in hand regardless.
7. Keep the gantry rope slack along the whole path once the robot stands: a rope that drags
   the robot shortens its steps and tilts it, and the box motion fails in ways that look
   like a policy problem.

## Gains and limits

Per-joint kp / kd and action scales are in `g1_box_deploy/layout.py` (`KP_DDS`, `KD_DDS`,
`ACTION_SCALE_DDS`): they are the training actuator model and must not be changed
independently of the policy. Commands are `mode_pr = 0` (pitch/roll ankle mode) with
CRC, as `unitree_sdk2py` expects.
