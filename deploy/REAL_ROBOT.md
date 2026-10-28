# Running the box student on a real G1

`run_policy.py --target real` speaks the same DDS interface as `sim2sim.py`, so the only
differences from sim2sim are the network, the depth source, the head-camera angle and a 2 s
(instead of 0.5 s) startup ramp, done in the gantry. Run sim2sim first (README, section 2)
and make sure it lifts the box on your machine.

```
D435 (head) ──USB 3──> PC: realsense_depth.py (GPU) ──rt/camera/depth──┐
                                                                       ├──> run_policy.py ──rt/lowcmd──> G1
G1 ──rt/lowstate (IMU, joints)──────────────────────────────────────────┘
                 all over DDS domain 0 on the Ethernet link to the robot (192.168.123.x)
```

* **Part A** (once per PC): network, camera library, stereo-depth engine, gamepad, a desk test.
* **Part B** (every session): a numbered checklist from power-on to the box motion.
* **Troubleshooting**, then **reference** (camera pitch, depth contract, gains).

## What you need

* Unitree G1, 29 DoF (`mode_machine` 5), firmware with the `unitree_hg` low-level interface.
  Hands are not used: the policy was trained with rigid box-shaped "dummy hands".
* The head Intel RealSense D435(i) (`d435_link`), USB cable to the PC.
* A PC with an NVIDIA GPU (RTX 3080-class or better, see README section 0) and an Ethernet
  port. It runs both the depth pipeline and `run_policy.py` (the policy needs only the CPU:
  ~0.2 ms per inference with 2 threads).
* The box: 205 × 125 × 280 mm, ~0.6 kg, standing on its 205 × 125 face. In the demo it
  starts 0.32 m in front of the pelvis and 7 cm to the right (`object_pos[0]` in
  `policy/motion.npz`); the student was trained with the box anywhere within ±0.2 m of
  that (wider in Stage 2) and at any yaw, so place it roughly there.
* A gantry or harness, the Unitree remote, and someone holding the e-stop.
* A Bluetooth or USB gamepad (Xbox layout).

---

## Part A: one-time setup of the PC

### A1. Install

```bash
git clone --recurse-submodules https://github.com/learnsyslab/g1-mocap-free-box-locomanipulation.git
cd g1-mocap-free-box-locomanipulation
pixi install -e real
```

### A2. Wired network to the robot (static IP)

The robot's internal network is `192.168.123.x` with no DHCP, so the PC's Ethernet port needs
a fixed address on it. Connect the PC to the robot with a LAN cable, then on the PC (Ubuntu):

1. **Settings → Network**, and under the wired connection click **+** to create a new profile.
2. **IPv4** tab: method **Manual**.
3. Address **`192.168.123.51`**, netmask **`24`** (= `255.255.255.0`); leave gateway and DNS
   empty. Any free `192.168.123.x` works; `.51` is the one Unitree's guide uses.
4. **Apply**, then switch the wired connection off and on (or unplug and replug the cable).

Check it, robot powered on:

```bash
ip -br addr | grep 192.168.123     # the interface name in the first column, e.g. enp3s0
ping -c 3 192.168.123.161          # the robot's locomotion computer must answer
```

Then tell the deploy commands which interface that is, in every terminal you use for the
robot (or once in `~/.bashrc`):

```bash
export G1_NETWORK_INTERFACE=enp3s0   # your name from `ip -br addr`
```

### A3. Camera library (pyrealsense2)

1. Plug the D435 **directly** into a USB 3 port of the PC. Check the link:

   ```bash
   lsusb | grep -i 8086               # "Intel Corp. RealSense ..."
   lsusb -t | grep -i -B1 video       # the camera's port must show 5000M (USB 3), not 480M
   ```

2. Let your user access it (librealsense's udev rules), then replug the camera:

   ```bash
   sudo curl -L -o /etc/udev/rules.d/99-realsense-libusb.rules \
     https://raw.githubusercontent.com/IntelRealSense/librealsense/v2.58.3/config/99-realsense-libusb.rules
   sudo udevadm control --reload-rules && sudo udevadm trigger
   ```

3. Build pyrealsense2 with the RSUSB (libusb) backend:

   ```bash
   pixi run -e real build-pyrealsense
   ```

   Every PyPI `pyrealsense2` wheel uses the V4L2 backend, which on our kernel fails with
   `xioctl(UVCIOC_CTRL_QUERY) failed ... Protocol error`; the script builds librealsense
   v2.58.3 with `-DFORCE_RSUSB_BACKEND=ON` and puts it first on the path. (If the wheel works
   on your machine, `pixi run -e real python -m pip install pyrealsense2==2.58.3` is enough.)

4. Check: open the camera, grab one frame, publish nothing:

   ```bash
   pixi run -e real realsense --backend rs --dry-run
   ```

   It prints the device, the stream geometry and the valid depth range of one frame.

### A4. Stereo-depth engine (Fast-FoundationStereo, TensorRT)

1. **Weights**: download NVIDIA's research checkpoint `23-36-37` (link in
   `deploy/third_party/Fast-FoundationStereo/README_ORIG.md`) into
   `deploy/third_party/Fast-FoundationStereo/weights/23-36-37/`.
2. **Engine**: `pixi run -e real build-engine`. Exports the ONNX (640 × 480, 8 iterations,
   max disparity 192) and builds an fp16 TensorRT 10.11 engine. An engine only runs on the GPU
   model and TensorRT version that built it, so build it on this PC.
3. **Check**: `pixi run -e real engine-check` (no camera needed). Runs a reference stereo pair
   at two resolutions and fails unless the metric depth agrees between them.

Already have an engine for this GPU (TensorRT 10.11.0.33)? Skip 1-2 and point at it:
`export FFS_ENGINE=/path/to/fast_foundationstereo.engine` (or `--engine` per command).

### A5. Gamepad

Pair it with the PC (Bluetooth settings, or plug it in). It must show up as a joystick:
`jstest /dev/input/js0` prints the axes and buttons as you move them.

### A6. Desk test (camera only, no robot)

```bash
pixi run -e real realsense --target sim        # terminal 1: depth on the local test bus
pixi run -e real depth-view --target sim       # terminal 2: what the policy will see
```

Point the camera at a box on the floor: a bright blob on a smooth near-to-far floor
gradient, black = invalid. The window shows the rate (~50 Hz on an RTX 5090, ~30 Hz on an
RTX 3080), the frame age and the valid fraction. `head-angle` cannot be tested this way: it
needs the robot's IMU (`rt/lowstate`).

---

## Part B: every session

Each robot terminal: `export G1_NETWORK_INTERFACE=<nic>` (A2) first.

1. **Robot in the gantry**, feet just touching the ground, powered on.
2. **Debug mode** with the Unitree remote: **L2 + B**, then **L2 + R2**. The built-in
   locomotion controller releases the motors (otherwise it fights `rt/lowcmd`).
3. **Network**: `ping -c 3 192.168.123.161` answers.
4. **Depth** (terminal 1): `pixi run -e real realsense`. Its 5 s status line should show
   ~50 Hz and an age at publish well under 100 ms.
5. **Look at it** (terminal 2): `pixi run -e real depth-view`. Same picture as in A6, now on
   the robot's bus. Close it when done (or leave it running; it only listens).
6. **Head angle** (terminal 2): `pixi run -e real head-angle`. Tilt the head by hand until it
   prints `OK (within 0.5 deg)`, let go, and check again after 2 s (reference R1).
7. **Box**: upright on the floor in front of the robot; you will walk the robot up to it.
8. **Dry run** (first run of the session): `pixi run -e real run-real --dry-run`. It reads the
   robot and the depth and runs every phase (`[` / Y to start the box motion) but sends
   nothing; once a second it prints the phase, torso tilt, depth age and the largest gap
   between the target it would send and the joints. Ctrl-C to quit.
9. **Run** (terminal 2): `pixi run -e real run-real` (`--input keyboard` also works; controls
   in the README).
   * type `real` to confirm: robot hung, area clear, e-stop in hand;
   * **STARTUP**: all joints ramp in 2 s to the standing pose, then the **locomotion** policy
     holds it at zero velocity;
   * lower the gantry until the robot stands on its own, then slacken the rope;
   * hold **LT** and use the sticks to walk up to the box until it is roughly at the
     demonstrated spot (0.32 m in front, 7 cm to the right of the pelvis);
   * press **Y**: a 0.1 s blend into the motion's first frame, then (once a fresh depth image
     is there) the student plays the 9.9 s motion: squat, grasp, lift, turn, place;
   * at the end it blends back to locomotion and stands. **B** returns to locomotion at any
     time during the motion.
10. **Stop**: **LT + RT** together (keyboard `0`) or Ctrl-C: damping (kp = 0, kd = 8) for
    0.5 s, then exit. The same happens on its own if `rt/lowstate` stops or a control tick is
    > 100 ms late; stale depth during the motion (> `--max_depth_age`) hands back to
    locomotion. Keep the e-stop in hand regardless.

Keep the gantry rope slack along the whole path once the robot stands: a rope that drags the
robot shortens its steps and tilts it, and the box motion fails in ways that look like a
policy problem.

`run-real` is the only command that sends motor commands, so it is the only one that asks
for `real` (anything else aborts; a non-interactive run refuses unless `--yes`). Every command
prints its bus first, e.g. `[dds] REAL HARDWARE: domain 0 on interface 'enp3s0'`.

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `RuntimeError: failed to set power state` (realsense) | another process holds the camera; or the PyPI pyrealsense2 is loaded; or no USB permission; or a weak USB link | close realsense-viewer / other camera scripts and replug; check `pixi run -e real python -c "import pyrealsense2 as rs; print(rs.__file__)"` points at the `build-pyrealsense` build (A3.3); install the udev rules (A3.2); plug directly into USB 3 |
| `xioctl(UVCIOC_CTRL_QUERY) failed ... Protocol error` | PyPI wheel (V4L2 backend) | `pixi run -e real build-pyrealsense` (A3.3) |
| camera disappears every few seconds | USB hub negotiating 10 Gb/s | plug directly, or a hub port at 5 Gb/s; the driver prints the link speed |
| `head-angle`: `waiting for rt/camera/depth and rt/lowstate ...` while depth-view shows depth | no robot state: robot off or not wired; or the depth runs on the test bus (`--target sim`) while head-angle listens on the robot's | robot on and `ping 192.168.123.161` (A2); start `realsense` **without** `--target sim`, all terminals with the same `G1_NETWORK_INTERFACE` |
| a deploy command refuses to start: no interface | `G1_NETWORK_INTERFACE` unset | `export G1_NETWORK_INTERFACE=<nic>` (A2) |
| `ping 192.168.123.161` fails | static IP not active, wrong port, robot not booted | `ip -br addr` shows a `192.168.123.x` address on the wired port (A2); wait for boot |
| `run-real` stays at `WAIT: holding frame 0 until a fresh depth image arrives` | no depth on the robot's bus, or frames older than `--max_depth_age` (0.15 s) | `realsense` running on the same bus (step 4), its status line's age |
| the robot reaches past / short of the box, never grasps | head not at 47.6° | `head-angle` until OK (step 6) |

---

## Reference

### R1. Camera pitch: 47.6° down

The policy has no other idea of where the camera looks, so the camera pitch has to match
training. The camera frame is `d435_link` = the **left IR imager**, at
`(0.0576, 0.0175, 0.4299)` m in `torso_link`, pitched 47.6° about +Y (quaternion
`wxyz = (0.9150, 0, 0.4035, 0)`), the G1's nominal head pose. A head left at another
angle (we once ran at ~62°) made the robot overshoot the box by ~23 cm and never grasp it.

The head is tilted **by hand** and holds its angle by friction, so it is set before every
session with `head-angle`:

```
pitch  53.4 deg  target  47.6  ->  tilt head UP 5.8 deg   roll +0.6  height 1.27 m  floor 2650 px
```

Robot standing still (or in the gantry with the feet on the floor), camera seeing open, flat
floor. It fits the floor plane in the depth image (RANSAC) and compares it with "up" from the
pelvis IMU and the waist joints, so it needs no motion capture; `height` (camera above the
floor, ~1.27 m standing) and `floor` (pixels on the plane) tell you whether it really found the
floor. Checked in sim2sim, where the true pitch is known: it reads 47.6 deg for the trained
mount and 55.0 deg (-> "tilt head UP 7.4 deg") for a camera tilted to 55 deg.

### R2. Depth stream contract

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

### R3. The depth pipeline: `realsense_depth.py`

IR1 + IR2 at 848 × 480, 60 fps, **projector off** → Fast-FoundationStereo as a TensorRT
engine (fp16, 640 × 480 input, 8 refinement iterations, ~13 ms on an RTX 5090) → disparity to
metric depth `Z = fx·baseline / d` in the left-IR frame → one nearest-neighbour ray resample
from the sensor intrinsics onto the 64 × 48 pinhole above → `rt/camera/depth` at 50 Hz,
stamped with the sensor's capture time. Measured age at publish: 22 ms median, 30 ms p95.
Settings are in `deploy/camera_d435.toml`.

```bash
pixi run -e real realsense --backend rs --dry-run      # geometry, one frame, no publish
pixi run -e real realsense --backend rs                # D435's own depth, projector on
pixi run -e real realsense                             # Fast-FoundationStereo
```

`--backend rs` exercises the whole chain (topic, format, resample, `run_policy.py`'s depth
freshness) without the GPU model, so plumbing problems and model problems stay separate.
The D435's own ASIC depth fits the contract and is fine for bring-up, but its noise and
holes differ from what the policy was trained on (the training noise model matches stereo
depth: thin, near-object holes, little temporal noise). Every 5 s the driver prints its rate,
compute time, age at publish, the fraction of valid pixels and the USB link speed.

Practical notes from our setup:

* Running the camera from the robot's onboard computer and streaming the IR pair over
  Ethernet was not reliable for us (the camera dropped off its USB within minutes); we wire
  it to the PC.
* Fast-FoundationStereo is NVIDIA's (https://github.com/NVlabs/Fast-FoundationStereo), under
  NVIDIA's own source-code license: **non-commercial / research use only**. It is included as
  a git submodule (our fork with the pixi / TensorRT build tasks); no weights or engines are
  included.
* Measure the depth rate as the policy sees it (fraction of 50 Hz ticks with a new image;
  ours is 96-99 %), not with a DDS recorder, which can drop large image messages.

### R4. Bus and safety

The real-robot commands (`run-real`, `realsense`, `head-angle`, `depth-view`) use DDS domain 0
on `$G1_NETWORK_INTERFACE`; `--interface <name>` overrides it per command. They refuse to
start without one and never accept the loopback `lo`. `--target sim` (domain 1 on `lo`) is
the sim2sim / desk-test bus.

### R5. Gains and limits

Per-joint kp / kd and action scales are in `g1_box_deploy/layout.py` (`KP_DDS`, `KD_DDS`,
`ACTION_SCALE_DDS`): they are the training actuator model and must not be changed
independently of the policy. Commands are `mode_pr = 0` (pitch/roll ankle mode) with
CRC, as `unitree_sdk2py` expects.
