# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""Publish rt/camera/depth from the head D435, for the depth student policy.

Hardware counterpart of sim2sim.py's depth renderer: same topic, same message, same units,
same frame -- run_policy.py subscribes to one thing and cannot tell which side of the
sim/real line it is on.

    python realsense_depth.py --engine-check            # verify the engine, no camera needed
    python realsense_depth.py --backend rs --dry-run    # camera geometry, publish nothing
    python realsense_depth.py --backend rs --interface enp3s0   # D435's own depth (bring-up)
    python realsense_depth.py --interface enp3s0        # Fast-FoundationStereo on the IR pair
    python realsense_depth.py --engine /path/to/fast_foundationstereo.engine   # reuse an engine

Settings live in camera_d435.toml; REAL_ROBOT.md has the setup, including the
librealsense backend trap (scripts/build_pyrealsense.sh).

THE RESAMPLE IS THE PART THAT MATTERS.

The policy was trained against a 64x48 pinhole with a 58 deg VERTICAL field of view, which
at 4:3 works out to fx = fy = 43.30 px and a 72.9 deg HORIZONTAL field of view. The D435's
IR stream is 87 deg horizontal. Vertical matches; horizontal does not.

So the real image cannot simply be resized down -- that would hand the policy a view ~20%
wider than anything it saw in training, with every object at the wrong apparent bearing,
and nothing would raise. It has to be RESAMPLED through the trained camera's intrinsics,
which crops and scales in one step and also absorbs any principal-point offset. The map is
built from librealsense's reported intrinsics rather than datasheet nominals, so it stays
correct if the stream resolution changes.

Nearest neighbour, never interpolation: averaging across a depth discontinuity invents a
surface at a distance where nothing exists. IsaacLab rendered the training images
point-sampled with no anti-aliasing, and this matches that.

Invalid pixels are published as NaN. The consumer maps non-finite to FAR, never near --
which matters because FoundationStereo's invalid region is a left-edge band that WIDENS as
objects approach (its own masking rule is `columns - disparity < 0`, so nearer means larger
disparity means a wider band). Filling those with zero would read as an obstacle at the
robot's face exactly while it reaches for the box.

Depth is metric Z (perpendicular distance to the image plane), matching both IsaacLab's
`distance_to_image_plane` and `fx * baseline / disparity`. Metres go on the wire;
normalising into [0,1] is the policy's business (g1_box_deploy.policy_io.preprocess_depth).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from g1_box_deploy.camera import CONTRACT  # noqa: E402

DEFAULT_DEVICE_CONFIG = HERE / "camera_d435.toml"


def load_device_config(path):
    with open(path, "rb") as fh:
        cfg = tomllib.load(fh)
    for section in ("stream", "backend", "publish"):
        if section not in cfg:
            raise ValueError(f"{path}: missing [{section}]")
    # Relative paths are relative to the config file, so the defaults point into the
    # Fast-FoundationStereo submodule wherever the repository is checked out.
    for key in ("engine", "repo"):
        value = cfg["backend"].get(key, "")
        if value and not Path(value).is_absolute():
            cfg["backend"][key] = str((Path(path).parent / value).resolve())
    return cfg


def build_resample_map(contract, fx, fy, cx, cy, width, height):
    """Nearest-neighbour source indices for each pixel of the trained camera.

    For every target pixel, take the ray it represents in the trained camera and ask which
    source pixel looks along the same ray. Crop, scale and principal point all fall out of
    that one step; doing them separately is how a silent half-pixel or off-by-a-crop error
    creeps in.
    """
    w, h = contract.width, contract.height
    u = (np.arange(w) + 0.5 - contract.cx) / contract.fx      # tan of the angle
    v = (np.arange(h) + 0.5 - contract.cy) / contract.fy
    su = np.rint(fx * u[None, :] + cx - 0.5).astype(np.int64)
    sv = np.rint(fy * v[:, None] + cy - 0.5).astype(np.int64)
    su = np.broadcast_to(su, (h, w))
    sv = np.broadcast_to(sv, (h, w))
    inside = (su >= 0) & (su < width) & (sv >= 0) & (sv < height)
    return np.clip(sv, 0, height - 1), np.clip(su, 0, width - 1), inside


def resample(depth_m, rows, cols, inside, valid_min_m, valid_max_m):
    """Full-resolution metric depth -> the trained camera's grid, in metres."""
    out = depth_m[rows, cols].astype(np.float32)
    bad = ~inside | ~np.isfinite(out) | (out < valid_min_m) | (out > valid_max_m)
    out[bad] = np.nan
    return out


class RealSenseSource:
    """D435 IR stereo, plus optionally its own onboard depth."""

    def __init__(self, width=848, height=480, fps=30, emitter=False, want_depth=False):
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
        config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)
        if want_depth:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipeline.start(config)

        device = self.profile.get_device()
        depth_sensor = device.first_depth_sensor()
        # The projector paints texture for the D435's own block matcher. A learned stereo
        # network keys on natural texture instead, and the dot pattern is at best neutral
        # for it -- but it is NOT neutral for the onboard depth backend, which needs it.
        # Default follows the backend.
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if emitter else 0.0)
        self.depth_scale = depth_sensor.get_depth_scale() if want_depth else None
        # Without this the device reports its own epoch, which no consumer can compare
        # against anything.
        if depth_sensor.supports(rs.option.global_time_enabled):
            depth_sensor.set_option(rs.option.global_time_enabled, 1.0)
        self.timestamp_domain = None

        ir1 = self.profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        ir2 = self.profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        self.intr = ir1.get_intrinsics()
        extr = ir2.get_extrinsics_to(ir1)
        self.baseline = abs(float(extr.translation[0]))
        self.want_depth = want_depth

    def describe(self):
        i = self.intr
        fovx = math.degrees(2 * math.atan((i.width / 2.0) / i.fx))
        fovy = math.degrees(2 * math.atan((i.height / 2.0) / i.fy))
        return (f"{i.width}x{i.height}  fx {i.fx:.1f} fy {i.fy:.1f} "
                f"cx {i.ppx:.1f} cy {i.ppy:.1f}  fov {fovx:.1f}x{fovy:.1f} deg  "
                f"baseline {self.baseline * 1000:.1f} mm")

    def read(self, timeout_ms=1000):
        """One frame set. Blocks; raises if the camera stops delivering.

        The timeout is 1 s rather than librealsense's 5 s so a camera that falls off the
        bus is reported while the runtime is still only faulting on staleness.
        """
        frames = self.pipeline.wait_for_frames(timeout_ms)
        # .copy() is load-bearing: these arrays are views into librealsense's frame
        # buffers, which are recycled the moment `frames` is released. The background
        # capture thread would otherwise overwrite the very pixels the network is reading.
        ir1 = frames.get_infrared_frame(1)
        left = np.asanyarray(ir1.get_data()).copy()
        right = np.asanyarray(frames.get_infrared_frame(2).get_data()).copy()
        depth = None
        if self.want_depth:
            d = frames.get_depth_frame()
            depth = np.asanyarray(d.get_data()).astype(np.float32) * self.depth_scale
            depth[depth <= 0] = np.nan
        return left, right, depth, self._capture_epoch(ir1)

    def _capture_epoch(self, frame):
        """The frame's own timestamp as host epoch seconds, or None.

        Better than marking the clock when wait_for_frames() returns, because that misses
        exposure, USB transfer and driver queueing -- tens of milliseconds on a D435, all
        of it real staleness the policy pays for. Returns None when the device reports its
        own unrelated clock, so the caller can fall back rather than publish a confidently
        wrong stamp.
        """
        rs = self.rs
        domain = frame.get_frame_timestamp_domain()
        if self.timestamp_domain is None:
            self.timestamp_domain = domain
        if domain in (rs.timestamp_domain.global_time, rs.timestamp_domain.system_time):
            return float(frame.get_timestamp()) / 1000.0
        return None

    def describe_clock(self):
        rs = self.rs
        if self.timestamp_domain == rs.timestamp_domain.global_time:
            return "global_time (host-synced; stamps include exposure and USB)"
        if self.timestamp_domain == rs.timestamp_domain.system_time:
            return "system_time (host clock; stamps include exposure and USB)"
        if self.timestamp_domain == rs.timestamp_domain.hardware_clock:
            return ("hardware_clock (device epoch, unusable) -- falling back to a host mark "
                    "at delivery, which EXCLUDES exposure and USB")
        return f"unknown ({self.timestamp_domain})"

    def close(self):
        self.pipeline.stop()


class LatestFrame:
    """Capture in a background thread and hand out only the NEWEST frame.

    THROUGHPUT. librealsense hands out frames in order, so a loop that reads then infers
    spends frame-interval PLUS inference every cycle -- 16.7 + 13 ms at 60 fps, which pins
    the publish rate near 30 Hz. Capturing concurrently overlaps the two.

    STALENESS. When inference falls behind, an in-order queue keeps handing over frames
    that are already old and the policy cannot tell -- it sees a fresh timestamp on stale
    pixels. Dropping everything but the latest frame keeps published depth as fresh as the
    camera can make it, which is what run_policy.py's --max_depth_age is measured against,
    and what the student's 0-40 ms trained lag assumes.
    """

    def __init__(self, source):
        self._source = source
        self._cv = threading.Condition()
        self._latest = None
        self._error = None
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="d435-capture", daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            try:
                frame = self._source.read()
            except Exception as exc:      # camera unplugged, pipeline stopped
                with self._cv:
                    self._error = exc
                    self._cv.notify_all()
                return
            with self._cv:
                if self._stopped:
                    return
                # Overwrite, never queue: an older frame has no value once a newer one exists.
                self._latest = (*frame, time.perf_counter())
                self._cv.notify()

    def get(self, timeout=2.0):
        """The newest unconsumed frame, as (left, right, onboard, sensor_epoch, t_capture)."""
        deadline = time.monotonic() + timeout
        with self._cv:
            while self._latest is None and self._error is None:
                if not self._cv.wait(max(0.0, deadline - time.monotonic())):
                    raise SystemExit(
                        f"[realsense] no frame for {timeout:g} s. The camera has most likely "
                        "dropped off the USB bus -- check `journalctl -k | grep 'usb '` for "
                        "disconnects, and see REAL_ROBOT.md."
                    )
            if self._error is not None:
                raise self._error
            frame, self._latest = self._latest, None
            return frame

    def close(self):
        with self._cv:
            self._stopped = True
        # The thread is a daemon parked in wait_for_frames; stopping the pipeline (the
        # caller's source.close()) is what actually releases it.


def usb_link_speed(vid="8086", pid="0b3a"):
    """(camera Mb/s, upstream hub Mb/s, sysfs name) for the camera, or None.

    Read from sysfs rather than asked of librealsense, because the number that matters is
    the HUB's upstream link -- which librealsense never sees.
    """
    root = Path("/sys/bus/usb/devices")

    def speed(path):
        try:
            return int(float((path / "speed").read_text().strip()))
        except (OSError, ValueError):
            return None

    for dev in sorted(root.glob("*-*")):
        try:
            if (dev / "idVendor").read_text().strip() != vid:
                continue
            if (dev / "idProduct").read_text().strip() != pid:
                continue
        except OSError:
            continue
        cam = speed(dev)
        if cam is None:
            continue
        # "4-5.1" hangs off hub "4-5"; a device straight on a root hub has no dot.
        hub = speed(root / dev.name.rsplit(".", 1)[0]) if "." in dev.name else None
        return cam, hub, dev.name
    return None


class TensorRTBackend:
    """Fast-FoundationStereo as a serialized TensorRT engine, on the IR pair.

    Full-resolution metric depth out, NaN where invalid. What is special is entirely a
    consequence of an engine being a FIXED-SIZE artifact.

    THE IMAGE IS RESIZED, SO DISPARITY MUST BE RESCALED.

    The engine is built for one input size (480x640) and the D435 streams another
    (848x480). Disparity is measured in PIXELS, so an image scaled horizontally by s comes
    back with its disparity scaled by s too. Dividing that back out before
    `fx * baseline / disp` is not optional: skip it and every distance is wrong by the
    constant 1/s -- 1.33x here -- while the disparity image still looks entirely correct.

    Only the HORIZONTAL factor enters. Rectification puts corresponding points on the same
    row, so resizing vertically resamples the disparity field without changing any of its
    values, while resizing horizontally rescales the values themselves. That is also why
    the resize is a plain stretch and not an aspect-preserving letterbox. Measured against
    the PyTorch model at native resolution on upstream's reference pair, the stretch agreed
    to 0.13% median depth error (p90 0.56%) and the letterbox to 0.16% (p90 0.85%) -- and
    the stretch spends all 480 engine rows on real image instead of 118 of them on padding.

    PREPROCESSING BELONGS TO THE ENGINE, NOT THE MODEL.

    make_single_onnx.py exports the network with normalisation OUTSIDE the graph, so the
    engine wants ImageNet-normalised NCHW float32. Hand it 0-255 and you get a
    plausible-looking, quietly wrong disparity field.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, engine, device="cuda"):
        import tensorrt as trt
        import torch

        self.torch = torch
        self.device = torch.device(device)

        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        if not Path(engine).exists():
            raise SystemExit(f"[realsense] engine does not exist: {engine}\n"
                             "            Build it first -- see REAL_ROBOT.md, 'Depth engine'.")
        with open(engine, "rb") as fh:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(fh.read())
        if self.engine is None:
            raise SystemExit(
                f"[realsense] failed to deserialize {engine}. A plan is not portable across "
                f"GPUs or TensorRT versions and this process has {trt.__version__}; rebuild "
                "the engine on this machine with the same TensorRT version."
            )
        self.context = self.engine.create_execution_context()

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        ins = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if len(ins) != 2 or len(outs) != 1:
            raise SystemExit(f"[realsense] expected 2 inputs and 1 output, got {ins} -> {outs}. "
                             "This does not look like a Fast-FoundationStereo engine.")
        # Prefer the names the export gives (left_image/right_image); fall back to
        # declaration order rather than guessing, because silently swapping left and right
        # yields a disparity field that looks fine and is nonsense.
        self.in_left = next((n for n in ins if "left" in n.lower()), ins[0])
        self.in_right = next((n for n in ins if "right" in n.lower()), ins[1])
        if self.in_left == self.in_right:
            raise SystemExit(f"[realsense] cannot tell the engine's inputs apart: {ins}")
        self.out_name = outs[0]

        shape = tuple(self.engine.get_tensor_shape(self.in_left))
        if any(d < 0 for d in shape):
            raise SystemExit(f"[realsense] engine input {self.in_left} has a dynamic shape "
                             f"{shape}; rebuild a fixed-size plan (make_single_onnx.py).")
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
            raise SystemExit(f"[realsense] unexpected engine input shape {shape}, want (1, 3, H, W)")
        self.eh, self.ew = int(shape[2]), int(shape[3])

        self.buf_left = torch.empty(shape, device=self.device, dtype=torch.float32)
        self.buf_right = torch.empty(shape, device=self.device, dtype=torch.float32)
        out_shape = tuple(self.context.get_tensor_shape(self.out_name))
        self.buf_out = torch.empty(out_shape, device=self.device, dtype=torch.float32)
        for name, buf in ((self.in_left, self.buf_left), (self.in_right, self.buf_right),
                          (self.out_name, self.buf_out)):
            self.context.set_tensor_address(name, int(buf.data_ptr()))

        self.mean = torch.tensor(self.IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        # Its own stream, not the default one: TensorRT warns that enqueueing on the
        # default stream costs extra synchronisation.
        self.stream = torch.cuda.Stream(device=self.device)
        self._cols = None

    def describe(self):
        return f"TensorRT {self.eh}x{self.ew} engine, inputs {self.in_left}/{self.in_right}"

    def _upload(self, img, buf):
        """uint8 (H, W) IR -> ImageNet-normalised (1, 3, eh, ew) in `buf`."""
        torch, F = self.torch, self.torch.nn.functional
        t = torch.from_numpy(np.ascontiguousarray(img)).to(self.device)
        t = t[None, None].float().div_(255.0)
        if t.shape[-2:] != (self.eh, self.ew):
            # Bilinear on the way DOWN: this is intensity, where averaging neighbours is
            # what you want. (Contrast the way back up, below.)
            t = F.interpolate(t, size=(self.eh, self.ew), mode="bilinear", align_corners=False)
        buf.copy_(t)  # broadcasts the one IR channel across all three
        buf.sub_(self.mean).div_(self.std)

    def depth(self, left, right, fx, baseline):
        torch, F = self.torch, self.torch.nn.functional
        src_h, src_w = left.shape[:2]
        with torch.cuda.stream(self.stream):
            self._upload(left, self.buf_left)
            self._upload(right, self.buf_right)
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("[realsense] TensorRT execute_async_v3 failed")

            disp = self.buf_out
            if disp.shape[-2:] != (src_h, src_w):
                # NEAREST on the way back up, not bilinear. Interpolating a disparity field
                # across an object boundary invents a value between near and far, which
                # becomes a surface at a distance where nothing exists.
                disp = F.interpolate(disp, size=(src_h, src_w), mode="nearest")
            # Engine pixels -> source pixels. Horizontal only; see the class docstring.
            disp = disp / (self.ew / src_w)

            if self._cols is None or self._cols.shape[-1] != src_w:
                self._cols = torch.arange(src_w, device=self.device,
                                          dtype=torch.float32).view(1, 1, 1, src_w)
            # A match whose disparity exceeds its column index came from off the right
            # image's edge: the left-edge band, which widens as objects come closer.
            ok = torch.isfinite(disp) & (disp > 0) & ((self._cols - disp) >= 0)
            depth = torch.where(ok, (fx * baseline) / disp.clamp(min=1e-6), torch.nan)
            out = depth.reshape(src_h, src_w).to("cpu")
        self.stream.synchronize()
        return out.numpy()


def engine_check(engine, repo):
    """Prove the engine runs and that its disparity rescale is right, no camera.

    Deployment depends on one silent invariant: metric depth must not depend on what
    resolution the source images arrived at, because the engine resizes them to its own
    fixed size and disparity is measured in pixels. So run upstream's reference pair twice
    at two different source widths and compare the metres. If the rescale were dropped,
    the two runs would disagree by the ratio of their scale factors -- 1.5x for this pair.
    """
    import torch
    from PIL import Image

    demo = Path(repo) / "demo_data"
    left_png, right_png, k_txt = demo / "left.png", demo / "right.png", demo / "K.txt"
    for f in (left_png, right_png, k_txt):
        if not f.exists():
            raise SystemExit(f"[realsense] engine check needs {f}. Is the Fast-FoundationStereo "
                             "submodule checked out (git submodule update --init)?")

    def gray(path):
        return np.array(Image.open(path).convert("L"), dtype=np.uint8)

    lines = k_txt.read_text().splitlines()
    K = np.array([float(x) for x in lines[0].split()], dtype=np.float64).reshape(3, 3)
    fx_native, baseline = float(K[0, 0]), float(lines[1])

    backend = TensorRTBackend(engine)
    print(f"[realsense] engine      : {backend.describe()}")
    left, right = gray(left_png), gray(right_png)
    src_h, src_w = left.shape
    print(f"[realsense] reference   : {src_w}x{src_h}, fx {fx_native:.1f}, baseline {baseline * 1000:.1f} mm")

    def resized(img, h, w):
        t = torch.from_numpy(img)[None, None].float()
        t = torch.nn.functional.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
        return t[0, 0].round().clamp(0, 255).to(torch.uint8).numpy()

    runs = {}
    # Native, and the engine's own size -- 1.5x apart in scale. Same scene, so the metres must match.
    for label, (h, w) in (("native", (src_h, src_w)), ("engine-size", (backend.eh, backend.ew))):
        lt, rt = (left, right) if (h, w) == (src_h, src_w) else (resized(left, h, w), resized(right, h, w))
        z = backend.depth(lt, rt, fx_native * (w / src_w), baseline)
        finite = z[np.isfinite(z)]
        if not finite.size:
            raise SystemExit(f"[realsense] engine check: {label} run produced NO valid depth")
        runs[label] = z
        print(f"[realsense] {label:11s} : {w}x{h} -> {100 * np.isfinite(z).mean():4.1f}% valid, "
              f"{finite.min():.3f}..{finite.max():.3f} m, median {np.median(finite):.3f} m")

    a, b = runs["native"], runs["engine-size"]
    b = np.asarray(Image.fromarray(b).resize((a.shape[1], a.shape[0]), Image.NEAREST))
    ok = np.isfinite(a) & np.isfinite(b)
    ratio = float(np.median(a[ok] / b[ok]))
    err = abs(ratio - 1.0) * 100
    print(f"[realsense] scale check : median depth ratio {ratio:.4f} ({err:.2f}% from 1.0) "
          f"over {ok.sum()} shared pixels")
    if err > 3.0:
        raise SystemExit("[realsense] FAIL: metric depth depends on the source resolution, so the "
                         "disparity rescale is wrong. Every distance is off by a constant factor.")

    t = []
    for _ in range(25):
        t0 = time.perf_counter()
        backend.depth(left, right, fx_native, baseline)
        t.append((time.perf_counter() - t0) * 1e3)
    t = np.array(t[5:])
    print(f"[realsense] latency     : median {np.median(t):.1f} ms, p95 {np.percentile(t, 95):.1f} ms "
          f"(at {src_w}x{src_h}; the D435 stream is smaller, so expect a little less)")
    print("[realsense] PASS: engine loads, runs, and its depth is resolution-invariant.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["sim", "real"], default="real",
                    help="DDS bus: real = domain 0 on --interface (the robot's network)")
    ap.add_argument("--interface", default=None, help="network interface wired to the G1 (default: $G1_NETWORK_INTERFACE)")
    ap.add_argument("--device", default=str(DEFAULT_DEVICE_CONFIG), help="device TOML (stream, backend, publish)")
    # Overrides. Everything defaults to the device config; these exist for one-off
    # experiments, not as the normal way to configure the camera.
    ap.add_argument("--backend", choices=("trt", "rs"), default=None)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--rate-hz", type=float, default=None)
    ap.add_argument("--emitter", choices=("on", "off"), default=None)
    ap.add_argument("--engine", default=None,
                    help="TensorRT engine to use instead of the device config's (default: $FFS_ENGINE, else the "
                         "submodule's build). Must have been built with the same TensorRT version as this env")
    ap.add_argument("--ffs-repo", default=None,
                    help="Fast-FoundationStereo checkout whose demo_data/ --engine-check reads "
                         "(default: $FFS_REPO, else the submodule)")
    ap.add_argument("--dry-run", action="store_true", help="open the camera, print the resample geometry, publish nothing")
    ap.add_argument("--engine-check", action="store_true",
                    help="verify the TensorRT engine on the reference stereo pair and exit; needs no camera")
    args = ap.parse_args(argv)

    contract = CONTRACT
    device = load_device_config(args.device)
    stream, backend_cfg, publish = device["stream"], device["backend"], device["publish"]
    # An existing engine (e.g. built earlier on this machine) can be reused without touching the
    # config: --engine / --ffs-repo, else $FFS_ENGINE / $FFS_REPO, else the config's paths.
    for key, flag, env in (("engine", args.engine, "FFS_ENGINE"), ("repo", args.ffs_repo, "FFS_REPO")):
        value = flag or os.environ.get(env)
        if value:
            backend_cfg[key] = str(Path(value).expanduser().resolve())
    backend_type = args.backend or backend_cfg["type"]
    fps = args.fps or int(stream["fps"])
    rate_hz = args.rate_hz or float(publish["rate_hz"])
    valid_min, valid_max = float(publish["valid_min_m"]), float(publish["valid_max_m"])
    print(f"[realsense] contract    : {contract.describe()}")
    print(f"[realsense] device cfg  : {args.device}")

    if args.engine_check:
        print(f"[realsense] engine      : {backend_cfg['engine']}")
        return engine_check(backend_cfg["engine"], backend_cfg["repo"])

    # The D435's own block matcher needs the projector's texture; a learned matcher keys on
    # natural texture instead. Config wins if it says so.
    if args.emitter is not None:
        emitter = args.emitter == "on"
    elif "emitter" in backend_cfg:
        emitter = bool(backend_cfg["emitter"])
    else:
        emitter = backend_type == "rs"

    source = RealSenseSource(int(stream["width"]), int(stream["height"]), fps,
                             emitter=emitter, want_depth=(backend_type == "rs"))
    print(f"[realsense] device      : {source.describe()}")
    print(f"[realsense] IR projector: {'ON' if emitter else 'OFF'}   stream {fps} fps")

    # The link speed is a safety property, not trivia: a camera behind a hub whose upstream
    # link negotiated 10 Gb/s re-enumerated about once a second in our setup and vanished
    # mid-stream. Print it every run so a silent renegotiation cannot go unnoticed.
    link = usb_link_speed()
    if link is None:
        print("[realsense] usb link    : could not read it from sysfs")
    else:
        cam_mbps, hub_mbps, name = link
        hub_txt = f", hub above it {hub_mbps / 1000:g} Gb/s" if hub_mbps else " (no hub)"
        print(f"[realsense] usb link    : {name} at {cam_mbps / 1000:g} Gb/s{hub_txt}")
        if hub_mbps and hub_mbps >= 10000:
            print("[realsense] WARNING: the hub has negotiated 10 Gb/s (SuperSpeed Plus). That link")
            print("            was unstable for us (the chain re-enumerates and depth stops mid-run);")
            print("            5 Gb/s is ~8x the bandwidth this stream needs. Use a USB 3.0 (5 Gb/s)")
            print("            upstream cable for the hub to cap the negotiation physically.")

    intr = source.intr
    rows, cols, inside = build_resample_map(contract, intr.fx, intr.fy, intr.ppx, intr.ppy,
                                            intr.width, intr.height)
    src_fovx = math.degrees(2 * math.atan((intr.width / 2.0) / intr.fx))
    src_fovy = math.degrees(2 * math.atan((intr.height / 2.0) / intr.fy))
    used = 100.0 * inside.mean()
    print(f"[realsense] resample    : {src_fovx:.1f}x{src_fovy:.1f} deg sensor -> "
          f"{contract.fovx_deg:.1f}x{contract.fovy_deg:.1f} deg contract, {used:.0f}% of target pixels in view")
    if used < 99.0:
        print("[realsense] WARNING: the trained field of view does not fit inside the sensor's.")
        print("            Out-of-view pixels publish as NaN (= far), so the policy sees a blind")
        print("            border it never trained against.")

    backend = None
    if backend_type == "trt":
        print(f"[realsense] loading TensorRT engine: {backend_cfg['engine']}")
        backend = TensorRTBackend(backend_cfg["engine"])
        print(f"[realsense] backend     : {backend.describe()}")

    if args.dry_run:
        left, right, onboard, _ = source.read()
        raw = onboard if onboard is not None else backend.depth(left, right, intr.fx, source.baseline)
        print(f"[realsense] frame clock : {source.describe_clock()}")
        out = resample(raw, rows, cols, inside, valid_min, valid_max)
        finite = out[np.isfinite(out)]
        if finite.size:
            print(f"[realsense] dry run     : {out.shape}, {100 * np.isfinite(out).mean():.1f}% valid, "
                  f"{finite.min():.3f}..{finite.max():.3f} m")
        else:
            print("[realsense] dry run     : NO valid pixels -- check the scene and lighting")
        source.close()
        return 0

    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.default import std_msgs_msg_dds__Header_

    from g1_box_deploy.dds_io import TOPIC_DEPTH, init_dds
    from g1_box_deploy.image_msg import Image_

    init_dds(args.target, args.interface)
    pub = ChannelPublisher(TOPIC_DEPTH, Image_)
    pub.Init()
    header = std_msgs_msg_dds__Header_()
    header.frame_id = contract.frame_id
    msg = Image_(header=header, height=contract.height, width=contract.width, encoding="32FC1",
                 is_bigendian=0, step=contract.width * 4, data=[])
    print(f"[realsense] publishing  : {TOPIC_DEPTH} at {rate_hz:g} Hz (32FC1 metres, frame {contract.frame_id})")
    # The student trained against a lag of up to 2 env steps -- 40 ms at 50 Hz.
    print("[realsense] budget      : keep 'age at publish' small; trained lag was 0-40 ms, "
          "run_policy.py faults past --max_depth_age")

    # Capture runs concurrently with inference from here on, so what gets published is
    # always built from the newest frame the camera has produced.
    camera = LatestFrame(source)
    period = 1.0 / rate_hz
    next_pub = time.monotonic()
    n, t_report, compute, age = 0, time.monotonic(), [], []
    try:
        while True:
            left, right, onboard, sensor_epoch, t_capture = camera.get()
            t0 = time.perf_counter()
            raw = onboard if onboard is not None else backend.depth(left, right, intr.fx, source.baseline)
            out = resample(raw, rows, cols, inside, valid_min, valid_max)
            compute.append((time.perf_counter() - t0) * 1e3)

            # Stamp with when the PIXELS were captured, not when the message was assembled:
            # the consumer's staleness check is against the age of the observation. Prefer
            # the sensor's own timestamp (it starts at exposure); the delivery mark is the
            # fallback, and misses everything before wait_for_frames() returned.
            stamp = sensor_epoch if sensor_epoch is not None else time.time() - (time.perf_counter() - t_capture)
            msg.header.stamp.sec = int(stamp)
            msg.header.stamp.nanosec = int((stamp % 1.0) * 1e9)
            msg.data = list(out.tobytes())
            pub.Write(msg)
            age.append((time.time() - stamp) * 1e3)

            n += 1
            now = time.monotonic()
            if now - t_report >= 5.0:
                c, a = np.array(compute), np.array(age)
                print(f"[realsense] {n / (now - t_report):5.1f} Hz   compute {np.median(c):5.1f} ms   "
                      f"age at publish {np.median(a):5.1f} ms (p95 {np.percentile(a, 95):5.1f})   "
                      f"{100 * np.isfinite(out).mean():4.1f}% valid")
                n, t_report, compute, age = 0, now, [], []

            # Pace to rate_hz. Sleeping rather than spinning is also what hands the capture
            # thread the CPU to have the next frame ready.
            next_pub += period
            slack = next_pub - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            else:
                next_pub = time.monotonic()   # behind: do not accumulate debt
    except KeyboardInterrupt:
        print("\n[realsense] stopping")
    finally:
        camera.close()
        source.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
