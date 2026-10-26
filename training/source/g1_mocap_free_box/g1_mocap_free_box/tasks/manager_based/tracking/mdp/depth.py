# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""Egocentric depth observation and its sensor-corruption model.

The distillation student replaces the two privileged object terms
(``object_pose_b`` / ``object_pose_error_b``) with this image, so everything the
student ever learns about where the box is has to come through here.

Corruption lives in the observation function rather than in a separate
NoiseModel because it needs three things the noise plumbing does not naturally
give: per-env state (hold-last-frame), a lag buffer, and an intensity that ramps
with training progress. Intensity is derived from ``env.common_step_counter``
directly, so there is no curriculum term to wire up and nothing to keep in sync.

The teacher never sees this image -- it reads the clean privileged state. That
asymmetry is the point: the student must learn invariances the teacher never
needed ("noisy student"), and the labels it chases stay clean throughout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Per-env corruption state, keyed by id(env): the held frame for dropped-frame
# emulation and the lag ring buffer. Keyed rather than stored on the env so the
# observation function stays a plain function and play/eval envs that never call
# it cost nothing.
_STATE: dict[int, dict] = {}


def _as_tensor(x):
    """IsaacLab 6.x hands back warp-backed ProxyArrays; .torch is the real tensor."""
    return x.torch if hasattr(x, "torch") else x


def _camera_depth_frame(
    env: ManagerBasedRLEnv,
    sensor_name: str = "depth_cam",
    cutoff_distance: float = 2.5,
    min_depth: float = 0.05,
    noise: bool = True,
    ramp_steps: int = 120_000,
    pixel_noise_std: float = 0.02,
    salt_prob: float = 0.01,
    max_rects: int = 2,
    rect_size_range: tuple[float, float] = (0.1, 0.3),
    hold_prob: float = 0.05,
    max_lag: int = 2,
    fill_value: float = 1.0,
) -> torch.Tensor:
    """Egocentric depth as a CNN-ready ``(num_envs, 1, H, W)`` image in [0, 1].

    Metric depth is clipped to ``[min_depth, cutoff_distance]`` and divided by
    the cutoff, so near is small and far is 1. Anything beyond the cutoff -- and
    anything the renderer returns as inf/nan, which is what the sky does --
    saturates to 1 rather than becoming a hole, so "nothing there" and "far away"
    read the same way to the CNN.

    ``fill_value`` defaults to 1.0 (far), NOT 0.0 (near). A dropped stereo pixel
    that reads as *near* would be an obstacle the student must avoid; reading it
    as *far* is the benign interpretation and matches how invalid depth is
    usually handled downstream.

    Corruption ramps in over ``ramp_steps`` environment steps so the student
    first learns to read clean depth and only then hardens against sensor noise.
    """
    sensor = env.scene[sensor_name]
    raw = _as_tensor(sensor.data.output["distance_to_image_plane"])
    num_envs = env.num_envs
    img = raw.reshape(num_envs, 1, sensor.cfg.height, sensor.cfg.width).float()

    # inf/nan (sky, no hit) -> far, before any arithmetic that would propagate it
    img = torch.nan_to_num(img, nan=cutoff_distance, posinf=cutoff_distance, neginf=cutoff_distance)
    img = img.clamp(min_depth, cutoff_distance) / cutoff_distance

    if not noise:
        return img

    intensity = 1.0
    if ramp_steps > 0:
        intensity = min(1.0, float(env.common_step_counter) / float(ramp_steps))
    if intensity <= 0.0:
        return img

    device = img.device
    _, _, height, width = img.shape
    st = _STATE.setdefault(id(env), {})

    # Per-pixel Gaussian noise, blended by intensity (identity at 0).
    if pixel_noise_std > 0.0:
        img = img + torch.randn_like(img) * (pixel_noise_std * intensity)

    # Salt dropout: isolated invalid pixels.
    p = min(salt_prob * intensity, 1.0)
    if p > 0.0:
        img = torch.where(torch.rand_like(img) < p, torch.full_like(img, fill_value), img)

    # Rectangle dropout: contiguous invalid patches, as specular/textureless
    # regions produce on a real stereo sensor.
    if max_rects > 0:
        ys = torch.arange(height, device=device).view(1, height, 1)
        xs = torch.arange(width, device=device).view(1, 1, width)
        n_rects = torch.floor(torch.rand(num_envs, device=device) * (max_rects + 1) * intensity)
        lo, hi = rect_size_range
        for r in range(max_rects):
            active = (n_rects > r).view(num_envs, 1, 1)
            cy = torch.rand(num_envs, 1, 1, device=device) * height
            cx = torch.rand(num_envs, 1, 1, device=device) * width
            rh = (lo + torch.rand(num_envs, 1, 1, device=device) * (hi - lo)) * height
            rw = (lo + torch.rand(num_envs, 1, 1, device=device) * (hi - lo)) * width
            mask = active & ((ys - cy).abs() < rh / 2) & ((xs - cx).abs() < rw / 2)
            img = torch.where(mask.unsqueeze(1), torch.full_like(img, fill_value), img)

    img = img.clamp(0.0, 1.0)

    # Latency: deliver a frame from up to `max_lag` steps ago. Real mocap-free
    # depth is never instantaneous, and a student trained only on fresh frames
    # learns a reflex the hardware cannot honour.
    if max_lag > 0:
        buf = st.get("lag")
        if buf is None or buf.shape[1:] != img.shape or buf.shape[0] != max_lag + 1:
            buf = img.unsqueeze(0).repeat(max_lag + 1, 1, 1, 1, 1)
            st["lag"] = buf
        buf = torch.roll(buf, shifts=1, dims=0)
        buf[0] = img
        st["lag"] = buf
        lag = torch.randint(0, max_lag + 1, (num_envs,), device=device)
        idx = lag.view(1, num_envs, 1, 1, 1).expand(1, num_envs, *img.shape[1:])
        img = torch.gather(buf, 0, idx).squeeze(0)

    # Hold-last-frame: a dropped frame repeats the previous delivery.
    held = st.get("held")
    if held is None or held.shape != img.shape:
        held = img.clone()
    hp = min(hold_prob * intensity, 1.0)
    if hp > 0.0:
        hold = torch.rand(num_envs, device=device) < hp
        img = torch.where(hold.view(num_envs, 1, 1, 1), held, img)
    st["held"] = img.clone()

    return img


def _just_reset(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Envs whose episode starts this step (all of them before the buffer exists)."""
    elb = getattr(env, "episode_length_buf", None)
    if elb is None:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return elb == 0


def _ring_push(st: dict, key: str, x: torch.Tensor, depth: int, fresh: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Push ``x`` into a per-env ring buffer of ``depth`` entries; envs in ``fresh``
    are refilled with ``x`` so a new episode never sees the previous one's history."""
    buf = st.get(key)
    if buf is None or buf.shape[0] != depth or buf.shape[1:] != x.shape:
        buf = x.unsqueeze(0).repeat(depth, *([1] * x.dim())).contiguous()
        st[key] = buf
        st[key + "_ptr"] = 0
    p = (st[key + "_ptr"] + 1) % depth
    st[key + "_ptr"] = p
    buf[p] = x
    if fresh.any():
        buf[:, fresh] = x[fresh]
    return buf, p


def camera_depth(
    env: ManagerBasedRLEnv,
    sensor_name: str = "depth_cam",
    cutoff_distance: float = 2.5,
    min_depth: float = 0.05,
    noise: bool = True,
    ramp_steps: int = 120_000,
    pixel_noise_std: float = 0.02,
    salt_prob: float = 0.01,
    max_rects: int = 2,
    rect_size_range: tuple[float, float] = (0.1, 0.3),
    hold_prob: float = 0.05,
    max_lag: int = 2,
    fill_value: float = 1.0,
    history_offsets: list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Egocentric depth, optionally as a stack of past DELIVERED frames.

    Without ``history_offsets`` this is exactly the single-frame observation
    (``_camera_depth_frame``), shape ``(N, 1, H, W)``.

    With e.g. ``history_offsets=[0, 8, 16, 24, 32, 40, 48]`` it returns
    ``(N, K, H, W)``: channel k is the frame the policy received ``offsets[k]``
    steps ago. The buffer sits AFTER the lag / frame-hold model, so every stacked
    frame is one the robot would actually have been given at that step -- a
    deployment runtime has to keep the same buffer of its own preprocessed frames.

    Why a strided stack: in the box reference the torso pitches ~88 deg at the
    grasp, and a steeply pitched camera can lose the box for up to ~0.9 s before
    the lift. Offsets reaching 48 steps back always contain a frame that saw it.
    (The shipped 47.6 deg students use a single frame.)
    A new episode starts with every slot filled by its first frame.
    """
    img = _camera_depth_frame(
        env, sensor_name, cutoff_distance, min_depth, noise, ramp_steps, pixel_noise_std,
        salt_prob, max_rects, rect_size_range, hold_prob, max_lag, fill_value,
    )
    if not history_offsets:
        return img
    offsets = list(history_offsets)
    depth = max(offsets) + 1
    st = _STATE.setdefault(id(env), {})
    buf, p = _ring_push(st, "depth_hist", img[:, 0], depth, _just_reset(env))
    idx = [(p - o) % depth for o in offsets]
    return buf[idx].permute(1, 0, 2, 3).contiguous()


