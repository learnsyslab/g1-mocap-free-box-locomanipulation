# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared evaluation metrics for the SBTO/DynaRetarget object-tracking task.

Deliberately framework-free -- pure torch, no isaaclab imports -- so the
metric definitions can be read and tested on their own.

Protocol -- DynaRetarget Section IV-E / Table V:

  "A policy roll-out is deemed successful if the object pose remains within a
   predefined threshold of the desired object pose at every timestep of the
   trajectory, since the policy may continue to imitate the robot motion even
   when object tracking fails."

Two consequences that make this different from mjlab's own
tasks/tracking/scripts/evaluate.py, which scores success as
``truncated & ~terminated``:

  1. Success is gated on the OBJECT, not just on staying alive. Under mjlab's
     criterion an episode that drops the box in the first second and then
     mimics the body motion perfectly counts as a success -- precisely the
     case the paper says it designed its criterion to exclude. That is not
     hypothetical: with the object termination removed (as the paper
     prescribes), a policy can sit at ~0.5 m object error with essentially no
     terminations.
  2. The error must stay inside the threshold at EVERY timestep, so this
     tracks a running per-env max, not a mean.

Threshold caveat: the paper says "a predefined threshold" without giving the
number. The defaults here (0.10 m, 25 deg) are SBTO's own object-success
numbers from Section IV-B / Table III. Those are stated there as bounds on the
*average* error of a refined trajectory, not as a per-timestep RL bound, so
treat them as a defensible convention rather than a reproduction of the
paper's exact figure -- and keep the threshold fixed when comparing runs.
"""

from __future__ import annotations

import math

import torch

DEFAULT_POS_THRESHOLD_M = 0.10
DEFAULT_ROT_THRESHOLD_RAD = math.radians(25.0)


class TrackingEvalAccumulator:
    """Accumulates per-episode tracking metrics across a batch of envs.

    One env == one episode. Metrics are averaged only over the steps an env
    was still running (``active``), so envs that finish early do not get
    padded with stale values.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        pos_threshold: float = DEFAULT_POS_THRESHOLD_M,
        rot_threshold: float = DEFAULT_ROT_THRESHOLD_RAD,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.pos_threshold = pos_threshold
        self.rot_threshold = rot_threshold

        z = lambda: torch.zeros(num_envs, device=device)  # noqa: E731
        self._active_steps = z()
        self._sum_mpkpe = z()
        self._sum_r_mpkpe = z()
        self._sum_obj_pos = z()
        self._sum_obj_rot = z()
        # Running max over timesteps -- the paper's "at every timestep" gate.
        self._max_obj_pos = z()
        self._max_obj_rot = z()
        # First step's error per episode. Since max >= first, an episode already
        # outside the gate on step one can never pass -- so these give the exact
        # ceiling on success_rate for whatever reset distribution actually ran.
        self._first_obj_pos = z()
        self._first_obj_rot = z()
        self._seen_first = torch.zeros(num_envs, dtype=torch.bool, device=device)

        self._done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._completed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._terminated_early = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # Which termination term ended each episode, and on which step. A high
        # terminated-early rate is only actionable once you know whether it is
        # the anchor falling, the orientation flipping, or an end effector
        # drifting -- and whether it happens on step 1 or step 200.
        self._term_cause: dict[str, torch.Tensor] = {}
        self._term_step = torch.full((num_envs,), float("nan"), device=device)
        self._step = 0

    def update(
        self,
        active: torch.Tensor,
        *,
        mpkpe: torch.Tensor,
        r_mpkpe: torch.Tensor,
        obj_pos_error: torch.Tensor,
        obj_rot_error: torch.Tensor,
    ) -> None:
        """Fold in one timestep. ``active`` masks envs still running."""
        self._step += 1
        a = active.float()
        self._active_steps += a
        self._sum_mpkpe += torch.where(active, mpkpe, torch.zeros_like(mpkpe))
        self._sum_r_mpkpe += torch.where(active, r_mpkpe, torch.zeros_like(r_mpkpe))
        self._sum_obj_pos += torch.where(active, obj_pos_error, torch.zeros_like(obj_pos_error))
        self._sum_obj_rot += torch.where(active, obj_rot_error, torch.zeros_like(obj_rot_error))
        # Only envs still running can worsen their max.
        self._max_obj_pos = torch.where(active, torch.maximum(self._max_obj_pos, obj_pos_error), self._max_obj_pos)
        self._max_obj_rot = torch.where(active, torch.maximum(self._max_obj_rot, obj_rot_error), self._max_obj_rot)
        first = active & ~self._seen_first
        self._first_obj_pos = torch.where(first, obj_pos_error, self._first_obj_pos)
        self._first_obj_rot = torch.where(first, obj_rot_error, self._first_obj_rot)
        self._seen_first |= first

    def mark_done(
        self,
        newly_done: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        causes: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Record how each freshly finished env ended.

        ``truncated & ~terminated`` means the rollout reached the time limit
        without a termination firing -- i.e. it played the whole trajectory.

        ``causes`` maps termination-term name -> per-env bool for this step
        (see :func:`termination_causes`); it is attributed only to episodes
        finishing right now, so each episode is charged to the term that
        actually ended it.
        """
        fresh = newly_done & ~self._done
        if not bool(fresh.any()):
            return
        self._completed |= fresh & truncated & ~terminated
        self._terminated_early |= fresh & terminated
        for name, fired in (causes or {}).items():
            slot = self._term_cause.setdefault(name, torch.zeros_like(self._done))
            slot |= fresh & fired.reshape(-1).bool()
        step_now = torch.full_like(self._term_step, float(self._step))
        self._term_step = torch.where(fresh & terminated, step_now, self._term_step)
        self._done |= fresh

    @property
    def all_done(self) -> torch.Tensor:
        return self._done

    def compute(self) -> dict[str, float]:
        steps = self._active_steps.clamp(min=1.0)
        object_ok = (self._max_obj_pos <= self.pos_threshold) & (self._max_obj_rot <= self.rot_threshold)
        # Exact upper bound on success_rate: episodes whose first step already
        # breaches the gate cannot pass whatever happens next.
        reachable = (self._first_obj_pos <= self.pos_threshold) & (self._first_obj_rot <= self.rot_threshold)
        success = self._completed & object_ok
        term_steps = self._term_step[~torch.isnan(self._term_step)]

        # How much of the success rate is the threshold rather than the policy?
        # The paper fixes the criterion ("within a predefined threshold ... at
        # every timestep") but never gives the number, so the single headline
        # figure is only as meaningful as our guess at it. Sweeping is free --
        # the per-episode running maxima are already accumulated -- and it turns
        # the one unverifiable constant into a curve the reader can locate the
        # paper's number on.
        mean_obj_pos = self._sum_obj_pos / steps
        mean_obj_rot = self._sum_obj_rot / steps
        sweep = {
            f"{int(t * 100)}cm": (self._completed & (self._max_obj_pos <= t) & (self._max_obj_rot <= self.rot_threshold))
            .float()
            .mean()
            .item()
            for t in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
        }
        # Table III's criterion is an AVERAGE over the trajectory (Eq. 2), not a
        # per-timestep max -- reported here so the two readings of "10 cm / 25
        # deg" can be compared directly on the same rollouts.
        avg_criterion = (
            self._completed & (mean_obj_pos <= self.pos_threshold) & (mean_obj_rot <= self.rot_threshold)
        )
        return {
            "success_rate_by_pos_threshold": sweep,
            "success_rate_avg_criterion": avg_criterion.float().mean().item(),
            "termination_causes": {n: v.float().mean().item() for n, v in self._term_cause.items()},
            "termination_step_mean": term_steps.mean().item() if term_steps.numel() else float("nan"),
            "termination_step_min": term_steps.min().item() if term_steps.numel() else float("nan"),
            "termination_step_median": term_steps.median().item() if term_steps.numel() else float("nan"),
            # Table V columns.
            "success_rate": success.float().mean().item(),
            "success_rate_ceiling": reachable.float().mean().item(),
            "mpkpe_cm": (self._sum_mpkpe / steps).mean().item() * 100.0,
            "object_pos_error_cm": (self._sum_obj_pos / steps).mean().item() * 100.0,
            "object_ori_error_rad": (self._sum_obj_rot / steps).mean().item(),
            # Diagnostics: separates "fell over" from "kept walking but lost the box".
            "completed_rate": self._completed.float().mean().item(),
            "object_within_threshold_rate": object_ok.float().mean().item(),
            "terminated_early_rate": self._terminated_early.float().mean().item(),
            "r_mpkpe_cm": (self._sum_r_mpkpe / steps).mean().item() * 100.0,
            "mean_object_pos_peak_cm": self._max_obj_pos.mean().item() * 100.0,
            "num_episodes": float(self.num_envs),
            "pos_threshold_m": self.pos_threshold,
            "rot_threshold_rad": self.rot_threshold,
        }


def termination_causes(manager) -> dict[str, torch.Tensor]:
    """Per-term done flags for the current step, as ``{name: bool tensor}``.

    Framework-free on purpose: IsaacLab and mjlab both expose ``active_terms``
    and ``get_term(name)`` on their termination managers, but neither is
    imported here, so this stays duck-typed. Falls back to the private
    ``_term_dones`` dict for manager versions without the accessor.
    """
    out: dict[str, torch.Tensor] = {}
    for name in list(getattr(manager, "active_terms", None) or []):
        try:
            out[name] = manager.get_term(name)
        except Exception:  # noqa: BLE001 -- diagnostics must never break a run
            dones = getattr(manager, "_term_dones", None)
            if isinstance(dones, dict) and name in dones:
                out[name] = dones[name]
    return out


def quat_error_magnitude_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Geodesic angle between two wxyz quaternions, shape (..., 4) -> (...).

    Sign-agnostic (q and -q are the same rotation), which matters because the
    two backends store quaternions in different orders and may disagree on
    sign after their own conversions.
    """
    dot = (q1 * q2).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def quat_error_magnitude_xyzw(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Same, for xyzw-ordered quaternions (IsaacLab 6.x convention).

    The formula is order-independent (it is just |dot|), so this simply names
    the convention at the call site to keep the two evaluators readable.
    """
    return quat_error_magnitude_wxyz(q1, q2)


def quat_error_magnitude_2fold_z_xyzw(q_ref: torch.Tensor, q_act: torch.Tensor) -> torch.Tensor:
    """Angle to the CLOSER of two references equivalent under a half turn about z.

    The mirror of motion_object_orientation_error_symmetric_exp on the metric
    side: a rectangular box is unchanged by a half turn about its own vertical,
    so scoring it against only one of the two equivalent references charges up
    to 180 degrees for a box that is physically already correct. Folding caps
    the error at 90 degrees:

        reduced = min(|theta|, 180 - |theta|)

    Off by default in the evaluator. Applying it silently would make numbers
    incomparable with everything measured before it, and it is only the right
    metric for a policy whose reward folds the same way.

    Post-multiplying the reference by a half turn about z is, in xyzw order,
    (x, y, z, w) -> (y, -x, w, -z); rotating about the box's OWN vertical rather
    than the world's, which matches the reward. The two coincide while the box
    is upright and diverge once it tilts.
    """
    direct = quat_error_magnitude_xyzw(q_ref, q_act)
    x, y, z, w = q_ref.unbind(dim=-1)
    flipped_ref = torch.stack((y, -x, w, -z), dim=-1)
    flipped = quat_error_magnitude_xyzw(flipped_ref, q_act)
    return torch.minimum(direct, flipped)


def print_report(metrics: dict[str, float], title: str) -> None:
    width = 62
    print("\n" + "=" * width)
    print(title)
    print("=" * width)
    print(f"  {'success_rate':<30} {metrics['success_rate'] * 100:8.2f} %")
    # success_rate gates on the MAXIMUM error over the episode, and the object
    # starts at whatever offset the reset applied -- so an episode that begins
    # outside the gate can never pass, no matter how well the policy then
    # behaves. Under wide randomisation that ceiling dominates the number, so
    # print it alongside. See success_rate_avg_criterion for a measure that is
    # not bounded this way.
    ceil = metrics.get("success_rate_ceiling")
    if ceil is not None:
        frac = metrics["success_rate"] / ceil if ceil > 0 else float("nan")
        print(f"  {'  ^ ceiling for this reset dist':<30} {ceil * 100:8.2f} %"
              f"   ({frac * 100:.0f}% of it)")
    print(f"  {'mpkpe':<30} {metrics['mpkpe_cm']:8.2f} cm")
    print(f"  {'object_pos_error':<30} {metrics['object_pos_error_cm']:8.2f} cm")
    print(f"  {'object_ori_error':<30} {metrics['object_ori_error_rad']:8.4f} rad")
    print("-" * width)
    print(f"  {'completed (no termination)':<30} {metrics['completed_rate'] * 100:8.2f} %")
    print(f"  {'object within threshold':<30} {metrics['object_within_threshold_rate'] * 100:8.2f} %")
    print(f"  {'terminated early':<30} {metrics['terminated_early_rate'] * 100:8.2f} %")
    print(f"  {'r_mpkpe':<30} {metrics['r_mpkpe_cm']:8.2f} cm")
    print(f"  {'mean peak object pos error':<30} {metrics['mean_object_pos_peak_cm']:8.2f} cm")
    causes = {n: v for n, v in metrics.get("termination_causes", {}).items() if v > 0.0}
    if causes:
        print("-" * width)
        print("  why episodes ended (share of all episodes):")
        for name, share in sorted(causes.items(), key=lambda kv: -kv[1]):
            print(f"    {name:<28} {share * 100:8.2f} %")
        if metrics["termination_step_mean"] == metrics["termination_step_mean"]:  # not NaN
            print(
                f"    {'termination step':<28} "
                f"min {metrics['termination_step_min']:.0f}, "
                f"median {metrics['termination_step_median']:.0f}, "
                f"mean {metrics['termination_step_mean']:.1f}"
            )
    sweep = metrics.get("success_rate_by_pos_threshold", {})
    if sweep:
        print("-" * width)
        print("  success_rate vs object position gate (rot gate held fixed):")
        for label, rate in sweep.items():
            print(f"    max error <= {label:<22} {rate * 100:8.2f} %")
        print(
            f"    {'mean error <= gate (Table III)':<28} "
            f"{metrics['success_rate_avg_criterion'] * 100:8.2f} %"
        )
    print("-" * width)
    print(
        f"  thresholds: pos {metrics['pos_threshold_m'] * 100:.1f} cm, "
        f"rot {math.degrees(metrics['rot_threshold_rad']):.1f} deg   "
        f"episodes: {int(metrics['num_episodes'])}"
    )
    print("=" * width)
