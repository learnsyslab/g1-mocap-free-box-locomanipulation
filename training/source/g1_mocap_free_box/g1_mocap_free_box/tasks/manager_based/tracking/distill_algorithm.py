# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
# Portions adapted from Atarilab's mjlab-distill (Copyright 2025 The mjlab Developers,
# Apache-2.0) and rsl_rl (Copyright ETH Zurich and NVIDIA CORPORATION, BSD-3-Clause);
# modified. See THIRD_PARTY_NOTICES.md.
#
# SPDX-License-Identifier: Apache-2.0

"""Distillation with distributional losses (NLL / KL), ported from Atarilab's mjlab-distill (Apache-2.0).

Stock rsl-rl ``Distillation`` implements only ``mse`` and ``huber``, both of
which regress the teacher's action as a point. mjlab-distill's July 2026
ablation found that choice dominates every other knob in the recipe:

    NLL vs MSE  -- joint error 0.74 vs 1.36 (halved), body-rot error down 56%,
                   ee-pos terminations 0.037 vs 2.65 (all but eliminated).
    KL          -- second best.
    MSE         -- worst, consistent with Rusu et al. 2016.

So using the stock algorithm unchanged would silently select the WORST variant.
This subclass adds the two distributional losses and an optional auxiliary
state-estimation head, overriding only ``_batch_losses`` and ``update``.

``update`` is an adapted copy of rsl-rl 5.0.1 ``Distillation.update`` with the
per-batch loss routed through ``_batch_losses``; the epoch loop, gradient
accumulation, hidden-state handling and multi-GPU reduction are unchanged.
Revisit it when the rsl-rl pin moves.

``nll``/``kl`` both require a plain ``GaussianDistribution`` (state-independent
std) on BOTH models: the stored teacher label is its deterministic forward
output, which is only the Gaussian mean under that assumption.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from rsl_rl.algorithms import Distillation
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.storage import RolloutStorage

_LOSS_TYPES = ("mse", "huber", "nll", "kl")


class DistributionalDistillation(Distillation):
    """``Distillation`` with NLL / KL losses and an optional aux head."""

    def __init__(
        self,
        student: MLPModel,
        teacher: MLPModel,
        storage: RolloutStorage,
        loss_type: str = "nll",
        aux_loss_coef: float = 0.0,
        aux_target_group: str = "aux_targets",
        **kwargs,
    ) -> None:
        if loss_type not in _LOSS_TYPES:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported: {list(_LOSS_TYPES)}")
        # The base class only knows mse/huber. Hand it a valid placeholder and
        # keep the real type here; self.loss_fn goes unused for nll/kl.
        base_loss = loss_type if loss_type in ("mse", "huber") else "mse"
        super().__init__(student, teacher, storage, loss_type=base_loss, **kwargs)
        self.loss_type = loss_type
        self.aux_loss_coef = aux_loss_coef
        self.aux_target_group = aux_target_group

        if loss_type in ("nll", "kl"):
            for name, model in (("student", student), ("teacher", teacher)):
                if type(model.distribution) is not GaussianDistribution:
                    raise ValueError(
                        f"loss_type={loss_type!r} needs the {name} to use a plain"
                        " GaussianDistribution (state-independent std), got"
                        f" {type(model.distribution).__name__}."
                    )
        if aux_loss_coef > 0.0 and not isinstance(getattr(student, "aux_head", None), nn.Module):
            raise ValueError("aux_loss_coef > 0 requires a student exposing an 'aux_head' module.")

    def _batch_losses(self, batch: RolloutStorage.Batch) -> dict[str, torch.Tensor]:
        obs = batch.observations
        teacher_actions = batch.privileged_actions
        assert obs is not None and teacher_actions is not None

        if self.aux_loss_coef > 0.0:
            # One shared latent pass for both heads, so a recurrent student
            # advances its hidden state exactly once -- as forward() would.
            latent = self.student.get_latent(obs)
            pred = self.student.mlp(latent)
        else:
            latent = None
            pred = self.student(obs)

        if self.loss_type in ("mse", "huber"):
            behavior = self.loss_fn(pred, teacher_actions)
        else:
            dist = self.student.distribution
            assert dist is not None
            dist.update(pred)
            if self.loss_type == "nll":
                behavior = -dist.log_prob(teacher_actions).mean()
            else:
                student_params = dist.params
                teacher_dist = self.teacher.distribution
                assert teacher_dist is not None
                with torch.no_grad():
                    teacher_dist.update(teacher_actions)
                    teacher_params = teacher_dist.params
                behavior = dist.kl_divergence(teacher_params, student_params).mean()

        losses = {"behavior": behavior}
        if self.aux_loss_coef > 0.0:
            assert latent is not None
            aux_head = cast(nn.Module, self.student.aux_head)
            aux_target = cast(torch.Tensor, obs[self.aux_target_group])
            losses["aux"] = nn.functional.mse_loss(aux_head(latent), aux_target)
        return losses

    def update(self) -> dict[str, float]:
        """Adapted from rsl-rl 5.0.1 Distillation.update; loss routed via _batch_losses."""
        self.num_updates += 1
        mean_losses = {"behavior": 0.0}
        if self.aux_loss_coef > 0.0:
            mean_losses["aux"] = 0.0
        loss: torch.Tensor | None = None
        cnt = 0

        for _ in range(self.num_learning_epochs):
            self.student.reset(hidden_state=self.last_hidden_states[0])
            self.teacher.reset(hidden_state=self.last_hidden_states[1])
            self.student.detach_hidden_state()
            for batch in self.storage.generator():
                losses = self._batch_losses(batch)
                batch_loss = losses["behavior"]
                if self.aux_loss_coef > 0.0:
                    batch_loss = batch_loss + self.aux_loss_coef * losses["aux"]

                loss = batch_loss if loss is None else loss + batch_loss
                for key, value in losses.items():
                    mean_losses[key] += value.item()
                cnt += 1

                if cnt % self.gradient_length == 0:
                    assert loss is not None
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.student.detach_hidden_state()
                    loss = None

                dones = batch.dones.view(-1)
                self.student.reset(dones)
                self.teacher.reset(dones)
                self.student.detach_hidden_state(dones)

        self.storage.clear()
        self.last_hidden_states = (
            self.student.get_hidden_state(),
            self.teacher.get_hidden_state(),
        )
        self.student.detach_hidden_state()
        return {key: value / max(cnt, 1) for key, value in mean_losses.items()}
