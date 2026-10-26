# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
# Portions adapted from Atarilab's mjlab-distill (Copyright 2025 The mjlab Developers,
# Apache-2.0) and rsl_rl (Copyright ETH Zurich and NVIDIA CORPORATION, BSD-3-Clause);
# modified. See THIRD_PARTY_NOTICES.md.
#
# SPDX-License-Identifier: Apache-2.0

"""Spatial-softmax CNN student, ported from Atarilab's mjlab-distill (Apache-2.0).

A plain CNN flattens its final feature map, which asks the MLP to recover
"where is the box" from an unstructured vector of activations. Spatial softmax
instead reduces each channel to its expected (x, y) location -- a soft argmax --
so the encoder hands the MLP explicit keypoint coordinates. That is the right
inductive bias here: the student's whole job is to localise one object in the
frame, and keypoints are exactly what the privileged terms it is replacing
(``object_pose_b`` / ``object_pose_error_b``) encoded directly.

The recurrent (GRU) variant is deliberately NOT ported. Their ablation found it
helps only anchor-position tracking (memory through occlusion), and it cannot be
used as a PPO actor later -- padded 5-D minibatches -- nor exported to ONNX. It
would foreclose the kickstarted fine-tune stage for a narrow gain.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models.cnn_model import CNNModel
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import CNN, MLP
from tensordict import TensorDict


class SpatialSoftmax(nn.Module):
    """Spatial soft-argmax over feature maps: ``(B, C, H, W) -> (B, C * 2)``."""

    def __init__(self, height: int, width: int, temperature: float = 1.0) -> None:
        super().__init__()
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height),
            torch.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        # Buffers, so they follow the module across devices.
        self.register_buffer("pos_x", pos_x.reshape(1, 1, -1))
        self.register_buffer("pos_y", pos_y.reshape(1, 1, -1))
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        features = x.reshape(B, C, -1)
        weights = torch.softmax(features / self.temperature, dim=-1)
        expected_x = (weights * self.pos_x).sum(dim=-1)
        expected_y = (weights * self.pos_y).sum(dim=-1)
        return torch.stack([expected_x, expected_y], dim=-1).reshape(B, C * 2)


class SpatialSoftmaxCNN(nn.Module):
    """``rsl_rl.modules.CNN`` with spatial-softmax pooling instead of flatten."""

    def __init__(
        self,
        input_dim: tuple[int, int],
        input_channels: int,
        temperature: float = 1.0,
        **cnn_kwargs: Any,
    ) -> None:
        super().__init__()
        # Spatial softmax replaces pooling/flattening, so those are forced.
        cnn_kwargs.pop("global_pool", None)
        cnn_kwargs.pop("flatten", None)
        self.cnn = CNN(
            input_dim=input_dim,
            input_channels=input_channels,
            global_pool="none",
            flatten=False,
            **cnn_kwargs,
        )
        out_h, out_w = self.cnn.output_dim
        num_channels: int = self.cnn.output_channels
        self.spatial_softmax = SpatialSoftmax(out_h, out_w, temperature)
        self._output_dim = num_channels * 2

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def output_channels(self) -> None:
        """Always None: the output is flattened keypoints, as CNNModel requires."""
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_softmax(self.cnn(x))


class SpatialSoftmaxCNNModel(CNNModel):
    """Drop-in ``CNNModel`` whose encoders use spatial softmax.

    ``aux_cfg`` optionally adds an auxiliary head over the latent that regresses
    a privileged observation group -- supervised state estimation alongside
    behaviour cloning. Their ablation found it neutral on its own at 6k
    iterations, so it is off by default and kept only for experimentation.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        cnn_cfg: dict[str, dict] | dict[str, Any],
        cnns: nn.ModuleDict | None = None,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict[str, Any] | None = None,
        aux_cfg: dict[str, Any] | None = None,
    ) -> None:
        # Splits 1D vs 2D groups; sets obs_groups_2d / obs_dims_2d / obs_channels_2d.
        self._get_obs_dim(obs, obs_groups, obs_set)

        if cnns is not None:
            if set(cnns.keys()) != set(self.obs_groups_2d):
                raise ValueError("Shared CNN encoders must cover the same 2D groups.")
            _cnns: dict[str, nn.Module] | nn.ModuleDict = cnns
        else:
            if not all(isinstance(v, dict) for v in cnn_cfg.values()):
                cnn_cfg = {group: cnn_cfg for group in self.obs_groups_2d}
            if len(cnn_cfg) != len(self.obs_groups_2d):
                raise ValueError("One CNN config per 2D observation group is required.")
            _cnns = {}
            for idx, group in enumerate(self.obs_groups_2d):
                group_cfg = dict(cnn_cfg[group])
                group_cfg.pop("spatial_softmax", None)
                temperature = group_cfg.pop("spatial_softmax_temperature", 1.0)
                _cnns[group] = SpatialSoftmaxCNN(
                    input_dim=self.obs_dims_2d[idx],
                    input_channels=self.obs_channels_2d[idx],
                    temperature=temperature,
                    **group_cfg,
                )

        self.cnn_latent_dim = 0
        for cnn in _cnns.values():
            if cnn.output_channels is not None:
                raise ValueError("CNN output must be flattened before the MLP.")
            self.cnn_latent_dim += int(cnn.output_dim)

        # Skip CNNModel.__init__ (it would build plain CNNs); go to MLPModel.
        MLPModel.__init__(
            self,
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        self.cnns = _cnns if isinstance(_cnns, nn.ModuleDict) else nn.ModuleDict(_cnns)

        self.aux_head: nn.Module | None = None
        if aux_cfg is not None:
            target_group = aux_cfg["target_group"]
            if target_group not in obs.keys():
                raise ValueError(
                    f"aux_cfg target_group {target_group!r} is not an observation group;"
                    f" available: {sorted(obs.keys())}"
                )
            target = obs[target_group]
            if target.ndim != 2:
                raise ValueError(
                    f"aux_cfg target_group {target_group!r} must be 1D, got {tuple(target.shape)}"
                )
            self.aux_head = MLP(
                self._get_latent_dim(),
                target.shape[-1],
                tuple(aux_cfg.get("hidden_dims", (128,))),
                aux_cfg.get("activation", "elu"),
            )
