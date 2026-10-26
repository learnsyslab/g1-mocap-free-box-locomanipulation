# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
# Portions adapted from unitree_rl_lab's tasks/mimic (Apache-2.0), itself based on
# whole_body_tracking (MIT); modified. See THIRD_PARTY_NOTICES.md.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    # Runtime asset/sensor classes pull in USD at import time. Hydra imports
    # this config before SimulationApp starts, and a second USD init aborts
    # the process inside TfEnum::_AddName -- so they stay type-only here.
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    action_term_name: str = "JointPositionAction",
):
    """Randomize the joint default positions, standing in for calibration error.

    The action term's offset has to move with the default pose, or the policy
    keeps commanding around the un-randomized one. Its offset is indexed by
    action column, not by articulation joint id -- the two only coincide when
    the action term drives every joint, so the columns are resolved explicitly.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    if pos_distribution_params is None:
        return

    joint_ids = asset_cfg.joint_ids
    if joint_ids == slice(None):
        joint_ids = list(range(asset.num_joints))
    joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=asset.device)

    pos = asset.data.default_joint_pos.to(asset.device).clone()
    pos = _randomize_prop_by_op(
        pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
    )[env_ids][:, joint_ids]
    asset.data.default_joint_pos[env_ids[:, None], joint_ids] = pos

    # Map the randomized joints onto the action term's columns, dropping any
    # joint the term does not drive. A residual action term has no tensor
    # offset to shift -- its offset is the motion's PD target, recomputed every
    # step -- so there is nothing to keep in sync.
    action_term = env.action_manager.get_term(action_term_name)
    if not isinstance(action_term._offset, torch.Tensor):
        return
    action_joint_ids = action_term._joint_ids
    if action_joint_ids == slice(None):
        action_joint_ids = list(range(asset.num_joints))
    column_of = {int(joint_id): column for column, joint_id in enumerate(action_joint_ids)}

    driven = [(i, column_of[int(j)]) for i, j in enumerate(joint_ids) if int(j) in column_of]
    if not driven:
        return
    src, columns = (torch.tensor(t, dtype=torch.long, device=asset.device) for t in zip(*driven))
    action_term._offset[env_ids[:, None], columns] = pos[:, src]


# randomize_rigid_body_com is intentionally not redefined here: IsaacLab 6.x
# ships its own ManagerTermBase version with the same (com_range, asset_cfg)
# signature, and the older free-function form broke when root_physx_view
# stopped returning torch tensors. The star-import in mdp/__init__.py picks
# up the built-in.


def hide_prim_visual(env, env_ids, prim_suffix: str = "Robot/torso_link/head_link") -> None:
    """Make a prim invisible in every environment, at startup.

    Used to hide the G1's head from its own depth camera. This is FAITHFUL, not
    a hack: the real RealSense sits behind the visor and looks out through it,
    while the mesh models the visor as solid geometry. Left in place, the depth
    image is a uniform wall at ~1.4 cm and carries no information at all.

    Every env is walked explicitly rather than relying on clone-time sharing --
    whether cloned prims reference env_0 or own their geometry depends on the
    cloner's copy_from_source setting, and being wrong there would silently
    leave most envs blind.
    """
    from pxr import UsdGeom

    stage = env.sim.stage
    hidden = 0
    for i in range(env.num_envs):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/{prim_suffix}")
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden += 1
    print(f"[INFO] hide_prim_visual: hid {prim_suffix} in {hidden}/{env.num_envs} envs")
