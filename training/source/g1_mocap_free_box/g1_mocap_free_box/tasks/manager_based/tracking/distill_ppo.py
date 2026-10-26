# Copyright (c) 2026 Jim Yun-Jin Li and the Learning Systems and Robotics Lab (LSY), Technical University of Munich.
# Portions adapted from Atarilab's mjlab-distill (Copyright 2025 The mjlab Developers,
# Apache-2.0) and rsl_rl (Copyright ETH Zurich and NVIDIA CORPORATION, BSD-3-Clause);
# modified. See THIRD_PARTY_NOTICES.md.
#
# SPDX-License-Identifier: Apache-2.0

"""Stage 3: PPO with an annealed teacher-imitation term ("kickstarting").

Distillation (stage 2) can only ever COPY the teacher, and it copies through a
worse observation, so it lands below it -- the imitation gap. Our stage-2 run hit
exactly that: the behaviour loss converged flat at full corruption while the task
error stalled, leaving the student at 1.56% against the teacher's 90.53%.

This breaks that ceiling by changing the objective. The loss becomes

    PPO(true task reward) + coef * KL(teacher || student)

with ``coef`` annealed linearly 0.5 -> 0 over ``kickstart_anneal_iterations``.
Early on the teacher term stops PPO from wrecking the distilled policy before
reward has taught it anything; once the coefficient reaches zero it is pure RL
from a strong initialisation, free to exceed the teacher.

The critic is ASYMMETRIC: it reads the privileged ``critic`` group (310-D, the
same group the Stage 4 teacher's critic trained on, so it strict-loads), while
the actor still sees only depth + proprioception. That is what makes the value
estimates good enough to be useful despite the actor's partial observation.

Ported from ``DistillPPO`` in Atarilab's mjlab-distill (Apache-2.0). That version targets rsl-rl
5.4.0 and copies that version's ``PPO.update``; we run 5.0.1, so ``update`` here
is generated from OUR installed PPO source with the kickstart term spliced into
the loss. Revisit on any rsl-rl pin bump -- if the upstream loss line moves, the
generator asserts rather than silently producing a no-op.

Also unlike theirs: ``_raw_actor``/``_raw_critic``/``_raw_teacher`` are 5.4.0
compile-wrapper internals that do not exist in 5.0.1, so the models are
addressed directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer
from tensordict import TensorDict

_KICKSTART_LOSS_TYPES = ("kl", "nll")


class DistillPPO(PPO):
    """PPO plus an annealed kickstarting loss against a frozen teacher."""

    teacher: MLPModel
    teacher_loaded: bool = False

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        teacher: MLPModel,
        storage: RolloutStorage,
        kickstart_coef_initial: float = 0.5,
        kickstart_coef_final: float = 0.0,
        kickstart_anneal_iterations: int = 1000,
        kickstart_loss_type: str = "kl",
        kickstart_mask_group: str | None = None,
        **kwargs,
    ) -> None:
        if kwargs.get("rnd_cfg") is not None:
            raise ValueError("RND is not compatible with DistillPPO.")
        if kwargs.get("symmetry_cfg") is not None:
            raise ValueError("Symmetry is not compatible with DistillPPO.")
        if actor.is_recurrent or critic.is_recurrent or teacher.is_recurrent:
            raise ValueError(
                "DistillPPO does not support recurrent models: the CNN actor cannot"
                " consume PPO's padded recurrent minibatches."
            )
        if kickstart_loss_type not in _KICKSTART_LOSS_TYPES:
            raise ValueError(
                f"Unknown kickstart loss type: {kickstart_loss_type}."
                f" Supported: {list(_KICKSTART_LOSS_TYPES)}"
            )
        super().__init__(actor, critic, storage, **kwargs)

        # Frozen: no grads, never in the optimizer, always eval.
        self.teacher = teacher.to(self.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.kickstart_coef_initial = kickstart_coef_initial
        self.kickstart_coef_final = kickstart_coef_final
        self.kickstart_anneal_iterations = kickstart_anneal_iterations
        self.kickstart_loss_type = kickstart_loss_type
        # Optional (N, 1) observation group in [0, 1]: the teacher term is then a
        # weighted mean over the samples where the teacher is trustworthy, and
        # zero elsewhere (see mdp.teacher_in_distribution). None: every sample.
        self.kickstart_mask_group = kickstart_mask_group
        self.num_updates = 0

    @property
    def kickstart_coef(self) -> float:
        """Linearly annealed kickstarting coefficient."""
        frac = min(1.0, self.num_updates / max(1, self.kickstart_anneal_iterations))
        return (
            self.kickstart_coef_initial
            + (self.kickstart_coef_final - self.kickstart_coef_initial) * frac
        )

    def _kickstart_loss(self, batch, distribution_params) -> torch.Tensor:
        """Teacher-imitation loss for the current minibatch.

        Called straight after the actor's minibatch forward, so the actor's
        distribution state matches ``distribution_params``.
        """
        assert batch.observations is not None
        with torch.no_grad():
            teacher_mean = self.teacher(batch.observations)
        if self.kickstart_loss_type == "kl":
            teacher_dist = self.teacher.distribution
            assert teacher_dist is not None
            with torch.no_grad():
                teacher_dist.update(teacher_mean)
                teacher_params = teacher_dist.params
            # Forward KL(teacher || student); grads flow through the student.
            per_sample = self.actor.get_kl_divergence(teacher_params, distribution_params)
        else:
            per_sample = -self.actor.get_output_log_prob(teacher_mean)
        if self.kickstart_mask_group is None:
            return per_sample.mean()
        mask = batch.observations[self.kickstart_mask_group].reshape(-1).to(per_sample.dtype)
        self.last_kickstart_mask_frac = mask.mean().item()
        # Normalised by the masked count, not the batch size, so the effective
        # coefficient on in-distribution samples does not shrink as the
        # distribution widens.
        return (per_sample.reshape(-1) * mask).sum() / mask.sum().clamp(min=1.0)

    def train_mode(self) -> None:
        super().train_mode()
        self.teacher.eval()  # the teacher never trains

    def eval_mode(self) -> None:
        super().eval_mode()
        self.teacher.eval()

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load with checkpoint-kind auto-detection.

        A stage-2 distillation checkpoint carries ``student_state_dict`` and
        ``teacher_state_dict``: the actor is initialised from the student and the
        teacher from the teacher. It carries NO critic, so the privileged critic
        must be loaded separately from the Stage 4 PPO run -- see
        ``load_critic_from``. Without that the critic starts from scratch and the
        first hundred iterations are spent fitting a value function while the
        kickstart term is still large.
        """
        if load_cfg is None and "student_state_dict" in loaded_dict:
            self.actor.load_state_dict(loaded_dict["student_state_dict"], strict=strict)
            self.teacher.load_state_dict(loaded_dict["teacher_state_dict"], strict=strict)
            self.teacher_loaded = True
            # A stage-2 checkpoint carries no critic. If one has been merged in
            # (see scripts/student/merge_finetune_checkpoint.py) take it: starting PPO
            # with a randomly initialised value function wastes the early
            # iterations fitting it, exactly when the kickstart term is largest
            # and the policy is most fragile.
            if "critic_state_dict" in loaded_dict:
                self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
                print("[INFO] privileged critic loaded from the merged checkpoint")
            else:
                print(
                    "[WARN] no critic_state_dict in this checkpoint -- the privileged"
                    " critic starts from scratch. Merge the Stage 4 critic in first."
                )
            return False  # fresh iteration counter: this is not a resume
        if load_cfg is None and "teacher_state_dict" in loaded_dict:
            self.teacher.load_state_dict(loaded_dict["teacher_state_dict"], strict=strict)
            self.teacher_loaded = True
            self.num_updates = loaded_dict.get("num_updates", 0)
            return super().load(loaded_dict, None, strict)
        if load_cfg is None:
            actor_sd = loaded_dict.get("actor_state_dict", {})
            # Distinguish OUR checkpoint from a raw Stage 4 PPO one by whether
            # the saved actor has CNN encoder weights. Falling back on "no
            # teacher_state_dict => raw PPO" is wrong for checkpoints written
            # before save() carried the teacher: those have a CNN actor, and
            # pushing it into the MLP teacher dies on a shape mismatch
            # (512x218 vs 512x172). Checking the weights themselves is robust to
            # how the file was produced.
            if any(k.startswith("cnns.") for k in actor_sd):
                self.actor.load_state_dict(actor_sd, strict=strict)
                if "critic_state_dict" in loaded_dict:
                    self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
                self.num_updates = loaded_dict.get("num_updates", 0)
                # The teacher is absent from such a checkpoint. Harmless for
                # EVALUATION (get_policy returns the actor), but it must not be
                # silently trusted for resumed TRAINING, where the kickstart term
                # would then imitate a randomly initialised network.
                print(
                    "[WARN] no teacher_state_dict in this checkpoint: fine for evaluation,"
                    " but do NOT resume training from it -- the kickstart target would be"
                    " a random network."
                )
                return True
            # Raw PPO checkpoint: pure kickstarting from a fresh actor.
            self.teacher.load_state_dict(actor_sd, strict=strict)
            self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
            self.teacher_loaded = True
            return False
        if load_cfg.get("teacher"):
            self.teacher.load_state_dict(loaded_dict["teacher_state_dict"], strict=strict)
            self.teacher_loaded = True
        if (
            load_cfg.get("actor")
            and "actor_state_dict" not in loaded_dict
            and "student_state_dict" in loaded_dict
        ):
            loaded_dict = dict(loaded_dict)
            loaded_dict["actor_state_dict"] = loaded_dict["student_state_dict"]
        ppo_cfg = {k: v for k, v in load_cfg.items() if k != "teacher"}
        return super().load(loaded_dict, ppo_cfg, strict)

    def load_critic_from(self, path: str, strict: bool = True) -> None:
        """Initialise the privileged critic from a Stage 4 PPO checkpoint.

        The critic reads the same 310-D privileged group the Stage 4 critic
        trained on, so it strict-loads. Do NOT remap it to the actor's group --
        the whole point of the asymmetric critic is that it keeps the privileged
        observation the actor gave up.
        """
        loaded = torch.load(path, map_location=self.device, weights_only=False)
        if "critic_state_dict" not in loaded:
            raise ValueError(f"{path} has no critic_state_dict")
        self.critic.load_state_dict(loaded["critic_state_dict"], strict=strict)
        print(f"[INFO] privileged critic loaded from {path}")

    def save(self) -> dict:
        """Add the teacher and the anneal clock to what PPO saves.

        NOT ``state_dict``: the runner calls ``alg.save()``. Overriding
        ``state_dict`` looks right and is dead code -- checkpoints then lack
        ``teacher_state_dict``, and reloading one falls through ``load``'s
        branches to the raw-PPO case, which tries to push the CNN actor's
        weights into the MLP teacher and dies on a shape mismatch. That is how
        this was found.
        """
        saved = super().save()
        saved["teacher_state_dict"] = self.teacher.state_dict()
        saved["num_updates"] = self.num_updates
        return saved

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "DistillPPO":
        """Build actor (depth student), privileged critic, and frozen teacher."""
        alg_class = resolve_callable(cfg["algorithm"].pop("class_name"))
        actor_class = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class = resolve_callable(cfg["critic"].pop("class_name"))
        teacher_class = resolve_callable(cfg["teacher"].pop("class_name"))

        default_sets = ["actor", "critic", "teacher"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        cfg["algorithm"]["rnd_cfg"] = None
        cfg["algorithm"]["symmetry_cfg"] = None
        # Stock PPO.construct_algorithm pops this before forwarding the rest to
        # PPO.__init__, which does not accept it. Sharing encoders is refused
        # here rather than silently ignored: the critic reads the privileged
        # group and has no image input, so there is no encoder to share, and a
        # config asking for it is a mistake worth surfacing.
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            raise ValueError(
                "share_cnn_encoders is not supported by DistillPPO: the privileged"
                " critic has no 2D observation group to share an encoder with."
            )

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        teacher = teacher_class(obs, cfg["obs_groups"], "teacher", env.num_actions, **cfg["teacher"]).to(device)
        print(f"Actor: {actor}\nCritic: {critic}\nTeacher: {teacher}")

        # "rl", NOT some other string: the training_type selects which fields the
        # storage allocates. PPO needs values / returns / advantages; passing an
        # unrecognised type yields a storage without them and the run dies on the
        # first update with AttributeError: 'RolloutStorage' object has no
        # attribute 'values'. Distillation uses "distillation" here.
        storage = RolloutStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )
        return alg_class(
            actor, critic, teacher, storage, device=device,
            **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"],
        )

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        # The anneal clock. PPO also has a LOCAL `num_updates` further down
        # (num_learning_epochs * num_mini_batches, used for averaging) -- these
        # are different quantities and must not be conflated: this one counts
        # update() CALLS, i.e. training iterations, which is what the kickstart
        # coefficient anneals over.
        self.num_updates += 1
        mean_value_loss = 0
        mean_kickstart_loss = 0.0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # Get mini batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                # Compute number of augmentations per sample
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            # Note: We only keep the distribution parameters and entropy of the first augmentation (the original one)
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # --- kickstarting -------------------------------------------------
            # The teacher term is what keeps early PPO from destroying the
            # distilled policy before it has learned anything from reward. It is
            # annealed to zero, after which this is plain PPO from a strong
            # initialisation -- that is the part that can EXCEED the teacher,
            # which distillation structurally cannot.
            kick_coef = self.kickstart_coef
            if kick_coef > 0.0:
                kickstart_loss = self._kickstart_loss(batch, distribution_params)
                loss = loss + kick_coef * kickstart_loss
                mean_kickstart_loss += kickstart_loss.item()

            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions = self.actor(batch.observations.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the batch.actions from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions[original_batch_size:], actions_mean_symm.detach()[original_batch_size:]
                )
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            if self.rnd:
                # Extract the rnd_state
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])  # type: ignore
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "kickstart": mean_kickstart_loss / max(num_updates, 1),
            "kickstart_coef": self.kickstart_coef,
            # Share of samples the teacher term applied to (1.0 without a mask).
            "kickstart_mask_frac": getattr(self, "last_kickstart_mask_frac", 1.0),
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict
