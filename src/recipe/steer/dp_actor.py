"""
STEER Actor: inherits EntropyControlledActor (for compute_log_prob_with_entropy)
and overrides update_policy to apply per-token STEER loss reweighting.

This means:
- Rollout phase: computes entropys (+ t1/t2/mass_k, unused but cheap) via parent
- Update phase: uses STEER's Ω-based token weights on the PPO loss
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from recipe.papo.dp_actor import EntropyControlledActor

from .core_algos import compute_policy_loss_steer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SteerActor(EntropyControlledActor):
    """Actor with STEER token reweighting in the policy loss."""

    @GPUMemoryLogger(role="steer actor update", logger=logger)
    def update_policy(self, data: DataProto):
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = self.config.use_dynamic_bsz

        steer_cfg = data.meta_info.get("steer", {})
        token_weight_min = float(steer_cfg.get("token_weight_min", 0.8))
        token_weight_max = float(steer_cfg.get("token_weight_max", 1.2))
        linear = bool(steer_cfg.get("linear", True))
        entropy_control_mode = str(steer_cfg.get("entropy_control_mode", "symmetric"))

        select_keys = [
            "responses", "input_ids", "attention_mask", "position_ids",
            "old_log_probs", "advantages", "entropys", "response_mask",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(self.config.ppo_mini_batch_size)

        self.actor_optimizer.zero_grad()
        metrics = {}

        n_micro = len(micro_batches)
        if not use_dynamic_bsz:
            self.gradient_accumulation = n_micro

        for i, micro_data in enumerate(micro_batches):
            if isinstance(micro_data, DataProto):
                micro_data = micro_data.batch
            micro_data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in micro_data.items()}

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                response_length = micro_data["responses"].size(-1)
                old_log_prob = micro_data["old_log_probs"]
                advantages = micro_data["advantages"]
                entropys = micro_data["entropys"]
                response_mask = micro_data.get("response_mask")
                if response_mask is None:
                    attention_mask = micro_data["attention_mask"]
                    response_mask = attention_mask[:, -response_length:]

                clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else self.config.clip_ratio
                clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else self.config.clip_ratio
                clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                entropy_coeff = self.config.entropy_coeff
                loss_agg_mode = self.config.loss_agg_mode

                entropy, log_prob = self._forward_micro_batch(
                    micro_batch=micro_data, temperature=temperature, calculate_entropy=True
                )

                pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, tw_stats = compute_policy_loss_steer(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    response_mask=response_mask,
                    entropys=entropys,
                    cliprange_low=clip_ratio_low,
                    cliprange_high=clip_ratio_high,
                    clip_ratio_c=clip_ratio_c,
                    loss_agg_mode=loss_agg_mode,
                    token_weight_min=token_weight_min,
                    token_weight_max=token_weight_max,
                    linear=linear,
                    entropy_control_mode=entropy_control_mode,
                )

                policy_loss = pg_loss
                if entropy_coeff != 0 and entropy is not None:
                    entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                    policy_loss = pg_loss - entropy_loss * entropy_coeff

                if self.config.use_kl_loss:
                    ref_log_prob = micro_data["ref_log_prob"]
                    kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)
                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics["actor/kl_loss"] = kl_loss.detach().item()

                if use_dynamic_bsz:
                    loss = policy_loss * (len(micro_data["responses"]) / self.config.ppo_mini_batch_size)
                else:
                    loss = policy_loss / self.gradient_accumulation
                loss.backward()

                step_data = {
                    "actor/pg_loss": pg_loss.detach().item(),
                    "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                    "actor/ppo_kl": ppo_kl.detach().item(),
                    "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                }
                step_data.update(tw_stats)
                append_to_dict(metrics, step_data)

        # Optimizer step AFTER all micro-batches (correct gradient accumulation)
        grad_norm = self._optimizer_step()
        append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
