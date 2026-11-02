"""
KL-Cov Actor: standard Verl actor with KL-Cov loss.

KL-Cov does NOT need entropy — it only needs (old_log_prob, log_prob, advantages).
We inherit DataParallelPPOActor and override update_policy to use our loss.
"""

import itertools
import logging
import os

import torch

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.workers.actor.dp_actor import DataParallelPPOActor

from .core_algos import compute_policy_loss_kl_cov

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class KLCovActor(DataParallelPPOActor):
    """Actor with KL-Cov loss (Entropy Mechanism paper)."""

    @GPUMemoryLogger(role="kl_cov actor update", logger=logger)
    def update_policy(self, data: DataProto):
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = self.config.use_dynamic_bsz

        kl_cov_cfg = data.meta_info.get("kl_cov", {})
        kl_cov_ratio = float(kl_cov_cfg.get("kl_cov_ratio", 0.0002))
        ppo_kl_coef = float(kl_cov_cfg.get("ppo_kl_coef", 1.0))

        select_keys = [
            "responses", "input_ids", "attention_mask", "position_ids",
            "old_log_probs", "advantages", "response_mask",
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

        for micro_data in micro_batches:
            if isinstance(micro_data, DataProto):
                micro_data = micro_data.batch
            micro_data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in micro_data.items()}

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                response_length = micro_data["responses"].size(-1)
                old_log_prob = micro_data["old_log_probs"]
                advantages = micro_data["advantages"]
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
                    micro_batch=micro_data, temperature=temperature, calculate_entropy=(entropy_coeff != 0)
                )

                pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, extra_stats = compute_policy_loss_kl_cov(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    response_mask=response_mask,
                    cliprange_low=clip_ratio_low,
                    cliprange_high=clip_ratio_high,
                    clip_ratio_c=clip_ratio_c,
                    loss_agg_mode=loss_agg_mode,
                    kl_cov_ratio=kl_cov_ratio,
                    ppo_kl_coef=ppo_kl_coef,
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
                step_data.update(extra_stats)
                append_to_dict(metrics, step_data)

        grad_norm = self._optimizer_step()
        append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
