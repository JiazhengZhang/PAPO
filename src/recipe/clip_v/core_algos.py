"""
Clip-Cov loss from "On the Entropy Dynamics in Reinforcement Fine-Tuning of LLMs"
(Trinity-RFT/examples/entropy)

Implementation: Clip-B (batch-normalized covariance clipping).
Selects tokens where Cov(advantage, log_prob) is in [lb, ub] and randomly zeros their
gradient contribution, effectively clipping the entropy-decreasing updates.

Ported from STEER/verl/trainer/ppo/core_algos.py @register_policy_loss("clip_cov").
"""

import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss


def compute_policy_loss_clip_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_low: float = 0.2,
    cliprange_high: float = 0.28,
    clip_ratio_c: float = 3.0,
    loss_agg_mode: str = "token-mean",
    clip_cov_ratio: float = 0.0002,
    clip_cov_ub: float = 5.0,
    clip_cov_lb: float = 1.0,
):
    """PPO loss with Clip-Cov: zero out gradient for tokens with high Cov(adv, log_prob).

    Identifies tokens where the batch-normalized covariance Cov(A, log π)
    falls in [clip_cov_lb, clip_cov_ub], then randomly selects up to
    clip_cov_ratio of valid tokens from this set and zeros their loss contribution.

    Returns: (pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, extra_stats)
    """
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)

    # Standard PPO clip: which tokens are clipped by the ratio constraint
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)
    pg_clipfrac_standard = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    # Compute batch-normalized covariance: Cov(A, log_prob) per token
    adv_mean = verl_F.masked_mean(advantages, response_mask)
    logp_mean = verl_F.masked_mean(log_prob.detach(), response_mask)
    cov_all = (advantages - adv_mean) * (log_prob - logp_mean)

    # Mask out invalid tokens and already-clipped tokens
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    # Select tokens with Cov in [lb, ub]
    eligible = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    eligible_idx = torch.nonzero(eligible)

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)

    # Random subset selection
    corr = torch.ones_like(advantages)
    if eligible_idx.numel() > 0:
        perm = torch.randperm(eligible_idx.size(0), device=eligible_idx.device)
        selected = eligible_idx[perm[:min(clip_num, eligible_idx.size(0))]]
        corr[selected[:, 0], selected[:, 1]] = 0

    pg_clipfrac_cov = verl_F.masked_mean((corr == 0).float(), response_mask)

    # Apply: standard PPO max + zero selected tokens
    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    extra_stats = {
        "clip_cov/pg_clipfrac_standard": pg_clipfrac_standard.detach().item(),
        "clip_cov/pg_clipfrac_cov": pg_clipfrac_cov.detach().item(),
        "clip_cov/clip_cov_ratio": clip_cov_ratio,
        "clip_cov/n_eligible": eligible_idx.size(0),
    }

    return pg_loss, pg_clipfrac_standard, ppo_kl, pg_clipfrac_lower, extra_stats
