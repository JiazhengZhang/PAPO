"""
KL-Cov loss from "The Entropy Mechanism of Reinforcement Learning for Reasoning LMs"
(PRIME-RL/Entropy-Mechanism-of-RL)

Selects top-k tokens by Cov(advantage, log_prob) and adds |KL| penalty to those tokens only.
No special actor needed — works with standard Verl actor's update_policy.

Ported from STEER/verl/trainer/ppo/core_algos.py @register_policy_loss("kl_cov").
"""

import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss


def compute_policy_loss_kl_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_low: float = 0.2,
    cliprange_high: float = 0.28,
    clip_ratio_c: float = 3.0,
    loss_agg_mode: str = "token-mean",
    kl_cov_ratio: float = 0.0002,
    ppo_kl_coef: float = 1.0,
):
    """PPO loss with KL-Cov: add |KL| penalty to top-k covariance tokens.

    The covariance is Cov(advantage, log_prob) per-token across the batch.
    Tokens with the highest covariance get an additional |KL| penalty,
    which targets entropy-decreasing updates on high-advantage tokens.

    Returns: (pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, extra_stats)
    """
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)

    # Standard PPO loss (no clip for kl_cov — follows original)
    pg_losses1 = -advantages * ratio
    # KL-penalized version
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1.clone()

    # Select top-k tokens by Cov(advantage, log_prob)
    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1)
    all_valid_logp = log_prob[all_valid].detach().reshape(-1)

    n_valid = all_valid_adv.numel()
    if n_valid > 0 and kl_cov_ratio > 0:
        cov_lst = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(n_valid * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst, k_percent_nums, largest=True).indices

        if large_cov_idxs.numel() > 0:
            flat_idxs = all_valid_idx[large_cov_idxs]
            seq_len = advantages.shape[1]
            row = flat_idxs // seq_len
            col = flat_idxs % seq_len
            pg_losses[row, col] = pg_losses_kl[row, col]

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    # Clipfrac (approximate — no actual clipping in kl_cov, report 0)
    pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    extra_stats = {
        "kl_cov/ppo_kl": ppo_kl.detach().item(),
        "kl_cov/kl_cov_ratio": kl_cov_ratio,
        "kl_cov/ppo_kl_coef": ppo_kl_coef,
    }

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, extra_stats
