"""
STEER: Stabilizing Token-level Entropy-changE via Reweighting

Core algorithm: compute per-token weights based on entropy change indicator Ω,
then apply them to the clipped PPO loss.

Reference: Hao et al., "Rethinking Entropy Interventions in RLVR: An Entropy Change Perspective"
"""

import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss


def compute_token_weights(
    advantages: torch.Tensor,
    entropys: torch.Tensor,
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    token_weight_min: float = 0.8,
    token_weight_max: float = 1.2,
    linear: bool = True,
    mode: str = "symmetric",
) -> torch.Tensor:
    """Compute per-token loss weights from STEER's Ω indicator.

    Ω = (A / π_old) · π(1−π) · (ln π + H)

    - symmetric: |Ω| large → weight small (suppress high entropy-change tokens)
    - asymmetric: Ω<0 (entropy-decreasing) → weight small; Ω>0 → weight large

    Returns weights in [token_weight_min, token_weight_max] for valid tokens, 0 elsewhere.
    """
    with torch.no_grad():
        x = torch.exp(log_prob)
        x = torch.clamp(x, min=1e-8, max=1.0 - 1e-8)

        f_x = x * (1 - x) * (torch.log(x) + entropys)

        old_prob = torch.exp(old_log_prob)
        old_prob = torch.clamp(old_prob, min=1e-8, max=1.0)
        advantage_over_old_prob = advantages / old_prob

        if mode == "asymmetric":
            metric = -advantage_over_old_prob * f_x
        else:
            metric = advantage_over_old_prob * f_x

        metric = torch.where(torch.isfinite(metric), metric, torch.zeros_like(metric))

        if mode == "symmetric":
            metric = torch.abs(metric)

        valid_mask = response_mask.bool()
        valid_metric = metric[valid_mask]
        if valid_metric.numel() == 0:
            return torch.zeros_like(response_mask, dtype=torch.float)

        metric_min = valid_metric.min()
        metric_max = valid_metric.max()

        token_weights = torch.zeros_like(metric, dtype=torch.float)

        if valid_mask.any():
            valid_metric = metric[valid_mask]

            if linear:
                if metric_max > metric_min:
                    scale_factor = (token_weight_max - token_weight_min) / (metric_max - metric_min)
                    if mode == "asymmetric":
                        valid_weights = token_weight_min + (valid_metric - metric_min) * scale_factor
                    else:
                        valid_weights = token_weight_max - (valid_metric - metric_min) * scale_factor
                else:
                    valid_weights = torch.full_like(valid_metric, (token_weight_min + token_weight_max) / 2)
                valid_weights = torch.clamp(valid_weights, min=token_weight_min, max=token_weight_max)
            else:
                abs_max = torch.maximum(
                    valid_metric.abs().max(),
                    torch.tensor(0.02, dtype=metric.dtype, device=metric.device),
                )
                if mode == "asymmetric":
                    k = -torch.log(torch.tensor(token_weight_min, dtype=metric.dtype, device=metric.device)) / abs_max
                    valid_weights = torch.exp(k * valid_metric)
                    valid_weights = torch.clamp(valid_weights, min=token_weight_min, max=token_weight_max)
                else:
                    k = -torch.log(torch.tensor(token_weight_min, dtype=metric.dtype, device=metric.device)) / \
                        torch.maximum(metric_max, torch.tensor(0.02, dtype=metric.dtype, device=metric.device))
                    valid_weights = torch.exp(-k * valid_metric)
                    valid_weights = torch.clamp(valid_weights, min=token_weight_min, max=1.0)

            token_weights[valid_mask] = valid_weights

        token_weights = token_weights * response_mask.float()
        return token_weights


def compute_policy_loss_steer(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    entropys: torch.Tensor,
    cliprange_low: float = 0.2,
    cliprange_high: float = 0.28,
    clip_ratio_c: float = 3.0,
    loss_agg_mode: str = "token-mean",
    token_weight_min: float = 0.8,
    token_weight_max: float = 1.2,
    linear: bool = True,
    entropy_control_mode: str = "symmetric",
):
    """PPO clipped loss with STEER token reweighting.

    Returns (pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, token_weight_stats).
    """
    token_weights = compute_token_weights(
        advantages=advantages,
        entropys=entropys,
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        response_mask=response_mask,
        token_weight_min=token_weight_min,
        token_weight_max=token_weight_max,
        linear=linear,
        mode=entropy_control_mode,
    )

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    weighted_pg_losses = pg_losses * token_weights

    pg_loss = agg_loss(loss_mat=weighted_pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    valid = response_mask > 0
    token_weight_stats = {
        "steer/token_weights_mean": token_weights[valid].mean().item() if valid.any() else 1.0,
        "steer/token_weights_min": token_weights[valid].min().item() if valid.any() else 1.0,
        "steer/token_weights_max": token_weights[valid].max().item() if valid.any() else 1.0,
    }

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, token_weight_stats
