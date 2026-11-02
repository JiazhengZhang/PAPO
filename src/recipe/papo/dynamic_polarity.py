"""
Dynamic Polarity Balancing (V1)

Stabilize token-level RL training by dynamically balancing positive (t12 > 0,
entropy-increasing) and negative (t12 < 0, entropy-decreasing) polarity
contributions.  A bang-bang controller adjusts per-polarity advantage weights
based on the *effective* (weight-adjusted) polarity contribution ratio rho_k.

Usage via config::

    trainer.entropy_mask.enable=true
    trainer.entropy_mask.policy=dynamic_polarity
    trainer.entropy_mask.dynamic.rho_low=0.30
    trainer.entropy_mask.dynamic.rho_high=0.45

See ``DynamicPolarityController`` for full parameter list.
"""

from typing import Any, Dict, Tuple

import torch


class DynamicPolarityController:
    """Stateful bang-bang controller for polarity weight adjustment.

    Maintains ``omega_pos`` and ``omega_neg`` across training steps.  Each call
    to :meth:`__call__` computes the polarity ratio rho_k from the current
    batch, updates the weights, and applies them to ``batch.batch["advantages"]``.

    The controller uses the **effective** (weight-adjusted) polarity
    contributions ``omega_pos * M+`` and ``omega_neg * M-`` to compute rho_k,
    so that the control signal reflects the actual gradient balance after
    reweighting.  Raw (unweighted) values are also logged for diagnostics.

    Parameters (read from ``config["dynamic"]`` sub-dict):
        rho_low         Lower bound of the healthy rho range (default 0.30).
        rho_high        Upper bound of the healthy rho range (default 0.45).
        delta           Step size for omega adjustment (default 0.02).
        omega_pos_init  Initial omega_pos (default 1.0).
        omega_neg_init  Initial omega_neg (default 1.0).
        omega_pos_max   Upper clamp for omega_pos (default 1.5).
        omega_pos_min   Lower clamp for omega_pos (default 0.5).
        omega_neg_max   Upper clamp for omega_neg (default 1.5).
        omega_neg_min   Lower clamp for omega_neg (default 0.5).
    """

    def __init__(self, config: Dict):
        cfg = config.get("dynamic", {})
        self.rho_low: float = float(cfg.get("rho_low", 0.30))
        self.rho_high: float = float(cfg.get("rho_high", 0.45))
        self.delta: float = float(cfg.get("delta", 0.02))

        self.omega_pos: float = float(cfg.get("omega_pos_init", 1.0))
        self.omega_neg: float = float(cfg.get("omega_neg_init", 1.0))

        self.omega_pos_max: float = float(cfg.get("omega_pos_max", 1.5))
        self.omega_pos_min: float = float(cfg.get("omega_pos_min", 0.5))
        self.omega_neg_max: float = float(cfg.get("omega_neg_max", 1.5))
        self.omega_neg_min: float = float(cfg.get("omega_neg_min", 0.5))

    # ------------------------------------------------------------------
    # Core API — matches the policy function signature in control_policy
    # ------------------------------------------------------------------

    def __call__(
        self,
        t12: torch.Tensor,
        response_mask: torch.Tensor,
        *,
        step: int,
        config: Dict,
        batch,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute rho, update omega, reweight advantages.

        Returns ``(keep_mask, info_dict)`` with the same interface as other
        policy functions in ``control_policy.py``.
        """
        # ----- 1. Compute raw polarity sums (batch-level) -----
        valid_t12 = t12[response_mask]
        pos_mask = valid_t12 > 0
        neg_mask = valid_t12 < 0

        m_pos_raw = valid_t12[pos_mask].sum().item() if pos_mask.any() else 0.0
        m_neg_raw = valid_t12[neg_mask].abs().sum().item() if neg_mask.any() else 0.0

        # Raw rho (for diagnostics only)
        denom_raw = m_pos_raw + m_neg_raw
        rho_raw = m_pos_raw / denom_raw if denom_raw > 0 else 0.5

        # ----- 2. Compute effective (weighted) polarity ratio -----
        m_pos_eff = self.omega_pos * m_pos_raw
        m_neg_eff = self.omega_neg * m_neg_raw
        denom_eff = m_pos_eff + m_neg_eff
        rho = m_pos_eff / denom_eff if denom_eff > 0 else 0.5

        # ----- 3. Bang-bang update of omega based on effective rho -----
        if rho < self.rho_low:
            # Collapse risk: boost exploration, suppress contraction
            self.omega_pos = min(self.omega_pos_max, self.omega_pos + self.delta)
            self.omega_neg = max(self.omega_neg_min, self.omega_neg - self.delta)
        elif rho > self.rho_high:
            # Over-exploration: boost contraction, suppress exploration
            self.omega_pos = max(self.omega_pos_min, self.omega_pos - self.delta)
            self.omega_neg = min(self.omega_neg_max, self.omega_neg + self.delta)
        # else: within range, hold

        # ----- 4. Reweight advantages -----
        w = torch.where(t12 > 0, self.omega_pos, self.omega_neg)
        w = torch.where(response_mask, w, torch.ones_like(t12))
        batch.batch["advantages"] = batch.batch["advantages"] * w

        # ----- 5. Return full mask (no token is dropped) + diagnostics -----
        info: Dict[str, Any] = {
            "policy": "dynamic_polarity",
            "keep_pos_ratio": 1.0,
            "keep_neg_ratio": 1.0,
            "dynamic/rho": rho,
            "dynamic/rho_raw": rho_raw,
            "dynamic/m_pos_raw": m_pos_raw,
            "dynamic/m_neg_raw": m_neg_raw,
            "dynamic/m_pos_eff": m_pos_eff,
            "dynamic/m_neg_eff": m_neg_eff,
            "dynamic/omega_pos": self.omega_pos,
            "dynamic/omega_neg": self.omega_neg,
        }
        return response_mask, info
