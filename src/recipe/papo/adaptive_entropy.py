"""
Adaptive Entropy-Slope Controller

Adjusts negative-branch advantage weight (omega_neg) based on the entropy
slope signal.  When entropy is collapsing (slope << 0), omega_neg is raised
to protect exploration; when entropy is recovering (slope → 0 or positive),
omega_neg is lowered to maximize exploitation.

The controller is **stateful** — it maintains an EMA of per-step entropy
deltas and estimates a reference collapse slope during a warmup phase.

Lifecycle::

    step 0 .. skip-1          : ignored (no EMA update)
    step skip .. skip+warmup-1: EMA accumulates, s_ref = EMA at end of warmup
    step skip+warmup ..        : adaptive omega_neg active

Usage via config (``config["adaptive"]`` sub-dict)::

    trainer.entropy_mask.policy=adaptive_topk
    trainer.entropy_mask.adaptive.skip=3
    trainer.entropy_mask.adaptive.warmup=12
    trainer.entropy_mask.adaptive.beta_warmup=0.95
    trainer.entropy_mask.adaptive.beta_run=0.9
    trainer.entropy_mask.adaptive.alpha=0.25
    trainer.entropy_mask.adaptive.omega_neg_max=1.3
    trainer.entropy_mask.adaptive.omega_neg_min=0.7
"""

from typing import Any, Dict

import torch


class AdaptiveEntropyController:
    """Stateful controller: entropy-slope → omega_neg.

    Parameters (read from ``config["adaptive"]`` sub-dict):
        skip            Steps to ignore at start (default 3).
        warmup          Steps to accumulate s_ref after skip (default 12).
        beta_warmup     EMA decay during warmup (default 0.95).
        beta_run        EMA decay after warmup (default 0.9).
        alpha           Scaling factor for omega_neg deviation (default 0.25).
        omega_neg_max   Upper clamp (default 1.3).
        omega_neg_min   Lower clamp (default 0.7).
        entropy_gate_ratio  When entropy drops below this fraction of the
                            warmup-end entropy, reweighting is disabled
                            (omega forced to 1.0).  0 = gate disabled.
                            Default 0.3.
    """

    def __init__(self, config: Dict):
        cfg = config.get("adaptive", {})
        self.skip: int = int(cfg.get("skip", 3))
        self.warmup: int = int(cfg.get("warmup", 12))
        self.beta_warmup: float = float(cfg.get("beta_warmup", 0.95))
        self.beta_run: float = float(cfg.get("beta_run", 0.9))
        self.alpha: float = float(cfg.get("alpha", 0.25))
        self.omega_neg_max: float = float(cfg.get("omega_neg_max", 1.3))
        self.omega_neg_min: float = float(cfg.get("omega_neg_min", 0.7))
        self.beta_omega: float = float(cfg.get("beta_omega", 0.9))
        self.entropy_gate_ratio: float = float(cfg.get("entropy_gate_ratio", 0.3))

        # State
        self._prev_h: float | None = None
        self._ema_slope: float = 0.0
        self._ema_h: float | None = None  # EMA of entropy for gate
        self._ema_omega: float = 1.0  # smoothed omega_neg output
        self._s_ref: float | None = None
        self._h_ref: float | None = None  # EMA entropy at warmup end for gate
        self._reweight_active: bool = True
        self._warmup_end_step: int = self.skip + self.warmup

    # ------------------------------------------------------------------

    def _mean_entropy(self, batch, response_mask: torch.Tensor) -> float:
        """Compute mean policy entropy over response tokens."""
        entropys = batch.batch["entropys"]  # (B, T)
        masked = entropys * response_mask.float()
        n_tokens = response_mask.sum().clamp(min=1)
        return (masked.sum() / n_tokens).item()

    def update(
        self,
        step: int,
        batch,
        response_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """Update EMA and return current adaptive state (omega_neg, diagnostics).

        Called once per training step.  Returns a dict with at least
        ``omega_neg`` and diagnostic keys.
        """
        h_t = self._mean_entropy(batch, response_mask)

        # --- EMA of entropy (use warmup beta before warmup ends, run beta after) ---
        if self._ema_h is None:
            self._ema_h = h_t
        else:
            beta_h = self.beta_warmup if step < self._warmup_end_step else self.beta_run
            self._ema_h = beta_h * self._ema_h + (1.0 - beta_h) * h_t

        info: Dict[str, Any] = {
            "adaptive/h_t": h_t,
            "adaptive/ema_h": self._ema_h,
            "adaptive/ema_slope": self._ema_slope,
            "adaptive/s_ref": self._s_ref if self._s_ref is not None else 0.0,
            "adaptive/omega_neg": 1.0,
            "adaptive/p_t": 0.0,
            "adaptive/h_ref": self._h_ref if self._h_ref is not None else 0.0,
            "adaptive/reweight_active": float(self._reweight_active),
        }

        # --- Phase 0: skip ---
        if step < self.skip:
            self._prev_h = h_t
            return info

        # --- Compute delta_h ---
        if self._prev_h is None:
            self._prev_h = h_t
            return info

        delta_h = h_t - self._prev_h
        self._prev_h = h_t

        info["adaptive/delta_h"] = delta_h

        # --- EMA update ---
        if step < self._warmup_end_step:
            # Warmup: only update EMA with negative slopes to capture
            # the true collapse rate, making s_ref steeper (more negative)
            # if delta_h < 0:
            self._ema_slope = self.beta_warmup * self._ema_slope + (1.0 - self.beta_warmup) * delta_h
        else:
            # Running phase: update with all slopes
            self._ema_slope = self.beta_run * self._ema_slope + (1.0 - self.beta_run) * delta_h

        info["adaptive/ema_slope"] = self._ema_slope

        # --- Phase 1: warmup (accumulate, record s_ref at end) ---
        if step < self._warmup_end_step:
            # Lock s_ref and h_ref at the last warmup step
            if step == self._warmup_end_step - 1:
                self._s_ref = self._ema_slope
                self._h_ref = self._ema_h
                info["adaptive/s_ref"] = self._s_ref
                info["adaptive/h_ref"] = self._h_ref
            return info

        # --- Phase 2: adaptive ---
        if self._s_ref is None:
            # Safety: should not happen, but handle gracefully
            self._s_ref = self._ema_slope
        if self._h_ref is None:
            self._h_ref = self._ema_h

        # --- Entropy gate: disable reweight when entropy drops too low ---
        if self.entropy_gate_ratio > 0 and self._ema_h <= self._h_ref * self.entropy_gate_ratio:
            self._reweight_active = False
        info["adaptive/reweight_active"] = float(self._reweight_active)
        info["adaptive/h_ref"] = self._h_ref

        s_ref = self._s_ref
        eps = 1e-8
        # p_t: 0 = still collapsing like reference, 1 = fully recovered or better
        p_t = max(0.0, min(1.0, (self._ema_slope - s_ref) / (-s_ref + eps)))

        # omega_neg: p_t=0 → min (collapse, reduce exploitation), p_t=1 → max (recovered, exploit hard)
        omega_neg_raw = self.omega_neg_min + (self.omega_neg_max - self.omega_neg_min) * (p_t ** 2)
        # Smooth omega_neg to prevent step-to-step oscillation
        self._ema_omega = self.beta_omega * self._ema_omega + (1.0 - self.beta_omega) * omega_neg_raw
        omega_neg = self._ema_omega

        # If gate tripped, force neutral reweight
        if not self._reweight_active:
            omega_neg = 1.0

        info["adaptive/p_t"] = p_t
        info["adaptive/s_ref"] = s_ref
        info["adaptive/omega_neg_raw"] = omega_neg_raw
        info["adaptive/omega_neg"] = omega_neg

        return info


class AdaptiveSymController:
    """Symmetric stateful controller: entropy-slope → (omega_pos, omega_neg).

    omega_neg follows the same schedule formula as AdaptiveEntropyController.
    omega_pos is the reciprocal: omega_pos = 1 / omega_neg.

    This guarantees:
        omega_neg = 1  →  omega_pos = 1  (neutral)
        omega_neg > 1  →  omega_pos < 1  (recovery: exploit + suppress exploration)
        omega_neg < 1  →  omega_pos > 1  (collapse: protect + boost exploration)

    Use small bounds near 1.0 (e.g. omega_neg_min=0.97, omega_neg_max=1.03)
    since both branches move simultaneously.

    Parameters (read from ``config["adaptive"]`` sub-dict, shared with AdaptiveEntropyController):
        skip            Steps to ignore at start (default 3).
        warmup          Steps to accumulate s_ref after skip (default 12).
        beta_warmup     EMA decay during warmup, negative-only (default 0.95).
        beta_run        EMA decay in running phase (default 0.9).
        beta_omega      EMA decay for omega smoothing (default 0.9).
        omega_neg_min   omega_neg lower bound — collapse phase (default 0.97).
        omega_neg_max   omega_neg upper bound — recovery phase (default 1.03).
        entropy_gate_ratio  When entropy drops below this fraction of the
                            warmup-end entropy, reweighting is disabled
                            (omega forced to 1.0).  0 = gate disabled.
                            Default 0.3.
    """

    def __init__(self, config: Dict):
        cfg = config.get("adaptive", {})
        self.skip: int = int(cfg.get("skip", 3))
        self.warmup: int = int(cfg.get("warmup", 12))
        self.beta_warmup: float = float(cfg.get("beta_warmup", 0.95))
        self.beta_run: float = float(cfg.get("beta_run", 0.9))
        self.beta_omega: float = float(cfg.get("beta_omega", 0.9))
        self.omega_neg_min: float = float(cfg.get("omega_neg_min", 0.97))
        self.omega_neg_max: float = float(cfg.get("omega_neg_max", 1.03))
        self.entropy_gate_ratio: float = float(cfg.get("entropy_gate_ratio", 0.3))

        # State
        self._prev_h: float | None = None
        self._ema_slope: float = 0.0
        self._ema_h: float | None = None  # EMA of entropy for gate
        self._ema_omega_pos: float = 1.0
        self._ema_omega_neg: float = 1.0
        self._s_ref: float | None = None
        self._h_ref: float | None = None  # EMA entropy at warmup end for gate
        self._reweight_active: bool = True
        self._warmup_end_step: int = self.skip + self.warmup

    def _mean_entropy(self, batch, response_mask: torch.Tensor) -> float:
        entropys = batch.batch["entropys"]
        n_tokens = response_mask.sum().clamp(min=1)
        return ((entropys * response_mask.float()).sum() / n_tokens).item()

    def update(self, step: int, batch, response_mask: torch.Tensor) -> Dict[str, Any]:
        h_t = self._mean_entropy(batch, response_mask)

        # --- EMA of entropy ---
        if self._ema_h is None:
            self._ema_h = h_t
        else:
            beta_h = self.beta_warmup if step < self._warmup_end_step else self.beta_run
            self._ema_h = beta_h * self._ema_h + (1.0 - beta_h) * h_t

        info: Dict[str, Any] = {
            "adaptive/h_t": h_t,
            "adaptive/ema_h": self._ema_h,
            "adaptive/ema_slope": self._ema_slope,
            "adaptive/s_ref": self._s_ref if self._s_ref is not None else 0.0,
            "adaptive/omega_pos": 1.0,
            "adaptive/omega_neg": 1.0,
            "adaptive/p_t": 0.0,
            "adaptive/h_ref": self._h_ref if self._h_ref is not None else 0.0,
            "adaptive/reweight_active": float(self._reweight_active),
        }

        # --- Phase 0: skip ---
        if step < self.skip:
            self._prev_h = h_t
            return info

        if self._prev_h is None:
            self._prev_h = h_t
            return info

        delta_h = h_t - self._prev_h
        self._prev_h = h_t
        info["adaptive/delta_h"] = delta_h

        # --- EMA update ---
        if step < self._warmup_end_step:
            self._ema_slope = self.beta_warmup * self._ema_slope + (1.0 - self.beta_warmup) * delta_h
            if step == self._warmup_end_step - 1:
                self._s_ref = self._ema_slope
                self._h_ref = self._ema_h
        else:
            self._ema_slope = self.beta_run * self._ema_slope + (1.0 - self.beta_run) * delta_h

        info["adaptive/ema_slope"] = self._ema_slope
        info["adaptive/s_ref"] = self._s_ref if self._s_ref is not None else 0.0

        if step < self._warmup_end_step:
            return info

        # --- Phase 2: adaptive ---
        if self._s_ref is None:
            self._s_ref = self._ema_slope
        if self._h_ref is None:
            self._h_ref = self._ema_h

        # --- Entropy gate: disable reweight when entropy drops too low ---
        if self.entropy_gate_ratio > 0 and self._ema_h <= self._h_ref * self.entropy_gate_ratio:
            self._reweight_active = False
        info["adaptive/reweight_active"] = float(self._reweight_active)
        info["adaptive/h_ref"] = self._h_ref

        eps = 1e-8
        p_t = max(0.0, min(1.0, (self._ema_slope - self._s_ref) / (-self._s_ref + eps)))

        omega_neg_min = self.omega_neg_min
        omega_neg_max = self.omega_neg_max
        omega_neg_raw = omega_neg_min + (omega_neg_max - omega_neg_min) * (p_t ** 2)
        omega_pos_raw = 1.0 / omega_neg_raw

        self._ema_omega_pos = self.beta_omega * self._ema_omega_pos + (1.0 - self.beta_omega) * omega_pos_raw
        self._ema_omega_neg = self.beta_omega * self._ema_omega_neg + (1.0 - self.beta_omega) * omega_neg_raw

        # If gate tripped, force neutral reweight
        if not self._reweight_active:
            omega_pos_out = 1.0
            omega_neg_out = 1.0
        else:
            omega_pos_out = self._ema_omega_pos
            omega_neg_out = self._ema_omega_neg

        info["adaptive/p_t"] = p_t
        info["adaptive/omega_pos_raw"] = omega_pos_raw
        info["adaptive/omega_neg_raw"] = omega_neg_raw
        info["adaptive/omega_pos"] = omega_pos_out
        info["adaptive/omega_neg"] = omega_neg_out

        return info
