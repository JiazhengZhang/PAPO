from typing import Dict, List, Tuple, Any

import torch

from recipe.papo.adaptive_entropy import AdaptiveEntropyController, AdaptiveSymController
from recipe.papo.dynamic_polarity import DynamicPolarityController


def _get_t12_and_response_mask(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    t12 = batch.batch["entropy_t12"]
    response_mask = batch.batch.get("response_mask")
    if response_mask is None:
        responses = batch.batch["responses"]
        response_length = responses.size(1)
        attention_mask = batch.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
    return t12, response_mask.bool()


def _clamp_ratio(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _smooth_alpha(alpha: float, mode: str) -> float:
    alpha = max(0.0, min(1.0, alpha))
    if mode == "smoothstep":
        return alpha * alpha * (3.0 - 2.0 * alpha)
    return alpha


def _read_schedule(config: Dict) -> List[Dict]:
    schedule = config.get("schedule", [])
    if not schedule:
        # Conservative default: start exploit-heavy, then shift to explore.
        schedule = [
            {"step": 0, "keep_pos_ratio": 0.1, "keep_neg_ratio": 0.8},
            {"step": 100, "keep_pos_ratio": 0.5, "keep_neg_ratio": 0.4},
            {"step": 200, "keep_pos_ratio": 0.8, "keep_neg_ratio": 0.1},
        ]
    schedule = sorted(schedule, key=lambda x: int(x["step"]))
    return schedule


def get_keep_ratios_by_step(step: int, config: Dict) -> Tuple[float, float]:
    schedule = _read_schedule(config)
    smooth_mode = config.get("smooth", "linear")
    cur_step = int(step)

    if cur_step <= int(schedule[0]["step"]):
        return _clamp_ratio(schedule[0]["keep_pos_ratio"]), _clamp_ratio(schedule[0]["keep_neg_ratio"])

    for i in range(len(schedule) - 1):
        left = schedule[i]
        right = schedule[i + 1]
        s0 = int(left["step"])
        s1 = int(right["step"])
        if s0 <= cur_step <= s1:
            if s1 == s0:
                alpha = 1.0
            else:
                alpha = (cur_step - s0) / float(s1 - s0)
            alpha = _smooth_alpha(alpha, smooth_mode)
            keep_pos = (1.0 - alpha) * float(left["keep_pos_ratio"]) + alpha * float(right["keep_pos_ratio"])
            keep_neg = (1.0 - alpha) * float(left["keep_neg_ratio"]) + alpha * float(right["keep_neg_ratio"])
            return _clamp_ratio(keep_pos), _clamp_ratio(keep_neg)

    last = schedule[-1]
    return _clamp_ratio(last["keep_pos_ratio"]), _clamp_ratio(last["keep_neg_ratio"])


def _topk_keep_mask(
    t12: torch.Tensor,
    eligible_mask: torch.Tensor,
    keep_ratio: float,
    *,
    largest: bool,
) -> torch.Tensor:
    B, _ = t12.shape
    keep = torch.zeros_like(eligible_mask, dtype=torch.bool)
    ratio = _clamp_ratio(keep_ratio)
    if ratio <= 0.0:
        return keep
    if ratio >= 1.0:
        return eligible_mask.bool()

    for b in range(B):
        idx = torch.nonzero(eligible_mask[b], as_tuple=False).squeeze(-1)
        n = int(idx.numel())
        if n == 0:
            continue
        k = int(n * ratio)
        if k <= 0:
            continue
        vals = t12[b, idx]
        top_local = torch.topk(vals, k=k, largest=largest, sorted=False).indices
        top_idx = idx[top_local]
        keep[b, top_idx] = True
    return keep


def _policy_vanilla(
    t12: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    step: int,
    config: Dict,
    batch=None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Vanilla policy (no entropy mask intervention).

    Config used:
    - None. `step` and `config` are ignored.

    Kept tokens:
    - Keep all tokens inside `response_mask`.
    - Equivalent to keep_pos_ratio=1.0 and keep_neg_ratio=1.0.
    """
    keep = response_mask
    return keep, {
        "policy": "vanilla",
        "keep_pos_ratio": 1.0,
        "keep_neg_ratio": 1.0,
    }


def _policy_phased_sign_topk(
    t12: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    step: int,
    config: Dict,
    batch=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Step-scheduled sign-topk policy.

    Config used:
    - `schedule`: list of {step, keep_pos_ratio, keep_neg_ratio}
      Ratios are interpolated by current `step`.
    - `smooth`: interpolation type, "linear" or "smoothstep".
    - `neg_inclusive`: False => negative set is t12<0; True => t12<=0.

    Kept tokens:
    - Positive side: from tokens where (response_mask & t12>0), keep top
      `keep_pos_ratio` by largest t12 value, per-sample.
    - Negative side: from tokens where (response_mask & t12<0 / <=0), keep top
      `keep_neg_ratio` by most negative t12 value, per-sample.
    - Final keep mask is union of positive and negative kept sets.

    Example cli override:
    python -m src.recipe.papo.main_ppo \
      trainer.entropy_mask.enable=true \
      trainer.entropy_mask.policy=phased_sign_topk \
      trainer.entropy_mask.schedule='[{step:0,keep_pos_ratio:0.0,keep_neg_ratio:0.8},{step:200,keep_pos_ratio:0.5,keep_neg_ratio:0.4},{step:500,keep_pos_ratio:0.8,keep_neg_ratio:0.1}]' \


    """
    keep_pos_ratio, keep_neg_ratio = get_keep_ratios_by_step(step=step, config=config)
    neg_inclusive = bool(config.get("neg_inclusive", False))  # False: t12<0, True: t12<=0

    pos_eligible = response_mask & (t12 > 0)
    if neg_inclusive:
        neg_eligible = response_mask & (t12 <= 0)
    else:
        neg_eligible = response_mask & (t12 < 0)

    keep_pos = _topk_keep_mask(t12, pos_eligible, keep_pos_ratio, largest=True)
    keep_neg = _topk_keep_mask(t12, neg_eligible, keep_neg_ratio, largest=False)
    keep = keep_pos | keep_neg

    return keep, {
        "policy": "phased_sign_topk",
        "keep_pos_ratio": keep_pos_ratio,
        "keep_neg_ratio": keep_neg_ratio,
    }


# ---------------------------------------------------------------------------
# Quadrant advantage reweighting
# ---------------------------------------------------------------------------

_QUADRANT_KEYS = ("correct_neg", "incorrect_neg", "correct_pos", "incorrect_pos")

_QUADRANT_DEFAULT_SCHEDULE = [
    {"step": 0, "correct_neg": 1.0, "incorrect_neg": 0.0, "correct_pos": 1.5, "incorrect_pos": 0.2},
    {"step": 300, "correct_neg": 0.5, "incorrect_neg": 0.0, "correct_pos": 1.5, "incorrect_pos": 0.8},
]


def _read_quadrant_schedule(config: Dict) -> List[Dict]:
    schedule = config.get("schedule", [])
    if not schedule:
        schedule = list(_QUADRANT_DEFAULT_SCHEDULE)
    schedule = sorted(schedule, key=lambda x: int(x["step"]))
    return schedule


def get_quadrant_weights_by_step(step: int, config: Dict) -> Dict[str, float]:
    """Interpolate 4 quadrant weights at the given *step*."""
    schedule = _read_quadrant_schedule(config)
    smooth_mode = config.get("smooth", "linear")
    cur_step = int(step)

    if cur_step <= int(schedule[0]["step"]):
        return {k: float(schedule[0][k]) for k in _QUADRANT_KEYS}

    for i in range(len(schedule) - 1):
        left, right = schedule[i], schedule[i + 1]
        s0, s1 = int(left["step"]), int(right["step"])
        if s0 <= cur_step <= s1:
            alpha = 1.0 if s1 == s0 else (cur_step - s0) / float(s1 - s0)
            alpha = _smooth_alpha(alpha, smooth_mode)
            return {
                k: (1.0 - alpha) * float(left[k]) + alpha * float(right[k])
                for k in _QUADRANT_KEYS
            }

    last = schedule[-1]
    return {k: float(last[k]) for k in _QUADRANT_KEYS}


def _policy_quadrant_reweight(
    t12: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    step: int,
    config: Dict,
    batch,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Stage-aware advantage reweighting by (correct/incorrect) x (t12 sign).

    Instead of masking tokens out, this policy multiplies per-token advantages
    by quadrant-specific weights that are interpolated from a step schedule.

    Four quadrants:
    - correct   & t12<0 : exploitation on correct trajectories
    - incorrect & t12<0 : erroneous reinforcement (typically zeroed)
    - correct   & t12>0 : reward-aligned exploration
    - incorrect & t12>0 : controlled exploration on incorrect trajectories

    Config used:
    - `schedule`: list of {step, correct_neg, incorrect_neg, correct_pos, incorrect_pos}
    - `smooth`: "linear" or "smoothstep"

    Example cli override:
    python -m src.recipe.papo.main_ppo \\
      trainer.entropy_mask.enable=true \\
      trainer.entropy_mask.policy=quadrant_reweight \\
      trainer.entropy_mask.schedule='[{step:0,correct_neg:1.0,incorrect_neg:0.0,correct_pos:1.5,incorrect_pos:0.2},{step:300,correct_neg:0.5,incorrect_neg:0.0,correct_pos:1.5,incorrect_pos:0.8}]'
    """
    weights = get_quadrant_weights_by_step(step, config)

    # correct / incorrect from token_level_scores
    scores = batch.batch["token_level_scores"].sum(-1)  # (B,)
    correct = (scores > 0).unsqueeze(-1).expand_as(t12)  # (B, T)

    pos_t12 = t12 > 0
    neg_t12 = ~pos_t12  # t12 <= 0

    w = torch.ones_like(t12)
    w[correct & neg_t12] = weights["correct_neg"]
    w[~correct & neg_t12] = weights["incorrect_neg"]
    w[correct & pos_t12] = weights["correct_pos"]
    w[~correct & pos_t12] = weights["incorrect_pos"]

    # Only reweight inside response_mask; leave prompt tokens untouched.
    w = torch.where(response_mask, w, torch.ones_like(w))
    batch.batch["advantages"] = batch.batch["advantages"] * w

    # Return full response_mask as keep mask (no token is dropped).
    ratios = {"policy": "quadrant_reweight"}
    ratios.update(weights)
    ratios["keep_pos_ratio"] = 1.0
    ratios["keep_neg_ratio"] = 1.0
    return response_mask, ratios


# ---------------------------------------------------------------------------
# Entropy top-k baseline (mask all but top X% highest-entropy tokens)
# ---------------------------------------------------------------------------

def _policy_entropy_topk(
    t12: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    step: int,
    config: Dict,
    batch,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Baseline: keep only the top X% highest raw-entropy tokens for RL gradient.

    Uses the original policy entropy ``batch.batch["entropys"]`` (H(pi) per
    token) rather than the derived t1/t2/t12 signals.  Selection is done at
    the **batch level** (pool all valid tokens across all rollouts, then pick
    the global top-k), not per-rollout.

    Config used:
    - ``keep_ratio``: fraction of response tokens to keep (default 0.2 = 20%).

    Example cli override::

        trainer.entropy_mask.enable=true
        trainer.entropy_mask.policy=entropy_topk
        trainer.entropy_mask.keep_ratio=0.2
    """
    keep_ratio = float(config.get("keep_ratio", 0.2))
    entropys = batch.batch["entropys"]  # (B, T)  raw policy entropy

    # Batch-level top-k: flatten all valid tokens, pick global top keep_ratio.
    flat_mask = response_mask.reshape(-1)                # (B*T,)
    flat_entropy = entropys.reshape(-1)                  # (B*T,)
    valid_indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1)
    n_valid = valid_indices.numel()
    k = max(1, int(n_valid * keep_ratio))

    keep = torch.zeros_like(flat_mask, dtype=torch.bool)
    if n_valid > 0 and k > 0:
        vals = flat_entropy[valid_indices]
        topk_local = torch.topk(vals, k=k, largest=True, sorted=False).indices
        keep[valid_indices[topk_local]] = True
    keep = keep.reshape_as(response_mask)

    return keep, {
        "policy": "entropy_topk",
        "keep_ratio": keep_ratio,
        "keep_pos_ratio": keep_ratio,
        "keep_neg_ratio": keep_ratio,
    }


# ---------------------------------------------------------------------------
# Dynamic polarity balancing (stateful — singleton instance)
# ---------------------------------------------------------------------------
_dynamic_polarity_instance: DynamicPolarityController | None = None


def _policy_dynamic_polarity(
    t12: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    step: int,
    config: Dict,
    batch,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Delegate to the stateful :class:`DynamicPolarityController`.

    The controller is lazily instantiated on first call and reused across
    steps.  Config is read from ``config["dynamic"]`` sub-dict.
    """
    global _dynamic_polarity_instance
    if _dynamic_polarity_instance is None:
        _dynamic_polarity_instance = DynamicPolarityController(config)
    return _dynamic_polarity_instance(
        t12=t12, response_mask=response_mask, step=step, config=config, batch=batch,
    )


# ---------------------------------------------------------------------------
# Composed: sign-topk masking + dynamic polarity reweighting
# ---------------------------------------------------------------------------
_dynamic_polarity_topk_instance: DynamicPolarityController | None = None


def _policy_dynamic_polarity_topk(
    t12: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    step: int,
    config: Dict,
    batch,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Two-stage policy: sign-topk selection then dynamic polarity reweighting.

    1. Use ``phased_sign_topk`` to select the top-k most positive and most
       negative t12 tokens (masking the rest from the loss).
    2. On the *kept* tokens only, apply ``DynamicPolarityController`` to
       dynamically adjust the advantage weights by polarity.

    Config used:
    - All ``phased_sign_topk`` config keys (schedule, smooth, neg_inclusive)
    - ``dynamic`` sub-dict for the polarity controller

    Example cli override::

        trainer.entropy_mask.enable=true
        trainer.entropy_mask.policy=dynamic_polarity_topk
        trainer.entropy_mask.schedule='[{step:0,keep_pos_ratio:0.2,keep_neg_ratio:0.2}]'
        trainer.entropy_mask.dynamic.rho_low=0.30
        trainer.entropy_mask.dynamic.rho_high=0.45
    """
    # Stage 1: sign-topk masking
    keep, topk_info = _policy_phased_sign_topk(
        t12=t12, response_mask=response_mask, step=step, config=config, batch=batch,
    )

    # Stage 2: dynamic polarity reweighting on kept tokens only
    global _dynamic_polarity_topk_instance
    if _dynamic_polarity_topk_instance is None:
        _dynamic_polarity_topk_instance = DynamicPolarityController(config)
    _, dyn_info = _dynamic_polarity_topk_instance(
        t12=t12, response_mask=keep, step=step, config=config, batch=batch,
    )

    # Merge diagnostics from both stages
    info: Dict[str, Any] = {}
    info.update(topk_info)
    info.update(dyn_info)
    info["policy"] = "dynamic_polarity_topk"
    return keep, info


# ---------------------------------------------------------------------------
# Composed: sign-topk masking + static scheduled advantage reweighting
# ---------------------------------------------------------------------------

def _get_omega_by_step(step: int, config: Dict) -> Tuple[float, float]:
    """Interpolate omega_pos / omega_neg from the schedule at *step*.

    Falls back to 1.0 for both if the schedule entries omit omega fields.
    """
    schedule = _read_schedule(config)
    smooth_mode = config.get("smooth", "linear")
    cur_step = int(step)

    def _omegas(entry: Dict) -> Tuple[float, float]:
        return float(entry.get("omega_pos", 1.0)), float(entry.get("omega_neg", 1.0))

    if cur_step <= int(schedule[0]["step"]):
        return _omegas(schedule[0])

    for i in range(len(schedule) - 1):
        left, right = schedule[i], schedule[i + 1]
        s0, s1 = int(left["step"]), int(right["step"])
        if s0 <= cur_step <= s1:
            alpha = 1.0 if s1 == s0 else (cur_step - s0) / float(s1 - s0)
            alpha = _smooth_alpha(alpha, smooth_mode)
            lp, ln = _omegas(left)
            rp, rn = _omegas(right)
            return (1.0 - alpha) * lp + alpha * rp, (1.0 - alpha) * ln + alpha * rn

    return _omegas(schedule[-1])


def _policy_static_reweight_topk(
    t12: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    step: int,
    config: Dict,
    batch,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Two-stage policy: sign-topk selection then static scheduled reweighting.

    1. ``phased_sign_topk`` selects the top-k most positive and most negative
       t12 tokens (the keep_pos_ratio / keep_neg_ratio part of the schedule).
    2. On the **kept** tokens, multiply advantages by interpolated
       ``omega_pos`` (for t12>0 tokens) and ``omega_neg`` (for t12<0 tokens).

    All six fields live in the same schedule entries::

        schedule:
          - step: 0
            keep_pos_ratio: 0.2
            keep_neg_ratio: 0.2
            omega_pos: 0.5
            omega_neg: 1.5
          - step: 200
            keep_pos_ratio: 0.2
            keep_neg_ratio: 0.2
            omega_pos: 1.5
            omega_neg: 0.5

    Example cli override::

        trainer.entropy_mask.enable=true
        trainer.entropy_mask.policy=static_reweight_topk
        'trainer.entropy_mask.schedule=[{step:0,keep_pos_ratio:0.2,keep_neg_ratio:0.2,omega_pos:0.5,omega_neg:1.5},{step:200,keep_pos_ratio:0.2,keep_neg_ratio:0.2,omega_pos:1.5,omega_neg:0.5}]'
    """
    # Stage 1: sign-topk masking
    keep, topk_info = _policy_phased_sign_topk(
        t12=t12, response_mask=response_mask, step=step, config=config, batch=batch,
    )

    # Stage 2: static scheduled advantage reweighting on kept tokens
    omega_pos, omega_neg = _get_omega_by_step(step, config)

    w = torch.where(t12 > 0, omega_pos, omega_neg)
    w = torch.where(keep, w, torch.ones_like(t12))
    batch.batch["advantages"] = batch.batch["advantages"] * w

    info: Dict[str, Any] = {}
    info.update(topk_info)
    info["policy"] = "static_reweight_topk"
    info["static/omega_pos"] = omega_pos
    info["static/omega_neg"] = omega_neg
    return keep, info


# ---------------------------------------------------------------------------
# Band ablation: keep a specific percentile band within one or both polarities
# ---------------------------------------------------------------------------

def _band_keep_mask(
    t12: torch.Tensor,
    eligible_mask: torch.Tensor,
    band_low: float,
    band_high: float,
    *,
    largest: bool,
) -> torch.Tensor:
    """Keep tokens whose |t12| rank falls in [band_low, band_high) percentile.

    Ranked by magnitude per sample within *eligible_mask*.
    ``largest=True``: rank by t12 descending (positive branch).
    ``largest=False``: rank by t12 ascending (negative branch).

    band_low=0.0, band_high=0.2 → top-20% strongest.
    band_low=0.8, band_high=1.0 → weakest 20%.
    """
    B, _ = t12.shape
    keep = torch.zeros_like(eligible_mask, dtype=torch.bool)
    bl = _clamp_ratio(band_low)
    bh = _clamp_ratio(band_high)
    if bl >= bh:
        return keep

    for b in range(B):
        idx = torch.nonzero(eligible_mask[b], as_tuple=False).squeeze(-1)
        n = int(idx.numel())
        if n == 0:
            continue
        start = int(n * bl)
        end = int(n * bh)
        if end <= start:
            continue
        vals = t12[b, idx]
        sorted_local = torch.argsort(vals, descending=largest)
        keep[b, idx[sorted_local[start:end]]] = True
    return keep


def _get_band_by_step(step: int, config: Dict) -> Tuple[float, float]:
    """Interpolate band_low / band_high from the schedule at *step*.

    Falls back to 0.0 / 0.2 (top-20%) if the schedule entries omit band fields.
    """
    schedule = _read_schedule(config)
    smooth_mode = config.get("smooth", "linear")
    cur_step = int(step)

    def _bands(entry: Dict) -> Tuple[float, float]:
        return float(entry.get("band_low", 0.0)), float(entry.get("band_high", 0.2))

    if cur_step <= int(schedule[0]["step"]):
        return _bands(schedule[0])

    for i in range(len(schedule) - 1):
        left, right = schedule[i], schedule[i + 1]
        s0, s1 = int(left["step"]), int(right["step"])
        if s0 <= cur_step <= s1:
            alpha = 1.0 if s1 == s0 else (cur_step - s0) / float(s1 - s0)
            alpha = _smooth_alpha(alpha, smooth_mode)
            ll, lh = _bands(left)
            rl, rh = _bands(right)
            return (
                _clamp_ratio((1.0 - alpha) * ll + alpha * rl),
                _clamp_ratio((1.0 - alpha) * lh + alpha * rh),
            )

    return _bands(schedule[-1])


def _policy_band_ablation(
    t12: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    step: int,
    config: Dict,
    batch,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Band ablation: keep a specific magnitude-percentile band per polarity.

    All parameters live in the schedule (reuses ``_read_schedule``):
    - ``band_low`` / ``band_high``: percentile boundaries (default 0.0 / 0.2)
    - ``band_polarity``: ``"pos"``, ``"neg"``, or ``"both"`` (default ``"both"``)

    ``band_polarity`` is read from the **first** schedule entry (not interpolated).

    Percentile 0.0 = strongest magnitude, 1.0 = weakest.

    Example cli overrides::

        # pos top-20% only
        trainer.entropy_mask.policy=band_ablation
        'trainer.entropy_mask.schedule=[{step:0,band_low:0.0,band_high:0.2,band_polarity:pos}]'

        # neg 40-60% band
        trainer.entropy_mask.policy=band_ablation
        'trainer.entropy_mask.schedule=[{step:0,band_low:0.4,band_high:0.6,band_polarity:neg}]'

        # both top-20%, static
        trainer.entropy_mask.policy=band_ablation
        'trainer.entropy_mask.schedule=[{step:0,band_low:0.0,band_high:0.2,band_polarity:both}]'
    """
    schedule = _read_schedule(config)
    polarity = str(schedule[0].get("band_polarity", "both"))
    neg_inclusive = bool(config.get("neg_inclusive", False))

    band_low, band_high = _get_band_by_step(step, config)

    pos_eligible = response_mask & (t12 > 0)
    neg_eligible = response_mask & (t12 <= 0) if neg_inclusive else response_mask & (t12 < 0)

    keep_pos = _band_keep_mask(t12, pos_eligible, band_low, band_high, largest=True) \
        if polarity in ("pos", "both") else torch.zeros_like(response_mask, dtype=torch.bool)
    keep_neg = _band_keep_mask(t12, neg_eligible, band_low, band_high, largest=False) \
        if polarity in ("neg", "both") else torch.zeros_like(response_mask, dtype=torch.bool)

    keep = keep_pos | keep_neg

    # --- diagnostics ---
    n_pos_eligible = pos_eligible.sum().item()
    n_neg_eligible = neg_eligible.sum().item()
    n_pos_kept = keep_pos.sum().item()
    n_neg_kept = keep_neg.sum().item()
    n_total_resp = response_mask.sum().item()

    # effective keep ratios (relative to eligible, not total)
    eff_pos_ratio = n_pos_kept / max(n_pos_eligible, 1)
    eff_neg_ratio = n_neg_kept / max(n_neg_eligible, 1)

    # mean |t12| inside the kept band vs all eligible (signal concentration)
    t12_abs = t12.abs()
    mean_t12_kept = (t12_abs * keep.float()).sum().item() / max(n_pos_kept + n_neg_kept, 1)
    mean_t12_all = (t12_abs * response_mask.float()).sum().item() / max(n_total_resp, 1)

    return keep, {
        "policy": "band_ablation",
        "keep_pos_ratio": eff_pos_ratio,
        "keep_neg_ratio": eff_neg_ratio,
        "band/polarity": polarity,
        "band/low": band_low,
        "band/high": band_high,
        "band/n_pos_kept": n_pos_kept,
        "band/n_neg_kept": n_neg_kept,
        "band/n_pos_eligible": n_pos_eligible,
        "band/n_neg_eligible": n_neg_eligible,
        "band/mean_abs_t12_kept": mean_t12_kept,
        "band/mean_abs_t12_all": mean_t12_all,
        "band/signal_ratio": mean_t12_kept / max(mean_t12_all, 1e-8),
    }


# ---------------------------------------------------------------------------
# Composed: sign-topk + adaptive entropy-slope negative reweighting
# ---------------------------------------------------------------------------
_adaptive_entropy_instance: AdaptiveEntropyController | None = None


def _policy_adaptive_topk(
    t12: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    step: int,
    config: Dict,
    batch,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Two-stage policy: sign-topk selection + adaptive entropy-slope reweighting.

    1. ``phased_sign_topk`` selects top-k pos / neg tokens (backbone unchanged).
    2. ``AdaptiveEntropyController`` computes omega_neg from the entropy slope
       EMA.  omega_neg is applied to negative-branch kept tokens only;
       positive-branch tokens are untouched (omega_pos = 1.0).

    Config used:
    - All ``phased_sign_topk`` keys (schedule with keep_pos_ratio / keep_neg_ratio)
    - ``adaptive`` sub-dict for the entropy-slope controller

    Example cli override::

        trainer.entropy_mask.enable=true
        trainer.entropy_mask.policy=adaptive_topk
        'trainer.entropy_mask.schedule=[{step:0,keep_pos_ratio:0.2,keep_neg_ratio:0.2}]'
        trainer.entropy_mask.adaptive.skip=3
        trainer.entropy_mask.adaptive.warmup=12
        trainer.entropy_mask.adaptive.alpha=0.25
        trainer.entropy_mask.adaptive.omega_neg_max=1.3
        trainer.entropy_mask.adaptive.omega_neg_min=0.7
    """
    # Stage 1: sign-topk masking
    keep, topk_info = _policy_phased_sign_topk(
        t12=t12, response_mask=response_mask, step=step, config=config, batch=batch,
    )

    # Stage 2: adaptive entropy-slope reweighting
    global _adaptive_entropy_instance
    if _adaptive_entropy_instance is None:
        _adaptive_entropy_instance = AdaptiveEntropyController(config)
    adaptive_info = _adaptive_entropy_instance.update(step=step, batch=batch, response_mask=response_mask)

    omega_neg = adaptive_info["adaptive/omega_neg"]
    # Only reweight negative branch; positive branch stays at 1.0
    w = torch.where(t12 > 0, 1.0, omega_neg)
    w = torch.where(keep, w, torch.ones_like(t12))
    batch.batch["advantages"] = batch.batch["advantages"] * w

    # --- diagnostics: kept token magnitude stats ---
    kept_pos = keep & (t12 > 0)
    kept_neg = keep & (t12 < 0)
    n_kept_pos = kept_pos.sum().item()
    n_kept_neg = kept_neg.sum().item()
    t12_abs = t12.abs()
    mean_abs_t12_pos = (t12_abs * kept_pos.float()).sum().item() / max(n_kept_pos, 1)
    mean_abs_t12_neg = (t12_abs * kept_neg.float()).sum().item() / max(n_kept_neg, 1)

    info: Dict[str, Any] = {}
    info.update(topk_info)
    info.update(adaptive_info)
    info["policy"] = "adaptive_topk"
    info["adaptive/n_kept_pos"] = n_kept_pos
    info["adaptive/n_kept_neg"] = n_kept_neg
    info["adaptive/mean_abs_t12_pos"] = mean_abs_t12_pos
    info["adaptive/mean_abs_t12_neg"] = mean_abs_t12_neg
    return keep, info


# ---------------------------------------------------------------------------
# Composed: sign-topk + symmetric adaptive reweighting
# ---------------------------------------------------------------------------
_adaptive_sym_instance: AdaptiveSymController | None = None


def _policy_adaptive_sym_topk(
    t12: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    step: int,
    config: Dict,
    batch,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """sign-topk + symmetric entropy-slope reweighting (single alpha bound).

    omega_pos and omega_neg move anti-correlated around 1.0 driven by p_t:
        collapse: omega_pos = 1+alpha (boost exploration), omega_neg = 1-alpha
        recovery: omega_pos = 1-alpha,                    omega_neg = 1+alpha

    Config used:
    - All ``phased_sign_topk`` keys (schedule with keep_pos_ratio / keep_neg_ratio)
    - ``adaptive_sym`` sub-dict for the symmetric controller

    Example cli override::

        trainer.entropy_mask.policy=adaptive_sym_topk
        'trainer.entropy_mask.schedule=[{step:0,keep_pos_ratio:0.2,keep_neg_ratio:0.2}]'
        trainer.entropy_mask.adaptive_sym.alpha=0.15
    """
    # Stage 1: sign-topk masking
    keep, topk_info = _policy_phased_sign_topk(
        t12=t12, response_mask=response_mask, step=step, config=config, batch=batch,
    )

    # Stage 2: symmetric adaptive reweighting
    global _adaptive_sym_instance
    if _adaptive_sym_instance is None:
        _adaptive_sym_instance = AdaptiveSymController(config)
    sym_info = _adaptive_sym_instance.update(step=step, batch=batch, response_mask=response_mask)

    omega_pos = sym_info["adaptive/omega_pos"]
    omega_neg = sym_info["adaptive/omega_neg"]
    w = torch.where(t12 > 0, omega_pos, omega_neg)
    w = torch.where(keep, w, torch.ones_like(t12))
    batch.batch["advantages"] = batch.batch["advantages"] * w

    info: Dict[str, Any] = {}
    info.update(topk_info)
    info.update(sym_info)
    info["policy"] = "adaptive_sym_topk"
    return keep, info


_POLICY_REGISTRY = {
    "vanilla": _policy_vanilla,
    "phased_sign_topk": _policy_phased_sign_topk,
    "quadrant_reweight": _policy_quadrant_reweight,
    "entropy_topk": _policy_entropy_topk,
    "dynamic_polarity": _policy_dynamic_polarity,
    "dynamic_polarity_topk": _policy_dynamic_polarity_topk,
    "static_reweight_topk": _policy_static_reweight_topk,
    "band_ablation": _policy_band_ablation,
    "adaptive_topk": _policy_adaptive_topk,
    "adaptive_sym_topk": _policy_adaptive_sym_topk,
}


def build_entropy_mask(
    batch,
    *,
    step: int,
    config: Dict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Route to a concrete entropy-mask policy by `config["policy"]`.

    Notes:
    - `keep` is a boolean mask where True means the token is retained.
    - Current policies:
      - `vanilla`
      - `phased_sign_topk`
      - `quadrant_reweight`
      - `entropy_topk`
      - `dynamic_polarity`
      - `dynamic_polarity_topk`
      - `static_reweight_topk`
      - `band_ablation`
      - `adaptive_topk`
    """
    t12, response_mask = _get_t12_and_response_mask(batch)
    policy_name = str(config.get("policy", "phased_sign_topk"))
    policy_fn = _POLICY_REGISTRY.get(policy_name)
    if policy_fn is None:
        raise ValueError(f"Unknown entropy mask policy: {policy_name}")
    return policy_fn(t12=t12, response_mask=response_mask, step=step, config=config, batch=batch)
