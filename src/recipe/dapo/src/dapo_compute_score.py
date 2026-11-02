"""
Standalone compute_score for the DAPO recipe.

Handles all train/val data sources used in our experiments using absolute
imports only — avoids the relative-import failure that occurs when a file
is loaded via importlib.util.spec_from_file_location (which gives the
module no package context).

Loaded via config:
    custom_reward_function.path=.../dapo_compute_score.py
    custom_reward_function.name=_default_compute_score
"""


def _normalize_ground_truth(ground_truth):
    """Normalize ground_truth to a plain string.

    Different datasets store ground_truth in different formats:
    - plain string:  '204'
    - dict:          {'question': '...', 'target': '27'}
    - numpy array:   array(['9.6\\n'], dtype=object)
    - list:          ['9.6']
    Extract the scalar answer string and strip whitespace.
    """
    import numpy as np
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("target", ground_truth.get("answer", ""))
    if isinstance(ground_truth, (np.ndarray,)):
        ground_truth = ground_truth.flat[0]
    elif isinstance(ground_truth, (list, tuple)):
        ground_truth = ground_truth[0] if ground_truth else ""
    return str(ground_truth).strip()


def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    ground_truth = _normalize_ground_truth(ground_truth)

    # ── MATH datasets (train + MATH-500 val) ─────────────────────────────────
    if data_source in [
        "lighteval/MATH",
        "DigitalLearningGmbH/MATH-lighteval",
        "TianHongZXY/MATH",
        "HuggingFaceH4/MATH-500",
    ]:
        from verl.utils.reward_score import math as verl_math
        res = verl_math.compute_score(solution_str, ground_truth)

    # ── AIME / AMC val datasets ───────────────────────────────────────────────
    elif data_source in [
        "TianHongZXY/AIME2024",
        "zjz/AIME2024",
        "TianHongZXY/AIME2025",
        "TianHongZXY/amc23",
        "TianHongZXY/aime-1983-2025",
    ]:
        from recipe.papo.reward.math import aime_compute_score
        res = aime_compute_score(solution_str, ground_truth)

    # ── Minerva / OlympiadBench val datasets ──────────────────────────────────
    elif data_source in [
        "zjz/minerva",
        "zjz/olympiadBench",
    ]:
        from recipe.papo.reward.math import compute_score as papo_math_score
        res = papo_math_score(solution_str, ground_truth)

    # ── DAPO-format math (math_dapo / aime* data_source strings) ─────────────
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from verl.utils.reward_score import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth)
        if isinstance(res, dict):
            return res

    # ── Other known sources ───────────────────────────────────────────────────
    elif data_source == "openai/gsm8k":
        from verl.utils.reward_score import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)

    elif data_source in [
        "numina_aops_forum", "numina_synthetic_math", "numina_amc_aime",
        "numina_synthetic_amc", "numina_cn_k12", "numina_olympiads",
    ]:
        from verl.utils.reward_score import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(
            f"Reward function is not implemented for {data_source=}. "
            "Add it to recipe/dapo/src/dapo_compute_score.py."
        )

    if isinstance(res, (int, float, bool)):
        return float(res)
    return float(res[0])
