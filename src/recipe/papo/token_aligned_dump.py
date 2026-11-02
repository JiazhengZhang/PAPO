"""
Per-step token-aligned rollout dump for qualitative case studies.

Each step appends a small subset of rollouts (correct/incorrect ~half each)
to a single JSONL file.  Every line is one rollout with the schema:

  {
    "step": <int>,
    "uid": "<str>",
    "data_source": "<str>",
    "score": <float>,          // sample-level reward (0.0 or 1.0)
    "correct": <bool>,
    "ground_truth": "<str>",   // from reward_model["ground_truth"] if available
    "question": "<decoded prompt>",
    "answer":   "<decoded response>",
    "tokens": [
      {"token_id": <int>, "token_str": "<str>", "t1": <float>, "t2": <float>, "t12": <float>},
      ...                               # zero-value tokens excluded
    ]
  }

Resume handling
---------------
Call `trim_token_aligned_file(path, current_step)` once at training startup
(after loading a checkpoint) to remove any entries with step >= current_step
that were written in a previous interrupted run.
"""

import json
import os
import tempfile
from typing import List, Optional

import torch

from verl import DataProto


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _decode_token(tokenizer, token_id: int) -> str:
    """Decode a single token id to its human-readable string.

    Uses tokenizer.decode() rather than convert_ids_to_tokens() so that
    BPE/SentencePiece internal symbols (Ġ, ▁, etc.) are converted to the
    actual characters they represent.
    """
    return tokenizer.decode([token_id], skip_special_tokens=False)


def _select_rollout_indices(batch: DataProto, n: int) -> List[int]:
    """Return up to n sample indices, aiming for ~half correct / ~half incorrect.

    Skips samples whose entropy signals are all zero (e.g., from GRPO groups
    where all responses got the same reward, so advantage=0 everywhere).
    Falls back to whatever is available when one side is short.
    """
    token_level_scores = batch.batch["token_level_scores"]  # (B, T)
    correct_mask = token_level_scores.sum(-1) == 1          # (B,)
    incorrect_mask = ~correct_mask

    # Exclude samples with no entropy signal (all t12 == 0)
    t12 = batch.batch["entropy_t12"]                        # (B, T)
    has_signal = (t12 != 0).any(dim=-1)                     # (B,)
    correct_mask = correct_mask & has_signal
    incorrect_mask = incorrect_mask & has_signal

    correct_idx = torch.nonzero(correct_mask, as_tuple=False).squeeze(-1).tolist()
    incorrect_idx = torch.nonzero(incorrect_mask, as_tuple=False).squeeze(-1).tolist()

    half = n // 2
    # take up to half from each side; fill remainder from the other if short
    n_correct = min(half, len(correct_idx))
    n_incorrect = min(n - n_correct, len(incorrect_idx))
    n_correct = min(n - n_incorrect, len(correct_idx))  # re-adjust if incorrect was short

    def _sample(lst, k):
        if k <= 0 or not lst:
            return []
        if k >= len(lst):
            return lst
        perm = torch.randperm(len(lst)).tolist()
        return [lst[i] for i in perm[:k]]

    selected = _sample(correct_idx, n_correct) + _sample(incorrect_idx, n_incorrect)
    return selected


def _build_token_list(token_ids, t1_row, t2_row, t12_row, tokenizer) -> list:
    """Build the per-token list for one rollout, excluding zero-value tokens."""
    result = []
    for pos, tid in enumerate(token_ids):
        tid_int = int(tid)
        v1 = float(t1_row[pos])
        v2 = float(t2_row[pos])
        v12 = float(t12_row[pos])
        if v1 == 0.0 and v2 == 0.0 and v12 == 0.0:
            continue
        result.append({
            "token_id": tid_int,
            "token_str": _decode_token(tokenizer, tid_int),
            "t1": v1,
            "t2": v2,
            "t12": v12,
        })
    return result


def _extract_ground_truth(reward_model_entry) -> str:
    """Extract ground truth string from a reward_model non-tensor entry.

    reward_model is a dict-like object (possibly numpy object array element)
    with at least a "ground_truth" key.  Returns empty string when unavailable.
    """
    if reward_model_entry is None:
        return ""
    try:
        gt = reward_model_entry.get("ground_truth", "")
        return str(gt) if gt is not None else ""
    except Exception:
        return str(reward_model_entry)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dump_token_aligned(
    batch: DataProto,
    step: int,
    path: str,
    tokenizer,
    n: int = 16,
) -> None:
    """Append up to `n` token-aligned rollout records for this step to `path`.

    Prerequisites: batch must contain ``entropy_t1``, ``entropy_t2``,
    ``entropy_t12``, ``responses``, ``prompts`` (or ``input_ids``), and
    ``token_level_scores``.
    """
    required = ("entropy_t1", "entropy_t2", "entropy_t12", "responses", "token_level_scores")
    for key in required:
        if key not in batch.batch:
            return  # silently skip if entropy tensors not yet computed

    indices = _select_rollout_indices(batch, n)
    if not indices:
        return

    responses = batch.batch["responses"]          # (B, resp_len)
    t1_all = batch.batch["entropy_t1"]            # (B, resp_len)
    t2_all = batch.batch["entropy_t2"]
    t12_all = batch.batch["entropy_t12"]
    scores_all = batch.batch["token_level_scores"].sum(-1)  # (B,) sample-level score

    # Decode questions
    if "prompts" in batch.batch:
        prompt_ids = batch.batch["prompts"]
    else:
        resp_len = responses.size(1)
        prompt_ids = batch.batch["input_ids"][:, :-resp_len]

    # Optional non-tensor metadata
    uid_arr = batch.non_tensor_batch.get("uid", None)
    data_source_arr = batch.non_tensor_batch.get("data_source", None)
    reward_model_arr = batch.non_tensor_batch.get("reward_model", None)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for idx in indices:
            score = float(scores_all[idx].item())
            correct = score >= 1.0

            ground_truth = ""
            if reward_model_arr is not None:
                ground_truth = _extract_ground_truth(reward_model_arr[idx])

            record = {
                "step": step,
                "uid": str(uid_arr[idx]) if uid_arr is not None else "",
                "data_source": str(data_source_arr[idx]) if data_source_arr is not None else "",
                "score": score,
                "correct": correct,
                "ground_truth": ground_truth,
                "question": tokenizer.decode(prompt_ids[idx].tolist(), skip_special_tokens=True),
                "answer": tokenizer.decode(responses[idx].tolist(), skip_special_tokens=True),
                "tokens": _build_token_list(
                    token_ids=responses[idx].tolist(),
                    t1_row=t1_all[idx].tolist(),
                    t2_row=t2_all[idx].tolist(),
                    t12_row=t12_all[idx].tolist(),
                    tokenizer=tokenizer,
                ),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def trim_token_aligned_file(path: str, current_step: int) -> None:
    """Remove entries with step >= current_step from the dump file.

    Called once at training startup after loading a checkpoint so that
    a resumed run does not produce duplicate step entries.
    Does nothing if the file does not exist.
    """
    if not os.path.exists(path):
        return

    kept_lines = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if int(record.get("step", -1)) < current_step:
                    kept_lines.append(line)
    except OSError:
        return

    # Write back atomically to avoid partial overwrites on failure
    dir_name = os.path.dirname(os.path.abspath(path))
    with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8", suffix=".tmp") as tmp:
        for line in kept_lines:
            tmp.write(line + "\n")
        tmp_path = tmp.name
    os.replace(tmp_path, path)
