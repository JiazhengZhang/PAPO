# replay_pool.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Sequence
import random
import logging

import torch
import numpy as np

from verl import DataProto

logger = logging.getLogger(__name__)


@dataclass
class Entry:
    """Inner entry for caching the pool
    """
    uid: str
    step: int
    last: int
    used: int
    
    # Tensors
    prompts: torch.Tensor
    responses: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    token_level_scores: Optional[torch.Tensor]
    response_mask: Optional[torch.Tensor]
    
    # Non Tensors
    non_tensor_info: Dict[str, Any]
    meta_info: Dict[str, Any]
    score: float
    
def build_dataproto(
    entries: List[Entry]
) -> DataProto:
    if not entries:
        raise ValueError("entries are empty")

    prompts = torch.stack([e.prompts for e in entries], dim=0)
    responses = torch.stack([e.responses for e in entries], dim=0)
    input_ids = torch.stack([e.input_ids for e in entries], dim=0)
    attention_mask = torch.stack([e.attention_mask for e in entries], dim=0)
    position_ids = torch.stack([e.position_ids for e in entries], dim=0)
    token_level_scores = torch.stack([e.token_level_scores for e in entries], dim=0)

    has_resp_mask = entries[0].response_mask is not None
    if has_resp_mask:
        response_mask = torch.stack([e.response_mask for e in entries], dim=0)
    else:
        response_mask = None

    batch_dict: Dict[str, Any] = {
        "prompts": prompts,
        "responses": responses,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "token_level_scores": token_level_scores,
    }
    if response_mask is not None:
        batch_dict["response_mask"] = response_mask

    uids = [e.uid for e in entries]
    batch_dict["uid"] = np.array(uids, dtype=object)

    all_keys = set()
    for e in entries:
        all_keys.update(e.non_tensor_info.keys())

    for k in all_keys:
        vals = [e.non_tensor_info.get(k, None) for e in entries]
        batch_dict[k] = np.array(vals, dtype=object)

    dp = DataProto.from_single_dict(batch_dict)


    global_token_num = attention_mask.sum(dim=-1).tolist()
    dp.meta_info["global_token_num"] = global_token_num

    return dp

def compute_sample_metrics(batch: DataProto, prefix = "controlled_sample", pos_thr: int = 0.5):
    """Compute metrics for a sampled batch from replay pool
    """
    metrics = {}
    token_level_scores : torch.Tensor = batch.batch["token_level_scores"]
    uids: np.ndarray = batch.non_tensor_batch["uid"]
    distinct_uids = set()
    for uid in uids:
        distinct_uids.add(uid)
        
    scores = token_level_scores.sum(dim=-1)

    total = scores.numel()
    if total == 0:
        return metrics

    pos_mask = scores > pos_thr
    num_pos = pos_mask.sum()
    num_neg = total - num_pos

    metrics[f"{prefix}/total"] = total
    metrics[f"{prefix}/pos"] = num_pos.item()
    metrics[f"{prefix}/neg"] = num_neg.item()
    metrics[f"{prefix}/pos_ratio"] = (num_pos.float() / total).item()
    metrics[f"{prefix}/valid_batch_size"] = len(distinct_uids)

    return metrics


class ReplayPool:

    def __init__(self, max_entries: int = 200_000) -> None:
        self.max_entries = max_entries
        self.by_uid: Dict[str, List[Entry]] = {}
        self.all: List[Entry] = []
        
    def metrics(self) -> Dict[str, Any]:
        metrics = {}
        metrics["replay_pool/total_entries"] = len(self.all)
        metrics["replay_pool/total_uids"] = len(self.by_uid)
        return metrics

    def add(self, step: int, batch: DataProto) -> None:
        """
        Add a batch to replay pool
          - batch.non_tensor_batch["uid"] must exist
          - batch.batch contains：
              - "prompts"
              - "responses"
              - "input_ids"
              - "attention_mask"
              - "position_ids"
            optional：
              - "token_level_scores"
              - "response_mask"
        """
        nt = batch.non_tensor_batch
        tb = batch.batch

        if "uid" not in nt:
            raise ValueError("ReplayPool.add: batch.non_tensor_batch['uid'] does not exist。")

        uids = nt["uid"]  #  np.array(dtype=object)
        
        prompts = tb["prompts"].detach().cpu()
        responses = tb["responses"].detach().cpu()
        input_ids = tb["input_ids"].detach().cpu()
        attention_mask = tb["attention_mask"].detach().cpu()
        position_ids = tb.get("position_ids", None)
        if position_ids is not None:
            position_ids = position_ids.detach().cpu()
        token_scores = tb.get("token_level_scores", None)
        if token_scores is not None:
            token_scores = token_scores.detach().cpu()
        else:
            raise ValueError("Token level scores must be computed for replay pool")
        response_mask = tb.get("response_mask", None)
        if response_mask is not None:
            response_mask = response_mask.detach().cpu()

        B = input_ids.size(0)

        # Check batch size consistency
        def _check(name: str, t: Optional[torch.Tensor]):
            if t is None:
                return
            if t.size(0) != B:
                raise ValueError(
                    f"{name}.size(0)={t.size(0)} is not the same as batch size {B} 。"
                )

        _check("prompts", prompts)
        _check("responses", responses)
        _check("attention_mask", attention_mask)
        _check("position_ids", position_ids)
        _check("token_level_scores", token_scores)
        _check("response_mask", response_mask)

        # Build entries
        for i in range(B):
            uid = str(uids[i])

            non_tensor_info: Dict[str, Any] = {}
            for k, v in nt.items():
                if k == "uid":
                    continue
                try:
                    if hasattr(v, "__len__") and len(v) == B:
                        non_tensor_info[k] = v[i]
                except TypeError:
                    pass


            meta_info: Dict[str, Any] = batch.meta_info

            # Calculate the full scores
            scores_i = token_scores[i]
            score = float(scores_i.sum().item())

            entry = Entry(
                uid=uid,
                step=step,
                last=step,
                used=0,
                prompts=prompts[i],
                responses=responses[i],
                input_ids=input_ids[i],
                attention_mask=attention_mask[i],
                position_ids=position_ids[i] if position_ids is not None else None,
                token_level_scores=scores_i,
                response_mask=response_mask[i] if response_mask is not None else None,
                non_tensor_info=non_tensor_info,
                meta_info=meta_info,
                score=score,
            )

            self._push(entry)

    def _push(self, e: Entry) -> None:
        self.by_uid.setdefault(e.uid, []).append(e)
        self.all.append(e)

        if len(self.all) > self.max_entries:
            self._pop()

    def _pop(self) -> None:
        if not self.all:
            return
        old = self.all.pop(0)
        lst = self.by_uid.get(old.uid, [])
        try:
            lst.remove(old)
        except ValueError:
            pass
        if not lst:
            self.by_uid.pop(old.uid, None)

    def __len__(self) -> int:
        return len(self.all)

    def num_uid(self, uid: str) -> int:
        return len(self.by_uid.get(uid, []))
    
    def should_stop_gen(
        self,
        uids: Sequence[str],
        rollout_n: int,
        pos_ratio: float,
        pos_thr: float,
        step: int,
    ) -> bool:
        uniq_uids = []
        seen = set()
        for u in uids:
            su = str(u)
            if su not in seen:
                seen.add(su)
                uniq_uids.append(su)

        uid_list = []
        uid_pos = []
        uid_neg = []

        for uid in uniq_uids:
            entries = self.by_uid.get(uid)
            if not entries:
                continue
            cur_entries = [e for e in entries if e.step == step]
            if not cur_entries:
                continue

            P_i = [e for e in cur_entries if e.score > pos_thr]
            N_i = [e for e in cur_entries if e.score <= pos_thr]

            if not P_i or not N_i:
                continue
            if len(P_i) + len(N_i) < rollout_n:
                continue

            uid_list.append(uid)
            uid_pos.append(P_i)
            uid_neg.append(N_i)

        M = len(uid_list)
        if M == 0:
            return False

        L = []
        U = []
        for P_i, N_i in zip(uid_pos, uid_neg):
            Li = max(0, rollout_n - len(N_i))
            Ui = min(rollout_n, len(P_i))

            if Li > Ui:
                return False

            L.append(Li)
            U.append(Ui)

        T_target = M * rollout_n 
        Pos_target_raw = int(round(T_target * pos_ratio))

        Pos_min = sum(L)
        Pos_max = sum(U)

        return Pos_min <= Pos_target_raw <= Pos_max
    
    def sample(
        self,
        uids: Sequence[str],
        rollout_n: int,
        pos_ratio: float,
        pos_thr: float,
        world_size: int,
        *,
        step: Optional[int] = None,
    ) -> List[Entry]:
        """Sample a list of entries from the replay pool.
        If samples are not enough for specific uid, it will discard the uid.

        Args:
            uids (Sequence[str]): candidate uids
            rollout_n (int): how many samples are selected for each uid
            pos_ratio (float): ratio of the positive samples
            pos_thr (float): score threshold to identify positive sample
            step (Optional[int], optional): global step. Defaults to None.

        Returns:
            List[Entry]: Selected samples
        """

        uniq_uids: List[str] = []
        seen = set()
        for u in uids:
            su = str(u)
            if su not in seen:
                seen.add(su)
                uniq_uids.append(su)

        uid_list: List[str] = []
        uid_pos: List[List[Entry]] = []
        uid_neg: List[List[Entry]] = []

        for uid in uniq_uids:
            entries = self.by_uid.get(uid)
            if not entries:
                continue

            P_i = [e for e in entries if e.score > pos_thr]
            N_i = [e for e in entries if e.score <= pos_thr]

            if not P_i or not N_i:
                continue

            if len(P_i) + len(N_i) < rollout_n:
                continue

            uid_list.append(uid)
            uid_pos.append(P_i)
            uid_neg.append(N_i)

        M = len(uid_list)
        if M == 0:
            return []

        L: List[int] = [] # per-sample lower limit
        U: List[int] = [] # per-sample upper limit
        for P_i, N_i in zip(uid_pos, uid_neg):
            Li = max(0, rollout_n - len(N_i))
            Ui = min(rollout_n, len(P_i))
            if Li > Ui:
                return []
            L.append(Li)
            U.append(Ui)

        # Target batch size 
        T_target = M * rollout_n

        # Pos num ratio
        Pos_target_raw = int(round(T_target * pos_ratio))

        Pos_min = sum(L)
        Pos_max = sum(U)

        # Clip postive nuber to max valid, default to the requested ratio
        Pos_target = min(max(Pos_target_raw, Pos_min), Pos_max)
        if Pos_target != Pos_target_raw:
            logger.warning(
                f"Adjust positive sample target from {Pos_target_raw} to {Pos_target} "
                f"due to per-uid sample limits."
            )

        # Per-sample pos num
        pos_i = L[:]
        cur_pos = Pos_min
        remain = Pos_target - cur_pos

        if remain < 0:
            remain = 0
        # Round Robin
        indices = list(range(M))
        while remain > 0:
            random.shuffle(indices)
            progressed = False
            for idx in indices:
                if remain <= 0:
                    break
                if pos_i[idx] < U[idx]:
                    pos_i[idx] += 1
                    remain -= 1
                    progressed = True
            if not progressed:
                break
        
        idx_all = list(range(M))
        if world_size is not None and world_size > 0:
            kept_M = M
            while kept_M > 0 and (kept_M * rollout_n) % world_size != 0:
                kept_M -= 1

            if kept_M == 0:
                logger.warning(
                    f"ReplayPool.sample: cannot find uid subset s.t. "
                    f"kept_M * rollout_n divisible by world_size={world_size}, return empty."
                )
                return []

            if kept_M < M:
                random.shuffle(idx_all)
                idx_keep = sorted(idx_all[:kept_M])
            else:
                idx_keep = idx_all
        else:
            idx_keep = idx_all
        
        # Final samples
        chosen: List[Entry] = []
        for idx in idx_keep:
            P_i = uid_pos[idx]
            N_i = uid_neg[idx]
            pi = pos_i[idx]
            ni = rollout_n - pi

            if pi > 0:
                pos_samples = random.sample(P_i, pi)
            else:
                pos_samples = []
            if ni > 0:
                neg_samples = random.sample(N_i, ni)
            else:
                neg_samples = []

            chosen.extend(pos_samples)
            chosen.extend(neg_samples)

        if step is not None:
            for e in chosen:
                e.used += 1
                e.last = step

        random.shuffle(chosen)
        assert len(chosen) % world_size == 0, "length of chosen samples must match world size"
        return chosen