import itertools
import logging
import os
from typing import Tuple

import torch
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.actor.dp_actor import DataParallelPPOActor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class EntropyControlledActor(DataParallelPPOActor):
    def _forward_micro_batch_with_entropy(self, micro_batch, temperature, calculate_entropy=True) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            entropy_t1 = None
            entropy_t2 = None
            entropy_mass_k = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled, inplace_backward=inplace_backward)  # (total_nnz)

                # compute entropy
                if calculate_entropy:
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                    vocab_size = logits_rmpad.size(-1)
                    topk = min(self.config.get("entropy_topk", 64), vocab_size)
                    if topk <= 0 or topk >= vocab_size:
                        log_probs_full = torch.nn.functional.log_softmax(logits_rmpad, dim=-1)  # (total_nnz, vocab_size)
                        probs_full = torch.exp(log_probs_full)
                        mu_rmpad = torch.sum(probs_full * log_probs_full, dim=-1)  # (total_nnz)
                        entropy_s_rmpad = torch.sum(probs_full * probs_full * (log_probs_full - mu_rmpad.unsqueeze(-1)), dim=-1)  # (total_nnz)
                        mass_k = torch.ones_like(mu_rmpad)  # (total_nnz)
                    else:
                        topk_logits, _ = torch.topk(logits_rmpad, k=topk, dim=-1)  # (total_nnz, topk)
                        topk_exp = torch.exp(topk_logits)
                        sum_exp_topk = topk_exp.sum(dim=-1)  # (total_nnz)
                        kth_logit = topk_logits[:, -1]  # (total_nnz)
                        tail_size = vocab_size - topk
                        tail_exp = torch.exp(kth_logit) * tail_size
                        log_z = torch.log(sum_exp_topk + tail_exp)
                        log_probs_topk = topk_logits - log_z.unsqueeze(-1)  # (total_nnz, topk)
                        probs_topk = torch.exp(log_probs_topk)  # (total_nnz, topk)
                        mass_k = probs_topk.sum(dim=-1)  # (total_nnz)
                        tail_mass = (1.0 - mass_k).clamp(min=1e-12)
                        p_tail = tail_mass / tail_size  # (total_nnz)
                        log_p_tail = torch.log(p_tail)  # (total_nnz)
                        mu_rmpad = torch.sum(probs_topk * log_probs_topk, dim=-1) + tail_size * p_tail * log_p_tail  # (total_nnz)
                        entropy_s_rmpad = (
                            torch.sum(probs_topk * probs_topk * (log_probs_topk - mu_rmpad.unsqueeze(-1)), dim=-1)
                            + tail_size * p_tail * p_tail * (log_p_tail - mu_rmpad)
                        )  # (total_nnz)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                        mu_rmpad = gather_outpus_and_unpad(mu_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                        entropy_s_rmpad = gather_outpus_and_unpad(entropy_s_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                        mass_k = gather_outpus_and_unpad(mass_k, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                    full_mu = pad_input(hidden_states=mu_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                    full_entropy_s = pad_input(hidden_states=entropy_s_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                    full_mass_k = pad_input(hidden_states=mass_k.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                    mu = full_mu.squeeze(-1)[:, -response_length - 1 : -1]
                    entropy_s = full_entropy_s.squeeze(-1)[:, -response_length - 1 : -1]
                    mass_k = full_mass_k.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                # t1 uses sampled token; t2 uses full distribution term from logits.
                entropy_t1 = torch.exp(log_probs) * (log_probs - mu)  # (bsz, response_length)
                entropy_t2 = entropy_s  # (bsz, response_length)
                entropy_mass_k = mass_k  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                raise ValueError("For entropy controlled training, use remove_padding = True")
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                if calculate_entropy:
                    entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs, entropy_t1, entropy_t2, entropy_mass_k
    
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob_with_entropy(self, data: DataProto) -> DataProto:
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        entropy_t1_lst = []
        entropy_t2_lst = []
        entropy_mass_k_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, entropy_t1, entropy_t2, entropy_mass_k = self._forward_micro_batch_with_entropy(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
            entropy_lst.append(entropy)
            entropy_t1_lst.append(entropy_t1)
            entropy_t2_lst.append(entropy_t2)
            entropy_mass_k_lst.append(entropy_mass_k)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = torch.concat(entropy_lst, dim=0)
        entropys_t1 = torch.concat(entropy_t1_lst, dim=0)
        entropys_t2 = torch.concat(entropy_t2_lst, dim=0)
        entropys_mass_k = torch.concat(entropy_mass_k_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            entropys = entropys[revert_indices]
            entropys_t1 = entropys_t1[revert_indices]
            entropys_t2 = entropys_t2[revert_indices]
            entropys_mass_k = entropys_mass_k[revert_indices]
        tensors = {
            "old_log_probs": log_probs,
            "entropys": entropys,
            "entropys_t1": entropys_t1,
            "entropys_t2": entropys_t2,
            "entropys_mass_k": entropys_mass_k,
        }
        
        results = DataProto.from_dict(
            tensors=tensors
        )
        return results
