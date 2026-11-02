# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
import logging
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type
import copy
import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, AdvantageEstimator, _timer, apply_kl_penalty, compute_advantage

from efficientrl.relay_pool import ReplayPool, build_dataproto, compute_sample_metrics

WorkerType = Type[Worker]

console_logger = logging.getLogger(__name__)


def reduce_actor_metrics(metrics):
    for key, val in metrics.items():
        if key in ["actor/train_involved_num", "actor/train_positive_num", "actor/train_negative_num", "actor/train_involved_token_sum", "actor/train_positive_token_sum", "actor/train_negative_token_sum"]:
            metrics[key] = np.sum(val)
        else:
            metrics[key] = np.mean(val)
    return metrics


def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]




def replace_bad_samples(mask_tensor):
    replace_idxs = torch.nonzero(mask_tensor == 0, as_tuple=True)[0]    
    candidate_idxs = torch.nonzero(mask_tensor == 1, as_tuple=True)[0]

    num_replacements = replace_idxs.size(0)    
    num_candidates = candidate_idxs.size(0)
    
    if num_candidates >= num_replacements:
        perm = torch.randperm(num_candidates)
        chosen_replacement_idxs = candidate_idxs[perm[:num_replacements]]
    else:
        indices = torch.randint(low=0, high=num_candidates, size=(num_replacements,))
        chosen_replacement_idxs = candidate_idxs[indices]
    
    return replace_idxs, chosen_replacement_idxs



def expand_tensor(tensor, n):
    range_tensor = torch.arange(n)    
    expanded = tensor.unsqueeze(1) * n + range_tensor
    return expanded.view(-1)

def compute_correctness_metrics(batch: DataProto):
    metrics = {}
    fully_solved_question_cnt = 0
    fully_unsolved_question_cnt = 0
    at_least_solve_one_time_question_cnt = 0

    correct_idx = batch.batch['token_level_scores'].sum(-1) == 1 
    incorrect_idx = batch.batch['token_level_scores'].sum(-1) == 0
    correct_count = correct_idx.sum()
    incorrect_count = incorrect_idx.sum()
    all_count = correct_count + incorrect_count
    assert correct_count + incorrect_count == batch.batch.batch_size[0], f"correct_count: {correct_count}; incorrect_count: {incorrect_count}, batch_size: {batch.batch.batch_size[0]}"
    print(f"correct_count: {correct_count}; incorrect_count: {incorrect_count}, batch_size: {batch.batch.batch_size[0]}")

    id2correct = defaultdict(lambda: [0, 0])  # [correct_count, total_count]

    bsz = batch.batch.batch_size[0]
    for i in range(bsz):
        uid = batch.non_tensor_batch['uid'][i]
        id2correct[uid][1] += 1  # Total count
        if correct_idx[i]:  # keep_mask[i] = True means correct, so False means incorrect
            id2correct[uid][0] += 1  # Correct count

    # Step 2: Compute correctness percentage
    id2correctness = {uid: correct / total for uid, (correct, total) in id2correct.items()}
    fully_solved_question_cnt += len([uid for uid, correctness in id2correctness.items() if correctness == 1.0])
    fully_unsolved_question_cnt += len([uid for uid, correctness in id2correctness.items() if correctness == 0.0])
    at_least_solve_one_time_question_cnt += len([uid for uid, correctness in id2correctness.items() if correctness > 0.0])

    # Step 3: Filter out UIDs exceeding the threshold
    fully_solved_uids = {uid for uid, correctness in id2correctness.items() if correctness >= 1}
    fully_unsolved_uids = {uid for uid, correctness in id2correctness.items() if correctness <= 0}
    one_time_solved_uids = {uid for uid, correctness in id2correctness.items() if 0 < correctness <= 1}
    tqdm.write(f"{len(fully_solved_uids)} out of {len(id2correctness)} UIDs in current batch are fully correct!")
    tqdm.write(f"{len(fully_unsolved_uids)} out of {len(id2correctness)} UIDs in current batch are fully incorrect!")
    tqdm.write(f"{len(one_time_solved_uids)} out of {len(id2correctness)} UIDs in current batch are at least solved one time!")


    metrics["effective_examples"] = batch.batch.batch_size[0]
    metrics["effective_ratio"] = batch.batch.batch_size[0] / all_count
    metrics["correct_count"] = correct_count
    metrics["incorrect_count"] = incorrect_count
    metrics["correct_ratio"] = correct_count / all_count
    metrics["train_rewards"] = correct_count / all_count
    metrics["incorrect_ratio"] = incorrect_count / all_count
    metrics["fully_solved_cnt"] = fully_solved_question_cnt
    metrics["fully_unsolved_cnt"] = fully_unsolved_question_cnt
    metrics["at_least_solve_one_time_cnt"] = at_least_solve_one_time_question_cnt
    metrics["fully_solved_ratio"] = fully_solved_question_cnt / len(id2correctness)
    metrics["fully_unsolved_ratio"] = fully_unsolved_question_cnt / len(id2correctness)
    metrics["at_least_solve_one_time_ratio"] = at_least_solve_one_time_question_cnt / len(id2correctness)

    return metrics


def compute_data_metrics(batch, use_critic=True):
    def _compute_response_info(batch):
        response_length = batch.batch['responses'].shape[-1]

        prompt_mask = batch.batch['attention_mask'][:, :-response_length]
        response_mask = batch.batch['attention_mask'][:, -response_length:]

        prompt_length = prompt_mask.sum(-1).float()
        response_length = response_mask.sum(-1).float()  # (batch_size,)

        return dict(
            response_mask=response_mask,
            prompt_length=prompt_length,
            response_length=response_length,
        )

    token_level_scores = batch.batch['token_level_scores']
    correct_idx = token_level_scores.sum(-1) == 1 
    incorrect_idx = token_level_scores.sum(-1) == 0


    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']
    # 分别统计correct_idx和incorrect_idx的平均resp长度
    if correct_idx.any():
        correct_response_length = response_length[correct_idx].mean().detach().item()
    else:
        correct_response_length = 0.0

    if incorrect_idx.any():
        incorrect_response_length = response_length[incorrect_idx].mean().detach().item()
    else:
        incorrect_response_length = 0.0

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/positive_mean':
            correct_response_length,
        'response_length/negative_mean':
            incorrect_response_length,
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


class SampleControlledTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping,
        resource_pool_manager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
    ):
        # assert torch.cuda.is_available(), 'cuda must be available on driver'
        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
            processor,
            reward_fn,
            val_reward_fn,
            train_dataset,
            val_dataset,
            collate_fn,
            train_sampler
        )
        self.replay_pool = ReplayPool(max_entries=self.config.trainer.get("replay_pool_size",8192))


    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path, extra_info: Optional[Dict] = None):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")
        extra_info = extra_info or {}

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v
                
        for k, v in extra_info.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch (interleave=False for load balancing)
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=False)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_inputs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # reorder back (since interleave is False)
            indices = torch.arange(len(test_output_gen_batch)).reshape(self.config.actor_rollout_ref.rollout.val_kwargs.n, -1).T.reshape(-1)
            test_batch.reorder(indices)
            test_output_gen_batch.reorder(indices)

            # The order for sample inputs, sample outputs, and sample scores should be the same as the order of test_batch (after reordering)
            sample_inputs.extend([input_texts[i] for i in indices])
            # print("Reordered gen_batch_output at test with indices:", indices)

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # result = self.val_reward_fn(test_batch, return_dict=True)
            # reward_tensor = result["reward_tensor"]
            reward_tensor = self.reward_fn(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            # if "reward_extra_info" in result:
            #     for key, lst in result["reward_extra_info"].items():
            #         reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        non_tensor_batch_keys_to_dump = ("uid","reward_model", "data_source")
        extra_info_to_dump = {}
        for nkey in non_tensor_batch_keys_to_dump:
            if nkey in test_batch.non_tensor_batch.keys():
                value = test_batch.non_tensor_batch[nkey]
                var_to_save = value
                if isinstance(value, np.ndarray):
                    var_to_save = value.tolist()
                extra_info_to_dump[nkey] = var_to_save
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
                extra_info=extra_info_to_dump
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))


    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                
                assert "uid" in batch.non_tensor_batch

                gen_batch = batch
                
                all_uids = []
                uids: np.ndarray = gen_batch.non_tensor_batch["uid"]
                for uid in uids:
                    if uid not in all_uids:
                        all_uids.append(uid)

                # repeat training batch (interleave=False for load balancing)
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=False)
                is_last_step = self.global_steps >= self.total_training_steps
                max_gen_repeats = self.config.trainer.get("max_gen_repeats", 1)
                is_first_epoch = (epoch == 0)

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        gen_round = 0
                        while gen_round < max_gen_repeats:
                            if not self.async_rollout_mode:
                                torch.cuda.empty_cache()
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                                torch.cuda.empty_cache()
                            else:
                                self.async_rollout_manager.wake_up()
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                                self.async_rollout_manager.sleep()

                            indices = torch.arange(len(gen_batch_output)).reshape(self.config.actor_rollout_ref.rollout.n, -1).T.reshape(-1)
                            # gen_batch is not reordered but it's not used anymore
                            # del gen_batch
                            gen_batch_output.reorder(indices)
                            # print("Reordered gen_batch_output at train with indices:", indices)

                            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                                raise NotImplementedError("REMAX is not implemented with interleave=False yet")
                                with _timer("gen_max", timing_raw):
                                    gen_baseline_batch = deepcopy(gen_batch)
                                    gen_baseline_batch.meta_info["do_sample"] = False
                                    gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                                    batch = batch.union(gen_baseline_output)
                                    reward_baseline_tensor = self.reward_fn(batch)
                                    reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                                    batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                                    batch.batch["reward_baselines"] = reward_baseline_tensor

                                    del gen_baseline_batch, gen_baseline_output

                            # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                            # repeat to align with repeated responses in rollout
                            batch_indices = copy.deepcopy(gen_batch_output.non_tensor_batch['index'])
                            # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                            # batch = batch.union(gen_batch_output)
                            rollout_n = self.config.actor_rollout_ref.rollout.n
                        
                            # memorize metrics before dynamic sampling
                            reward_tensor = self.reward_fn(gen_batch_output)
                            sum_reward_tensor = reward_tensor.sum(-1) 
                            reward_reshaped = sum_reward_tensor.reshape(-1, rollout_n) # (batch, rollout_n)
                            reward_batch = reward_reshaped.sum(dim=1) /  rollout_n # (batch, )
                            
                            idx2score = []

                            # save to sink
                            save_reward = reward_batch.numpy().tolist()

                            idx2score.append({"index": batch_indices.tolist(), "score": save_reward})
                            # Handle train_files as a list - take the first file for saving
                            if hasattr(self.config.data.train_files, '__len__') and len(self.config.data.train_files) > 0:
                                # Handle both ListConfig and regular list types
                                train_files = list(self.config.data.train_files) if hasattr(self.config.data.train_files, '__iter__') else [self.config.data.train_files]
                                parquet_file = train_files[0].replace(".parquet", "")
                            else:
                                parquet_file = str(self.config.data.train_files).replace(".parquet", "")
                            save_dir = f"{parquet_file}/"
                            os.makedirs(save_dir, exist_ok=True)
                            with open(f"{save_dir}/{self.config.trainer.experiment_name}.jsonl", "a") as f:
                                for i in range(len(idx2score)):
                                    f.write(json.dumps(idx2score[i]) + "\n")
                            gen_batch_output.batch["token_level_scores"] = reward_tensor
                            gen_round += 1
                            self.replay_pool.add(self.global_steps, gen_batch_output)
                            if self.replay_pool.should_stop_gen(
                                uids=all_uids,
                                rollout_n=self.config.actor_rollout_ref.rollout.n,
                                pos_ratio=self.config.trainer.get("pos_ratio", 0.5),
                                pos_thr=0.5,
                                step=self.global_steps,
                            ):
                                console_logger.info(f"Global step: {self.global_steps} generation stop at turn {gen_round}")
                                break
                            
                    metrics["controlled_sample/gen_round"]=gen_round
                    world_size = self.actor_rollout_wg.world_size
                    entries = self.replay_pool.sample(
                        uids=all_uids,
                        rollout_n=self.config.actor_rollout_ref.rollout.n,
                        pos_ratio=self.config.trainer.get("pos_ratio", 0.5),
                        pos_thr=0.5,
                        world_size=world_size,
                        step=self.global_steps
                    )
                    
                    
                    
                    if not entries:
                        console_logger.warning(f"No valid entries for batch, skip global step {self.global_steps}")
                        continue
                    
                    batch = build_dataproto(entries)
                    
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    
                    metrics.update(compute_sample_metrics(batch))
                    metrics.update(self.replay_pool.metrics())

                    metrics.update(compute_correctness_metrics(batch))
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()


                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    torch.cuda.empty_cache()
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_actor_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)


                    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))

                    non_tensor_batch_keys_to_dump = ("uid","reward_model", "data_source")
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            extra_info_to_dump = {}
                            for nkey in non_tensor_batch_keys_to_dump:
                                if nkey in batch.non_tensor_batch.keys():
                                    value = batch.non_tensor_batch[nkey]
                                    var_to_save = value
                                    if isinstance(value, np.ndarray):
                                        var_to_save = value.tolist()
                                    extra_info_to_dump[nkey] = var_to_save

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict={},
                                dump_path=self.config.trainer.rollout_data_dir,
                                extra_info=extra_info_to_dump,
                            )
                    
                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
