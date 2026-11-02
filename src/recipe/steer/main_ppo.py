"""
STEER recipe entry point.
Adapted from recipe.papo.main_ppo — uses SteerActorRolloutRefWorker + SteerRayPPOTrainer.
"""

import os
import copy

import hydra
import ray
import torch

from verl import DataProto
from verl.utils.reward_score import gsm8k, math
from recipe.papo.reward import _default_compute_score
from recipe.papo.rl_dataset_spec import RLHFDatasetSpec


@ray.remote(num_cpus=1)
def _grade_chunk(chunk):
    """Grade a shard of samples on a Ray worker (true multi-core; sympy grading is GIL-bound).
    chunk: list of (k, data_source, sequences_str, ground_truth) -> list of (k, score)."""
    from recipe.papo.reward import _default_compute_score as _score
    return [(k, _score(ds, s, gt)) for (k, ds, s, gt) in chunk]


class RewardFn():
    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        import time
        from concurrent.futures import ThreadPoolExecutor

        # --- Phase A: extract valid token-id sequences (serial, cheap) ---
        _t = time.time()
        seq_ids = []
        meta = []  # (i, ground_truth, data_source, valid_response_length)
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            seq_ids.append(sequences.tolist())
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            meta.append((i, ground_truth, data_source, valid_response_length))
        t_prep = time.time() - _t

        # --- Phase B: batch-decode all sequences at once (one GIL acquisition, fast tokenizer) ---
        _t = time.time()
        sequences_strs = self.tokenizer.batch_decode(seq_ids)
        t_decode = time.time() - _t

        # --- Phase C: grade in parallel across Ray workers (true multi-core; ProcessPool is unsafe under Ray) ---
        _t = time.time()
        tasks = [(k, meta[k][2], sequences_strs[k], meta[k][1]) for k in range(len(meta))]  # (k, data_source, str, gt)
        n_chunks = min(64, len(tasks)) if tasks else 0
        results = []
        if n_chunks:
            chunks = [tasks[j::n_chunks] for j in range(n_chunks)]  # round-robin for balance
            for part in ray.get([_grade_chunk.remote(c) for c in chunks if c]):
                results.extend(part)
        t_grade = time.time() - _t

        for k, score in results:
            i, ground_truth, data_source, valid_response_length = meta[k]
            reward_tensor[i, valid_response_length - 1] = score

        print(f"[RewardFn timing] n={len(data)} prep={t_prep:.2f}s batch_decode={t_decode:.2f}s grade={t_grade:.2f}s (ray x{n_chunks})", flush=True)

        return reward_tensor


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        debug_mode = config["trainer"]["debug"]
        wandb_env = {
            'WANDB_MODE': 'offline',
            'WANDB_API_KEY': '',
            'WANDB_ANONYMOUS': 'must',
        }
        if not debug_mode:
            ray.init(runtime_env={'env_vars': {'NCCL_DEBUG': 'WARN', **wandb_env}})
        else:
            ray.init(runtime_env={"env_vars": {"RAY_DEBUG": "legacy", 'NCCL_DEBUG': 'INFO', **wandb_env}})

    n_node = config["trainer"]["nnodes"]
    while len(ray.nodes()) < n_node:
        import time
        print(f"Waiting for workers... {len(ray.nodes())}/{n_node} nodes connected")
        time.sleep(5)

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        from verl.utils import hf_processor, hf_tokenizer
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import CriticWorker
            from .worker import SteerActorRolloutRefWorker

            actor_rollout_cls = SteerActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from .ray_trainer import RayPPOTrainer as SteerTrainer

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(SteerActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = RewardFn(tokenizer=tokenizer, num_examine=0)
        val_reward_fn = RewardFn(tokenizer=tokenizer, num_examine=1)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn
        val_data_config = copy.deepcopy(config.data)
        val_data_config.add_template = True
        config.data.add_template = True

        train_dataset = _create_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = _create_dataset(config.data.val_files, val_data_config, tokenizer, processor)
        train_sampler = _create_sampler(config.data, train_dataset)

        trainer = SteerTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


def _create_dataset(data_paths, data_config, tokenizer, processor):
    from torch.utils.data import Dataset
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    else:
        dataset_cls = RLHFDatasetSpec
    return dataset_cls(data_files=data_paths, tokenizer=tokenizer, processor=processor, config=data_config)


def _create_sampler(data_config, dataset):
    from torch.utils.data import RandomSampler, SequentialSampler
    if data_config.shuffle:
        g = torch.Generator()
        g.manual_seed(data_config.get("seed", 1))
        return RandomSampler(data_source=dataset, generator=g)
    return SequentialSampler(data_source=dataset)


if __name__ == "__main__":
    main()
