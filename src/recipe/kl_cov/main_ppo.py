"""
KL-Cov recipe entry point (Entropy Mechanism paper baseline).
"""

import os
import copy

import hydra
import ray
import torch

from verl import DataProto
from recipe.papo.reward import _default_compute_score
from recipe.papo.rl_dataset_spec import RLHFDatasetSpec


@ray.remote(num_cpus=1)
def _grade_chunk(chunk):
    """Grade a shard of samples on a Ray worker (true multi-core; sympy grading is GIL-bound)."""
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

        _t = time.time()
        seq_ids = []
        meta = []
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

        _t = time.time()
        sequences_strs = self.tokenizer.batch_decode(seq_ids)
        t_decode = time.time() - _t

        _t = time.time()
        tasks = [(k, meta[k][2], sequences_strs[k], meta[k][1]) for k in range(len(meta))]
        n_chunks = min(64, len(tasks)) if tasks else 0
        results = []
        if n_chunks:
            chunks = [tasks[j::n_chunks] for j in range(n_chunks)]
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
        wandb_env = {'WANDB_MODE': 'offline', 'WANDB_API_KEY': '', 'WANDB_ANONYMOUS': 'must'}
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

        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import CriticWorker
        from .worker import KLCovActorRolloutRefWorker
        from .ray_trainer import RayPPOTrainer as KLCovTrainer
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        actor_rollout_cls = KLCovActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        mapping = {Role.ActorRollout: global_pool_id, Role.Critic: global_pool_id}

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(KLCovActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = RewardFn(tokenizer=tokenizer, num_examine=0)
        val_reward_fn = RewardFn(tokenizer=tokenizer, num_examine=1)
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn
        val_data_config = copy.deepcopy(config.data)
        val_data_config.add_template = True
        config.data.add_template = True

        train_dataset = RLHFDatasetSpec(data_files=config.data.train_files, tokenizer=tokenizer, processor=processor, config=config.data)
        val_dataset = RLHFDatasetSpec(data_files=config.data.val_files, tokenizer=tokenizer, processor=processor, config=val_data_config)

        from torch.utils.data import RandomSampler
        g = torch.Generator()
        g.manual_seed(config.data.get("seed", 1))
        train_sampler = RandomSampler(data_source=train_dataset, generator=g)

        trainer = KLCovTrainer(
            config=config, tokenizer=tokenizer, processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn, val_reward_fn=val_reward_fn,
            train_dataset=train_dataset, val_dataset=val_dataset,
            collate_fn=collate_fn, train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
