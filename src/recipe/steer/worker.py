"""STEER worker: uses SteerActor for both compute_log_prob_with_entropy and update_policy."""

import logging
import os

import torch
from omegaconf import open_dict

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fsdp_utils import (
    fsdp_version,
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
)
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from .dp_actor import SteerActor


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))


class SteerActorRolloutRefWorker(ActorRolloutRefWorker):

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if self._is_actor:
            self.actor = SteerActor(
                config=self.config.actor,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob_with_entropy(self, data: DataProto):
        """Recompute log_prob + entropy (inherits EntropyControlledActor logic)."""
        assert self._is_actor
        assert isinstance(self.actor, SteerActor)
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        data = data.to(torch.cuda.current_device())
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.actor.compute_log_prob_with_entropy(data=data)
            output.meta_info["temperature"] = self.config.rollout.temperature
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output
