"""Clip-Cov worker. Adds compute_log_prob_with_entropy for SampleControlledTrainer compatibility."""

import logging
import os

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from .dp_actor import ClipCovActor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))


class ClipCovActorRolloutRefWorker(ActorRolloutRefWorker):
    """Worker using ClipCovActor — standard rollout, Clip-Cov loss in update."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if self._is_actor:
            self.actor = ClipCovActor(
                config=self.config.actor,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob_with_entropy(self, data: DataProto):
        """Delegate to parent's compute_log_prob which already returns old_log_probs + entropys."""
        return self.compute_log_prob(data)
