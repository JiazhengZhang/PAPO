"""KL-Cov trainer: inherits SampleControlledTrainer, injects kl_cov config, disables entropy mask."""

from recipe.papo.balanced_ray_trainer import SampleControlledTrainer


class RayPPOTrainer(SampleControlledTrainer):
    """KL-Cov baseline trainer. Balanced sampling + KL-Cov loss."""

    def _inject_kl_cov_meta(self, batch):
        kl_cov_cfg = self.config.trainer.get("kl_cov", {})
        batch.meta_info["kl_cov"] = {
            "kl_cov_ratio": float(kl_cov_cfg.get("kl_cov_ratio", 0.0002)),
            "ppo_kl_coef": float(kl_cov_cfg.get("ppo_kl_coef", 1.0)),
        }

    def fit(self):
        original_update_actor = self.actor_rollout_wg.update_actor

        def _update_actor_with_kl_cov(batch):
            self._inject_kl_cov_meta(batch)
            return original_update_actor(batch)

        self.actor_rollout_wg.update_actor = _update_actor_with_kl_cov
        return super().fit()
