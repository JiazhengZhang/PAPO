"""Clip-Cov trainer: injects clip_cov config, disables entropy mask."""
from recipe.papo.balanced_ray_trainer import SampleControlledTrainer


class RayPPOTrainer(SampleControlledTrainer):
    def _inject_clip_cov_meta(self, batch):
        cfg = self.config.trainer.get("clip_cov", {})
        batch.meta_info["clip_cov"] = {
            "clip_cov_ratio": float(cfg.get("clip_cov_ratio", 0.0002)),
            "clip_cov_ub": float(cfg.get("clip_cov_ub", 5.0)),
            "clip_cov_lb": float(cfg.get("clip_cov_lb", 1.0)),
        }

    def fit(self):
        original_update_actor = self.actor_rollout_wg.update_actor

        def _update_with_clip_cov(batch):
            self._inject_clip_cov_meta(batch)
            return original_update_actor(batch)

        self.actor_rollout_wg.update_actor = _update_with_clip_cov
        return super().fit()
