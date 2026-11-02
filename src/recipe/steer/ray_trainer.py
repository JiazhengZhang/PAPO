"""
STEER Ray Trainer — thin wrapper over balanced_ray_trainer.

Inherits all balanced sampling / entropy logging, but:
- Forces entropy_mask.enable=False (STEER handles things at loss level, not mask level)
- Injects STEER config into batch.meta_info before actor update
"""

from recipe.papo.balanced_ray_trainer import SampleControlledTrainer


class RayPPOTrainer(SampleControlledTrainer):
    """STEER baseline trainer.

    Reuses the full balanced trainer pipeline (rollout, balanced sampling,
    entropy stats, rollout dump) but skips entropy mask application.
    STEER's token weighting is handled inside the SteerActor.update_policy.
    """

    def _inject_steer_meta(self, batch):
        """Put STEER hyperparams into batch.meta_info so the actor can read them."""
        steer_cfg = self.config.trainer.get("steer", {})
        batch.meta_info["steer"] = {
            "token_weight_min": float(steer_cfg.get("token_weight_min", 0.8)),
            "token_weight_max": float(steer_cfg.get("token_weight_max", 1.2)),
            "linear": bool(steer_cfg.get("linear", True)),
            "entropy_control_mode": str(steer_cfg.get("entropy_control_mode", "symmetric")),
        }

    def fit(self):
        # Monkey-patch: wrap the parent's actor update to inject STEER meta.
        # This avoids copy-pasting the entire 400-line fit() loop.
        original_update_actor = self.actor_rollout_wg.update_actor

        def _update_actor_with_steer(batch):
            self._inject_steer_meta(batch)
            return original_update_actor(batch)

        self.actor_rollout_wg.update_actor = _update_actor_with_steer
        return super().fit()
