# Training configuration

The tables below record the settings reported in the paper and map them to this
codebase. The launch template is a general starting point; its defaults differ
from these settings.

## Shared math settings

| Setting | Paper |
| --- | --- |
| Backbones | Qwen2.5-7B-Base; Qwen2.5-14B-Base |
| Training data | DAPO-Math-17K |
| Hardware | 16 × H20 GPUs |
| Training epochs | 6 |
| Learning rate | 1e-6 |
| Sampling temperature | 1.0 |
| Generation batch size | 256 |
| Training batch size | 128 |
| Gradient accumulation | 64 steps |
| Responses per training prompt | 8 |
| Maximum completion length | 8,192 tokens |
| PPO clipping offsets, where used | low = 0.2; high = 0.28 |
| Evaluation | Mean@32, for both math and OOD |

PAPO uses no additional KL penalty. KL-Cov instead applies its token-selective
penalty with coefficient 1.

Training and evaluation sampling counts are separate:
`actor_rollout_ref.rollout.n` controls training rollouts, while
`actor_rollout_ref.rollout.val_kwargs.n` controls validation rollouts.
The provided AIME evaluation scripts have their own `--n` argument.

## PAPO controller

The main components are:

| Component | File |
| --- | --- |
| Token/distribution statistics | [`dp_actor.py`](../src/recipe/papo/dp_actor.py) |
| Advantage-weighted polarity (`entropy_t12`) | [`balanced_ray_trainer.py`](../src/recipe/papo/balanced_ray_trainer.py) |
| Branch selection and advantage rescaling | [`control_policy.py`](../src/recipe/papo/control_policy.py) |
| Symmetric entropy-slope controller | [`AdaptiveSymController`](../src/recipe/papo/adaptive_entropy.py) |

Set `trainer.entropy_mask.enable=True` and
`trainer.entropy_mask.policy=adaptive_sym_topk` to select the symmetric controller.
It reads the `trainer.entropy_mask.adaptive` configuration block.

| Paper setting | Hydra key | Value |
| --- | --- | --- |
| Polarity vocabulary approximation | `actor_rollout_ref.actor.entropy_topk` | 2048 |
| Negative-branch lower bound | `trainer.entropy_mask.adaptive.omega_neg_min` | 0.98 |
| Negative-branch upper bound | `trainer.entropy_mask.adaptive.omega_neg_max` | 1.03 (7B); 1.02 (14B) |
| Warmup entropy-slope EMA | `trainer.entropy_mask.adaptive.beta_warmup` | 0.95 |
| Subsequent entropy-slope EMA | `trainer.entropy_mask.adaptive.beta_run` | 0.9 |
| Entropy gate ratio | `trainer.entropy_mask.adaptive.entropy_gate_ratio` | 0.3 |

The paper reports 20 neutral warmup steps. The implementation exposes both
`adaptive.skip` and `adaptive.warmup`, with the transition at their sum; setting
`warmup=20` alone also retains the configured initial skip. The controller also
exposes `adaptive.beta_omega`, which smooths each branch's weight after computing
the reciprocal target weights.

Token selection is configured independently through
`trainer.entropy_mask.schedule`: `keep_pos_ratio` and `keep_neg_ratio` specify
the fraction retained within each polarity branch. Setting both to `1.0` retains
all nonzero-polarity response tokens. The generic template contains a changing
selection schedule, so enabling the controller alone does not select this mode.
`apply_to_response_mask` controls whether the selection masks the policy loss.

The configuration file currently defaults to `entropy_topk=256`, disabled
polarity control, weight bounds `[0.7, 1.3]`, and a disabled entropy gate. These are
template defaults; use the settings above when configuring PAPO.

## Additional baselines

### STEER

Entry: [`recipe.steer.main_ppo`](../src/recipe/steer/main_ppo.py).
Configuration: [`ppo_trainer.yaml`](../src/recipe/steer/config/ppo_trainer.yaml).

```yaml
trainer:
  steer:
    token_weight_min: 0.8
    token_weight_max: 1.2
    linear: true
```

`linear` selects the token-weight mapping, not the learning-rate schedule.

### Clip-Cov

Entry: [`recipe.clip_v.main_ppo`](../src/recipe/clip_v/main_ppo.py).
Configuration: [`ppo_trainer.yaml`](../src/recipe/clip_v/config/ppo_trainer.yaml).

```yaml
trainer:
  clip_cov:
    clip_cov_ratio: 0.0002
    clip_cov_lb: 1.0
    clip_cov_ub: 5.0
```

The bounds apply to the covariance score. The selection ratio is 0.02% of tokens.

### KL-Cov

Entry: [`recipe.kl_cov.main_ppo`](../src/recipe/kl_cov/main_ppo.py).
Configuration: [`ppo_trainer.yaml`](../src/recipe/kl_cov/config/ppo_trainer.yaml).

```yaml
trainer:
  kl_cov:
    kl_cov_ratio: 0.0002
    ppo_kl_coef: 1.0
```

The coefficient applies to the token-selective KL term relative to the rollout
policy, rather than a separate fixed reference model.

## Runtime and evaluation utilities

Expose this repository's `src` directory on `PYTHONPATH`. This selects the modified
`src/verl` implementation alongside `recipe` and `deepscaler`.
The vendored [verl package metadata](../src/third_party/verl/setup.py) and inherited
[reward dependency list](../src/tools/setup.py) document dependency families;
they are not a locked GPU environment for the reported experiments.

The included evaluation utilities generate and grade AIME24/AIME25 responses.
Run them from `src`, since their benchmark paths are relative to that directory.
The [math/OOD results](results.md) are paper results; the included AIME
utilities are only part of that evaluation workflow.
