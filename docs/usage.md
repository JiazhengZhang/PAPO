# Using the code

[Back to README](../README.md)

The training code uses Linux/CUDA with PyTorch, Ray, vLLM, FlashAttention, and Hydra.
The modified `verl` implementation is in `src/verl`; expose `src` on `PYTHONPATH`
so that recipe imports use this implementation:

```bash
# From the repository root, in your training environment.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

| Method | Entry point / implementation |
| --- | --- |
| **PAPO** | [`recipe.papo.main_ppo`](../src/recipe/papo/main_ppo.py) |
| DAPO | [`recipe.dapo.src.main_dapo`](../src/recipe/dapo/src/main_dapo.py) |
| Ent-Reg | [`actor.entropy_coeff`](../src/recipe/papo/dp_actor.py), with polarity control disabled |
| 80/20 | [`entropy_topk`](../src/recipe/papo/control_policy.py), with `keep_ratio=0.2` |
| STEER | [`recipe.steer.main_ppo`](../src/recipe/steer/main_ppo.py) |
| Clip-Cov | [`recipe.clip_v.main_ppo`](../src/recipe/clip_v/main_ppo.py) |
| KL-Cov | [`recipe.kl_cov.main_ppo`](../src/recipe/kl_cov/main_ppo.py) |

The [PAPO launch template](../src/recipe/papo/train_singlenode.sh) accepts `MODEL_PATH`,
`TRAIN_DATA`, `VAL_DATA`, `SAVE_BASE_DIR`, and `LOG_DIR`, plus trailing Hydra
overrides. Its base configuration leaves polarity control disabled. To inspect a
configuration with the symmetric controller enabled, without starting training:

```bash
MODEL_PATH=/path/to/model \
TRAIN_DATA=/path/to/train.parquet \
VAL_DATA=/path/to/validation.parquet \
bash src/recipe/papo/train_singlenode.sh \
  trainer.entropy_mask.enable=True \
  trainer.entropy_mask.policy=adaptive_sym_topk \
  --cfg job
```

This previews the template with two controller overrides. The [configuration
guide](configuration.md) lists the paper's settings and the separate token
selection, warmup, and controller options that must be set for a training run.

The [evaluation directory](../src/evaluation) contains AIME24/AIME25 generation and
grading utilities. Their relative data paths assume execution from `src`.
The [reported results](results.md) cover six math and three OOD benchmarks.

## Configuration

The paper uses a top-$k$ polarity approximation with $k=2048$, eight responses per
training prompt, and a maximum completion length of 8,192 tokens.

| PAPO controller setting | 7B | 14B |
| --- | ---: | ---: |
| Negative-branch weight bounds | [0.98, 1.03] | [0.98, 1.02] |
| Neutral warmup | 20 steps | 20 steps |
| Entropy-slope EMA, warmup / after warmup | 0.95 / 0.9 | 0.95 / 0.9 |
| Entropy gate, relative to warmup reference | 0.3 | 0.3 |

The additional baselines use the following settings:

| Baseline | Settings |
| --- | --- |
| STEER | Linear token-weight mapping; weights in [0.8, 1.2] |
| Clip-Cov | Token-selection ratio 0.0002 (0.02%); covariance bounds [1, 5] |
| KL-Cov | Selected-token ratio 0.0002 (0.02%); KL coefficient 1 |

See [configuration details](configuration.md) for the corresponding Hydra
keys and shared training settings.
