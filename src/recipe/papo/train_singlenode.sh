#!/usr/bin/env bash
# Launch template; supply the exact overrides used for your experiment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
: "${MODEL_PATH:?Set MODEL_PATH to the model directory or Hugging Face ID}"
: "${TRAIN_DATA:?Set TRAIN_DATA to a training parquet path}"
: "${VAL_DATA:?Set VAL_DATA to a validation parquet path or Hydra list}"
PROJECT_NAME="${PROJECT_NAME:-PAPO}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-papo-run}"
SAVE_BASE_DIR="${SAVE_BASE_DIR:-${SRC_DIR}/../outputs}"
LOG_DIR="${LOG_DIR:-${SRC_DIR}/../logs}"
ROLLOUT_N="${ROLLOUT_N:-8}"
CONFIG_DIR="${SCRIPT_DIR}/config"
CONFIG_NAME=ppo_trainer
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

PROJECT_DIR=${PROJECT_NAME}/${EXPERIMENT_NAME}
export WANDB_DIR=${SAVE_BASE_DIR}/${PROJECT_DIR}

N_NODE=${N_NODE:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}



mkdir -p "$LOG_DIR" "$WANDB_DIR"

# Legacy template settings; not a verified paper reproduction configuration.
"$PYTHON_BIN" -m recipe.papo.main_ppo --config-path="$CONFIG_DIR" --config-name="$CONFIG_NAME" \
    algorithm.adv_estimator=grpo \
    "data.train_files=${TRAIN_DATA}" \
    "data.val_files=${VAL_DATA}" \
    data.train_batch_size=32 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=39936 \
    "actor_rollout_ref.model.path=$MODEL_PATH"  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
    actor_rollout_ref.rollout.max_num_batched_tokens=40960 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.5 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.2 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=${ROLLOUT_N} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    "trainer.logger=[console,wandb]" \
    "trainer.project_name=${PROJECT_NAME}" \
    "trainer.default_local_dir=${SAVE_BASE_DIR}/${PROJECT_DIR}" \
    "trainer.experiment_name=$EXPERIMENT_NAME" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$N_NODE \
    trainer.debug=False \
    trainer.dyn_sampling_polaris=True \
    "trainer.rollout_data_dir=${SAVE_BASE_DIR}/${PROJECT_DIR}/rollout_data"  \
    "trainer.validation_data_dir=${SAVE_BASE_DIR}/${PROJECT_DIR}/validation_data" \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=100 \
    trainer.max_gen_repeats=4 \
    trainer.pos_ratio=0.5 \
    "$@" 2>&1 | tee -a "${LOG_DIR}/${PROJECT_NAME}_${EXPERIMENT_NAME}.log"
