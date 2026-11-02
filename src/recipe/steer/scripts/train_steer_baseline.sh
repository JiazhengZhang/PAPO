#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SRC_DIR:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${SRC_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
: "${MODEL_PATH:?Set MODEL_PATH to a local model directory}"
export VLLM_USE_V1=1
export WANDB_DISABLED=false
export WANDB_MODE=offline
export WANDB_ANONYMOUS=must

# Requires an already running multi-node Ray cluster.
: "${NODE_IP_LIST:?Set NODE_IP_LIST to the cluster node list}"
: "${INDEX:?Set INDEX to the node rank}"

# === Experiment config ===
N_NODE=2
NGPUS_PER_NODE=8

cd "$SRC_DIR/src"
LOG_DIR="${LOG_DIR:-${SRC_DIR}/log}"
mkdir -p "$LOG_DIR"

dapo14k_train=${SRC_DIR}/data/dapo14k/train_processed.parquet
aime2025_test_path=${SRC_DIR}/data/aime2025/test.parquet
aime2024_test_path=${SRC_DIR}/data/aime2024/test.parquet
amc23_test_path=${SRC_DIR}/data/amc23/test.parquet
minerva_test_path=${SRC_DIR}/data/minerva/test.parquet
olympiadbench_test_path=${SRC_DIR}/data/olympiadbench/test.parquet
math500_test_path=${SRC_DIR}/data/math500/test.parquet

train_files="['$dapo14k_train']"
test_files="['$aime2024_test_path','$aime2025_test_path','$amc23_test_path','$minerva_test_path','$olympiadbench_test_path','$math500_test_path']"

# STEER hyperparams (paper defaults: symmetric linear w_min=0.8 w_max=1.2)
STEER_TOKEN_WEIGHT_MIN=0.8
STEER_TOKEN_WEIGHT_MAX=1.2
STEER_LINEAR=True
STEER_MODE=symmetric   # symmetric | asymmetric

advantage="grpo"
kl_coef=0.0
lr=1e-6
prompt_template_type="qwen3_thinking"
model_name="$MODEL_PATH"
TRAIN_ROLLOUT_N=8
VAL_ROLLOUT_N=8
positive_ratio=0.5
per_batch_max_gen_repeats=1

CONFIG_DIR=${SRC_DIR}/src/recipe/steer/config
CONFIG_NAME=ppo_trainer
PROJECT_NAME=Efficient-RL-STEER-Baseline
EXPERIMENT_NAME=steer-linear-sym-qwen25-7b-base-dapo14k
SAVE_BASE_DIR="${SAVE_BASE_DIR:-${SRC_DIR}/outputs/${EXPERIMENT_NAME}}"
export WANDB_DIR="${WANDB_DIR:-${SRC_DIR}/wandb/${EXPERIMENT_NAME}}"
export WANDB_RUN_DIR=${WANDB_DIR}

mkdir -p "$WANDB_DIR"
mkdir -p "$SAVE_BASE_DIR"

TMP_DIR="${TMP_DIR:-${SRC_DIR}/tmp/steer}"
mkdir -p "$TMP_DIR"

echo "$NODE_IP_LIST" > "$TMP_DIR/env.text" 2>&1
sed "s/:/ slots=/g" "$TMP_DIR/env.text" | sed "s/,/\n/g" > "$TMP_DIR/hostfile"
sed "s/:.//g" "$TMP_DIR/env.text" | sed "s/,/\n/g" > "$TMP_DIR/pssh.hosts"
main_ip=$(head -n 1 "$TMP_DIR/pssh.hosts")
VC_MASTER_HOSTS=$main_ip
PET_MASTER_PORT=6379
DASHBOARD_PORT=8265

export RAY_DASHBOARD_ADDRESS="http://127.0.0.1:$DASHBOARD_PORT"

echo "============================================"
echo "Master IP: $VC_MASTER_HOSTS"
echo "RAY_DASHBOARD_ADDRESS: $RAY_DASHBOARD_ADDRESS"
echo "Current Node INDEX: $INDEX"
echo "STEER: mode=$STEER_MODE linear=$STEER_LINEAR w_min=$STEER_TOKEN_WEIGHT_MIN w_max=$STEER_TOKEN_WEIGHT_MAX"
echo "============================================"

wait_for_ray_dashboard() {
    local max_attempts=60
    local attempt=1
    echo "Waiting for Ray Dashboard at $RAY_DASHBOARD_ADDRESS ..."
    while [ $attempt -le $max_attempts ]; do
        response=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$RAY_DASHBOARD_ADDRESS/api/version" 2>/dev/null)
        if [ "$response" = "200" ]; then
            echo "Ray Dashboard is ready! (HTTP 200)"
            return 0
        fi
        echo "  Attempt $attempt/$max_attempts - HTTP Status: $response, waiting 5s..."
        sleep 5
        attempt=$((attempt + 1))
    done
    echo "Ray Dashboard not ready after $max_attempts attempts"
    return 1
}

if [[ "$INDEX" == "0" ]]; then
    echo "Waiting for Ray cluster to be ready..."
    if ! wait_for_ray_dashboard; then
        echo "ERROR: Ray cluster is not ready!"
        exit 1
    fi

    echo "========== Ray Cluster Status =========="
    ray status
    echo "========================================"

    "$PYTHON_BIN" -m recipe.steer.main_ppo --config-path="$CONFIG_DIR" --config-name="$CONFIG_NAME" \
    algorithm.adv_estimator=$advantage \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.gen_batch_size=256 \
    data.val_batch_size=2048 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.entropy_topk=2048 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.prompt_template_type='qwen3_thinking' \
    trainer.use_balanced_sample=True \
    trainer.steer.token_weight_min=${STEER_TOKEN_WEIGHT_MIN} \
    trainer.steer.token_weight_max=${STEER_TOKEN_WEIGHT_MAX} \
    trainer.steer.linear=${STEER_LINEAR} \
    trainer.steer.entropy_control_mode=${STEER_MODE} \
    trainer.max_gen_repeats=${per_batch_max_gen_repeats} \
    trainer.pos_ratio=${positive_ratio} \
    actor_rollout_ref.rollout.n=${TRAIN_ROLLOUT_N} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    "actor_rollout_ref.model.path=$model_name" \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=49152 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=48000 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=64000 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=64000 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_ROLLOUT_N} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    +actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    trainer.critic_warmup=0 \
    "trainer.logger=[console,wandb]" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    "trainer.rollout_data_dir=${SAVE_BASE_DIR}/rollout_data" \
    "trainer.validation_data_dir=${SAVE_BASE_DIR}/validation_data" \
    "trainer.default_local_dir=${SAVE_BASE_DIR}" \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.val_before_train=False \
    trainer.nnodes=$N_NODE \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.resume_mode=auto \
    trainer.total_epochs=12 "$@" \
    2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}.log"

else
    echo "worker, rank:$INDEX, waiting..."
    sleep infinity
fi
