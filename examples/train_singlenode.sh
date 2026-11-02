set -x
# Set conda prefix
export CONDA_PREFIX=
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export PATH="$CONDA_PREFIX/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_TIMEOUT=3600  
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAY_BACKEND_LOG_LEVEL=DEBUG



# verl path in src
SRC_DIR=""

# No need to modify this
BASE_PATH=$CONDA_PREFIX/lib/python3.10/site-packages
export PYTHONPATH="${SRC_DIR}:${BASE_PATH}:$PYTHONPATH"
echo "PYTHONPATH = $PYTHONPATH"


PROJECT_NAME=
EXPERIMENT_NAME=
TRAIN_DATA=
VAL_DATA=
MODEL_PATH=
SAVE_BASE_DIR=
LOG_DIR=
ROLLOUT_N=8
CONFIG_DIR=${SRC_DIR}/verl/trainer/config
CONFIG_NAME=ppo_trainer # verl config name

PROJECT_DIR=${PROJECT_NAME}/${EXPERIMENT_NAME}
export WANDB_DIR=${SAVE_BASE_DIR}/${PROJECT_DIR}

N_NODE=1
NGPUS_PER_NODE=8



# Train over a single node, 8 H800-80GB GPUs.
python3 -m verl.trainer.main_ppo --config-path=$CONFIG_DIR --config-name=$CONFIG_NAME \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=32 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=39936 \
    actor_rollout_ref.model.path=$MODEL_PATH  \
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
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.default_local_dir=${SAVE_BASE_DIR}/${PROJECT_DIR} \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$N_NODE \
    trainer.debug=False \
    trainer.dyn_sampling_polaris=True \
    trainer.rollout_data_dir=${SAVE_BASE_DIR}/${PROJECT_DIR}/rollout_data  \
    trainer.validation_data_dir=${SAVE_BASE_DIR}/${PROJECT_DIR}/validation_data \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=100 "${@:1}" \
     2>&1 | tee -a ${LOG_DIR}/${PROJECT_NAME}_${EXPERIMENT_NAME}.log



# Echo the command for verification
echo "Command to be executed:"
echo "$CMD"





# Execute the command
eval "$CMD"
