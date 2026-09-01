#!/bin/bash

# ==========================================================
# GPU
# ==========================================================

CUDA_VISIBLE_DEVICES=0,1,2
NPROC_PER_NODE=3

# ==========================================================
# Path
# ==========================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REWARD_FILE="${PROJECT_ROOT}/scripts/financial_rlvr/verifiable_reward.py"
DATASET="${PROJECT_ROOT}/data/金融领域中文数据集-选择题-all-processed-dedup-cleared-shuffled-with_qwen35_9B-selected.jsonl"

# ==========================================================
# System
# ==========================================================

SYSTEM_PROMPT="""你是 Cornucopia 聚宝盆大模型，是擅长推理思考的金融领域专家。"""

# ==========================================================
# Model
# ==========================================================

MODEL=Qwen/Qwen3.5-2B

# ==========================================================
# Reward
# ==========================================================

REWARD_FUNCS=verifiable_reward

# ==========================================================
# Thinking
# ==========================================================

ENABLE_THINKING=true

# ==========================================================
# vLLM
# ==========================================================

USE_VLLM=true
VLLM_MODE=colocate
VLLM_GPU_MEMORY_UTILIZATION=0.5
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_MAX_MODEL_LEN=32768
SLEEP_LEVEL=1

# ==========================================================
# LoRA
# ==========================================================

TUNER_TYPE=lora
LORA_RANK=8
LORA_ALPHA=32
TARGET_MODULES=all-linear

# ==========================================================
# Train
# ==========================================================

TORCH_DTYPE=bfloat16
LOAD_FROM_CACHE_FILE=true
MAX_LENGTH=4096
MAX_COMPLETION_LENGTH=16384
NUM_TRAIN_EPOCHS=2
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=5e-5
LR_SCHEDULER_TYPE=cosine
SAVE_STEPS=10
SAVE_TOTAL_LIMIT=10
LOGGING_STEPS=1
WARMUP_RATIO=0.0
DATALOADER_NUM_WORKERS=4
DEEPSPEED=zero2

# ==========================================================
# GRPO
# ==========================================================

RLHF_TYPE=grpo
NUM_GENERATIONS=8
TEMPERATURE=0.8
LOG_COMPLETIONS=true
REPORT_TO="tensorboard swanlab"
MAX_GRAD_NORM=1.0
EPSILON=0.2
EPSILON_HIGH=0.28
SCALE_REWARDS=none

# ==========================================================
# Launch
# ==========================================================

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
NPROC_PER_NODE=${NPROC_PER_NODE} \
swift rlhf \
--rlhf_type ${RLHF_TYPE} \
--model ${MODEL} \
--external_plugins ${REWARD_FILE} \
--reward_funcs ${REWARD_FUNCS} \
--enable_thinking ${ENABLE_THINKING} \
--use_vllm ${USE_VLLM} \
--vllm_mode ${VLLM_MODE} \
--vllm_gpu_memory_utilization ${VLLM_GPU_MEMORY_UTILIZATION} \
--vllm_tensor_parallel_size ${VLLM_TENSOR_PARALLEL_SIZE} \
--vllm_max_model_len ${VLLM_MAX_MODEL_LEN} \
--sleep_level ${SLEEP_LEVEL} \
--tuner_type ${TUNER_TYPE} \
--lora_rank ${LORA_RANK} \
--lora_alpha ${LORA_ALPHA} \
--target_modules ${TARGET_MODULES} \
--torch_dtype ${TORCH_DTYPE} \
--dataset ${DATASET} \
--load_from_cache_file ${LOAD_FROM_CACHE_FILE} \
--max_length ${MAX_LENGTH} \
--max_completion_length ${MAX_COMPLETION_LENGTH} \
--num_train_epochs ${NUM_TRAIN_EPOCHS} \
--per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
--gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
--learning_rate ${LEARNING_RATE} \
--lr_scheduler_type ${LR_SCHEDULER_TYPE} \
--save_steps ${SAVE_STEPS} \
--save_total_limit ${SAVE_TOTAL_LIMIT} \
--logging_steps ${LOGGING_STEPS} \
--warmup_ratio ${WARMUP_RATIO} \
--dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
--num_generations ${NUM_GENERATIONS} \
--temperature ${TEMPERATURE} \
--system "${SYSTEM_PROMPT}" \
--deepspeed ${DEEPSPEED} \
--log_completions ${LOG_COMPLETIONS} \
--max_grad_norm ${MAX_GRAD_NORM} \
--epsilon ${EPSILON} \
--epsilon_high ${EPSILON_HIGH} \
--scale_rewards ${SCALE_REWARDS} \
--report_to ${REPORT_TO}
