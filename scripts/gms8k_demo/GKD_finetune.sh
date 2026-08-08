#!/bin/bash

# ==========================================================
# GPU: RTX4090/24G
# ==========================================================

export CUDA_VISIBLE_DEVICES=1,2
export NPROC_PER_NODE=2

# ==========================================================
# Path
# ==========================================================

MODEL_PATH="/data/xyj/models/Qwen3.5-2B"
DATASET_PATH='modelscope/gsm8k'
OUTPUT_DIR="./output/gkd_gsm8k_model"

# ==========================================================
# Teacher Server
# ==========================================================

TEACHER_MODEL_SERVER="http://localhost:8000"

# ==========================================================
# LoRA
# ==========================================================

TUNER_TYPE=lora
LORA_RANK=8
LORA_ALPHA=32
TARGET_MODULES=all-linear

# ==========================================================
# GKD
# ==========================================================

GKD_LOGITS_TOPK=64
ENABLE_THINKING=false
LAMBDA=1
BETA=0.5

# ==========================================================
# vLLM
# ==========================================================

USE_VLLM=true
VLLM_MODE=colocate
VLLM_GPU_MEMORY_UTILIZATION=0.5
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_MAX_MODEL_LEN=8192
SLEEP_LEVEL=0

# ==========================================================
# Train
# ==========================================================

TORCH_DTYPE=bfloat16

PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=16

LEARNING_RATE=5e-5
WARMUP_RATIO=0.1

MAX_LENGTH=2048
MAX_COMPLETION_LENGTH=6144

LOGGING_STEPS=1
SAVE_STEPS=50
SAVE_TOTAL_LIMIT=10

SAVE_ONLY_MODEL=true

DATALOADER_NUM_WORKERS=4
DATASET_NUM_PROC=4

TRUNCATION_STRATEGY='delete'

DEEPSPEED=zero2
SEQUENCE_PARALLEL_SIZE=1

# ==========================================================
# Launch
# ==========================================================

NPROC_PER_NODE=${NPROC_PER_NODE} \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments\:True' \
swift rlhf \
--rlhf_type gkd \
--model ${MODEL_PATH} \
--dataset ${DATASET_PATH} \
--output_dir ${OUTPUT_DIR} \
--teacher_model_server ${TEACHER_MODEL_SERVER} \
--gkd_logits_topk ${GKD_LOGITS_TOPK} \
--enable_thinking ${ENABLE_THINKING} \
--tuner_type ${TUNER_TYPE} \
--lora_rank ${LORA_RANK} \
--lora_alpha ${LORA_ALPHA} \
--target_modules ${TARGET_MODULES} \
--use_vllm ${USE_VLLM} \
--vllm_mode ${VLLM_MODE} \
--vllm_gpu_memory_utilization ${VLLM_GPU_MEMORY_UTILIZATION} \
--vllm_tensor_parallel_size ${VLLM_TENSOR_PARALLEL_SIZE} \
--vllm_max_model_len ${VLLM_MAX_MODEL_LEN} \
--sleep_level ${SLEEP_LEVEL} \
--lmbda ${LAMBDA} \
--beta ${BETA} \
--torch_dtype ${TORCH_DTYPE} \
--per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
--gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
--learning_rate ${LEARNING_RATE} \
--logging_steps ${LOGGING_STEPS} \
--save_steps ${SAVE_STEPS} \
--save_total_limit ${SAVE_TOTAL_LIMIT} \
--max_length ${MAX_LENGTH} \
--max_completion_length ${MAX_COMPLETION_LENGTH} \
--truncation_strategy ${TRUNCATION_STRATEGY} \
--deepspeed ${DEEPSPEED} \
--sequence_parallel_size ${SEQUENCE_PARALLEL_SIZE} \
--warmup_ratio ${WARMUP_RATIO} \
--save_only_model ${SAVE_ONLY_MODEL} \
--dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
--dataset_num_proc ${DATASET_NUM_PROC} \
--report_to tensorboard swanlab
