#!/bin/bash

# GPU: RTX4090/24G
# teacher server
export MODEL_PATH=/data/xyj/models/Qwen3.5-4B
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift deploy \
    --model $MODEL_PATH \
    --served-model-name Qwen3.5-4B \
    --infer_backend vllm \
    --host 0.0.0.0 \
    --port 8000 \
    --max_logprobs 64 \
    --vllm_max_model_len 8192 \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 1
