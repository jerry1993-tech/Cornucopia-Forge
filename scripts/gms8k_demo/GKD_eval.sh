#!/bin/bash

# GPU: RTX4090/24G
# eval GKD model on GSM8K dataset
CUDA_VISIBLE_DEVICES=1,2 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift eval \
    --model "/data/xyj/models/Qwen3.5-2B" \
    --adapters /data/xyj/output/gkd_gsm8k_model/v10-20260711-133659/checkpoint-200 \
    --merge_lora true \
    --enable_thinking false \
    --eval_dataset gsm8k \
    --eval_backend Native --infer_backend vllm \
    --eval_generation_config '{"max_tokens":8192,"temperature":0.0,"do_sample":false}'


# merge lora adapters into the base model
swift export \
    --model "/data/xyj/models/Qwen3.5-2B" \
    --adapters "/data/xyj/output/gkd_gsm8k_model/v10-20260711-133659/checkpoint-200" \
    --merge_lora true \
    --device_map auto


2026-07-11 19:18:43 - evalscope - INFO: *** Report table ***
┌────────────┬───────────┬──────────┬──────────┬───────┬─────────┬─────────┐
│ Model      │ Dataset   │ Metric   │ Subset   │   Num │   Score │ Steps   │
├────────────┼───────────┼──────────┼──────────┼───────┼─────────┼─────────┤
│ Qwen3.5-2B │ gsm8k     │ mean_acc │ main     │  1319 │  0.8074 │ GKD 200 steps │
└────────────┴───────────┴──────────┴──────────┴───────┴─────────┴─────────┘
