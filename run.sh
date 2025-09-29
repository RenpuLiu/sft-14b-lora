#!/usr/bin/env bash
set -e

# Example: Qwen2.5-14B (change to any 14B you have access to)
MODEL="Qwen/Qwen2.5-14B-Instruct"
OUTDIR="out-qwen14b-lora"
DATASET="HuggingFaceH4/ultrachat_200k"   # demo; replace with your own

# If you have a local jsonl with {"text": ...}, use --data_path /path/to.jsonl
# and --data_field text below.

CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_name $MODEL \
  --output_dir $OUTDIR \
  --data_path $DATASET \
  --data_split train \
  --data_field text \
  --max_seq_len 2048 \
  --packing true \
  --lora_target attention_only \
  --lora_target_ratio 0.01 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --warmup_ratio 0.03 \
  --save_steps 1000 \
  --log_steps 20 \
  --bf16 true \
  --flash_attn true \
  --gradient_checkpointing true \
  --optimizer paged_adamw_32bit \
  --lr_scheduler cosine \
  --seed 42
