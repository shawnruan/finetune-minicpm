#!/bin/bash

MODEL="/Users/ruanxiaoyang/Desktop/repo/finetune-minicpm/models/MiniCPM-V-2_6"
DATA="/Users/ruanxiaoyang/Desktop/repo/finetune-minicpm/helmet_detection_data.json"
EVAL_DATA="/Users/ruanxiaoyang/Desktop/repo/finetune-minicpm/helmet_detection_eval_data.json"
LLM_TYPE="qwen"
MODEL_MAX_Length=800 # Further reduced for minimal memory usage

export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

python finetune.py \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 false \
    --bf16_full_eval false \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --do_eval \
    --tune_vision false \
    --tune_llm false \
    --use_lora true \
    --lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)" \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 1 \
    --max_steps 5 \
    --eval_steps 2 \
    --output_dir output/output_lora_minimal \
    --logging_dir output/output_lora_minimal \
    --logging_strategy "steps" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 5 \
    --learning_rate 1e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --gradient_checkpointing true \
    --dataloader_num_workers 0 