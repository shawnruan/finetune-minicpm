@echo off
setlocal

REM 添加环境变量禁用 libuv
set USE_LIBUV=0

REM 设置环境变量
set GPUS_PER_NODE=1
set NNODES=1
set NODE_RANK=0
set MASTER_ADDR=localhost
set MASTER_PORT=6001

REM 设置模型和数据路径
set MODEL=E:\recongize_helmet\models\MiniCPM-V-2_6
set DATA=E:\recongize_helmet\LLaMA-Factory\data\mllm_demo.json
set EVAL_DATA=E:\recongize_helmet\LLaMA-Factory\data\mllm_demo.json
set LLM_TYPE=qwen
set MODEL_MAX_Length=2048

REM 直接使用 python 运行
python finetune.py ^
    --model_name_or_path %MODEL% ^
    --llm_type %LLM_TYPE% ^
    --data_path %DATA% ^
    --eval_data_path %EVAL_DATA% ^
    --remove_unused_columns false ^
    --label_names "labels" ^
    --prediction_loss_only false ^
    --bf16 true ^
    --bf16_full_eval true ^
    --fp16 false ^
    --fp16_full_eval false ^
    --do_train ^
    --do_eval ^
    --tune_vision true ^
    --tune_llm false ^
    --model_max_length %MODEL_MAX_Length% ^
    --max_slice_nums 9 ^
    --max_steps 10000 ^
    --eval_steps 1000 ^
    --output_dir output/output_minicpmv26 ^
    --logging_dir output/output_minicpmv26 ^
    --logging_strategy "steps" ^
    --per_device_train_batch_size 1 ^
    --per_device_eval_batch_size 1 ^
    --gradient_accumulation_steps 1 ^
    --evaluation_strategy "steps" ^
    --save_strategy "steps" ^
    --save_steps 1000 ^
    --save_total_limit 10 ^
    --learning_rate 1e-6 ^
    --weight_decay 0.1 ^
    --adam_beta2 0.95 ^
    --warmup_ratio 0.01 ^
    --lr_scheduler_type "cosine" ^
    --logging_steps 1 ^
    --gradient_checkpointing true ^
    --deepspeed ds_config_zero3.json ^
    --report_to "tensorboard"

endlocal