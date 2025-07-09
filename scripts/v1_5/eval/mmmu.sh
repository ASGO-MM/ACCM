#!/bin/bash

CKPT_NAME=llava-v1.5-7b-Qformer-11-lora-finetune_from_vicuna-proto-mlp-2_text=30_test
MODEL_PATH=/public/home/renwu04/fmy/LLaVA-PruMerge-text/checkpoints/llava-v1.5-7b-Qformer-11-lora-finetune_from_vicuna-proto-mlp-2

export CUDA_VISIBLE_DEVICES=0
python /public/home/renwu04/fmy/LLaVA-PruMerge-text/llava/eval/mmmu/run_llava.py \
    --model_base /public/home/renwu04/fmy/data/vicuna-7b-v1.5_2 \
    --model_path $MODEL_PATH \
    --data_path /public/home/renwu04/fmy/LLaVA-PruMerge-text/playground/data/eval/MMMU \
    --output_path /public/home/renwu04/fmy/LLaVA-PruMerge-text/llava/eval/mmmu/example_outputs/$CKPT_NAME.json \
    --config_path /public/home/renwu04/fmy/LLaVA-PruMerge-text/llava/eval/mmmu/configs/llava1.5.yaml \
    --split "test" \
# "validation"


python /public/home/renwu04/fmy/LLaVA-PruMerge-text/llava/eval/mmmu/main_eval_only.py \
    --output_path /public/home/renwu04/fmy/LLaVA-PruMerge-text/llava/eval/mmmu/example_outputs/$CKPT_NAME.json \
    --answer_path /public/home/renwu04/fmy/LLaVA-PruMerge-text/llava/eval/mmmu/answer_dict_val.json
