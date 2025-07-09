#!/bin/bash

CKPT_NAME=llava-v1.5-7b-direct-test_rule_36tokens
MODEL_PATH=/data16tb/fmy/llava-v1.5-7b

export CUDA_VISIBLE_DEVICES=5
python -m llava.eval.model_vqa \
    --model-path $MODEL_PATH \
    --question-file /datanew/fmy/LLaVA-main/playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /datanew/fmy/LLaVA-main/playground/data/eval/mm-vet/mm-vet/images \
    --answers-file /datanew/fmy/LLaVA-main/playground/data/eval/mm-vet/answers/$CKPT_NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /datanew/fmy/LLaVA-main/playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src /datanew/fmy/LLaVA-main/playground/data/eval/mm-vet/answers/$CKPT_NAME.jsonl \
    --dst /datanew/fmy/LLaVA-main/playground/data/eval/mm-vet/results/$CKPT_NAME.jsonl


#     --model-base /public/home/renwu04/fmy/data/vicuna-7b-v1.5 \