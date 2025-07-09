#!/bin/bash

CKPT_NAME=llava-v1.5-7b_36-tokens_scene-graph
MODEL_PATH=/home/fmy/data/llava-v1.5-7b

export CUDA_VISIBLE_DEVICES=1
python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file /home/fmy/data/LLaVA-main/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /home/fmy/data/LLaVA-main/playground/data/eval/pope/val2014 \
    --answers-file /home/fmy/data/LLaVA-main/playground/data/eval/pope/answers/$CKPT_NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --add_proto false \
    --proto_num 0

python llava/eval/eval_pope.py \
    --annotation-dir /home/fmy/data/LLaVA-main/playground/data/eval/pope/coco \
    --question-file /home/fmy/data/LLaVA-main/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file /home/fmy/data/LLaVA-main/playground/data/eval/pope/answers/$CKPT_NAME.jsonl

# --model-base /public/home/renwu04/fmy/data/vicuna-7b-v1.5 \