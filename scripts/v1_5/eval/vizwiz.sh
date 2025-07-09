#!/bin/bash

CKPT_NAME=llava-v1.5-7b-direct-test_36tokens-text-20
MODEL_PATH=/datanew/fmy/llava-v1.5-7b

export CUDA_VISIBLE_DEVICES=0
python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file /datanew/fmy/LLaVA-main/playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder /datanew/fmy/LLaVA-main/playground/data/eval/vizwiz/test \
    --answers-file /datanew/fmy/LLaVA-main/playground/data/eval/vizwiz/answers/$CKPT_NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file /datanew/fmy/LLaVA-main/playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file /datanew/fmy/LLaVA-main/playground/data/eval/vizwiz/answers/$CKPT_NAME.jsonl \
    --result-upload-file /datanew/fmy/LLaVA-main/playground/data/eval/vizwiz/answers_upload/$CKPT_NAME.json

# --model-base /public/home/renwu04/fmy/data/vicuna-7b-v1.5 \