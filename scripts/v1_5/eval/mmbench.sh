#!/bin/bash

CKPT_NAME=llava-v1.5-7b_short-caption-clipcap_72-tokens+selector_4400
MODEL_PATH=/home/fmy/data/llava-v1.5-7b

SPLIT="mmbench_dev_20230712"

export CUDA_VISIBLE_DEVICES=0
python -m llava.eval.model_vqa_mmbench \
    --model-path $MODEL_PATH \
    --question-file /home/fmy/data/LLaVA-main/playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file /home/fmy/data/LLaVA-main/playground/data/eval/mmbench/answers/$SPLIT/$CKPT_NAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --add_proto false \
    --proto_num 0

mkdir -p /home/fmy/data/LLaVA-main/playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file /home/fmy/data/LLaVA-main/playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir /home/fmy/data/LLaVA-main/playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir /home/fmy/data/LLaVA-main/playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $CKPT_NAME


#--model-base /public/home/renwu04/fmy/data/vicuna-7b-v1.5 \