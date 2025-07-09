#!/bin/bash

CKPT_NAME=llava-v1.5-7b_short-caption-clipcap_question_72-tokens_wo-normalize+selector_4400
MODEL_PATH=/home/fmy/data/llava-v1.5-7b

export CUDA_VISIBLE_DEVICES=0
python -W ignore llava/eval/mmvp/mmvp_eval.py --model_path $MODEL_PATH \
   --answers_file llava/eval/mmvp/answers/$CKPT_NAME.jsonl \
   --add_proto false \
   --proto_num 0

python llava/eval/mmvp/mmvp_test.py --answers_file llava/eval/mmvp/answers/${CKPT_NAME}_0.jsonl \
   --output_file llava/eval/mmvp/answers/${CKPT_NAME}_incorrect.jsonl \
   --csv_file llava/eval/mmvp/answers/${CKPT_NAME}_experiments.csv