#!/bin/bash

CKPT_NAME=llava-v1.5-7b_36-tokens
MODEL_PATH=/home/fmy/data/llava-v1.5-7b

export CUDA_VISIBLE_DEVICES=1
python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file /home/fmy/data/LLaVA-main/playground/data/eval/MME/llava_mme.jsonl \
    --image-folder /home/fmy/data/LLaVA-main/playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file /home/fmy/data/LLaVA-main/playground/data/eval/MME/answers/$CKPT_NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 


cd /home/fmy/data/LLaVA-main/playground/data/eval/MME

python convert_answer_to_mme.py --experiment $CKPT_NAME

cd eval_tool

python calculation.py --results_dir answers/$CKPT_NAME


## --model-base /public/home/renwu04/fmy/data/vicuna-7b-v1.5 \
