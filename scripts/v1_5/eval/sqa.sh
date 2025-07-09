#!/bin/bash

CKPT_NAME=llava-v1.5-7b-Qformer-11-lora-finetune_from_vicuna-proto-mlp-2-104_tokens_text=20
MODEL_PATH=/public/home/renwu04/fmy/LLaVA-PruMerge-text/checkpoints/llava-v1.5-7b-Qformer-11-lora-finetune_from_vicuna-proto-mlp-2-104_tokens

export CUDA_VISIBLE_DEVICES=1
python -m llava.eval.model_vqa_science \
    --model-base /public/home/renwu04/fmy/data/vicuna-7b-v1.5 \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/$CKPT_NAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$CKPT_NAME.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${CKPT_NAME}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${CKPT_NAME}_result.json
