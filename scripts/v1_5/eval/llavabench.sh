#!/bin/bash

CKPT_NAME=llava-v1.5-7b-Qformer-11-lora-finetune_from_vicuna-proto-mlp-2
MODEL_PATH=/public/home/renwu04/fmy/LLaVA-PruMerge-text/checkpoints/llava-v1.5-7b-Qformer-11-lora-finetune_from_vicuna-proto-mlp-2

export CUDA_VISIBLE_DEVICES=1
python -m llava.eval.model_vqa \
    --model-base /public/home/renwu04/fmy/data/vicuna-7b-v1.5_2 \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/$CKPT_NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

# mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

# python llava/eval/eval_gpt_review_bench.py \
#     --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
#     --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
#     --rule llava/eval/table/rule.json \
#     --answer-list \
#         playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
#         playground/data/eval/llava-bench-in-the-wild/answers/llava-v1.5-13b.jsonl \
#     --output \
#         playground/data/eval/llava-bench-in-the-wild/reviews/llava-v1.5-13b.jsonl

# python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/llava-v1.5-13b.jsonl
