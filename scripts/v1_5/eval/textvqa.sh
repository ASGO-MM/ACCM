#!/bin/bash    

CKPT_NAME=llava-v1.5-7b-rule-direct-test_36tokens
MODEL_PATH=/datanew/fmy/llava-v1.5-7b

export CUDA_VISIBLE_DEVICES=6
python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file /datanew/fmy/LLaVA-main/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /datanew/fmy/LLaVA-main/playground/data/eval/textvqa/train_images \
    --answers-file /datanew/fmy/LLaVA-main/playground/data/eval/textvqa/answers/$CKPT_NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file /datanew/fmy/LLaVA-main/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file /datanew/fmy/LLaVA-main/playground/data/eval/textvqa/answers/$CKPT_NAME.jsonl

# --model-base /public/home/renwu04/fmy/data/vicuna-7b-v1.5 \