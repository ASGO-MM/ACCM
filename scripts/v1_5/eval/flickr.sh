#!/bin/bash

CKPT_NAME=llava-v1.5-7b_short-caption-clipcap_72-tokens+selector_4400
MODEL_PATH=/home/fmy/data/llava-v1.5-7b

export CUDA_VISIBLE_DEVICES=0
python -m llava.eval.model_vqa_caption \
    --model-path $MODEL_PATH \
    --question-file /home/fmy/data/LLaVA-main/playground/data/eval/Flickr/questions-3k.jsonl \
    --image-folder  /home/fmy/data/LLaVA-main/playground/data/eval/Flickr/flickr30k-images \
    --answers-file /home/fmy/data/LLaVA-main/playground/data/eval/Flickr/answers/$CKPT_NAME.jsonl \
    --temperature 0 \
    --add_proto false \
    --proto_num 0 \
    --conv-mode vicuna_v1


python scripts/eval_flickr.py \
    --answer_file /home/fmy/data/LLaVA-main/playground/data/eval/Flickr/answers/$CKPT_NAME.jsonl \
    --anno_file /home/fmy/data/LLaVA-main/playground/data/eval/Flickr/annotations-3k.jsonl
