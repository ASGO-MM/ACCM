#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-1}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_PATH=/datanew/fmy/llava-v1.5-7b
CKPT="llava-v1.5-7b-short-cap_greedy_36tokens"

SPLIT="llava_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file /datanew/fmy/LLaVA-main/playground/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder /datanew/fmy/LLaVA-main/playground/data/eval/vqav2/test2015 \
        --answers-file /datanew/fmy/LLaVA-main/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=/datanew/fmy/LLaVA-main/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /datanew/fmy/LLaVA-main/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT --dir /datanew/fmy/LLaVA-main/playground/data/eval/vqav2

# --model-base /public/home/renwu04/fmy/data/vicuna-7b-v1.5_2 \