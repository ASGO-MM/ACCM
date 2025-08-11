#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=llava-v1.5-7b_short-caption-clipcap_72-tokens
MODEL_PATH=/home/fmy/data/llava-v1.5-7b

SPLIT="llava_gqa_testdev_balanced"
GQADIR="/home/fmy/data/LLaVA-main/playground/data/eval/gqa/data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file /home/fmy/data/LLaVA-main/playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder /home/fmy/data/LLaVA-main/playground/data/eval/gqa/data/images \
        --answers-file /home/fmy/data/LLaVA-main/playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=/home/fmy/data/LLaVA-main/playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /home/fmy/data/LLaVA-main/playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced

# --model-base /public/home/renwu04/fmy/data/vicuna-7b-v1.5 \
