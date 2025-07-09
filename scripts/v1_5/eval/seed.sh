#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=llava-v1.5-7b_short-caption-clipcap_question_72-tokens_wo-normalize+selector_4400
MODEL_PATH=/home/fmy/data/llava-v1.5-7b


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file /home/fmy/data/LLaVA-main/playground/data/eval/seed_bench/llava-seed-bench.jsonl \
        --image-folder /home/fmy/data/LLaVA-main/playground/data/eval/seed_bench \
        --answers-file /home/fmy/data/LLaVA-main/playground/data/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --add_proto false \
        --proto_num 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=/home/fmy/data/LLaVA-main/playground/data/eval/seed_bench/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /home/fmy/data/LLaVA-main/playground/data/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python scripts/convert_seed_for_submission.py \
    --annotation-file /home/fmy/data/LLaVA-main/playground/data/eval/seed_bench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file /home/fmy/data/LLaVA-main/playground/data/eval/seed_bench/answers_upload/$CKPT.jsonl

# --model-base /public/home/renwu04/fmy/data/vicuna-7b-v1.5 \