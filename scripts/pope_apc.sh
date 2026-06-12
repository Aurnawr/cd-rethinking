#!/bin/sh

datasets=(coco aokvqa gqa)
types=(random popular adversarial)

# directory to keep continuous logs of each run
log_dir=./outputs/pope/apc/logs
mkdir -p "$log_dir"

# total number of runs for the overall progress counter
total=$(( ${#datasets[@]} * ${#types[@]} ))
current=0

for dataset in ${datasets[@]}; do
    for type in ${types[@]}; do

        current=$(( current + 1 ))
        log_file="${log_dir}/llava-7b-${dataset}-${type}.log"

        echo "============================================================"
        echo "[${current}/${total}] dataset=${dataset} type=${type}"
        echo "logging to ${log_file}"
        echo "============================================================"

        # stdbuf -oL keeps tqdm/print output line-buffered so it shows live,
        # tee writes it to the screen AND to the log file continuously.
        stdbuf -oL python ./inference/pope_infer_apc.py \
            --model-path ./pretrained_models/llava-v1.5-7b \
            --question-file ./data/pope/${dataset}/${dataset}_pope_${type}.json \
            --image-folder ./data/pope/${dataset}/images \
            --answers-file ./outputs/pope/apc/llava-7b-${dataset}-${type}-sample.jsonl \
            --temperature 1 \
            --conv-mode vicuna_v1 \
            --use-apc 2>&1 | tee "$log_file"
    done
done

echo "All ${total} runs finished. Logs are in ${log_dir}"
