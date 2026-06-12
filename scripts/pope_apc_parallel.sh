#!/bin/bash
# Data-parallel version of pope_apc.sh.
# Each (dataset, type) split is sharded across multiple GPUs using the
# inference script's built-in --num-chunks / --chunk-idx options, then the
# per-chunk outputs are concatenated (in chunk order, preserving the original
# question order) into the single answers file the eval scripts expect.
set -u

datasets=(coco aokvqa gqa)
types=(random popular adversarial)

# GPUs to shard each split across (one python process per GPU).
GPUS=(0 1 2 4)
N=${#GPUS[@]}

out_dir=./outputs/pope/apc
log_dir=${out_dir}/logs
tmp_dir=${out_dir}/chunks
mkdir -p "$log_dir" "$tmp_dir"

total=$(( ${#datasets[@]} * ${#types[@]} ))
current=0

for dataset in "${datasets[@]}"; do
    for type in "${types[@]}"; do
        current=$(( current + 1 ))
        final=${out_dir}/llava-7b-${dataset}-${type}-sample.jsonl

        echo "============================================================"
        echo "[${current}/${total}] dataset=${dataset} type=${type}"
        echo "sharding across ${N} GPUs: ${GPUS[*]}"
        echo "============================================================"

        pids=()
        for k in $(seq 0 $((N - 1))); do
            gpu=${GPUS[$k]}
            chunk_out=${tmp_dir}/llava-7b-${dataset}-${type}-chunk${k}.jsonl
            log_file=${log_dir}/llava-7b-${dataset}-${type}-gpu${gpu}.log

            CUDA_VISIBLE_DEVICES=${gpu} stdbuf -oL python ./inference/pope_infer_apc.py \
                --model-path ./pretrained_models/llava-v1.5-7b \
                --question-file ./data/pope/${dataset}/${dataset}_pope_${type}.json \
                --image-folder ./data/pope/${dataset}/images \
                --answers-file "${chunk_out}" \
                --num-chunks ${N} \
                --chunk-idx ${k} \
                --temperature 1 \
                --conv-mode vicuna_v1 \
                --use-apc > "${log_file}" 2>&1 &
            pids+=($!)
        done

        # Wait for all chunk processes of this split; fail fast if any errors.
        fail=0
        for pid in "${pids[@]}"; do
            wait "${pid}" || fail=1
        done
        if [ "${fail}" -ne 0 ]; then
            echo "ERROR: a chunk failed for ${dataset} ${type}; see logs in ${log_dir}" >&2
            exit 1
        fi

        # Merge chunks in order to reconstruct the original question order.
        : > "${final}"
        for k in $(seq 0 $((N - 1))); do
            cat "${tmp_dir}/llava-7b-${dataset}-${type}-chunk${k}.jsonl" >> "${final}"
        done
        rm -f "${tmp_dir}"/llava-7b-${dataset}-${type}-chunk*.jsonl
        echo "merged -> ${final} ($(wc -l < "${final}") lines)"
    done
done

echo "All ${total} splits finished. Results in ${out_dir}"
