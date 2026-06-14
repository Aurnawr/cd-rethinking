#!/bin/bash
# POPE Table 5 (sample strategy) for Qwen2.5-VL-7B-Instruct.
# 5 methods x 3 datasets x 3 types, sharded across GPUS, resumable.
# Output layout matches the eval/table tooling: outputs/pope_qwen/<subdir>/qwen-7b-<ds>-<type>-sample.jsonl
set -u

types=(random popular adversarial)
GPUS=(2 3 4 5)
N=${#GPUS[@]}
EXPECTED_LINES=3000

MODEL=./pretrained_models/Qwen2.5-VL-7B-Instruct
TAG=qwen-7b
ROOT=./outputs/pope_qwen
SCRIPT=./inference_qwen/qwen_pope_infer.py

# method subdir -> --method flag value
declare -A METHOD_FLAG=( [baseline]=sample [vcd]=vcd [icd]=icd [sid]=sid [apc]=apc )
METHODS=(baseline vcd icd sid apc)

# shard_split <subdir> <dataset> <type>
shard_split() {
    local subdir="$1" dataset="$2" type="$3"
    local method="${METHOD_FLAG[$subdir]}"
    local out_dir="${ROOT}/${subdir}"
    local log_dir="${out_dir}/logs"
    local tmp_dir="${out_dir}/chunks"
    mkdir -p "$log_dir" "$tmp_dir"
    local final="${out_dir}/${TAG}-${dataset}-${type}-sample.jsonl"

    if [ -f "${final}" ] && [ "$(wc -l < "${final}")" -eq "${EXPECTED_LINES}" ]; then
        echo "  -> ${subdir}: ${dataset}/${type}  (skip: complete)"
        return
    fi
    echo "  -> ${subdir}: ${dataset}/${type}  (shard x${N} on GPUs ${GPUS[*]})"
    local pids=() k gpu chunk_out log_file
    for k in $(seq 0 $((N - 1))); do
        gpu=${GPUS[$k]}
        chunk_out="${tmp_dir}/${TAG}-${dataset}-${type}-chunk${k}.jsonl"
        log_file="${log_dir}/${TAG}-${dataset}-${type}-gpu${gpu}.log"
        CUDA_VISIBLE_DEVICES=${gpu} stdbuf -oL python "${SCRIPT}" \
            --model-path "${MODEL}" \
            --question-file ./data/pope/${dataset}/${dataset}_pope_${type}.json \
            --image-folder ./data/pope/${dataset}/images \
            --answers-file "${chunk_out}" \
            --method "${method}" \
            --num-chunks ${N} \
            --chunk-idx ${k} \
            --temperature 1 > "${log_file}" 2>&1 &
        pids+=($!)
    done
    local fail=0 pid
    for pid in "${pids[@]}"; do
        wait "${pid}" || fail=1
    done
    if [ "${fail}" -ne 0 ]; then
        echo "ERROR: a chunk failed for ${subdir} ${dataset}/${type}; see ${log_dir}" >&2
        exit 1
    fi
    : > "${final}"
    for k in $(seq 0 $((N - 1))); do
        cat "${tmp_dir}/${TAG}-${dataset}-${type}-chunk${k}.jsonl" >> "${final}"
    done
    rm -f "${tmp_dir}/${TAG}-${dataset}-${type}-chunk"*.jsonl
    echo "     merged -> ${final} ($(wc -l < "${final}") lines)"
}

for subdir in "${METHODS[@]}"; do
    echo "### ${subdir} ###"
    for dataset in coco aokvqa gqa; do
        for type in "${types[@]}"; do
            shard_split "${subdir}" "${dataset}" "${type}"
        done
    done
done

echo "All Qwen2.5-VL Table 5 runs finished. Results in ${ROOT}"
