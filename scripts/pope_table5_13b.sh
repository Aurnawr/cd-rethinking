#!/bin/bash
# Reproduce Table 5 (sample strategy) for LLaVA-1.6-vicuna-13B using the AnyRes
# inference scripts (pope_infer_*16.py). Each (method, dataset, type) split is
# sharded across GPUS and merged into outputs/pope_13b/<method>/.
#
# NOTE: a single 13B AnyRes process needs ~31 GB of GPU memory, so only GPUs
# with >=31 GB free should be listed in GPUS.
set -u

types=(random popular adversarial)

# GPUs to shard each split across (must each have >=31GB free for 13B AnyRes).
GPUS=(2 4 5)
N=${#GPUS[@]}

# Number of questions expected in a completed split (used for resume/skip).
EXPECTED_LINES=3000

MODEL=./pretrained_models/llava-v1.6-vicuna-13b
TAG=llava-13b
ROOT=./outputs/pope_13b

# shard_split <infer_script> <out_subdir> <extra_flags> <dataset> <type>
shard_split() {
    local script="$1" outsub="$2" extra="$3" dataset="$4" type="$5"
    local out_dir="${ROOT}/${outsub}"
    local log_dir="${out_dir}/logs"
    local tmp_dir="${out_dir}/chunks"
    mkdir -p "$log_dir" "$tmp_dir"
    local final="${out_dir}/${TAG}-${dataset}-${type}-sample.jsonl"

    # Resume: skip splits that already completed.
    if [ -f "${final}" ] && [ "$(wc -l < "${final}")" -eq "${EXPECTED_LINES}" ]; then
        echo "  -> ${outsub}: ${dataset}/${type}  (skip: already complete, ${EXPECTED_LINES} lines)"
        return
    fi

    echo "  -> ${outsub}: ${dataset}/${type}  (shard x${N} on GPUs ${GPUS[*]})"
    local pids=() k gpu chunk_out log_file
    for k in $(seq 0 $((N - 1))); do
        gpu=${GPUS[$k]}
        chunk_out="${tmp_dir}/${TAG}-${dataset}-${type}-chunk${k}.jsonl"
        log_file="${log_dir}/${TAG}-${dataset}-${type}-gpu${gpu}.log"
        CUDA_VISIBLE_DEVICES=${gpu} stdbuf -oL python "${script}" \
            --model-path "${MODEL}" \
            --question-file ./data/pope/${dataset}/${dataset}_pope_${type}.json \
            --image-folder ./data/pope/${dataset}/images \
            --answers-file "${chunk_out}" \
            --num-chunks ${N} \
            --chunk-idx ${k} \
            --temperature 1 \
            --conv-mode vicuna_v1 \
            ${extra} > "${log_file}" 2>&1 &
        pids+=($!)
    done
    local fail=0 pid
    for pid in "${pids[@]}"; do
        wait "${pid}" || fail=1
    done
    if [ "${fail}" -ne 0 ]; then
        echo "ERROR: a chunk failed for ${outsub} ${dataset}/${type}; see ${log_dir}" >&2
        exit 1
    fi
    : > "${final}"
    for k in $(seq 0 $((N - 1))); do
        cat "${tmp_dir}/${TAG}-${dataset}-${type}-chunk${k}.jsonl" >> "${final}"
    done
    rm -f "${tmp_dir}/${TAG}-${dataset}-${type}-chunk"*.jsonl
    echo "     merged -> ${final} ($(wc -l < "${final}") lines)"
}

echo "### baseline sample ###"
for dataset in coco aokvqa gqa; do
    for type in "${types[@]}"; do
        shard_split ./inference/pope_infer_base16.py baseline "" "${dataset}" "${type}"
    done
done

echo "### VCD sample ###"
for dataset in coco aokvqa gqa; do
    for type in "${types[@]}"; do
        shard_split ./inference/pope_infer_cd16.py vcd "--use-vcd" "${dataset}" "${type}"
    done
done

echo "### ICD sample ###"
for dataset in coco aokvqa gqa; do
    for type in "${types[@]}"; do
        shard_split ./inference/pope_infer_cd16.py icd "--use-icd" "${dataset}" "${type}"
    done
done

echo "### SID sample ###"
for dataset in coco aokvqa gqa; do
    for type in "${types[@]}"; do
        shard_split ./inference/pope_infer_cd16.py sid "--use-sid" "${dataset}" "${type}"
    done
done

echo "### APC sample ###"
for dataset in coco aokvqa gqa; do
    for type in "${types[@]}"; do
        shard_split ./inference/pope_infer_apc16.py apc "--use-apc" "${dataset}" "${type}"
    done
done

echo "All 13B Table 5 runs finished. Results in ${ROOT}"
