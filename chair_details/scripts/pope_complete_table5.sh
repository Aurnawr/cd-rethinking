#!/bin/bash
# Generate the remaining sample-strategy (temperature 1) results needed to
# complete Table 5 for LLaVA-v1.5-7B:
#   - baseline `sample` for aokvqa + gqa   (coco baseline already exists)
#   - VCD / ICD / SID `sample` for coco + aokvqa + gqa
# Each split is sharded across GPUS and merged (in chunk order) into the single
# answers file the eval/table scripts expect.
set -u

types=(random popular adversarial)
GPUS=(0 1 2 4)
N=${#GPUS[@]}

MODEL=./pretrained_models/llava-v1.5-7b
TAG=llava-7b

# shard_split <infer_script> <out_subdir> <extra_flags> <dataset> <type>
shard_split() {
    local script="$1" outsub="$2" extra="$3" dataset="$4" type="$5"
    local out_dir="./outputs/pope/${outsub}"
    local log_dir="${out_dir}/logs"
    local tmp_dir="${out_dir}/chunks"
    mkdir -p "$log_dir" "$tmp_dir"
    local final="${out_dir}/${TAG}-${dataset}-${type}-sample.jsonl"

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

echo "### baseline sample (aokvqa, gqa) ###"
for dataset in aokvqa gqa; do
    for type in "${types[@]}"; do
        shard_split ./inference/pope_infer_base.py baseline "" "${dataset}" "${type}"
    done
done

echo "### VCD sample (coco, aokvqa, gqa) ###"
for dataset in coco aokvqa gqa; do
    for type in "${types[@]}"; do
        shard_split ./inference/pope_infer_cd.py vcd "--use-vcd" "${dataset}" "${type}"
    done
done

echo "### ICD sample (coco, aokvqa, gqa) ###"
for dataset in coco aokvqa gqa; do
    for type in "${types[@]}"; do
        shard_split ./inference/pope_infer_cd.py icd "--use-icd" "${dataset}" "${type}"
    done
done

echo "### SID sample (coco, aokvqa, gqa) ###"
for dataset in coco aokvqa gqa; do
    for type in "${types[@]}"; do
        shard_split ./inference/pope_infer_cd.py sid "--use-sid" "${dataset}" "${type}"
    done
done

echo "All remaining Table 5 runs finished."
