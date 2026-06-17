datasets=(gqa)
types=(random popular adversarial)
methods=(baseline vcd icd sid pba olm)

for dataset in ${datasets[@]}; do
    for type in ${types[@]}; do
        for method in ${methods[@]}; do

            echo "=== method: ${method}, dataset: ${dataset}, type: ${type} ==="
            python ./eval/pope_eval_base.py \
                --ref-files ./data/${dataset}/${dataset}_pope_${type}.json \
                --res-files ./outputs/pope/${method}/Qwen2.5-7b-${dataset}-${type}-greedy.jsonl
        done
    done
done
