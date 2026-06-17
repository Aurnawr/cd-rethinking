datasets=(gqa)
types=(random popular adversarial)
cd_methods=(vcd icd sid pba olm)

for dataset in ${datasets[@]}; do
    for type in ${types[@]}; do
        for method in ${cd_methods[@]}; do

            echo "=== baseline vs ${method}, dataset: ${dataset}, type: ${type} ==="
            python ./eval/pope_eval_transfer.py \
                --ref-files ./data/${dataset}/${dataset}_pope_${type}.json \
                --res-rg-files ./outputs/pope/baseline/Qwen2.5-7b-${dataset}-${type}-greedy.jsonl \
                --res-cd-files ./outputs/pope/${method}/Qwen2.5-7b-${dataset}-${type}-greedy.jsonl
        done
    done
done
