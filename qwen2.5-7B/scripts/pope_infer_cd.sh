datasets=(gqa)
types=(random popular adversarial)

## vcd
for dataset in ${datasets[@]}; do 
    for type in ${types[@]}; do

        python ./inference/pope_infer_cd.py \
            --model-path /teamspace/lightning_storage/model/Qwen2.5_7b \
            --question-file ./data/${dataset}/${dataset}_pope_${type}.json \
            --image-folder /teamspace/lightning_storage/datasets/gqa/images \
            --answers-file ./outputs/pope/vcd/Qwen2.5-7b-${dataset}-${type}-greedy.jsonl \
            --temperature 0 \
            --conv-mode qwen_chat \
            --use-vcd
    done
done



## icd

for dataset in ${datasets[@]}; do 
    for type in ${types[@]}; do

        python ./inference/pope_infer_cd.py \
            --model-path /teamspace/lightning_storage/model/Qwen2.5_7b \
            --question-file ./data/${dataset}/${dataset}_pope_${type}.json \
            --image-folder /teamspace/lightning_storage/datasets/gqa/images \
            --answers-file ./outputs/pope/icd/Qwen2.5-7b-${dataset}-${type}-greedy.jsonl \
            --temperature 0 \
            --conv-mode qwen_chat \
            --use-icd
    done
done




## sid

for dataset in ${datasets[@]}; do 
    for type in ${types[@]}; do

        python ./inference/pope_infer_cd.py \
            --model-path /teamspace/lightning_storage/model/Qwen2.5_7b \
            --question-file ./data/${dataset}/${dataset}_pope_${type}.json \
            --image-folder /teamspace/lightning_storage/datasets/gqa/images \
            --answers-file ./outputs/pope/sid/Qwen2.5-7b-${dataset}-${type}-greedy.jsonl \
            --temperature 0 \
            --conv-mode qwen_chat \
            --use-sid
    done
done

