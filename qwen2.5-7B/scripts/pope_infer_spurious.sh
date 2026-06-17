datasets=(gqa)
types=(random popular adversarial)

## pba

for dataset in ${datasets[@]}; do 
    for type in ${types[@]}; do

        python ./inference/pope_infer_pba.py \
            --model-path /teamspace/lightning_storage/model/Qwen2.5_7b \
            --question-file ./data/${dataset}/${dataset}_pope_${type}.json \
            --image-folder /teamspace/lightning_storage/datasets/gqa/images \
            --answers-file ./outputs/pope/pba/Qwen2.5-7b-${dataset}-${type}-greedy.jsonl \
            --temperature 0 \
            --conv-mode qwen_chat
    done
done


## olm

for dataset in ${datasets[@]}; do 
    for type in ${types[@]}; do

        python ./inference/pope_infer_olm.py \
            --model-path /teamspace/lightning_storage/model/Qwen2.5_7b \
            --question-file ./data/${dataset}/${dataset}_pope_${type}.json \
            --image-folder /teamspace/lightning_storage/datasets/gqa/images \
            --answers-file ./outputs/pope/olm/Qwen2.5-7b-${dataset}-${type}-greedy.jsonl \
            --temperature 0 \
            --conv-mode qwen_chat \
            --use-olm
    done
done

