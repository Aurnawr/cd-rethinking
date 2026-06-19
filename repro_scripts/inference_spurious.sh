datasets=(coco)
types=(random popular adversarial)

## pba
for dataset in ${datasets[@]}; do 
    for type in ${types[@]}; do
        python ./qwen-v2.5-vl-7b_inference/infer_pba.py \
            --model-path /teamspace/lightning_storage/model/Qwen2.5_7b \
            --question-file ./data/${dataset}/${dataset}_pope_${type}.json \
            --image-folder /teamspace/lightning_storage/datasets/coco/val2014 \
            --answers-file ./repro_outputs/coco/qwen2.5-vl-7b/pba/qwen-7b-${dataset}-${type}-greedy.jsonl \
            --temperature 0 \
            --conv-mode qwen_chat 
    done
done

## olm
for dataset in ${datasets[@]}; do 
    for type in ${types[@]}; do
        python ./qwen-v2.5-vl-7b_inference/infer_olm.py \
            --model-path /teamspace/lightning_storage/model/Qwen2.5_7b \
            --question-file ./data/${dataset}/${dataset}_pope_${type}.json \
            --image-folder /teamspace/lightning_storage/datasets/coco/val2014 \
            --answers-file ./repro_outputs/coco/qwen2.5-vl-7b/olm/qwen-7b-${dataset}-${type}-greedy.jsonl \
            --temperature 0 \
            --conv-mode qwen_chat \
            --use-olm
    done
done