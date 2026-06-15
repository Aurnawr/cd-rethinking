datasets=(coco)
types=(random popular adversarial)

## baseline
for dataset in ${datasets[@]}; do 
    for type in ${types[@]}; do
        python ./inference/pope_infer_base.py \
            --model-path /teamspace/lightning_storage/model/llava-v1.5-7b \
            --question-file ./data/${dataset}/${dataset}_pope_${type}.json \
            --image-folder /teamspace/lightning_storage/datasets/coco/val2014 \
            --answers-file ./repro_outputs/coco/llava-v1.5-7b/baseline/llava-7b-${dataset}-${type}-greedy.jsonl \
            --temperature 0 \
            --conv-mode vicuna_v1
    done
done