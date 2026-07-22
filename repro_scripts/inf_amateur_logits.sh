declare -A image_folders=(
    [coco]="/teamspace/lightning_storage/datasets/coco/val2014"
    [gqa]="/teamspace/lightning_storage/datasets/gqa/images"
    [aokvqa]="/teamspace/lightning_storage/datasets/coco/val2014"
)

datasets=(coco gqa aokvqa)
types=(random popular adversarial)

## vcd amateur audit
for dataset in ${datasets[@]}; do
    for type in ${types[@]}; do
        python ./inference/llava_amateur_logits.py \
            --model-path /teamspace/lightning_storage/model/llava-v1.5-7b \
            --question-file ./data/${dataset}/${dataset}_pope_${type}.json \
            --image-folder ${image_folders[${dataset}]} \
            --answers-file ./llava_amateur_logits/${dataset}/vcd/${dataset}-${type}-amateur-deltas.jsonl \
            --conv-mode vicuna_v1 \
            --use-vcd
    done
done

## icd amateur audit
for dataset in ${datasets[@]}; do
    for type in ${types[@]}; do
        python ./inference/llava_amateur_logits.py \
            --model-path /teamspace/lightning_storage/model/llava-v1.5-7b \
            --question-file ./data/${dataset}/${dataset}_pope_${type}.json \
            --image-folder ${image_folders[${dataset}]} \
            --answers-file ./llava_amateur_logits/${dataset}/icd/${dataset}-${type}-amateur-deltas.jsonl \
            --conv-mode vicuna_v1 \
            --use-icd
    done
done