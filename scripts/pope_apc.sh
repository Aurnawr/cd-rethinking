datasets=(coco aokvqa gqa)
types=(random popular adversarial)

declare -A image_subdirs=(
    [coco]=val2014
    [aokvqa]=val2014
    [gqa]=images
)

for dataset in ${datasets[@]}; do 
    for type in ${types[@]}; do

        python ./inference/pope_infer_apc.py \
            --model-path ./llava-v1.5-7b \
            --question-file ./data/pope/${dataset}/${dataset}_pope_${type}.json \
            --image-folder ./data/pope/${dataset}/images/${image_subdirs[$dataset]} \
            --answers-file ./outputs/pope/apc/llava-7b-${dataset}-${type}-sample.jsonl \
            --temperature 1 \
            --conv-mode vicuna_v1 \
            --use-apc
    done
done