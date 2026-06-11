datasets=(coco aokvqa gqa)
types=(random popular adversarial)

declare -A image_subdirs=(
    [coco]=val2014
    [aokvqa]=val2014
    [gqa]=images
)

## baseline
for dataset in ${datasets[@]}; do 
    for type in ${types[@]}; do
        python ./inference/pope_infer_base.py \
            --model-path ./llava-v1.5-7b \
            --question-file ./data/pope/${dataset}/${dataset}_pope_${type}.json \
            --image-folder ./data/pope/${dataset}/images/${image_subdirs[$dataset]} \
            --answers-file ./outputs/pope/baseline/llava-7b-${dataset}-${type}-greedy.jsonl \
            --temperature 0 \
            --conv-mode vicuna_v1
    done
done

for dataset in ${datasets[@]}; do 
    for type in ${types[@]}; do
        python ./inference/pope_infer_base.py \
            --model-path ./llava-v1.5-7b \
            --question-file ./data/pope/${dataset}/${dataset}_pope_${type}.json \
            --image-folder ./data/pope/${dataset}/images/${image_subdirs[$dataset]} \
            --answers-file ./outputs/pope/baseline/llava-7b-${dataset}-${type}-sample.jsonl \
            --temperature 1 \
            --conv-mode vicuna_v1
    done
done