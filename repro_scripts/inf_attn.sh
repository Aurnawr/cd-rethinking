declare -A image_folders=(
    [coco]="/teamspace/lightning_storage/dataset/val2014"
    [gqa]="/teamspace/lightning_storage/dataset/gqa"
    [aokvqa]="/teamspace/lightning_storage/dataset/val2014"
)

datasets=(coco gqa aokvqa)
types=(random popular adversarial)

for dataset in ${datasets[@]}; do
    for type in ${types[@]}; do
        python ./inference/attention_exp_inf.py \
            --model-path /teamspace/lightning_storage/data/models/llava-v1.5-7b \
            --question-file ./data/${dataset}/${dataset}_pope_${type}.json \
            --image-folder ${image_folders[${dataset}]} \
            --answers-file ./attn/ans/${dataset}/vcd/${dataset}-${type}.jsonl \
            --attn-file ./attn/eval/${dataset}/vcd/${dataset}-${type}.jsonl \
            --warnings-file ./attn/warnings/${dataset}/vcd/${dataset}-${type}.jsonl \
            --conv-mode vicuna_v1 \
            --use-vcd
    done
done

for dataset in ${datasets[@]}; do
    for type in ${types[@]}; do
        python ./inference/attention_exp_inf.py \
            --model-path /teamspace/lightning_storage/data/models/llava-v1.5-7b \
            --question-file ./data/${dataset}/${dataset}_pope_${type}.json \
            --image-folder ${image_folders[${dataset}]} \
            --answers-file ./attn/ans/${dataset}/icd/${dataset}-${type}.jsonl \
            --attn-file ./attn/eval/${dataset}/icd/${dataset}-${type}.jsonl \
            --warnings-file ./attn/warnings/${dataset}/icd/${dataset}-${type}.jsonl \
            --conv-mode vicuna_v1 \
            --use-icd
    done
done

for dataset in ${datasets[@]}; do
    for type in ${types[@]}; do
        python ./inference/attention_exp_inf.py \
            --model-path /teamspace/lightning_storage/data/models/llava-v1.5-7b \
            --question-file ./data/${dataset}/${dataset}_pope_${type}.json \
            --image-folder ${image_folders[${dataset}]} \
            --answers-file ./attn/ans/${dataset}/sid/${dataset}-${type}.jsonl \
            --attn-file ./attn/eval/${dataset}/sid/${dataset}-${type}.jsonl \
            --warnings-file ./attn/warnings/${dataset}/sid/${dataset}-${type}.jsonl \
            --conv-mode vicuna_v1 \
            --use-sid
    done
done