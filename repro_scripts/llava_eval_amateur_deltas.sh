#!/bin/bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
cd "${REPO_ROOT}"

# Define your parameters
methods=(vcd icd)
datasets=(coco gqa aokvqa)
splits=(random popular adversarial)

# Create the CSV file and write the header row with Dataset first, then Split
output_file="${OUT_ROOT}/llava_amateur_deltas_results.csv"
echo "Dataset,Split,Method,MeanDelta,Accuracy,YesRatio" > "$output_file"

echo "Aggregating amateur deltas and writing to $output_file..."

# Loop through datasets, then splits, then methods
for dataset in "${datasets[@]}"; do
    for split in "${splits[@]}"; do
        for method in "${methods[@]}"; do

            # Define path -- matches inf_amateur_logits.sh's answers-file convention
            res_file="${REPO_ROOT}/llava_amateur_logits/${dataset}/${method}/${dataset}-${split}-amateur-deltas.jsonl"
            ref_file="${DATA_DIR}/${dataset}/${dataset}_pope_${split}.json"

            # Check if the file actually exists before running
            if [ ! -f "$res_file" ]; then
                echo "$dataset,${split^},$method,Missing File,-,-" >> "$output_file"
                continue
            fi

            # Run the backend python script
            output=$(python ./eval/llava_eval_amateur_deltas.py \
                --res-file "$res_file" \
                --ref-file "$ref_file" \
                2>/dev/null)

            # Same kind of forgiving key:value parser as pope_eval_base.py's wrapper
            metrics=$(echo "$output" | python3 -c "
import sys, re
res = {}
for line in sys.stdin:
    match = re.search(r'^([^:]+):\s*([0-9.eE+\-]+)', line.strip())
    if match:
        k = match.group(1).lower().strip()
        v = float(match.group(2))
        if 'mean_delta' in k:
            res['meandelta'] = f'{v:.4f}'
        elif 'accuracy' in k:
            res['accuracy'] = f'{v*100:.2f}%'
        elif 'yes_ratio' in k:
            res['yesratio'] = f'{v*100:.2f}%'

print(f\"{res.get('meandelta', '-')},{res.get('accuracy', '-')},{res.get('yesratio', '-')}\")
")

            # Read the comma-separated values into bash variables
            IFS=',' read -r mean_delta accuracy yes_ratio <<< "$metrics"

            # Append the row directly to the CSV file
            echo "$dataset,${split^},$method,$mean_delta,$accuracy,$yes_ratio" >> "$output_file"

        done
    done
done

echo "Done! Amateur deltas aggregated and saved to $output_file"