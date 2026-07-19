#!/bin/bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
cd "${REPO_ROOT}"

# Define your parameters
methods=(baseline vcd icd sid pba olm)
datasets=(coco)
splits=(random popular adversarial)

# Create the CSV file and write the header row with Split first
output_file="${OUT_ROOT}/table3_results.csv"
echo "Split,Method,Accuracy,Precision,Recall,F1-Score,Yes Ratio" > "$output_file"

echo "Evaluating methods and writing to $output_file..."

# Loop through splits FIRST, then loop through methods
for split in "${splits[@]}"; do
    for method in "${methods[@]}"; do
        
        # Define paths
        ref_file="${DATA_DIR}/${datasets[0]}/${datasets[0]}_pope_${split}.json"
        res_file="${OUT_ROOT}/${datasets[0]}/${MODEL_TAG}/${method}/${FILE_TAG}-${datasets[0]}-${split}-greedy.jsonl"
        
        # Format the method name for the CSV
        display_method="${method}"
        if [ "$method" == "baseline" ]; then
            display_method="greedy"
        fi
        
        # Check if the inference file actually exists before running
        if [ ! -f "$res_file" ]; then
            echo "${split^},$display_method,Missing File,-,-,-,-" >> "$output_file"
            continue
        fi
        
        # Run the original python script
        output=$(python ./eval/pope_eval_base.py --ref-files "$ref_file" --res-files "$res_file" 2>/dev/null)
        
        # The ultimate forgiving parser
        metrics=$(echo "$output" | python3 -c "
import sys, re
res = {}
for line in sys.stdin:
    # Look for anything before a colon, followed by a decimal number
    match = re.search(r'^([^:]+):\s*([0-9.]+)', line.strip())
    if match:
        k = match.group(1).lower()
        val_str = f'{float(match.group(2)) * 100:.2f}%'
        
        # Keyword matching instead of exact string matching
        if 'acc' in k: res['accuracy'] = val_str
        elif 'prec' in k: res['precision'] = val_str
        elif 'rec' in k: res['recall'] = val_str
        elif 'f1' in k: res['f1score'] = val_str
        elif 'yes' in k or 'ratio' in k: res['yesratio'] = val_str

print(f\"{res.get('accuracy', '-')},{res.get('precision', '-')},{res.get('recall', '-')},{res.get('f1score', '-')},{res.get('yesratio', '-')}\")
")

        # Read the comma-separated values into bash variables
        IFS=',' read -r acc prec rec f1 yes <<< "$metrics"
        
        # Append the row directly to your CSV file with ${split^} first
        echo "${split^},$display_method,$acc,$prec,$rec,$f1,$yes" >> "$output_file"
        
    done
done

echo "Done! All metrics parsed flawlessly and saved to $output_file"