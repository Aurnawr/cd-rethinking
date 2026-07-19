#!/bin/bash

# Define your parameters
methods=(vcd)
datasets=(coco gqa aokvqa)
splits=(popular)

# Create the CSV file and write the header row with Dataset first, then Split
output_file="./repro_outputs/amateur_deltas_results_t999.csv"
echo "Dataset,Split,Method,MeanDelta,N" > "$output_file"

echo "Aggregating amateur deltas and writing to $output_file..."

# Loop through datasets, then splits, then methods
for dataset in "${datasets[@]}"; do
    for split in "${splits[@]}"; do
        for method in "${methods[@]}"; do

            # Define path -- matches inference_cd.sh's answers-file convention
            res_file="./t_999/${dataset}/${method}/${dataset}-${split}-amateur-deltas.jsonl"

            # Check if the file actually exists before running
            if [ ! -f "$res_file" ]; then
                echo "$dataset,${split^},$method,Missing File,-" >> "$output_file"
                continue
            fi

            # Run the backend python script
            output=$("${PY_BIN}" ./eval/eval_amateur_deltas.py --res-file "$res_file" 2>/dev/null)

            # Same kind of forgiving key:value parser as pope_eval_base.py's wrapper
            metrics=$(echo "$output" | python3 -c "
import sys, re
res = {}
for line in sys.stdin:
    match = re.search(r'^([^:]+):\s*([0-9.eE+-]+)', line.strip())
    if match:
        k = match.group(1).lower()
        if 'mean_delta' in k:
            res['meandelta'] = f'{float(match.group(2)):.4f}'
        elif k == 'n':
            res['n'] = match.group(2)

print(f\"{res.get('meandelta', '-')},{res.get('n', '-')}\")
")

            # Read the comma-separated values into bash variables
            IFS=',' read -r mean_delta n <<< "$metrics"

            # Append the row directly to the CSV file
            echo "$dataset,${split^},$method,$mean_delta,$n" >> "$output_file"

        done
    done
done

echo "Done! Amateur deltas aggregated and saved to $output_file"