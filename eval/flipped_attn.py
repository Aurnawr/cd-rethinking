import json
import csv
from pathlib import Path

# Set the base directory based on your folder structure
base_dir = Path("/teamspace/studios/this_studio/cd-rethinking/attn/eval")

# The datasets we want to iterate through
datasets = ["coco", "gqa", "aokvqa"]

# Name of the output file
output_file = "/teamspace/studios/this_studio/cd-rethinking/attn/visual_grounding_flipped_metrics.csv"

# Initialize a list to hold the rows for our CSV
csv_data = []

for dataset in datasets:
    # Path to the vcd folder within the dataset
    dataset_path = base_dir / dataset / "vcd"
    
    # Check if the directory exists to avoid errors
    if not dataset_path.exists():
        print(f"Directory not found: {dataset_path}")
        continue

    # Iterate through all .jsonl files in the current dataset's vcd folder
    for file_path in dataset_path.glob("*.jsonl"):
        split_name = file_path.stem 
        
        fraction_expert_vals = []
        fraction_cd_vals = []
        
        # Read the JSONL file
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                data = json.loads(line)
                
                # Check condition: only process if cd_flipped_yes_to_no is True
                if data.get("cd_flipped_yes_to_no") is True:
                    fraction_expert_vals.append(data.get("fraction_expert", 0.0))
                    fraction_cd_vals.append(data.get("fraction_cd", 0.0))
        
        num_flipped = len(fraction_expert_vals)
        
        # Calculate metrics if there are valid samples
        if num_flipped > 0:
            mean_expert = sum(fraction_expert_vals) / num_flipped
            mean_cd = sum(fraction_cd_vals) / num_flipped
            
            if mean_expert != 0:
                percentage_increase = ((mean_cd - mean_expert) / mean_expert) * 100
            else:
                percentage_increase = 0.0
                
            # Append the calculated data as a row for the CSV
            csv_data.append([
                dataset.upper(), 
                split_name, 
                num_flipped, 
                round(mean_expert, 6), 
                round(mean_cd, 6), 
                round(percentage_increase, 2)
            ])
            print(f"Processed {split_name} - Found {num_flipped} flipped samples.")
        else:
            # Handle cases where a file exists but has no flipped samples
            csv_data.append([
                dataset.upper(), 
                split_name, 
                0, 
                "N/A", 
                "N/A", 
                "N/A"
            ])
            print(f"Processed {split_name} - No flipped samples found.")

# Write all collected data to the CSV file
with open(output_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    # Write the header row
    writer.writerow(["Dataset", "Split", "Flipped Samples (N)", "Mean Fraction Expert", "Mean Fraction CD", "Percentage Increase (%)"])
    
    # Write the data rows
    writer.writerows(csv_data)

print(f"\nSuccess! All metrics have been saved to '{output_file}'")