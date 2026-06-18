import pandas as pd
import json
import os

def load_jsonl_answers(filepath):
    """
    Reads a JSONL file and extracts the answers.
    Assumes standard LLaVA/MME output format with 'question_id' and 'text'.
    Returns a dictionary mapping question_id to the cleaned text answer.
    """
    answers = {}
    if not os.path.exists(filepath):
        print(f"Warning: File not found {filepath}")
        return answers
        
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Standard keys are usually 'question_id' and 'text'
            q_id = data.get('question_id')
            
            # Clean the answer: lower case, strip whitespace and basic punctuation
            ans_text = data.get('text', '').strip().lower().replace('.', '')
            
            # We are only interested if the model said 'yes' or 'no'
            # Sometimes models output "Yes, there is..." so we take the first word
            first_word = ans_text.split(' ')[0] if ans_text else ""
            
            answers[q_id] = first_word
            
    return answers

def main():
    # 1. Load the evaluation results
    results_df = pd.read_csv('/teamspace/studios/this_studio/cd_rethink/outputs/mme/mme_eval_results.csv')
    
    # 2. Separate the baseline from the CD methods
    baseline_df = results_df[results_df['method'] == 'baseline (greedy)'].copy()
    cd_df = results_df[results_df['method'] != 'baseline (greedy)'].copy()
    
    # Create a mapping of subtask -> baseline answer file
    baseline_files = dict(zip(baseline_df['subtask'], baseline_df['answer_file']))
    
    # 3. List to store our final processed rows
    final_data = []
    
    # 4. Iterate over each CD method run
    for _, row in cd_df.iterrows():
        subtask = row['subtask']
        method = row['method']
        cd_file = row['answer_file']
        yes_pct = row['yes_pct']
        accuracy = row['accuracy']
        
        baseline_file = baseline_files.get(subtask)
        
        if not baseline_file:
            print(f"No baseline found for subtask: {subtask}")
            continue
            
        # Load the answers from the JSONL files
        baseline_answers = load_jsonl_answers(baseline_file)
        cd_answers = load_jsonl_answers(cd_file)
        
        # Initialize transition counters
        yes_to_no = 0
        no_to_yes = 0
        
        # Compare question by question
        for q_id, base_ans in baseline_answers.items():
            if q_id in cd_answers:
                cd_ans = cd_answers[q_id]
                
                # Check transitions based on raw outputs
                if base_ans == 'yes' and cd_ans == 'no':
                    yes_to_no += 1
                elif base_ans == 'no' and cd_ans == 'yes':
                    no_to_yes += 1
                    
        # Append the results
        final_data.append({
            'subtask': subtask,
            'method': method,
            'yes_to_no': yes_to_no,
            'no_to_yes': no_to_yes,
            'yes_pct': yes_pct,
            'accuracy': accuracy
        })
        
    # 5. Create final DataFrame and save it
    final_df = pd.DataFrame(final_data)
    
    # Sort by subtask then method for clean grouping
    final_df = final_df.sort_values(by=['subtask', 'method']).reset_index(drop=True)
    
    # Save to a new CSV file
    output_filename = 'mme_transitions_summary.csv'
    final_df.to_csv(output_filename, index=False)
    
    print("\nProcessing complete! Here is a preview of the table:")
    print(final_df.head(10))
    print(f"\nFull table saved to '{output_filename}'.")

if __name__ == "__main__":
    main()