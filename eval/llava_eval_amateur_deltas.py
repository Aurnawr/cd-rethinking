import argparse
import json
import statistics
import sys


def main(args):
    # Load ground truth labels from the question file, keyed by question_id
    ground_truth = {}
    with open(args.ref_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ground_truth[row["question_id"]] = row["label"].strip().lower()

    deltas = []
    correct = 0
    yes_count = 0
    total = 0

    try:
        with open(args.res_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                qid = row["question_id"]
                delta = row["delta_amateur"]  # logit_yes - logit_no
                deltas.append(delta)

                # Greedy prediction: yes if delta > 0, no otherwise
                prediction = "yes" if delta > 0 else "no"
                yes_count += 1 if prediction == "yes" else 0

                if qid in ground_truth:
                    if prediction == ground_truth[qid]:
                        correct += 1

                total += 1

    except FileNotFoundError:
        print(f"error: file not found: {args.res_file}", file=sys.stderr)
        sys.exit(1)

    if not deltas:
        print(f"error: file present but empty: {args.res_file}", file=sys.stderr)
        sys.exit(1)

    mean_delta = statistics.mean(deltas)
    accuracy  = correct / total
    yes_ratio = yes_count / total

    print(f"mean_delta: {mean_delta}")
    print(f"accuracy: {accuracy}")
    print(f"yes_ratio: {yes_ratio}")
    print(f"n: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--res-file", type=str, required=True,
                        help="Path to a single *-amateur-deltas.jsonl file")
    parser.add_argument("--ref-file", type=str, required=True,
                        help="Path to the POPE question jsonl file containing ground truth labels")
    args = parser.parse_args()
    main(args)