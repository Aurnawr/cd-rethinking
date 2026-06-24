import argparse
import json
import statistics
import sys


def main(args):
    deltas = []
    try:
        with open(args.res_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                deltas.append(row["delta_amateur"])
    except FileNotFoundError:
        print(f"error: file not found: {args.res_file}", file=sys.stderr)
        sys.exit(1)

    if not deltas:
        print(f"error: file present but empty: {args.res_file}", file=sys.stderr)
        sys.exit(1)

    mean_delta = statistics.mean(deltas)
    n = len(deltas)

    # Plain "key: value" lines, same convention as pope_eval_base.py's output,
    # so the same kind of regex parser in bash can pull these out.
    print(f"mean_delta: {mean_delta}")
    print(f"n: {n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--res-file", type=str, required=True,
                         help="Path to a single *-amateur-deltas.jsonl file")
    args = parser.parse_args()
    main(args)