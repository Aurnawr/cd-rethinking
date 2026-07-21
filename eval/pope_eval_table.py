import os
import argparse
from typing import Dict, List, Tuple

# Reuse the exact metric logic already used for the reproduction tables.
from pope_eval_base import load_json_lines, evaluate_binary, compute_metrics

# Display name -> output-file model prefix (as used in repro_outputs/*/<prefix>-...).
MODELS: List[Tuple[str, str]] = [
    ("LLaVA-v1.5-7B", "llava-7b"),
    ("Qwen2.5-7B", "Qwen2.5_7b"),
]

# Method display label -> repro_outputs subdirectory. Order matches the paper tables.
METHODS: List[Tuple[str, str]] = [
    ("Greedy (Baseline)", "baseline"),
    ("VCD", "vcd"),
    ("SID", "sid"),
    ("PBA", "pba"),
    ("OLM", "olm"),
    ("ICD", "icd"),
]

CATEGORIES: List[str] = ["random", "popular", "adversarial"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate POPE metrics (Accuracy, Yes%, Precision, F1) across methods."
    )
    parser.add_argument(
        "--ref-dir",
        type=str,
        default="./data/pope/aokvqa",
        help="Directory holding aokvqa_pope_<category>.json ground-truth files.",
    )
    parser.add_argument(
        "--res-root",
        type=str,
        default="./repro_outputs",
        help="Root directory with one subfolder per method (baseline, vcd, ...).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="aokvqa",
        help="Dataset name used in the reference/result filenames.",
    )
    parser.add_argument(
        "--decoding",
        type=str,
        default="greedy",
        help="Decoding tag in the result filename, e.g. 'greedy' or 'sample'.",
    )
    return parser.parse_args()


def metrics_for(ref_path: str, res_path: str) -> Dict[str, float]:
    """Compute the full metric set for one (reference, result) pair."""
    ref_data = load_json_lines(ref_path)
    res_data = load_json_lines(res_path)
    if len(ref_data) != len(res_data):
        raise ValueError(
            f"Length mismatch: {ref_path} ({len(ref_data)}) vs {res_path} ({len(res_data)})"
        )
    tp, tn, fp, fn, unknown = evaluate_binary(ref_data, res_data)
    return compute_metrics(tp, tn, fp, fn, len(ref_data), unknown)


def signed(value: float) -> str:
    """Format a percentage-point delta with an explicit up/down arrow."""
    arrow = "^" if value >= 0 else "v"
    return f"{arrow}{abs(value):.1f}"


def print_model_table(
    model_name: str,
    model_prefix: str,
    ref_dir: str,
    res_root: str,
    dataset: str,
    decoding: str,
) -> None:
    header = f"{'Category':<12} {'Method':<18} {'Acc':>6} {'dAcc':>7} {'Yes%':>6} {'dYes':>7} {'Prec':>6} {'dPrec':>7} {'F1':>6} {'dF1':>7}"
    print("=" * len(header))
    print(model_name)
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for category in CATEGORIES:
        ref_path = os.path.join(ref_dir, f"{dataset}_pope_{category}.json")
        baseline: Dict[str, float] = {}

        for method_label, method_dir in METHODS:
            res_path = os.path.join(
                res_root,
                method_dir,
                f"{model_prefix}-{dataset}-{category}-{decoding}.jsonl",
            )
            if not os.path.exists(os.path.expanduser(res_path)):
                print(f"{category:<12} {method_label:<18} {'-- missing: ' + res_path}")
                continue

            m = metrics_for(ref_path, res_path)
            # Round to the displayed precision first, so deltas-vs-baseline match
            # what the reader sees in the table (the paper's convention).
            acc = round(m["Accuracy"] * 100, 1)
            yes = round(m["Yes_Proportion"] * 100, 1)
            prec = round(m["Precision"] * 100, 1)
            f1 = round(m["F1"] * 100, 1)

            if method_dir == "baseline":
                baseline = {"acc": acc, "yes": yes, "prec": prec, "f1": f1}

            print(
                f"{category:<12} {method_label:<18} "
                f"{acc:6.1f} {signed(acc - baseline['acc']):>7} "
                f"{yes:6.1f} {signed(yes - baseline['yes']):>7} "
                f"{prec:6.1f} {signed(prec - baseline['prec']):>7} "
                f"{f1:6.1f} {signed(f1 - baseline['f1']):>7}"
            )
        print("-" * len(header))
    print()


def main() -> None:
    args = parse_args()
    for model_name, model_prefix in MODELS:
        print_model_table(
            model_name=model_name,
            model_prefix=model_prefix,
            ref_dir=args.ref_dir,
            res_root=args.res_root,
            dataset=args.dataset,
            decoding=args.decoding,
        )


if __name__ == "__main__":
    main()
