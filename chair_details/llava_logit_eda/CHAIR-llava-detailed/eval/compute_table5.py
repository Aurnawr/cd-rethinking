#!/usr/bin/env python3
"""
Reproduce Table 5 (Independent Application of the Adaptive Plausibility
Constraint) for a single model, e.g. LLaVA-v1.5-7B.

For each method it reads the per-(dataset, type) answer files, computes
Accuracy and Yes-proportion using the same yes/no matching rule as
eval/pope_eval_base.py, then averages across datasets for each POPE type
(Random / Popular / Adversarial). Deltas are reported relative to the
`sample` baseline.

Only methods whose answer files exist are shown, so it can be run before all
inference is finished.
"""
import os
import json
import argparse

DATASETS = ["coco", "aokvqa", "gqa"]
TYPES = ["random", "popular", "adversarial"]

# Display label -> output sub-directory under --results-dir.
# `sample` is the baseline (used for the deltas).
METHODS = [
    ("sample", "baseline"),
    ("VCD", "vcd"),
    ("ICD", "icd"),
    ("SID", "sid"),
    ("sample\u2020", "apc"),  # sample-dagger = APC
]


def load_jsonl(path):
    rows = []
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metrics_for_file(ref_path, res_path):
    """Return (accuracy_pct, yes_pct) for one answer file, or None if missing."""
    if not (os.path.isfile(ref_path) and os.path.isfile(res_path)):
        return None
    ref = load_jsonl(ref_path)
    res = load_jsonl(res_path)
    if len(ref) != len(res):
        raise ValueError(f"length mismatch: {ref_path} ({len(ref)}) vs {res_path} ({len(res)})")

    tp = tn = fp = fn = 0
    for r, p in zip(ref, res):
        if r["question_id"] != p["question_id"]:
            raise ValueError(f"id mismatch in {res_path}: {r['question_id']} != {p['question_id']}")
        gt = r["label"].strip().lower()
        pred = p["text"].strip().lower()
        if gt == "yes":
            tp += 1 if "yes" in pred else 0
            fn += 0 if "yes" in pred else 1
        elif gt == "no":
            tn += 1 if "no" in pred else 0
            fp += 0 if "no" in pred else 1
    total = len(ref)
    acc = 100.0 * (tp + tn) / total
    yes = 100.0 * (tp + fp) / total
    return acc, yes


def method_type_avg(results_dir, data_dir, subdir, model_tag, ptype):
    """Average Accuracy and Yes% across datasets for one method+type.
    Returns (acc, yes, n_datasets) or None if no files present."""
    accs, yeses = [], []
    for ds in DATASETS:
        ref = os.path.join(data_dir, ds, f"{ds}_pope_{ptype}.json")
        res = os.path.join(results_dir, subdir, f"{model_tag}-{ds}-{ptype}-sample.jsonl")
        m = metrics_for_file(ref, res)
        if m is not None:
            accs.append(m[0])
            yeses.append(m[1])
    if not accs:
        return None
    return sum(accs) / len(accs), sum(yeses) / len(yeses), len(accs)


def fmt_delta(val, base):
    if base is None:
        return ""
    d = val - base
    arrow = "\u2191" if d >= 0 else "\u2193"
    return f"{arrow}{abs(d):.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="./outputs/pope",
                    help="dir containing method subdirs (baseline, vcd, icd, sid, apc)")
    ap.add_argument("--data-dir", default="./data/pope",
                    help="dir containing <dataset>/<dataset>_pope_<type>.json")
    ap.add_argument("--model-tag", default="llava-7b",
                    help="answer-file prefix, e.g. llava-7b")
    args = ap.parse_args()

    # Pre-compute baseline (`sample`) per type for deltas.
    baseline = {}
    for ptype in TYPES:
        r = method_type_avg(args.results_dir, args.data_dir, "baseline", args.model_tag, ptype)
        baseline[ptype] = r  # (acc, yes, n) or None

    col = f"{args.model_tag} (sample strategy)"
    print(f"\nTable 5 — {col}\n")
    header = f"{'Category':<12} {'Method':<8} {'Accuracy':>16} {'Yes (%)':>16}   datasets"
    print(header)
    print("-" * len(header))

    for ptype in TYPES:
        cat = ptype.capitalize()
        base = baseline[ptype]
        base_acc = base[0] if base else None
        base_yes = base[1] if base else None
        printed_any = False
        for label, subdir in METHODS:
            r = method_type_avg(args.results_dir, args.data_dir, subdir, args.model_tag, ptype)
            if r is None:
                continue
            acc, yes, n = r
            is_base = (subdir == "baseline")
            acc_str = f"{acc:5.1f} {('' if is_base else fmt_delta(acc, base_acc)):>6}"
            yes_str = f"{yes:5.1f} {('' if is_base else fmt_delta(yes, base_yes)):>6}"
            print(f"{cat:<12} {label:<8} {acc_str:>16} {yes_str:>16}   {n}/3")
            printed_any = True
        if not printed_any:
            print(f"{cat:<12} {'(no results found)'}")
        print()


if __name__ == "__main__":
    main()
