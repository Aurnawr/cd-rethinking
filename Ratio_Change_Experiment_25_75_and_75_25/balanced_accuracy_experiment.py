"""
Balanced Accuracy (= Binary AUC) Experiment.

Standard accuracy on POPE is class-balance-sensitive: the same model predictions
produce different accuracy values depending on the YES/NO ratio.

Balanced Accuracy = (TPR + TNR) / 2 = (Sensitivity + Specificity) / 2

For a binary classifier, this is mathematically identical to the AUC-ROC:
  AUC(binary) = 0.5*(1 + TPR - FPR) = 0.5*(TP/P + TN/N) = Balanced Accuracy

It is class-imbalance-immune: a pure class-balance shift with identical predictions
does NOT change balanced accuracy. A pure threshold shift DOES change it.

Hypothesis:
  - Standard accuracy shows a large V-shape swing across 25/75 and 75/25.
  - Balanced accuracy shows much smaller variation for baseline (no threshold change).
  - YES-inflating methods (VCD, SID, PBA, OLM) show DECREASED balanced accuracy
    in BOTH regimes vs baseline, proving threshold shift hurts discrimination.
  - ICD shows increased balanced accuracy in the NO-heavy regime (correct direction
    for a NO-biased threshold shift).

All counts are computed exactly from filtered files — no rounding from table values.
"""

import json, os
from collections import defaultdict

EXPERIMENTS = {
    "25_75": {
        "YES_gt": 500,
        "NO_gt":  1500,
        "configs": [
            ("LLava7B", "GQA",     "25_75_experiments/25_75exp_LLava7B_GQA",    "gqa",     "llava-7b-gqa-{split}-greedy.jsonl",   ["baseline","vcd","sid","icd","pba","olm"]),
            ("LLava7B", "COCO",    "25_75_experiments/25_75exp_LLava7B_COCO",   "coco",    "llava-7b-coco-{split}-greedy.jsonl",  ["baseline","vcd","sid","pba","olm"]),
            ("LLava7B", "AOKVQA",  "25_75_experiments/25_75exp_LLava7B_AOKVQA", "aokvqa",  "aokvqa-{split}-greedy.jsonl",         ["baseline","vcd","sid","icd","pba","olm"]),
            ("qwen7B",  "GQA",     "25_75_experiments/25_75exp_qwen7B_GQA",     "gqa",     "qwen-7b-gqa-{split}-greedy.jsonl",    ["baseline","vcd","sid","icd","pba","olm"]),
            ("qwen7B",  "COCO",    "25_75_experiments/25_75exp_qwen7B_COCO",    "coco",    "qwen-7b-coco-{split}-greedy.jsonl",   ["baseline","vcd","sid","icd","pba","olm"]),
            ("qwen7B",  "AOKVQA",  "25_75_experiments/25_75exp_qwen7B_AOKVQA",  "aokvqa",  "aokvqa-{split}-greedy.jsonl",         ["baseline","vcd","sid","icd","pba","olm"]),
        ]
    },
    "75_25": {
        "YES_gt": 1500,
        "NO_gt":  500,
        "configs": [
            ("LLava7B", "GQA",     "75_25_experiments/75_25exp_LLava7B_GQA",    "gqa",     "llava-7b-gqa-{split}-greedy.jsonl",   ["baseline","vcd","sid","icd","pba","olm"]),
            ("LLava7B", "COCO",    "75_25_experiments/75_25exp_LLava7B_COCO",   "coco",    "llava-7b-coco-{split}-greedy.jsonl",  ["baseline","vcd","sid","pba","olm"]),
            ("LLava7B", "AOKVQA",  "75_25_experiments/75_25exp_LLava7B_AOKVQA", "aokvqa",  "aokvqa-{split}-greedy.jsonl",         ["baseline","vcd","sid","icd","pba","olm"]),
            ("qwen7B",  "GQA",     "75_25_experiments/75_25exp_qwen7B_GQA",     "gqa",     "qwen-7b-gqa-{split}-greedy.jsonl",    ["baseline","vcd","sid","icd","pba","olm"]),
            ("qwen7B",  "COCO",    "75_25_experiments/75_25exp_qwen7B_COCO",    "coco",    "qwen-7b-coco-{split}-greedy.jsonl",   ["baseline","vcd","sid","icd","pba","olm"]),
            ("qwen7B",  "AOKVQA",  "75_25_experiments/75_25exp_qwen7B_AOKVQA",  "aokvqa",  "aokvqa-{split}-greedy.jsonl",         ["baseline","vcd","sid","icd","pba","olm"]),
        ]
    }
}

BASE = "/teamspace/studios/this_studio/cd_rethink"
SPLITS = ["random", "popular", "adversarial"]

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def compute_counts(ref_path, res_path):
    ref = load_jsonl(ref_path)
    res = load_jsonl(res_path)
    tp = tn = fp = fn = 0
    for r, s in zip(ref, res):
        assert r["question_id"] == s["question_id"], f"ID mismatch {r['question_id']} vs {s['question_id']}"
        gt  = r["label"].strip().lower()
        ans = s["text"].strip().lower()
        if gt == "yes":
            if "yes" in ans: tp += 1
            else:            fn += 1
        else:
            if "no" in ans:  tn += 1
            else:            fp += 1
    return tp, tn, fp, fn

def metrics(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    std_acc  = (tp + tn) / total * 100
    tpr      = tp / (tp + fn)          # sensitivity / recall
    tnr      = tn / (tn + fp)          # specificity
    fpr      = fp / (fp + tn)          # false positive rate
    bal_acc  = (tpr + tnr) / 2 * 100  # = binary AUC * 100
    yes_pct  = (tp + fp) / total * 100
    return std_acc, bal_acc, tpr, tnr, fpr, yes_pct

# ── Collect all results ───────────────────────────────────────────────────────

all_results = {}   # (ratio, model, dataset, method, split) -> (std_acc, bal_acc, tpr, tnr, fpr, yes_pct)

for ratio, info in EXPERIMENTS.items():
    for (model, dataset, exp_rel, anno_key, out_template, methods) in info["configs"]:
        exp_dir = os.path.join(BASE, exp_rel)

        for split in SPLITS:
            # Reference file
            ref_path = os.path.join(exp_dir, "data", anno_key, f"{anno_key}_pope_{split}.json")
            if not os.path.exists(ref_path):
                continue

            for method in methods:
                # Find the output file — try the template, and also a flat name
                candidates = [
                    os.path.join(exp_dir, "outputs", "pope", method, out_template.format(split=split)),
                    os.path.join(exp_dir, "outputs", "pope", method, f"aokvqa-{split}-greedy.jsonl"),
                ]
                res_path = None
                for c in candidates:
                    if os.path.exists(c):
                        res_path = c
                        break

                if res_path is None:
                    # Try listing the directory
                    d = os.path.join(exp_dir, "outputs", "pope", method)
                    if os.path.isdir(d):
                        files = os.listdir(d)
                        matches = [f for f in files if split in f]
                        if matches:
                            res_path = os.path.join(d, matches[0])

                if res_path is None:
                    continue

                try:
                    tp, tn, fp, fn = compute_counts(ref_path, res_path)
                    key = (ratio, model, dataset, method, split)
                    all_results[key] = metrics(tp, tn, fp, fn)
                except Exception as e:
                    print(f"ERROR {ratio}/{model}/{dataset}/{method}/{split}: {e}")

print(f"Computed {len(all_results)} result entries\n")

# ── Print comprehensive table ─────────────────────────────────────────────────

METHOD_ORDER = ["baseline", "vcd", "sid", "icd", "pba", "olm"]

for model in ["LLava7B", "qwen7B"]:
    for dataset in ["GQA", "COCO", "AOKVQA"]:
        has_data = any((r, model, dataset, "baseline", s) in all_results
                       for r in ["25_75","75_25"] for s in SPLITS)
        if not has_data:
            continue

        print(f"\n{'='*80}")
        print(f"  MODEL: {model}   DATASET: {dataset}")
        print(f"{'='*80}")
        print(f"  {'Method':<10} {'Split':<12} "
              f"{'25/75 StdAcc':>13} {'25/75 BalAcc':>13} "
              f"{'75/25 StdAcc':>13} {'75/25 BalAcc':>13} "
              f"{'Std Swing':>10} {'Bal Swing':>10}")
        print(f"  {'-'*95}")

        for method in METHOD_ORDER:
            for split in SPLITS:
                k25 = ("25_75", model, dataset, method, split)
                k75 = ("75_25", model, dataset, method, split)

                if k25 not in all_results or k75 not in all_results:
                    continue

                std25, bal25, tpr25, tnr25, fpr25, yes25 = all_results[k25]
                std75, bal75, tpr75, tnr75, fpr75, yes75 = all_results[k75]

                std_swing = std75 - std25
                bal_swing = bal75 - bal25

                print(f"  {method:<10} {split:<12} "
                      f"{std25:>12.2f}% {bal25:>12.2f}% "
                      f"{std75:>12.2f}% {bal75:>12.2f}% "
                      f"{std_swing:>+10.2f} {bal_swing:>+10.2f}")

# ── Summary: how much does each metric swing vs baseline ─────────────────────

print(f"\n\n{'='*80}")
print("  SUMMARY: Balanced Accuracy vs Baseline — Averaged across splits")
print("  Positive = method is ABOVE baseline; Negative = method is BELOW baseline")
print(f"{'='*80}")

for model in ["LLava7B", "qwen7B"]:
    for dataset in ["GQA", "COCO", "AOKVQA"]:
        results_25 = {}
        results_75 = {}

        for split in SPLITS:
            kb = ("25_75", model, dataset, "baseline", split)
            if kb not in all_results:
                continue
            _, bal_base25, _, _, _, _ = all_results[kb]
            kb2 = ("75_25", model, dataset, "baseline", split)
            if kb2 not in all_results:
                continue
            _, bal_base75, _, _, _, _ = all_results[kb2]

            for method in METHOD_ORDER:
                k25 = ("25_75", model, dataset, method, split)
                k75 = ("75_25", model, dataset, method, split)
                if k25 in all_results:
                    _, bal25, _, _, _, _ = all_results[k25]
                    results_25.setdefault(method, []).append(bal25 - bal_base25)
                if k75 in all_results:
                    _, bal75, _, _, _, _ = all_results[k75]
                    results_75.setdefault(method, []).append(bal75 - bal_base75)

        if not results_25:
            continue

        print(f"\n  {model} / {dataset}")
        print(f"  {'Method':<10} {'BalAcc vs Base (25/75)':>22} {'BalAcc vs Base (75/25)':>22}")
        print(f"  {'-'*56}")
        for method in METHOD_ORDER:
            if method == "baseline":
                print(f"  {'baseline':<10} {'(reference)':>22} {'(reference)':>22}")
                continue
            avg25 = sum(results_25.get(method,[])) / max(1, len(results_25.get(method,[])))
            avg75 = sum(results_75.get(method,[])) / max(1, len(results_75.get(method,[])))
            print(f"  {method:<10} {avg25:>+21.2f}% {avg75:>+21.2f}%")

# ── Key insight: V-shape magnitude comparison ─────────────────────────────────

print(f"\n\n{'='*80}")
print("  V-SHAPE MAGNITUDE: Standard Acc vs Balanced Acc (Adversarial split only)")
print("  Shows how much each metric swings from 25/75 to 75/25")
print("  A genuine discrimination improvement would show ZERO swing in balanced acc")
print(f"{'='*80}")

print(f"\n  {'Config':<22} {'Method':<10} {'Std Swing':>12} {'Bal Swing':>12} {'Ratio (Std/Bal)':>16}")
print(f"  {'-'*74}")

for model in ["LLava7B", "qwen7B"]:
    for dataset in ["GQA", "COCO", "AOKVQA"]:
        split = "adversarial"
        for method in METHOD_ORDER:
            k25 = ("25_75", model, dataset, method, split)
            k75 = ("75_25", model, dataset, method, split)
            if k25 not in all_results or k75 not in all_results:
                continue
            std25, bal25, *_ = all_results[k25]
            std75, bal75, *_ = all_results[k75]
            std_swing = std75 - std25
            bal_swing = bal75 - bal25
            ratio = abs(std_swing) / max(0.01, abs(bal_swing))
            config = f"{model}/{dataset}"
            print(f"  {config:<22} {method:<10} {std_swing:>+11.2f}% {bal_swing:>+11.2f}% {ratio:>15.1f}x")
        print()
