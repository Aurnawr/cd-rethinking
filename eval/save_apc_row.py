#!/usr/bin/env python3
"""Compute the Table 5 `sample-dagger` (APC) row for LLaVA-v1.5-7B from the
APC answer files and save it (CSV + Markdown + per-dataset breakdown)."""
import os, json, csv

DATA = "data/pope"
OUT = "outputs/pope"
APC = os.path.join(OUT, "apc")
TYPES = ["random", "popular", "adversarial"]
DATASETS = ["coco", "aokvqa", "gqa"]
MODEL_TAG = "llava-7b"


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def metrics(ref, res):
    R, P = load(ref), load(res)
    tp = tn = fp = fn = 0
    for r, p in zip(R, P):
        gt = r["label"].strip().lower()
        pr = p["text"].strip().lower()
        if gt == "yes":
            tp += "yes" in pr
            fn += "yes" not in pr
        elif gt == "no":
            tn += "no" in pr
            fp += "no" not in pr
    n = len(R)
    return 100 * (tp + tn) / n, 100 * (tp + fp) / n


per_dataset = []   # (category, dataset, acc, yes)
row_avg = []       # (category, acc, yes)
for t in TYPES:
    accs, yeses = [], []
    for ds in DATASETS:
        ref = f"{DATA}/{ds}/{ds}_pope_{t}.json"
        res = f"{APC}/{MODEL_TAG}-{ds}-{t}-sample.jsonl"
        acc, yes = metrics(ref, res)
        per_dataset.append((t.capitalize(), ds, acc, yes))
        accs.append(acc); yeses.append(yes)
    row_avg.append((t.capitalize(), sum(accs) / len(accs), sum(yeses) / len(yeses)))

os.makedirs(OUT, exist_ok=True)

# CSV (averaged row)
csv_path = os.path.join(OUT, "table5_llava-7b_apc_row.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Category", "Method", "Model", "Accuracy", "Yes(%)", "n_datasets", "datasets"])
    for cat, acc, yes in row_avg:
        w.writerow([cat, "sample\u2020 (APC)", "LLaVA-v1.5-7B", f"{acc:.1f}", f"{yes:.1f}", 3, "coco+aokvqa+gqa"])

# CSV (per-dataset breakdown)
csv_pd_path = os.path.join(OUT, "table5_llava-7b_apc_per_dataset.csv")
with open(csv_pd_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Category", "Dataset", "Method", "Accuracy", "Yes(%)"])
    for cat, ds, acc, yes in per_dataset:
        w.writerow([cat, ds, "sample\u2020 (APC)", f"{acc:.1f}", f"{yes:.1f}"])

# Markdown
md_path = os.path.join(OUT, "table5_llava-7b_apc_row.md")
with open(md_path, "w") as f:
    f.write("# Table 5 — LLaVA-v1.5-7B, `sample\u2020` (APC) row\n\n")
    f.write("Averaged across coco, aokvqa, gqa (sampling strategy, temperature 1).\n\n")
    f.write("| Category | Method | Accuracy | Yes (%) |\n")
    f.write("|---|---|---|---|\n")
    for cat, acc, yes in row_avg:
        f.write(f"| {cat} | sample\u2020 (APC) | {acc:.1f} | {yes:.1f} |\n")
    f.write("\n## Per-dataset breakdown\n\n")
    f.write("| Category | Dataset | Accuracy | Yes (%) |\n")
    f.write("|---|---|---|---|\n")
    for cat, ds, acc, yes in per_dataset:
        f.write(f"| {cat} | {ds} | {acc:.1f} | {yes:.1f} |\n")

# Console output
print("Table 5 - LLaVA-v1.5-7B - sample\u2020 (APC) row  [avg over coco+aokvqa+gqa]\n")
print(f"{'Category':<12} {'Method':<14} {'Accuracy':>9} {'Yes(%)':>8}")
print("-" * 45)
for cat, acc, yes in row_avg:
    print(f"{cat:<12} {'sample\u2020 (APC)':<14} {acc:>9.1f} {yes:>8.1f}")

print("\nSaved:")
for p in (csv_path, csv_pd_path, md_path):
    print(" ", p)
