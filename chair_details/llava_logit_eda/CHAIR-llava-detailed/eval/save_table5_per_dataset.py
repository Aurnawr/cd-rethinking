#!/usr/bin/env python3
"""Compute Table 5 (LLaVA-v1.5-7B, sample strategy) separately for each
dataset (coco, aokvqa, gqa). Saves one CSV + one Markdown per dataset and a
combined Markdown, with deltas vs the `sample` baseline."""
import os, json, csv

DATA = "data/pope"
OUT = "outputs/pope"
TYPES = ["random", "popular", "adversarial"]
DATASETS = ["coco", "aokvqa", "gqa"]
TAG = "llava-7b"
METHODS = [("sample", "baseline"), ("VCD", "vcd"), ("ICD", "icd"),
           ("SID", "sid"), ("sample\u2020", "apc")]


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def file_metrics(ref, res):
    if not (os.path.isfile(ref) and os.path.isfile(res)):
        return None
    R, P = load(ref), load(res)
    tp = tn = fp = fn = 0
    for r, p in zip(R, P):
        gt = r["label"].strip().lower(); pr = p["text"].strip().lower()
        if gt == "yes":
            tp += "yes" in pr; fn += "yes" not in pr
        elif gt == "no":
            tn += "no" in pr; fp += "no" not in pr
    n = len(R)
    return 100 * (tp + tn) / n, 100 * (tp + fp) / n


def delta(v, b):
    s = v - b
    return f"{'+' if s >= 0 else '-'}{abs(s):.1f}"


def rows_for_dataset(ds):
    rows = []  # (category, method, acc, yes, dacc, dyes)
    for ptype in TYPES:
        ref = f"{DATA}/{ds}/{ds}_pope_{ptype}.json"
        base = file_metrics(ref, f"{OUT}/baseline/{TAG}-{ds}-{ptype}-sample.jsonl")
        for label, subdir in METHODS:
            m = file_metrics(ref, f"{OUT}/{subdir}/{TAG}-{ds}-{ptype}-sample.jsonl")
            if m is None:
                continue
            acc, yes = m
            if subdir == "baseline" or base is None:
                rows.append((ptype.capitalize(), label, acc, yes, "", ""))
            else:
                rows.append((ptype.capitalize(), label, acc, yes,
                             delta(acc, base[0]), delta(yes, base[1])))
    return rows


combined_md = ["# Table 5 — LLaVA-v1.5-7B (sample strategy), per dataset\n",
               "Accuracy and Yes(%) per dataset; deltas vs the `sample` baseline.\n"]

for ds in DATASETS:
    rows = rows_for_dataset(ds)

    csv_path = f"{OUT}/table5_{TAG}_{ds}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Category", "Method", "Accuracy", "dAccuracy", "Yes(%)", "dYes(%)"])
        for cat, m, acc, yes, da, dy in rows:
            w.writerow([cat, m, f"{acc:.1f}", da, f"{yes:.1f}", dy])

    md_path = f"{OUT}/table5_{TAG}_{ds}.md"
    block = [f"# Table 5 — LLaVA-v1.5-7B, dataset = {ds} (sample strategy)\n",
             "| Category | Method | Accuracy | \u0394Acc | Yes (%) | \u0394Yes |",
             "|---|---|---|---|---|---|"]
    for cat, m, acc, yes, da, dy in rows:
        block.append(f"| {cat} | {m} | {acc:.1f} | {da} | {yes:.1f} | {dy} |")
    with open(md_path, "w") as f:
        f.write("\n".join(block) + "\n")

    combined_md.append(f"\n## dataset = {ds}\n")
    combined_md += block[1:]

    print(f"\n=== {ds} ===")
    print(f"{'Category':<12} {'Method':<9} {'Acc':>6} {'dAcc':>6} {'Yes%':>6} {'dYes':>6}")
    for cat, m, acc, yes, da, dy in rows:
        print(f"{cat:<12} {m:<9} {acc:>6.1f} {da:>6} {yes:>6.1f} {dy:>6}")

with open(f"{OUT}/table5_{TAG}_per_dataset.md", "w") as f:
    f.write("\n".join(combined_md) + "\n")

print("\nSaved per-dataset CSV/MD and combined:")
for ds in DATASETS:
    print(f"  {OUT}/table5_{TAG}_{ds}.csv  /  .md")
print(f"  {OUT}/table5_{TAG}_per_dataset.md")
