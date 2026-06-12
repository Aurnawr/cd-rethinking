#!/usr/bin/env python3
"""Compute the full Table 5 (LLaVA-v1.5-7B, sample strategy) and save to
CSV + Markdown, with deltas vs the `sample` baseline."""
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


def avg(subdir, ptype):
    accs, yeses = [], []
    for ds in DATASETS:
        ref = f"{DATA}/{ds}/{ds}_pope_{ptype}.json"
        res = f"{OUT}/{subdir}/{TAG}-{ds}-{ptype}-sample.jsonl"
        if os.path.isfile(ref) and os.path.isfile(res):
            a, y = file_metrics(ref, res)
            accs.append(a); yeses.append(y)
    if not accs:
        return None
    return sum(accs) / len(accs), sum(yeses) / len(yeses)


def d(v, b):
    s = v - b
    return f"{'+' if s >= 0 else '-'}{abs(s):.1f}"


rows = []  # (category, method, acc, yes, dacc, dyes)
for ptype in TYPES:
    base = avg("baseline", ptype)
    for label, subdir in METHODS:
        r = avg(subdir, ptype)
        if r is None:
            continue
        acc, yes = r
        if subdir == "baseline":
            rows.append((ptype.capitalize(), label, acc, yes, "", ""))
        else:
            rows.append((ptype.capitalize(), label, acc, yes,
                         d(acc, base[0]), d(yes, base[1])))

# CSV
csv_path = f"{OUT}/table5_{TAG}_full.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Category", "Method", "Accuracy", "dAccuracy", "Yes(%)", "dYes(%)"])
    for cat, m, acc, yes, da, dy in rows:
        w.writerow([cat, m, f"{acc:.1f}", da, f"{yes:.1f}", dy])

# Markdown
md_path = f"{OUT}/table5_{TAG}_full.md"
with open(md_path, "w") as f:
    f.write("# Table 5 — LLaVA-v1.5-7B (sample strategy)\n\n")
    f.write("Accuracy and Yes(%) averaged across coco/aokvqa/gqa; deltas vs the `sample` baseline.\n\n")
    f.write("| Category | Method | Accuracy | \u0394Acc | Yes (%) | \u0394Yes |\n")
    f.write("|---|---|---|---|---|---|\n")
    for cat, m, acc, yes, da, dy in rows:
        f.write(f"| {cat} | {m} | {acc:.1f} | {da} | {yes:.1f} | {dy} |\n")

print("Saved:")
print(" ", csv_path)
print(" ", md_path)
