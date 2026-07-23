#!/usr/bin/env python3
"""Compute Table 5 (sample strategy) for a given model and save:
  - full averaged table (CSV + MD)
  - per-dataset tables (CSV + MD) + a combined per-dataset MD
Deltas are vs the `sample` baseline. Parameterized by --results-dir / --model-tag
so it works for both 7B (outputs/pope, llava-7b) and 13B (outputs/pope_13b, llava-13b).
"""
import os, json, csv, argparse

TYPES = ["random", "popular", "adversarial"]
DATASETS = ["coco", "aokvqa", "gqa"]
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


def d(v, b):
    s = v - b
    return f"{'+' if s >= 0 else '-'}{abs(s):.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--data-dir", default="./data/pope")
    ap.add_argument("--model-tag", required=True)
    args = ap.parse_args()
    R, DATA, TAG = args.results_dir, args.data_dir, args.model_tag

    def avg(subdir, ptype):
        accs, yeses = [], []
        for ds in DATASETS:
            m = file_metrics(f"{DATA}/{ds}/{ds}_pope_{ptype}.json",
                             f"{R}/{subdir}/{TAG}-{ds}-{ptype}-sample.jsonl")
            if m:
                accs.append(m[0]); yeses.append(m[1])
        return (sum(accs) / len(accs), sum(yeses) / len(yeses)) if accs else None

    # ---- full averaged table ---- #
    rows = []
    for ptype in TYPES:
        base = avg("baseline", ptype)
        for label, subdir in METHODS:
            r = avg(subdir, ptype)
            if r is None:
                continue
            acc, yes = r
            if subdir == "baseline" or base is None:
                rows.append((ptype.capitalize(), label, acc, yes, "", ""))
            else:
                rows.append((ptype.capitalize(), label, acc, yes, d(acc, base[0]), d(yes, base[1])))

    with open(f"{R}/table5_{TAG}_full.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Category", "Method", "Accuracy", "dAccuracy", "Yes(%)", "dYes(%)"])
        for cat, m, acc, yes, da, dy in rows:
            w.writerow([cat, m, f"{acc:.1f}", da, f"{yes:.1f}", dy])
    with open(f"{R}/table5_{TAG}_full.md", "w") as f:
        f.write(f"# Table 5 - {TAG} (sample strategy)\n\n")
        f.write("Accuracy and Yes(%) averaged across coco/aokvqa/gqa; deltas vs `sample` baseline.\n\n")
        f.write("| Category | Method | Accuracy | \u0394Acc | Yes (%) | \u0394Yes |\n|---|---|---|---|---|---|\n")
        for cat, m, acc, yes, da, dy in rows:
            f.write(f"| {cat} | {m} | {acc:.1f} | {da} | {yes:.1f} | {dy} |\n")

    # ---- per-dataset tables ---- #
    combined = [f"# Table 5 - {TAG} (sample strategy), per dataset\n",
                "Accuracy and Yes(%) per dataset; deltas vs `sample` baseline.\n"]
    for ds in DATASETS:
        ds_rows = []
        for ptype in TYPES:
            ref = f"{DATA}/{ds}/{ds}_pope_{ptype}.json"
            base = file_metrics(ref, f"{R}/baseline/{TAG}-{ds}-{ptype}-sample.jsonl")
            for label, subdir in METHODS:
                m = file_metrics(ref, f"{R}/{subdir}/{TAG}-{ds}-{ptype}-sample.jsonl")
                if m is None:
                    continue
                acc, yes = m
                if subdir == "baseline" or base is None:
                    ds_rows.append((ptype.capitalize(), label, acc, yes, "", ""))
                else:
                    ds_rows.append((ptype.capitalize(), label, acc, yes, d(acc, base[0]), d(yes, base[1])))
        with open(f"{R}/table5_{TAG}_{ds}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Category", "Method", "Accuracy", "dAccuracy", "Yes(%)", "dYes(%)"])
            for cat, m, acc, yes, da, dy in ds_rows:
                w.writerow([cat, m, f"{acc:.1f}", da, f"{yes:.1f}", dy])
        block = [f"\n## dataset = {ds}\n",
                 "| Category | Method | Accuracy | \u0394Acc | Yes (%) | \u0394Yes |",
                 "|---|---|---|---|---|---|"]
        for cat, m, acc, yes, da, dy in ds_rows:
            block.append(f"| {cat} | {m} | {acc:.1f} | {da} | {yes:.1f} | {dy} |")
        with open(f"{R}/table5_{TAG}_{ds}.md", "w") as f:
            f.write(f"# Table 5 - {TAG}, dataset = {ds} (sample strategy)\n" + "\n".join(block[1:]) + "\n")
        combined += block

    with open(f"{R}/table5_{TAG}_per_dataset.md", "w") as f:
        f.write("\n".join(combined) + "\n")

    # ---- console ---- #
    print(f"=== Table 5 - {TAG} (avg over coco/aokvqa/gqa) ===")
    print(f"{'Category':<12} {'Method':<9} {'Acc':>6} {'dAcc':>6} {'Yes%':>6} {'dYes':>6}")
    for cat, m, acc, yes, da, dy in rows:
        print(f"{cat:<12} {m:<9} {acc:>6.1f} {da:>6} {yes:>6.1f} {dy:>6}")
    print("\nSaved CSV/MD:")
    for suffix in ["full", "coco", "aokvqa", "gqa"]:
        print(f"  {R}/table5_{TAG}_{suffix}.csv  /  .md")
    print(f"  {R}/table5_{TAG}_per_dataset.md")


if __name__ == "__main__":
    main()
