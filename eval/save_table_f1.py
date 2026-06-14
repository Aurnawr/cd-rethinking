#!/usr/bin/env python3
"""Compute Accuracy / F1 / Yes(%) (Table 11 style) for a given model from the
logged answer files (no re-inference). Saves averaged + per-dataset CSV/MD.

F1 uses the same yes/no matching rule as eval/pope_eval_base.py.
Parameterized by --results-dir / --model-tag (7B: outputs/pope, llava-7b;
13B: outputs/pope_13b, llava-13b).
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
    acc = 100 * (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 100 * (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    yes = 100 * (tp + fp) / n
    return acc, f1, yes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--data-dir", default="./data/pope")
    ap.add_argument("--model-tag", required=True)
    args = ap.parse_args()
    R, DATA, TAG = args.results_dir, args.data_dir, args.model_tag

    def avg(subdir, ptype):
        a, f, y = [], [], []
        for ds in DATASETS:
            m = file_metrics(f"{DATA}/{ds}/{ds}_pope_{ptype}.json",
                             f"{R}/{subdir}/{TAG}-{ds}-{ptype}-sample.jsonl")
            if m:
                a.append(m[0]); f.append(m[1]); y.append(m[2])
        return (sum(a)/len(a), sum(f)/len(f), sum(y)/len(y)) if a else None

    def write_table(path_csv, path_md, title, get):
        rows = []
        for ptype in TYPES:
            for label, subdir in METHODS:
                m = get(subdir, ptype)
                if m is None:
                    continue
                rows.append((ptype.capitalize(), label, m[0], m[1], m[2]))
        with open(path_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["Category", "Method", "Accuracy", "F1", "Yes(%)"])
            for cat, mth, acc, f1, yes in rows:
                w.writerow([cat, mth, f"{acc:.1f}", f"{f1:.1f}", f"{yes:.1f}"])
        with open(path_md, "w") as fh:
            fh.write(f"# {title}\n\n| Category | Method | Accuracy | F1-Score | Yes (%) |\n|---|---|---|---|---|\n")
            for cat, mth, acc, f1, yes in rows:
                fh.write(f"| {cat} | {mth} | {acc:.1f} | {f1:.1f} | {yes:.1f} |\n")
        return rows

    # averaged over datasets
    rows = write_table(f"{R}/table_f1_{TAG}_full.csv", f"{R}/table_f1_{TAG}_full.md",
                       f"Accuracy / F1 / Yes(%) - {TAG} (avg over coco/aokvqa/gqa)", avg)

    # per dataset (individual files + one combined file)
    combined = [f"# Accuracy / F1 / Yes(%) - {TAG} (sample strategy), per dataset\n"]
    for ds in DATASETS:
        def get_ds(subdir, ptype, ds=ds):
            return file_metrics(f"{DATA}/{ds}/{ds}_pope_{ptype}.json",
                                f"{R}/{subdir}/{TAG}-{ds}-{ptype}-sample.jsonl")
        write_table(f"{R}/table_f1_{TAG}_{ds}.csv", f"{R}/table_f1_{TAG}_{ds}.md",
                    f"Accuracy / F1 / Yes(%) - {TAG}, dataset = {ds}", get_ds)
        combined.append(f"\n## dataset = {ds}\n")
        combined.append("| Category | Method | Accuracy | F1-Score | Yes (%) |")
        combined.append("|---|---|---|---|---|")
        for ptype in TYPES:
            for label, subdir in METHODS:
                m = get_ds(subdir, ptype)
                if m is None:
                    continue
                combined.append(f"| {ptype.capitalize()} | {label} | {m[0]:.1f} | {m[1]:.1f} | {m[2]:.1f} |")
    with open(f"{R}/table_f1_{TAG}_per_dataset.md", "w") as fh:
        fh.write("\n".join(combined) + "\n")

    print(f"=== Accuracy / F1 / Yes(%) - {TAG} (avg over coco/aokvqa/gqa) ===")
    print(f"{'Category':<12} {'Method':<9} {'Acc':>6} {'F1':>6} {'Yes%':>6}")
    for cat, mth, acc, f1, yes in rows:
        print(f"{cat:<12} {mth:<9} {acc:>6.1f} {f1:>6.1f} {yes:>6.1f}")
    print("\nSaved CSV/MD:")
    for suffix in ["full", "coco", "aokvqa", "gqa"]:
        print(f"  {R}/table_f1_{TAG}_{suffix}.csv  /  .md")
    print(f"  {R}/table_f1_{TAG}_per_dataset.md")


if __name__ == "__main__":
    main()
