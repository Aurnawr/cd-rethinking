#!/usr/bin/env python3
"""
Build Table 4 (LLaVA-Bench column) from Gemini review JSONL files.

For each method the relative score = model_score / ref_score * 100, averaged
over all 60 questions, and broken down by category (conv / detail / complex).
Also saves a delta vs Greedy baseline (matching the ↑↓ notation in Table 4).

Output: CSV + Markdown saved to outputs/llava_bench/.
"""
import os, json, csv, argparse
from collections import defaultdict


def load_scores(review_path):
    """Return {category: [(ref, model), ...], 'all': [...]}."""
    cat_scores = defaultdict(list)
    for line in open(review_path):
        r = json.loads(line)
        scores = r.get("tuple") or r.get("score")
        if not scores or len(scores) != 2 or -1 in scores:
            continue
        cat = r.get("category", "all").replace("llava_bench_", "")
        cat_scores[cat].append(scores)
        cat_scores["all"].append(scores)
    return cat_scores


def relative(pairs):
    if not pairs:
        return 0.0
    ref_avg = sum(p[0] for p in pairs) / len(pairs)
    mod_avg = sum(p[1] for p in pairs) / len(pairs)
    return mod_avg / ref_avg * 100 if ref_avg > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="./outputs/llava_bench")
    ap.add_argument("--methods", nargs="+", default=["greedy", "vcd", "sid"])
    args = ap.parse_args()
    R = args.results_dir
    CATS = ["conv", "detail", "complex", "all"]

    # Collect per-method scores
    data = {}
    for m in args.methods:
        path = os.path.join(R, f"review_{m}.jsonl")
        if not os.path.isfile(path):
            print(f"[warn] {path} not found — skipping {m}")
            continue
        data[m] = load_scores(path)

    if not data:
        print("No review files found. Run llava_bench_gemini_review.py first.")
        return

    # Baseline for deltas (greedy if present, else first method)
    base_key = "greedy" if "greedy" in data else next(iter(data))
    base = {cat: relative(data[base_key].get(cat, [])) for cat in CATS}

    def d(v, b):
        s = v - b
        return f"{'+'  if s >= 0 else '-'}{abs(s):.1f}"

    # ── console ──────────────────────────────────────────────────────────────
    header = f"{'Method':<8} {'conv':>6} {'detail':>7} {'complex':>8} {'overall':>8}"
    print(f"\n=== Table 4 – LLaVA-Bench (relative score, %) ===")
    print(f"{'Method':<8} {'conv':>6} {'detail':>7} {'complex':>8} {'overall':>8}   (delta vs {base_key})")
    print("-" * 55)
    rows = []
    for m in args.methods:
        if m not in data:
            continue
        scores = {cat: relative(data[m].get(cat, [])) for cat in CATS}
        is_base = (m == base_key)
        row = [m] + [f"{scores[c]:.1f}" for c in CATS]
        if not is_base:
            row += [d(scores[c], base[c]) for c in CATS]
        else:
            row += [""] * len(CATS)
        rows.append(row)
        delta_str = "" if is_base else f"  Δall={d(scores['all'], base['all'])}"
        print(f"{m:<8} {scores['conv']:>6.1f} {scores['detail']:>7.1f} "
              f"{scores['complex']:>8.1f} {scores['all']:>8.1f}{delta_str}")

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = os.path.join(R, "table4_llava_bench.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "conv", "detail", "complex", "overall",
                    "d_conv", "d_detail", "d_complex", "d_overall"])
        for row in rows:
            w.writerow(row)

    # ── Markdown ─────────────────────────────────────────────────────────────
    md_path = os.path.join(R, "table4_llava_bench.md")
    with open(md_path, "w") as f:
        f.write("# Table 4 — LLaVA-Bench (LLaVA-v1.5-7B, greedy decoding)\n\n")
        f.write("Relative score (model / GPT-4 reference × 100). "
                f"Δ = delta vs {base_key}.\n\n")
        f.write("| Method | conv | detail | complex | **Overall** | Δconv | Δdetail | Δcomplex | **Δoverall** |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for row in rows:
            # row = [method, conv, detail, complex, all, d_conv, d_detail, d_complex, d_all]
            m_val = row[0]
            scores_part = row[1:5]
            delta_part = row[5:9]
            f.write(f"| {m_val} | " + " | ".join(scores_part) +
                    " | " + " | ".join(delta_part) + " |\n")

    print(f"\nSaved:\n  {csv_path}\n  {md_path}")


if __name__ == "__main__":
    main()
