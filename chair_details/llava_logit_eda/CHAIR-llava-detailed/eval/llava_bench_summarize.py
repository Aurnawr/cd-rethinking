"""
Step 3 of LLaVA-Bench evaluation: summarise GPT-4 review scores.

Reads the review JSONL produced by llava_bench_gpt_review.py and prints:
  - per-category relative score  = model_score / ref_score * 100
  - overall (all categories) relative score
This matches the LLaVA-Bench column in Table 4 of the paper.

Usage:
    python eval/llava_bench_summarize.py \
        --review ./outputs/llava_bench/review_llava7b.jsonl
"""
import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", required=True)
    args = ap.parse_args()

    cat_scores = defaultdict(list)  # category -> list of [ref_score, model_score]

    for line in open(args.review):
        r = json.loads(line)
        scores = r.get("tuple") or r.get("score")
        if scores is None or len(scores) != 2 or -1 in scores:
            print(f"[warn] skipping id={r.get('id')}: bad scores {scores}")
            continue
        cat = r.get("category", "all").replace("llava_bench_", "")
        cat_scores[cat].append(scores)
        cat_scores["all"].append(scores)

    print(f"{'Category':<12} {'Ref':>6} {'Model':>6} {'Relative':>10}")
    print("-" * 36)
    for cat in sorted(cat_scores):
        pairs = cat_scores[cat]
        ref_avg = sum(p[0] for p in pairs) / len(pairs)
        model_avg = sum(p[1] for p in pairs) / len(pairs)
        relative = model_avg / ref_avg * 100 if ref_avg > 0 else 0.0
        print(f"{cat:<12} {ref_avg*10:>6.1f} {model_avg*10:>6.1f} {relative:>9.1f}%")


if __name__ == "__main__":
    main()
