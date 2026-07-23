"""
Turn the Claude judge's score pairs into a review JSONL compatible with
llava_bench_summarize.py.

--scores is a JSON file: a list of 60 [assistant1, assistant2] score pairs,
in question order (id 1..60), as produced by the in-session judge.

Usage:
  python eval/finalize_review.py \
      --prompts outputs/llava/_judge/greedy.prompts.jsonl \
      --scores  outputs/llava/_judge/greedy.scores.json \
      --out     outputs/llava/_judge/greedy.review.jsonl
"""
import argparse, json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts = [json.loads(l) for l in open(args.prompts)]
    scores = json.load(open(args.scores))
    assert len(prompts) == len(scores) == 60, f"len mismatch {len(prompts)} vs {len(scores)}"

    with open(args.out, "w") as f:
        for p, sc in zip(prompts, scores):
            assert len(sc) == 2, f"bad pair {sc}"
            f.write(json.dumps({
                "id": p["id"], "question_id": p["question_id"],
                "category": p["category"], "tuple": [float(sc[0]), float(sc[1])],
            }) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
