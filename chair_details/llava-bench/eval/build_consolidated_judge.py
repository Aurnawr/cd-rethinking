"""
Build ONE consolidated judging file for the in-session Claude judge.

For each of the 60 questions it emits a block with the shared context, the
question, the GPT-4 reference answer (scored once), and every method's candidate
answer. The judge scores the reference and each candidate on the LLaVA-Bench 1-10
scale in a single calibrated pass, then per-method relative = mean(cand)/mean(ref).

This is a single-judge variant of the official protocol (which re-scores the
reference in every pairwise call); scoring the reference once removes position
noise and keeps all methods mutually calibrated.

Usage:
  python eval/build_consolidated_judge.py --model llava \
      --methods greedy sample sampledagger_apc_b0.100 vcd_sample vcd_greedy sid_greedy sid_sample \
      --out outputs/llava/_judge/consolidated.jsonl
"""
import argparse, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "llava_bench")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="subdir under outputs/ (llava or qwen)")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ques = [json.loads(l) for l in open(os.path.join(DATA, "questions.jsonl"))]
    ref = [json.loads(l) for l in open(os.path.join(DATA, "answers_gpt4.jsonl"))]
    ctx = {c["image"]: c for c in (json.loads(l) for l in open(os.path.join(DATA, "context.jsonl")))}

    method_ans = {}
    for m in args.methods:
        path = os.path.join(ROOT, "outputs", args.model, f"{m}.jsonl")
        rows = [json.loads(l) for l in open(path)]
        assert len(rows) == 60, f"{m}: {len(rows)} rows (need 60)"
        method_ans[m] = rows

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for i, (q, r) in enumerate(zip(ques, ref)):
            cap = ctx.get(q["image"], {}).get("caption", "")
            if isinstance(cap, list):
                cap = "\n".join(cap)
            block = {
                "id": i + 1,
                "question_id": q["question_id"],
                "category": q["category"],
                "context": cap,
                "question": q["text"],
                "reference": r["text"],
                "candidates": {m: method_ans[m][i]["text"] for m in args.methods},
            }
            f.write(json.dumps(block) + "\n")
    print(f"wrote {args.out}: 60 blocks x ({len(args.methods)} methods + reference)")


if __name__ == "__main__":
    main()
