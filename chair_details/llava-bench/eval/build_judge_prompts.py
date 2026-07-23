"""
Build the exact LLaVA-Bench judge prompts for a given method's answers, so the
Claude judge (in-session) can score them. This reproduces the content string that
llava_bench_gpt_review.py sends to gpt-4-0314 — same context caption, same rule.json
system prompt, GPT-4 reference = Assistant 1, model answer = Assistant 2.

Output: one JSON line per question with {id, question_id, category, content}.
The judge reads `content`, emits two 1-10 scores (Assistant 1, Assistant 2), and
those pairs are written back via finalize_review.py.

Usage:
  python eval/build_judge_prompts.py \
      --answer-model outputs/llava/greedy.jsonl \
      --out          outputs/llava/_judge/greedy.prompts.jsonl
"""
import argparse, json, os

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "llava_bench")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default=os.path.join(DATA, "questions.jsonl"))
    ap.add_argument("--context", default=os.path.join(DATA, "context.jsonl"))
    ap.add_argument("--rule", default=os.path.join(DATA, "rule.json"))
    ap.add_argument("--answer-ref", default=os.path.join(DATA, "answers_gpt4.jsonl"))
    ap.add_argument("--answer-model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rule = json.load(open(args.rule))
    ques = [json.loads(l) for l in open(args.question)]
    ans1 = [json.loads(l) for l in open(args.answer_ref)]
    ans2 = [json.loads(l) for l in open(args.answer_model)]
    ctx = {c["image"]: c for c in (json.loads(l) for l in open(args.context))}

    assert len(ques) == len(ans1) == len(ans2) == 60, f"len mismatch: {len(ques)},{len(ans1)},{len(ans2)}"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for i, (q, a1, a2) in enumerate(zip(ques, ans1, ans2)):
            cap = ctx.get(q["image"], {}).get("caption", "")
            if isinstance(cap, list):
                cap = "\n".join(cap)
            cat = "llava_bench_" + q["category"]
            r = rule[cat]
            content = (
                f"[Context]\n{cap}\n\n"
                f"[Question]\n{q['text']}\n\n"
                f"[{r['role']} 1]\n{a1['text']}\n\n[End of {r['role']} 1]\n\n"
                f"[{r['role']} 2]\n{a2['text']}\n\n[End of {r['role']} 2]\n\n"
                f"[System]\n{r['prompt']}\n\n"
            )
            f.write(json.dumps({"id": i + 1, "question_id": q["question_id"],
                                "category": cat, "content": content}) + "\n")
    print(f"wrote {args.out}: 60 judge prompts")


if __name__ == "__main__":
    main()
