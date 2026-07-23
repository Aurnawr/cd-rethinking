"""
Step 2 of LLaVA-Bench evaluation: GPT-4 judge.
Compares GPT-4 reference answers (ans1) vs your model's answers (ans2),
using the image caption from context.jsonl and the scoring prompts from rule.json.

Uses the openai >= 1.0 API. Requires OPENAI_API_KEY env variable.

Usage:
    python eval/llava_bench_gpt_review.py \
        --question  ./data/llava_bench/questions.jsonl \
        --context   ./data/llava_bench/context.jsonl \
        --rule      ./data/llava_bench/rule.json \
        --answer-ref ./data/llava_bench/answers_gpt4.jsonl \
        --answer-model ./outputs/llava_bench/answers_llava7b.jsonl \
        --output    ./outputs/llava_bench/review_llava7b.jsonl
"""
import argparse
import json
import os
import time

from openai import OpenAI

NUM_SECONDS_TO_SLEEP = 0.5


def get_eval(client: OpenAI, content: str, max_tokens: int = 1024) -> str:
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4-0314",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful and precise assistant for checking the quality of the answer.",
                    },
                    {"role": "user", "content": content},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[retry] {type(e).__name__}: {e}")
            time.sleep(NUM_SECONDS_TO_SLEEP)


def parse_score(review: str):
    try:
        first_line = review.split("\n")[0].replace(",", " ")
        sp = first_line.split()
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        print(f"[parse_score] unexpected format: {review}")
        return [-1, -1]
    except Exception as e:
        print(f"[parse_score] error: {e} | review: {review}")
        return [-1, -1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="./data/llava_bench/questions.jsonl")
    ap.add_argument("--context", default="./data/llava_bench/context.jsonl")
    ap.add_argument("--rule", default="./data/llava_bench/rule.json")
    ap.add_argument("--answer-ref", required=True, help="GPT-4 reference answers (ans1)")
    ap.add_argument("--answer-model", required=True, help="Your model's answers (ans2)")
    ap.add_argument("--output", required=True, help="Output review JSONL")
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    rule_dict = json.load(open(args.rule))

    questions = [json.loads(l) for l in open(args.question)]
    ans1_list = [json.loads(l) for l in open(args.answer_ref)]
    ans2_list = [json.loads(l) for l in open(args.answer_model)]
    context_list = {c["image"]: c for c in (json.loads(l) for l in open(args.context))}

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    existing = []
    if os.path.isfile(args.output):
        existing = [json.loads(l) for l in open(args.output)]

    out_file = open(args.output, "a")
    for idx, (ques, ans1, ans2) in enumerate(zip(questions, ans1_list, ans2_list)):
        if idx < len(existing):
            print(f"Skipping {idx} (already reviewed)")
            continue

        ctx = context_list.get(ques["image"], {})
        caption = ctx.get("caption", "")
        if isinstance(caption, list):
            caption = "\n".join(caption)

        category = "llava_bench_" + ques["category"]
        if category not in rule_dict:
            raise ValueError(f"Category not found in rule file: {category}")
        rule = rule_dict[category]

        content = (
            f"[Context]\n{caption}\n\n"
            f"[Question]\n{ques['text']}\n\n"
            f"[{rule['role']} 1]\n{ans1['text']}\n\n[End of {rule['role']} 1]\n\n"
            f"[{rule['role']} 2]\n{ans2['text']}\n\n[End of {rule['role']} 2]\n\n"
            f"[System]\n{rule['prompt']}\n\n"
        )

        review = get_eval(client, content, args.max_tokens)
        scores = parse_score(review)

        record = {
            "id": idx + 1,
            "question_id": ques["question_id"],
            "answer1_id": ans1.get("answer_id", ans1["question_id"]),
            "answer2_id": ans2.get("answer_id", ans2.get("question_id")),
            "category": category,
            "content": review,
            "tuple": scores,
        }
        out_file.write(json.dumps(record) + "\n")
        out_file.flush()
        print(f"[{idx+1}/{len(questions)}] {ques['category']} -> {scores}")

    out_file.close()


if __name__ == "__main__":
    main()
