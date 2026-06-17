"""
Step 2 of LLaVA-Bench evaluation: Gemini-Flash judge (free-tier replacement
for the GPT-4 judge in llava_bench_gpt_review.py).

Uses the google-genai SDK (pip install google-genai).
Requires GEMINI_API_KEY env variable (free from aistudio.google.com).

The prompt format and output JSONL are identical to llava_bench_gpt_review.py,
so llava_bench_summarize.py works unchanged on the output.

Usage:
    export GEMINI_API_KEY="AIza..."
    python eval/llava_bench_gemini_review.py \
        --question     ./data/llava_bench/questions.jsonl \
        --context      ./data/llava_bench/context.jsonl \
        --rule         ./data/llava_bench/rule.json \
        --answer-ref   ./data/llava_bench/answers_gpt4.jsonl \
        --answer-model ./outputs/llava_bench/answers_greedy.jsonl \
        --output       ./outputs/llava_bench/review_greedy.jsonl
"""
import argparse
import json
import os
import time

from google import genai
from google.genai import types

SYSTEM_PROMPT = (
    "You are a helpful and precise assistant for checking the quality of the answer."
)
NUM_SECONDS_TO_SLEEP = 1.0   # rate-limit buffer for free tier (15 rpm)


def get_eval(client: genai.Client, content: str, model: str, max_tokens: int) -> str:
    for attempt in range(10):
        try:
            response = client.models.generate_content(
                model=model,
                contents=content,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text
        except Exception as e:
            wait = NUM_SECONDS_TO_SLEEP * (2 ** attempt)
            print(f"[retry {attempt+1}] {type(e).__name__}: {e} — waiting {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError("Gemini call failed after 10 retries")


def parse_score(review: str):
    try:
        first_line = review.strip().split("\n")[0].replace(",", " ")
        sp = first_line.split()
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        print(f"[parse_score] unexpected format: {review[:80]}")
        return [-1, -1]
    except Exception as e:
        print(f"[parse_score] error: {e} | review: {review[:80]}")
        return [-1, -1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question",      default="./data/llava_bench/questions.jsonl")
    ap.add_argument("--context",       default="./data/llava_bench/context.jsonl")
    ap.add_argument("--rule",          default="./data/llava_bench/rule.json")
    ap.add_argument("--answer-ref",    required=True, help="GPT-4 reference answers (ans1)")
    ap.add_argument("--answer-model",  required=True, help="Your model's answers (ans2)")
    ap.add_argument("--output",        required=True, help="Output review JSONL")
    ap.add_argument("--max-tokens",    type=int, default=1024)
    ap.add_argument("--gemini-model",  default="gemini-1.5-flash",
                    help="Gemini model name (e.g. gemini-1.5-flash, gemini-2.0-flash)")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY env variable not set. "
                               "Get a free key at aistudio.google.com.")

    client = genai.Client(api_key=api_key)
    rule_dict = json.load(open(args.rule))

    questions  = [json.loads(l) for l in open(args.question)]
    ans1_list  = [json.loads(l) for l in open(args.answer_ref)]
    ans2_list  = [json.loads(l) for l in open(args.answer_model)]
    context_map = {c["image"]: c for c in (json.loads(l) for l in open(args.context))}

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    existing = []
    if os.path.isfile(args.output):
        existing = [json.loads(l) for l in open(args.output)]

    out_file = open(args.output, "a")
    for idx, (ques, ans1, ans2) in enumerate(zip(questions, ans1_list, ans2_list)):
        if idx < len(existing):
            print(f"[{idx+1}/{len(questions)}] skip (already reviewed)")
            continue

        ctx = context_map.get(ques["image"], {})
        caption = ctx.get("caption", "")
        if isinstance(caption, list):
            caption = "\n".join(caption)

        category = "llava_bench_" + ques["category"]
        if category not in rule_dict:
            raise ValueError(f"Category not in rule file: {category}")
        rule = rule_dict[category]

        content = (
            f"[Context]\n{caption}\n\n"
            f"[Question]\n{ques['text']}\n\n"
            f"[{rule['role']} 1]\n{ans1['text']}\n\n[End of {rule['role']} 1]\n\n"
            f"[{rule['role']} 2]\n{ans2['text']}\n\n[End of {rule['role']} 2]\n\n"
            f"[System]\n{rule['prompt']}\n\n"
        )

        review = get_eval(client, content, args.gemini_model, args.max_tokens)
        scores = parse_score(review)

        record = {
            "id": idx + 1,
            "question_id": ques["question_id"],
            "answer1_id": ans1.get("answer_id", ans1.get("question_id")),
            "answer2_id": ans2.get("answer_id", ans2.get("question_id")),
            "category": category,
            "content": review,
            "tuple": scores,
        }
        out_file.write(json.dumps(record) + "\n")
        out_file.flush()
        print(f"[{idx+1}/{len(questions)}] {ques['category']:8s} scores={scores}")

        time.sleep(NUM_SECONDS_TO_SLEEP)   # respect free-tier rate limit (15 rpm)

    out_file.close()
    print(f"\nReview saved to {args.output}")


if __name__ == "__main__":
    main()
