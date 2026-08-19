"""
generate_qwen_bench_icd.py -- ICD (Instruction Contrastive Decoding,
arXiv:2403.18715) on the 60-question LLaVA-Bench-in-the-Wild set, for
Qwen2.5-VL-7B-Instruct.

This is the other missing leg identified in the LLaVA-Bench audit: no raw
ICD generations existed anywhere in the repo for Qwen on this benchmark
(LLaVA's ICD data already exists via generate_llava_bench_icd.py -- this is
the Qwen counterpart of that script). Same 5 disturbance prompts, same
cd_alpha/cd_beta convention, run as 5 SEPARATE full passes (one prompt per
run), matching the official ICD reference methodology and this project's
own generate_llava_bench_icd.py / generate_qwen_icd.py precedent.

Reuses the verified KV-cached Branch class from generate_qwen.py unchanged
-- this is exactly the case Branch was built to handle safely: the amateur
branch here has a DIFFERENT prompt length than the expert (the disturbance
prefix), so the two branches' multimodal RoPE deltas genuinely differ, and
Branch computes/stores each independently rather than through the model's
shared mutable rope_deltas attribute. See generate_qwen.py's module
docstring for the full explanation.

Output schema matches generate_llava_bench_icd.py exactly (same disturbance
prompt pool, same field names), so both models' ICD llava-bench answers
drop into the same judging pipeline unchanged.

Usage:
  python generate_qwen_bench_icd.py --prompt-key p1 --decode sample --seed 0 \
      --out ../../jaccard_on_qwen/outputs/llava_bench/icd/qwen_icd_p1.jsonl
"""
import argparse, json, math, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from generate_qwen import MODEL_PATH, Branch, build_inputs, select_token, set_seed

MAX_NEW_TOKENS = 512
CD_ALPHA = 1.0
CD_BETA = 0.1  # matches generate_llava_bench_icd.py's llava-bench convention
LOG_BETA = math.log(CD_BETA)
NEG_INF = float("-inf")

# Verbatim from generate_llava_bench_icd.py / generate_qwen_icd.py -- the
# official ICD repo's 5 disturbance prompts (github.com/p1k0pan/ICD).
DISTURBANCE_PROMPTS = {
    "p1": "You are an object detector to recognize every different objects.",
    "p2": "You are an object detector to recognize every different objects by focusing the shapes, colors and relationships of objects.",
    "n1": "I want you avoid any specific identification or categorization of the objects depicted.",
    "n2": "You are a confused objects detector to provide a fuzzy overview or impression of the image.",
    "p3": "You are an object detector to provide a general overview or impression of the image.",
}

BENCH_DATA = HERE.parent.parent / "jaccard_experiment" / "data" / "llava_bench"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-key", required=True, choices=list(DISTURBANCE_PROMPTS.keys()))
    ap.add_argument("--decode", choices=["greedy", "sample"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--question-file", default=str(BENCH_DATA / "questions.jsonl"))
    ap.add_argument("--image-folder", default=str(BENCH_DATA / "images"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="only process the first N questions (smoke test)")
    args = ap.parse_args()

    set_seed(args.seed)
    disturbance_text = DISTURBANCE_PROMPTS[args.prompt_key]
    print(f"[qwen-bench-icd] prompt_key={args.prompt_key} disturbance={disturbance_text!r} "
          f"decode={args.decode} cd_alpha={CD_ALPHA} cd_beta={CD_BETA}", flush=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager").eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    device = model.device
    eos_id = processor.tokenizer.eos_token_id

    questions = [json.loads(l) for l in open(args.question_file)]
    if args.limit:
        questions = questions[: args.limit]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = {json.loads(l)["question_id"] for l in open(args.out)} if os.path.exists(args.out) else set()
    out_f = open(args.out, "a")

    import time; t0 = time.time(); n = 0
    for q in questions:
        qid = q["question_id"]
        if qid in done:
            continue
        image_file, qs = q["image"], q["text"]

        iseed = (args.seed * 1_000_003 + qid) % (2**31 - 1)
        torch.manual_seed(iseed)

        img = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        inp_e = build_inputs(processor, img, qs, device)
        inp_a = build_inputs(processor, img, disturbance_text + " " + qs, device)

        expert = Branch(model, inp_e, device)
        amateur = Branch(model, inp_a, device)

        chosen_ids = []
        for step in range(MAX_NEW_TOKENS):
            E, A = expert.logits, amateur.logits
            cutoff = LOG_BETA + E.max().item()
            mask = E < cutoff
            scored = (1 + CD_ALPHA) * E - CD_ALPHA * A
            scored = scored.clone(); scored[mask] = NEG_INF

            chosen_id = select_token(scored, args.decode)
            chosen_ids.append(chosen_id)
            if chosen_id == eos_id:
                break
            expert.step(chosen_id)
            amateur.step(chosen_id)

        outputs = processor.tokenizer.decode(chosen_ids, skip_special_tokens=True).strip()

        out_f.write(json.dumps({
            "question_id": qid, "prompt": qs, "text": outputs,
            "answer_id": f"icd-{args.prompt_key}-{qid}", "model_id": "qwen2.5-vl-7b-icd",
            "metadata": {"prompt_key": args.prompt_key, "disturbance": disturbance_text, "decode": args.decode},
        }) + "\n")
        out_f.flush()
        n += 1
        if n % 10 == 0:
            el = time.time() - t0
            print(f"  [{n}] {el:.0f}s ({el/n:.1f}s/q)", flush=True)

    out_f.close()
    print(f"[qwen-bench-icd] DONE {args.prompt_key} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
