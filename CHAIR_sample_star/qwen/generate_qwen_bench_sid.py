"""
generate_qwen_bench_sid.py -- SID (Self-Introspective Decoding, arXiv:2408.02032)
on the 60-question LLaVA-Bench-in-the-Wild set, for Qwen2.5-VL-7B-Instruct.

This is the missing leg identified in the LLaVA-Bench audit: no raw SID
generations existed anywhere in the repo for Qwen on this benchmark. Reuses
the verified, KV-cached, attention-based SID machinery from generate_qwen.py
unchanged (Branch, SIDState, install_sid_hooks -- least-attended 10% of image
tokens by REAL captured attention weights at AGG_LAYER=2, topk(largest=False),
not a random subset) -- see generate_qwen.py's module docstring for the full
correctness story and verify_qwen_cache.py for the cache verification this
mechanism was checked against before any real run.

Hyperparameters match generate_llava_bench_icd.py's llava-bench convention
(cd_alpha=1.0, cd_beta=0.1, max_new_tokens=512, sample decode) rather than
this package's CHAIR/COCO convention (max_new_tokens=256) -- llava-bench
questions expect longer, more open-ended answers than a single "describe
this image" caption.

Output schema matches the existing llava-bench answer files in this repo
({question_id, prompt, text, answer_id, model_id, metadata}), so it drops
into the same judging pipeline (build_consolidated_judge.py) unchanged.

Usage:
  python generate_qwen_bench_sid.py --decode sample --seed 0 \
      --out ../../jaccard_on_qwen/outputs/llava_bench/sid/qwen_sid.jsonl
"""
import argparse, json, math, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from generate_qwen import (MODEL_PATH, Branch, build_inputs, select_token, set_seed,
                            SIDState, install_sid_hooks)

MAX_NEW_TOKENS = 512
CD_ALPHA = 1.0
CD_BETA = 0.1  # matches generate_llava_bench_icd.py's llava-bench convention
LOG_BETA = math.log(CD_BETA)
NEG_INF = float("-inf")

BENCH_DATA = HERE.parent.parent / "jaccard_experiment" / "data" / "llava_bench"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode", choices=["greedy", "sample"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--question-file", default=str(BENCH_DATA / "questions.jsonl"))
    ap.add_argument("--image-folder", default=str(BENCH_DATA / "images"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="only process the first N questions (smoke test)")
    args = ap.parse_args()

    set_seed(args.seed)
    print(f"[qwen-bench-sid] decode={args.decode} seed={args.seed} "
          f"cd_alpha={CD_ALPHA} cd_beta={CD_BETA}", flush=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager").eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    device = model.device
    eos_id = processor.tokenizer.eos_token_id

    sid_state = SIDState()
    install_sid_hooks(model, sid_state)

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
        inp = build_inputs(processor, img, qs, device)

        expert = Branch(model, inp, device)
        sid_state.active = True
        sid_state.image_positions = expert.image_positions
        sid_state.masked_cols = None
        amateur = Branch(model, inp, device)
        sid_state.active = False

        chosen_ids = []
        for step in range(MAX_NEW_TOKENS):
            E = expert.logits
            sid_state.active = True
            A = amateur.logits
            sid_state.active = False

            cutoff = LOG_BETA + E.max().item()
            mask = E < cutoff
            scored = (1 + CD_ALPHA) * E - CD_ALPHA * A
            scored = scored.clone(); scored[mask] = NEG_INF

            chosen_id = select_token(scored, args.decode)
            chosen_ids.append(chosen_id)
            if chosen_id == eos_id:
                break

            expert.step(chosen_id)
            sid_state.active = True
            sid_state.image_positions = expert.image_positions
            sid_state.masked_cols = None
            amateur.step(chosen_id)
            sid_state.active = False

        outputs = processor.tokenizer.decode(chosen_ids, skip_special_tokens=True).strip()

        out_f.write(json.dumps({
            "question_id": qid, "prompt": qs, "text": outputs,
            "answer_id": f"sid-{qid}", "model_id": "qwen2.5-vl-7b-sid",
            "metadata": {"cd_alpha": CD_ALPHA, "cd_beta": CD_BETA, "decode": args.decode},
        }) + "\n")
        out_f.flush()
        n += 1
        if n % 10 == 0:
            el = time.time() - t0
            print(f"  [{n}] {el:.0f}s ({el/n:.1f}s/q)", flush=True)

    out_f.close()
    print(f"[qwen-bench-sid] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
