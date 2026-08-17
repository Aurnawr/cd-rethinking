"""
generate_qwen_icd.py -- ICD (Instruction Contrastive Decoding) on the
CHAIR/COCO captioning task, for Qwen2.5-VL-7B-Instruct. Mirrors
generate_llava_icd.py exactly (same disturbance prompts, same cd_alpha/beta,
same 5-separate-passes methodology, same capture schema).

This is exactly the case generate_qwen.py's `Branch` class was built to
handle safely: the amateur branch here has a DIFFERENT prompt length than
the expert (the disturbance prefix), so the two branches' multimodal RoPE
deltas genuinely differ. `Branch` computes and stores each branch's delta
independently and never touches the model's shared `rope_deltas` attribute,
so this is safe by construction -- see generate_qwen.py's module docstring
for the full explanation.
"""
import argparse, json, math, os, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from generate_qwen import MODEL_PATH, IMAGE_DIR, IMAGE_IDS, Branch, build_inputs, select_token, set_seed

MAX_NEW_TOKENS = 256
CD_ALPHA = 1.0
CD_BETA = 0.1  # Mirage paper's Table 6 convention, matches this package's VCD/SID/Sample* -- see generate_llava_icd.py's docstring
LOG_BETA = math.log(CD_BETA)
TOPK = 10
PROMPT = "Describe this image in detail."
NEG_INF = float("-inf")

DISTURBANCE_PROMPTS = {
    "p1": "You are an object detector to recognize every different objects.",
    "p2": "You are an object detector to recognize every different objects by focusing the shapes, colors and relationships of objects.",
    "n1": "I want you avoid any specific identification or categorization of the objects depicted.",
    "n2": "You are a confused objects detector to provide a fuzzy overview or impression of the image.",
    "p3": "You are an object detector to provide a general overview or impression of the image.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-key", required=True, choices=list(DISTURBANCE_PROMPTS.keys()))
    ap.add_argument("--decode", choices=["greedy", "sample"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    disturbance_text = DISTURBANCE_PROMPTS[args.prompt_key]
    print(f"[qwen-icd] prompt_key={args.prompt_key} decode={args.decode} seed={args.seed} "
          f"n={args.n_images} cd_alpha={CD_ALPHA} cd_beta={CD_BETA}", flush=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager").eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    device = model.device
    eos_id = processor.tokenizer.eos_token_id

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = {json.loads(l)["image_id"] for l in open(args.out)} if os.path.exists(args.out) else set()
    out_f = open(args.out, "a")

    import time; t0 = time.time(); n = 0
    for image_id in IMAGE_IDS[: args.n_images]:
        if image_id in done:
            continue
        iseed = (args.seed * 1_000_003 + image_id) % (2**31 - 1)
        torch.manual_seed(iseed)

        img = Image.open(IMAGE_DIR / f"{image_id:012d}.jpg").convert("RGB")
        inp_e = build_inputs(processor, img, PROMPT, device)
        inp_a = build_inputs(processor, img, disturbance_text + " " + PROMPT, device)

        expert = Branch(model, inp_e, device)
        amateur = Branch(model, inp_a, device)

        chosen_ids, steps_out = [], []
        for step in range(MAX_NEW_TOKENS):
            E, A = expert.logits, amateur.logits
            cutoff = LOG_BETA + E.max().item()
            mask = E < cutoff
            scored = (1 + CD_ALPHA) * E - CD_ALPHA * A
            scored = scored.clone(); scored[mask] = NEG_INF

            chosen_id = select_token(scored, args.decode)

            ev, ei = torch.topk(E, TOPK); av, ai = torch.topk(A, TOPK)
            cd_pre = (1 + CD_ALPHA) * E - CD_ALPHA * A
            cv, ci = torch.topk(cd_pre, TOPK)
            steps_out.append({
                "step": step, "chosen_id": chosen_id, "apc_cutoff": round(cutoff, 4),
                "expert_top_ids": ei.tolist(), "expert_top_logits": [round(v, 4) for v in ev.tolist()],
                "amateur_top_ids": ai.tolist(), "amateur_top_logits": [round(v, 4) for v in av.tolist()],
                "cd_top_ids": ci.tolist(), "cd_top_pre_apc_logits": [round(v, 4) for v in cv.tolist()],
                "cd_top_survives_apc": (~mask[ci]).tolist(),
            })

            chosen_ids.append(chosen_id)
            if chosen_id == eos_id:
                break
            expert.step(chosen_id)
            amateur.step(chosen_id)

        rec = {"image_id": image_id, "method": "icd", "prompt_key": args.prompt_key, "decode": args.decode,
               "caption": processor.tokenizer.decode(chosen_ids, skip_special_tokens=True).strip(),
               "steps": steps_out}
        out_f.write(json.dumps(rec) + "\n"); out_f.flush()
        n += 1
        if n % 10 == 0:
            el = time.time() - t0
            print(f"  [{n}] {el:.0f}s ({el/n:.1f}s/img)", flush=True)

    out_f.close()
    print(f"[qwen-icd] DONE {args.prompt_key}/{args.decode} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
