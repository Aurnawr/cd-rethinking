"""
generate_internvl_icd.py -- ICD (Instruction Contrastive Decoding,
arXiv:2403.18715) on the CHAIR captioning task for InternVL3-8B.

ICD needs its own file because its amateur branch differs from the expert in
the PROMPT rather than the image: the disturbance instruction is prefixed to
the question, which changes the token sequence and therefore the prompt length.
Everything else -- the KV-cached Branch, the capture schema, the APC mask --
is imported unchanged from generate_internvl.py.

Following the official implementation, the disturbance prompts are used as
SEPARATE full passes over the dataset (one prompt per run), not mixed within a
run, so --prompt-key selects one.

Note the differing prompt lengths between branches are harmless here:
InternVL3-8B has a Qwen2.5-7B backbone with ordinary 1D RoPE, so each branch
just advances its own positions inside its own cache. (This is the case that
caused real trouble on Qwen2.5-VL, whose multimodal RoPE cached a shared
position delta on the model object; InternVL has no such shared state.)

Usage:
  python generate_internvl_icd.py --prompt-key p1 --decode sample --cd-beta 0.1 --out out.jsonl
"""
import argparse, json, math, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch
from PIL import Image

from generate_internvl import (
    MODEL_PATH, IMAGE_DIR, IMAGE_IDS_FILE, PROMPT, MAX_NEW_TOKENS, CD_ALPHA,
    TOPK, NEG_INF, Branch, build_inputs, load_model_and_processor,
    select_token, set_seed,
)

# Verbatim from the official ICD repo (github.com/p1k0pan/ICD), identical to
# the pool used by this project's LLaVA and Qwen ICD runs.
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
    ap.add_argument("--decode", choices=["greedy", "sample"], default="sample")
    ap.add_argument("--cd-beta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--model-path", default=MODEL_PATH)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    log_beta = math.log(args.cd_beta) if args.cd_beta > 0 else NEG_INF
    disturbance = DISTURBANCE_PROMPTS[args.prompt_key]
    set_seed(args.seed)
    print(f"[internvl-icd] prompt_key={args.prompt_key} disturbance={disturbance!r} "
          f"decode={args.decode} cd_beta={args.cd_beta} seed={args.seed} "
          f"n={args.n_images}", flush=True)

    model, processor = load_model_and_processor(args.model_path)
    device = model.device
    eos_id = processor.tokenizer.eos_token_id

    image_ids = json.load(open(IMAGE_IDS_FILE))[: args.n_images]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = {json.loads(l)["image_id"] for l in open(args.out)} if os.path.exists(args.out) else set()
    out_f = open(args.out, "a")

    t0, n = time.time(), 0
    for image_id in image_ids:
        if image_id in done:
            continue
        torch.manual_seed((args.seed * 1_000_003 + image_id) % (2**31 - 1))

        img = Image.open(IMAGE_DIR / f"{image_id:012d}.jpg").convert("RGB")
        inp_e = build_inputs(processor, model, img, PROMPT)
        inp_a = build_inputs(processor, model, img, disturbance + " " + PROMPT)

        expert = Branch(model, inp_e, device)
        amateur = Branch(model, inp_a, device)

        chosen_ids, steps_out = [], []
        for step in range(MAX_NEW_TOKENS):
            E, A = expert.logits, amateur.logits
            cutoff = log_beta + E.max().item()
            mask = E < cutoff
            cd_pre = (1 + CD_ALPHA) * E - CD_ALPHA * A
            scored = cd_pre.clone(); scored[mask] = NEG_INF

            chosen_id = select_token(scored, args.decode)

            ev, ei = torch.topk(E, TOPK)
            av, ai = torch.topk(A, TOPK)
            cv, ci = torch.topk(cd_pre, TOPK)
            steps_out.append({
                "step": step, "chosen_id": chosen_id, "apc_cutoff": round(cutoff, 4),
                "expert_top_ids": ei.tolist(),
                "expert_top_logits": [round(v, 4) for v in ev.tolist()],
                "amateur_top_ids": ai.tolist(),
                "amateur_top_logits": [round(v, 4) for v in av.tolist()],
                "cd_top_ids": ci.tolist(),
                "cd_top_pre_apc_logits": [round(v, 4) for v in cv.tolist()],
                "cd_top_survives_apc": (~mask[ci]).tolist(),
            })

            chosen_ids.append(chosen_id)
            if chosen_id == eos_id:
                break
            expert.step(chosen_id)
            amateur.step(chosen_id)

        out_f.write(json.dumps({
            "image_id": image_id, "method": "icd", "prompt_key": args.prompt_key,
            "decode": args.decode, "cd_beta": args.cd_beta,
            "caption": processor.tokenizer.decode(chosen_ids, skip_special_tokens=True).strip(),
            "steps": steps_out,
        }) + "\n")
        out_f.flush()
        n += 1
        if n % 10 == 0:
            el = time.time() - t0
            print(f"  [{n}] {el:.0f}s ({el/n:.1f}s/img)", flush=True)

    out_f.close()
    print(f"[internvl-icd] DONE {args.prompt_key}/{args.decode} beta={args.cd_beta} "
          f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
