"""
generate_llava_icd.py -- ICD (Instruction Contrastive Decoding, arXiv:2403.18715)
on the CHAIR / COCO captioning task (500 images, "Describe this image in
detail."), for LLaVA-1.5-7B.

Faithful to the official reference implementation (github.com/p1k0pan/ICD):
the amateur branch is the SAME image with a disturbance-prefixed instruction
instead of the base instruction (no image perturbation, no attention
pruning -- purely a text-side amateur). The 5 disturbance prompts are run as
5 SEPARATE full passes over the dataset (one prompt per run, via
--prompt-key), not randomly mixed within one run -- this was verified
against the official repo earlier in this project (the previously-vendored
icd_utils.py's random.choice() mixing was the bug; this script does not
repeat it).

Hyperparameters and schema are held IDENTICAL to this package's
generate_llava.py (cd_alpha=1.0, cd_beta=0.1, max_new_tokens=256, same 500
images, same top-10 capture fields) so the SAME analysis scripts
(agreement_analysis.py / contrastive_analysis.py) run on ICD's captures
unmodified, and ICD is directly comparable to VCD/SID/Sample* in this
package. cd_beta=0.1 matches the Mirage paper's Table 6 convention (VCD's
own paper default), used uniformly across Sample*/VCD/SID/ICD here so the
comparison isolates the amateur branch's contribution at a fixed mask
strength -- see generate_llava.py's module docstring.

--decode {greedy,sample} applies here exactly as in generate_llava.py.
"""
import argparse, json, math, os, sys, types, random
from pathlib import Path

HERE = Path(__file__).resolve().parent          # common/
REPO = HERE.parent                              # CHAIR_sampling+ICD/
sys.path.insert(0, str(HERE))
import torch
from PIL import Image

from transformers.models.auto.configuration_auto import CONFIG_MAPPING
if "llava" in CONFIG_MAPPING._mapping:
    del CONFIG_MAPPING._mapping["llava"]
_mpt = types.ModuleType("llava.model.language_model.llava_mpt")
_mpt.LlavaMPTForCausalLM = type("LlavaMPTForCausalLM", (), {})
_mpt.LlavaMPTConfig = type("LlavaMPTConfig", (), {})
sys.modules["llava.model.language_model.llava_mpt"] = _mpt

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

MODEL_PATH = str(REPO / "models" / "llava-v1.5-7b")
IMAGE_DIR = REPO / "data" / "coco" / "val2017"
IMAGE_IDS = json.load(open(REPO / "image_ids_500.json"))

MAX_NEW_TOKENS = 256
CD_ALPHA = 1.0
CD_BETA = 0.1  # Mirage paper's Table 6 convention, matches this package's VCD/SID/Sample* -- see module docstring
LOG_BETA = math.log(CD_BETA)
TOPK = 10
PROMPT = "Describe this image in detail."

# Verbatim from the official ICD repo (p1k0pan/ICD), byte-identical to the
# prompts already verified and used for this project's llava-bench ICD run.
DISTURBANCE_PROMPTS = {
    "p1": "You are an object detector to recognize every different objects.",
    "p2": "You are an object detector to recognize every different objects by focusing the shapes, colors and relationships of objects.",
    "n1": "I want you avoid any specific identification or categorization of the objects depicted.",
    "n2": "You are a confused objects detector to provide a fuzzy overview or impression of the image.",
    "p3": "You are an object detector to provide a general overview or impression of the image.",
}


def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_token(scored, decode):
    if decode == "greedy":
        return int(scored.argmax().item())
    probs = torch.softmax(scored, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def build_prompt_ids(tokenizer, model, qs):
    if model.config.mm_use_im_start_end:
        from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        qs_full = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs_full = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], qs_full)
    conv.append_message(conv.roles[1], None)
    return tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)


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
    print(f"[icd] prompt_key={args.prompt_key} decode={args.decode} seed={args.seed} "
          f"n={args.n_images} cd_alpha={CD_ALPHA} cd_beta={CD_BETA}", flush=True)

    model_name = get_model_name_from_path(MODEL_PATH)
    tokenizer, model, image_processor, _ = load_pretrained_model(MODEL_PATH, None, model_name)
    model.eval()
    device = next(model.parameters()).device
    dtype = model.lm_head.weight.dtype
    eos_id = tokenizer.eos_token_id or 2

    prompt_ids_base = build_prompt_ids(tokenizer, model, PROMPT)
    prompt_ids_cd_base = build_prompt_ids(tokenizer, model, disturbance_text + " " + PROMPT)

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
        img_tensor = image_processor.preprocess(img, return_tensors="pt")["pixel_values"].to(dtype).to(device)
        prompt_ids = prompt_ids_base.to(device)
        prompt_ids_cd = prompt_ids_cd_base.to(device)

        expert_pkv = amateur_pkv = attn_e = attn_a = None
        chosen_ids, steps_out = [], []

        for step in range(MAX_NEW_TOKENS):
            if step == 0:
                expert_inp = dict(input_ids=prompt_ids, images=img_tensor, use_cache=True, return_dict=True)
                amateur_inp = dict(input_ids=prompt_ids_cd, images=img_tensor, use_cache=True, return_dict=True)
            else:
                new_tok = torch.tensor([[chosen_ids[-1]]], dtype=torch.long, device=device)
                expert_inp = dict(input_ids=new_tok, past_key_values=expert_pkv, attention_mask=attn_e,
                                   use_cache=True, return_dict=True)
                amateur_inp = dict(input_ids=new_tok, past_key_values=amateur_pkv, attention_mask=attn_a,
                                    use_cache=True, return_dict=True)

            with torch.no_grad():
                expert_out = model(**expert_inp)
                E = expert_out.logits[0, -1].float()
                amateur_out = model(**amateur_inp)
                A = amateur_out.logits[0, -1].float()

            cutoff = LOG_BETA + E.max().item()
            mask = E < cutoff
            scored = (1 + CD_ALPHA) * E - CD_ALPHA * A
            scored = scored.clone(); scored[mask] = float("-inf")

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

            expert_pkv = expert_out.past_key_values
            amateur_pkv = amateur_out.past_key_values
            if step == 0:
                attn_e = torch.ones(1, expert_pkv[0][0].shape[2] + 1, device=device, dtype=torch.long)
                attn_a = torch.ones(1, amateur_pkv[0][0].shape[2] + 1, device=device, dtype=torch.long)
            else:
                attn_e = torch.cat([attn_e, torch.ones(1, 1, device=device, dtype=torch.long)], dim=-1)
                attn_a = torch.cat([attn_a, torch.ones(1, 1, device=device, dtype=torch.long)], dim=-1)

            chosen_ids.append(chosen_id)
            if chosen_id == eos_id:
                break

        rec = {"image_id": image_id, "method": "icd", "prompt_key": args.prompt_key, "decode": args.decode,
               "caption": tokenizer.decode(chosen_ids, skip_special_tokens=True).strip(),
               "steps": steps_out}
        out_f.write(json.dumps(rec) + "\n"); out_f.flush()
        n += 1
        if n % 25 == 0:
            el = time.time() - t0
            print(f"  [{n}] {el:.0f}s ({el/n:.1f}s/img)", flush=True)

    out_f.close()
    print(f"[icd] DONE {args.prompt_key}/{args.decode} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
