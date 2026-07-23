"""
gen_llava.py -- generate 500 CHAIR captions for LLaVA-1.5-7B under one of four
decoding methods, with a fixed random seed. Caption-only (no top-30 capture),
so files stay small and many seeds are cheap.

Methods
  greedy : plain expert argmax (undefended baseline; deterministic).
  vcd    : real Visual Contrastive Decoding. amateur = diffusion-noised image
           (noise_step=500). CD score C = 2E - A, then APC, then argmax.
  sid    : real Self-Introspective Decoding. amateur = attention-pruned image
           (vendored llava use_sid path). C = 2E - A, then APC, then argmax.
  proxy  : NO amateur branch. Start from expert logits E, add independent
           Gaussian noise N(mu, sigma^2) to EVERY vocabulary logit (full
           vocabulary, not just object tokens), then apply the SAME APC (gate
           on the raw expert logit) and argmax. mu, sigma come from
           proxy_stats_llava.json for the chosen --stats-source (vcd or sid).

Seeding: --seed sets torch/np/python RNG, so vcd's noise draw, sid's kept
subset, and proxy's Gaussian noise all vary with the seed. Greedy ignores it.

Usage:
  python gen_llava.py --method proxy --stats-source vcd --seed 0 --out OUT.jsonl
  python gen_llava.py --method vcd --seed 0 --out OUT.jsonl
  python gen_llava.py --method greedy --out OUT.jsonl
Add --n-images 3 for a quick smoke test.
"""
import argparse, json, math, os, sys, types, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                          # gaussian_proxy_handoff/
sys.path.insert(0, str(HERE))               # so "llava" and "eval" import
import torch
from PIL import Image

# --- vendored llava package shim (same as the original capture script) ---
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

MODEL_PATH = str(ROOT / "models" / "llava-v1.5-7b")
IMAGE_DIR = ROOT / "data" / "coco" / "val2017"
IMAGE_IDS = json.load(open(ROOT / "image_ids_500.json"))
STATS = json.load(open(ROOT / "proxy_stats_llava.json"))

MAX_NEW_TOKENS = 256
CD_ALPHA = 1.0
CD_BETA = 0.2
LOG_BETA = math.log(CD_BETA)
NOISE_STEP = 500
PROMPT = "Describe this image in detail."


def add_diffusion_noise(image_tensor, noise_step):
    # identical schedule to the original VCD implementation
    num_steps = 1000
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0).to(image_tensor.device)

    def q_x(x_0, t):
        noise = torch.randn_like(x_0)
        a = alphas_prod[t].sqrt()
        am1 = (1 - alphas_prod[t]).sqrt()
        return a * x_0 + am1 * noise

    return q_x(image_tensor.clone(), noise_step)


def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np; np.random.seed(seed)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["greedy", "vcd", "sid", "proxy"], required=True)
    ap.add_argument("--stats-source", choices=["vcd", "sid"], default="vcd",
                    help="which measured d distribution the proxy noise is matched to")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    print(f"[gen_llava] method={args.method} stats={args.stats_source} seed={args.seed} "
          f"n={args.n_images}", flush=True)

    model_name = get_model_name_from_path(MODEL_PATH)
    tokenizer, model, image_processor, _ = load_pretrained_model(MODEL_PATH, None, model_name)
    model.eval()
    device = next(model.parameters()).device
    dtype = model.lm_head.weight.dtype
    vocab_size = model.config.vocab_size

    # proxy noise generator (seeded, reproducible) on the model device
    noise_gen = torch.Generator(device=device).manual_seed(args.seed)
    if args.method == "proxy":
        mu = float(STATS[args.stats_source]["pooled_mean"])
        sigma = float(STATS[args.stats_source]["pooled_std"])
        print(f"[gen_llava] proxy noise N(mu={mu:.4f}, sigma={sigma:.4f}) over full vocab", flush=True)

    qs = DEFAULT_IMAGE_TOKEN + "\n" + PROMPT
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    prompt_ids_base = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    eos_id = tokenizer.eos_token_id or 2

    ids = IMAGE_IDS[: args.n_images]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["image_id"] for l in open(args.out)}
    out_f = open(args.out, "a")

    import time; t0 = time.time(); n = 0
    for image_id in ids:
        if image_id in done:
            continue
        # per-image reseed: reproducible and resume-safe; still varies with --seed
        iseed = (args.seed * 1_000_003 + image_id) % (2**31 - 1)
        torch.manual_seed(iseed)
        noise_gen.manual_seed(iseed)
        img = Image.open(IMAGE_DIR / f"{image_id:012d}.jpg").convert("RGB")
        img_tensor = image_processor.preprocess(img, return_tensors="pt")["pixel_values"].to(dtype).to(device)

        prompt_ids = prompt_ids_base.to(device)
        two_branch = args.method in ("vcd", "sid")
        noisy = None
        if args.method == "vcd":
            noisy = add_diffusion_noise(img_tensor.squeeze(0).float(), NOISE_STEP).unsqueeze(0).to(dtype).to(device)

        expert_pkv = amateur_pkv = None
        attn_e = attn_a = None
        chosen_ids = []

        for step in range(MAX_NEW_TOKENS):
            if step == 0:
                expert_inp = dict(input_ids=prompt_ids, images=img_tensor, use_cache=True, return_dict=True)
                if two_branch:
                    if args.method == "vcd":
                        amateur_inp = dict(input_ids=prompt_ids, images=noisy, use_cache=True, return_dict=True)
                    else:
                        amateur_inp = dict(input_ids=prompt_ids, images=img_tensor, use_sid=True,
                                           use_cache=True, return_dict=True)
            else:
                new_tok = torch.tensor([[chosen_ids[-1]]], dtype=torch.long, device=device)
                expert_inp = dict(input_ids=new_tok, past_key_values=expert_pkv, attention_mask=attn_e,
                                  use_cache=True, return_dict=True)
                if two_branch:
                    amateur_inp = dict(input_ids=new_tok, past_key_values=amateur_pkv, attention_mask=attn_a,
                                       use_cache=True, return_dict=True)
                    if args.method == "sid":
                        amateur_inp["use_sid"] = True

            with torch.no_grad():
                expert_out = model(**expert_inp)
                E = expert_out.logits[0, -1].float()
                if two_branch:
                    amateur_out = model(**amateur_inp)
                    A = amateur_out.logits[0, -1].float()

            cutoff = LOG_BETA + E.max().item()
            mask = E < cutoff                       # APC gates on RAW expert logit
            if args.method == "greedy":
                scored = E.clone()                  # plain baseline: no APC, no noise
            elif two_branch:
                scored = (1 + CD_ALPHA) * E - CD_ALPHA * A
                scored[mask] = float("-inf")
            else:  # proxy: full-vocabulary Gaussian noise on expert logits + APC
                noise = torch.randn(E.shape[-1], generator=noise_gen, device=device) * sigma + mu
                scored = E + noise.float()
                scored[mask] = float("-inf")
            chosen_id = int(scored.argmax().item())

            expert_pkv = expert_out.past_key_values
            if two_branch:
                amateur_pkv = amateur_out.past_key_values
            if step == 0:
                exp_len = expert_pkv[0][0].shape[2]
                attn_e = torch.ones(1, exp_len + 1, device=device, dtype=torch.long)
                if two_branch:
                    attn_a = torch.ones(1, amateur_pkv[0][0].shape[2] + 1, device=device, dtype=torch.long)
            else:
                attn_e = torch.cat([attn_e, torch.ones(1, 1, device=device, dtype=torch.long)], dim=-1)
                if two_branch:
                    attn_a = torch.cat([attn_a, torch.ones(1, 1, device=device, dtype=torch.long)], dim=-1)

            chosen_ids.append(chosen_id)
            if chosen_id == eos_id:
                break

        caption = tokenizer.decode(chosen_ids, skip_special_tokens=True).strip()
        out_f.write(json.dumps({"image_id": image_id, "caption": caption}) + "\n")
        out_f.flush()
        n += 1
        if n % 50 == 0:
            el = time.time() - t0
            print(f"  [{n}/{len(ids)}] {el:.0f}s ({el/n:.1f}s/img)", flush=True)

    out_f.close()
    print(f"[gen_llava] DONE {args.method} seed={args.seed} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
