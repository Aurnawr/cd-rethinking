"""
gen_qwen.py -- generate 500 CHAIR captions for Qwen2.5-VL-7B-Instruct under one
of four decoding methods, with a fixed random seed. Caption-only.

Methods
  greedy : plain expert argmax (undefended baseline; deterministic).
  vcd    : real VCD. amateur = diffusion-noised image (noise_step=500), computed
           cache-less each step inside a LogitsProcessor. C = 2E - A, then APC.
  sid    : real SID (official recipe). amateur = clean image with attention
           pruned from layer AGG_LAYER onward to a random 72/576 subset of image
           tokens, injected as a 4D mask under sdpa. C = 2E - A, then APC.
  proxy  : NO amateur branch. add independent Gaussian noise N(mu, sigma^2) to
           EVERY vocabulary logit of the expert, then the SAME APC, then argmax.
           mu, sigma from proxy_stats_qwen.json for the chosen --stats-source.

Seeding: --seed sets torch/np/python RNG so vcd's noise, sid's kept subset, and
proxy's Gaussian noise vary with the seed. Greedy ignores it.

Usage:
  python gen_qwen.py --method proxy --stats-source vcd --seed 0 --out OUT.jsonl
  python gen_qwen.py --method sid --seed 0 --out OUT.jsonl
Add --n-images 3 for a quick smoke test.
"""
import argparse, json, math, os, sys, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
import torch
from PIL import Image
from transformers import (Qwen2_5_VLForConditionalGeneration, AutoProcessor,
                          LogitsProcessor, LogitsProcessorList)
from qwen_vl_utils import process_vision_info

MODEL_PATH = str(ROOT / "models" / "Qwen2.5-VL-7B-Instruct")
IMAGE_DIR = ROOT / "data" / "coco" / "val2017"
IMAGE_IDS = json.load(open(ROOT / "image_ids_500.json"))
STATS = json.load(open(ROOT / "proxy_stats_qwen.json"))

IMAGE_TOKEN_ID = 151655
AGG_LAYER = 2
SID_KEEP_FRAC = 72.0 / 576.0
MAX_NEW_TOKENS = 256
CD_ALPHA, CD_BETA = 1.0, 0.2
LOG_BETA = math.log(CD_BETA)
PROMPT = "Describe this image in detail."
NEG_INF = float("-inf")


def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np; np.random.seed(seed)
    except Exception:
        pass


def add_diffusion_noise(x, t=500):
    betas = torch.sigmoid(torch.linspace(-6, 6, 1000)) * (0.5e-2 - 1e-5) + 1e-5
    ap = torch.cumprod(1 - betas, 0)
    return torch.sqrt(ap[t]) * x + torch.sqrt(1 - ap[t]) * torch.randn_like(x)


class SIDState:
    def __init__(self):
        self.active = False
        self.masked_cols = None


def install_sid_hooks(model, state):
    """Per-layer 4D attention-mask injection under sdpa: from AGG_LAYER onward,
    block all image key-columns except the kept subset (official SID recipe)."""
    minv = torch.finfo(model.dtype).min

    def mk(idx):
        def hook(mod, args, kwargs):
            if not state.active or idx < AGG_LAYER or state.masked_cols is None:
                return None
            hs = args[0] if args else kwargs.get("hidden_states")
            L = hs.shape[1]
            m4 = torch.zeros(1, 1, L, L, dtype=model.dtype, device=hs.device)
            m4[0, 0][torch.triu(torch.ones(L, L, dtype=torch.bool, device=hs.device), 1)] = minv
            m4[0, 0][:, state.masked_cols] = minv
            kwargs["attention_mask"] = m4
            return args, kwargs
        return hook
    for i, layer in enumerate(model.model.language_model.layers):
        layer.register_forward_pre_hook(mk(i), with_kwargs=True)


class CDProcessor(LogitsProcessor):
    """vcd/sid: recompute the amateur cache-less each step, C = 2E - A, then APC."""
    def __init__(self, model, method, pv_cd, thw, sid_state):
        self.model, self.method, self.pv_cd, self.thw, self.sid = model, method, pv_cd, thw, sid_state

    def __call__(self, input_ids, scores):
        E = scores[0].float()
        if self.method == "sid":
            self.sid.active = True
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                             pixel_values=self.pv_cd, image_grid_thw=self.thw, use_cache=False)
        if self.method == "sid":
            self.sid.active = False
        A = out.logits[0, -1].float()
        cd = (1 + CD_ALPHA) * E - CD_ALPHA * A
        cutoff = LOG_BETA + E.max().item()
        cd[E < cutoff] = NEG_INF
        return cd.unsqueeze(0).to(scores.dtype)


class ProxyProcessor(LogitsProcessor):
    """proxy: expert + full-vocabulary Gaussian noise, then APC. No amateur."""
    def __init__(self, mu, sigma, gen, device):
        self.mu, self.sigma, self.gen, self.device = mu, sigma, gen, device

    def __call__(self, input_ids, scores):
        E = scores[0].float()
        V = E.shape[-1]
        noise = torch.randn(V, generator=self.gen, device=self.device) * self.sigma + self.mu
        scored = E + noise
        cutoff = LOG_BETA + E.max().item()
        scored[E < cutoff] = NEG_INF
        return scored.unsqueeze(0).to(scores.dtype)


def build(proc, img, device):
    m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    ii, vi = process_vision_info(m)
    return proc(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt").to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["greedy", "vcd", "sid", "proxy"], required=True)
    ap.add_argument("--stats-source", choices=["vcd", "sid"], default="vcd")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    print(f"[gen_qwen] method={args.method} stats={args.stats_source} seed={args.seed} "
          f"n={args.n_images}", flush=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda", attn_implementation="sdpa").eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=256*28*28, max_pixels=1280*28*28)
    device = model.device

    sid_state = SIDState()
    if args.method == "sid":
        install_sid_hooks(model, sid_state)

    noise_gen = torch.Generator(device=device).manual_seed(args.seed)
    if args.method == "proxy":
        mu = float(STATS[args.stats_source]["pooled_mean"])
        sigma = float(STATS[args.stats_source]["pooled_std"])
        print(f"[gen_qwen] proxy noise N(mu={mu:.4f}, sigma={sigma:.4f}) over full vocab", flush=True)

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
        inp = build(processor, img, device)

        if args.method == "greedy":
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
        else:
            if args.method == "vcd":
                pv_cd = add_diffusion_noise(inp["pixel_values"].float(), 500).to(inp["pixel_values"].dtype)
                proc = CDProcessor(model, "vcd", pv_cd, inp["image_grid_thw"], sid_state)
            elif args.method == "sid":
                pv_cd = inp["pixel_values"]
                pos = (inp.input_ids[0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
                keep = max(1, int(round(SID_KEEP_FRAC * pos.numel())))
                perm = torch.randperm(pos.numel(), device=device)
                sid_state.masked_cols = pos[perm[keep:]]
                proc = CDProcessor(model, "sid", pv_cd, inp["image_grid_thw"], sid_state)
            else:  # proxy
                proc = ProxyProcessor(mu, sigma, noise_gen, device)
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True,
                                     logits_processor=LogitsProcessorList([proc]))

        cap = processor.tokenizer.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
        out_f.write(json.dumps({"image_id": image_id, "caption": cap}) + "\n")
        out_f.flush()
        n += 1
        if n % 25 == 0:
            el = time.time() - t0
            print(f"  [{n}/{len(ids)}] {el:.0f}s ({el/n:.1f}s/img)", flush=True)

    out_f.close()
    print(f"[gen_qwen] DONE {args.method} seed={args.seed} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
