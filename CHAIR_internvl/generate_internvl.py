"""
generate_internvl.py -- CHAIR generation for InternVL3-8B, covering the four
single-file conditions:

  --mode sample          plain expert branch, no APC mask, no amateur branch.
                         The direct-sampling baseline.
  --mode sample_star     expert branch with the APC mask applied to the raw
                         expert distribution (cutoff = log(beta) + max(E)).
                         No amateur branch, no contrastive term. This is
                         "Sample*" from Yin et al.'s Mirage paper.
  --mode capture --method vcd    real VCD: amateur branch sees a
                         diffusion-noised copy of the image.
  --mode capture --method sid    real SID: amateur branch sees the SAME image
                         but with all except the least-attended 10% of image
                         tokens blocked from attention.

ICD lives in generate_internvl_icd.py (its amateur branch needs a different
prompt, so it has a different input-construction path).

Every mode records the expert's top-10 tokens and logits at each step, and the
two-branch modes additionally record the amateur top-10 and the pre-APC
contrastive score C = (1+alpha)E - alpha*A, matching the capture schema used
by the LLaVA and Qwen packages so the same downstream analysis runs unchanged.

MODEL FACTS (established by probe_internvl.py against the real checkpoint,
not assumed -- see CHAIR_internvl/probe_internvl.py):
  * decoder layers live at  model.language_model.layers  (28 Qwen2DecoderLayer)
  * the image token is <IMG_CONTEXT>, id 151667; a 12-tile image expands to
    3328 image tokens inside a ~3380-token prompt
  * attn_implementation="eager" does return real attention weights, shape
    (batch, heads, q_len, kv_len) -- SDPA never materialises them, so eager is
    mandatory for SID
  * a KV-cached continuation reproduces a stateless recompute to within bf16
    noise with matching argmax, so both branches can be cached

Because InternVL3-8B uses a Qwen2.5-7B LLM backbone with ordinary 1D RoPE,
there is none of the multimodal-RoPE position bookkeeping that Qwen2.5-VL
needed: a branch is just its own KV cache, and positions advance normally.

CRITICAL SETTING: this checkpoint ships "crop_to_patches": false in
preprocessor_config.json. Left at that default the processor skips dynamic
tiling and squashes the whole image into ONE 448x448 tile (256 visual tokens
instead of up to 3328), silently, with no warning. It is forced on and then
verified below.

Usage:
  python generate_internvl.py --mode sample      --decode sample --out out.jsonl
  python generate_internvl.py --mode sample_star --decode sample --cd-beta 0.1 --out out.jsonl
  python generate_internvl.py --mode capture --method vcd --decode sample --cd-beta 0.1 --out out.jsonl
  python generate_internvl.py --mode capture --method sid --decode sample --cd-beta 0.2 --out out.jsonl
"""
import argparse, json, math, os, random, time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

MODEL_PATH = str(Path.home() / "models" / "InternVL3-8B-hf")
IMAGE_DIR = REPO / "CHAIR_sample_star" / "data" / "coco" / "val2017"
IMAGE_IDS_FILE = REPO / "CHAIR_sample_star" / "image_ids_500.json"

IMAGE_TOKEN_ID = 151667          # <IMG_CONTEXT>, verified by probe
AGG_LAYER = 2                    # same layer index used for LLaVA and Qwen SID
SID_KEEP_FRAC = 0.10             # paper: keep the least-attended 10% of image tokens
NOISE_STEP = 500                 # VCD forward-diffusion step
MAX_NEW_TOKENS = 256
CD_ALPHA = 1.0
TOPK = 10
PROMPT = "Describe this image in detail."
NEG_INF = float("-inf")

# Same system prompt the InternVL Jaccard runs used, so all InternVL work in
# this project is under identical instructions.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Always respond in English only, in plain text "
    "without any markdown formatting (no bold, no headers, no bullet symbols)."
)


def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_diffusion_noise(x, t=NOISE_STEP):
    """Forward-diffusion noise, elementwise and therefore shape-agnostic --
    applies unchanged to InternVL's (num_tiles, 3, 448, 448) pixel_values."""
    betas = torch.sigmoid(torch.linspace(-6, 6, 1000)) * (0.5e-2 - 1e-5) + 1e-5
    ap = torch.cumprod(1 - betas, 0)
    return torch.sqrt(ap[t]) * x + torch.sqrt(1 - ap[t]) * torch.randn_like(x)


class SIDState:
    """Shared mutable state for the SID hooks. `active` gates everything, so
    the hooks are inert for the expert branch and only fire on the amateur
    branch's forward calls."""
    def __init__(self):
        self.active = False
        self.image_positions = None   # absolute column indices of image tokens
        self.masked_cols = None       # this step's pruned columns


def install_sid_hooks(model, state, agg_layer=AGG_LAYER, keep_frac=SID_KEEP_FRAC):
    """Capture attention at AGG_LAYER, pick the least-attended image tokens,
    and block every other image-token column in all later layers.

    Selection is the paper's mechanism: average the attention over heads, read
    the row for the current query position, restrict to image-token columns,
    and keep the bottom `keep_frac` by attention (topk(largest=False)). This is
    NOT a random subset -- an earlier Qwen capture in this project used
    torch.randperm here and produced a different method under the same name.
    """
    layers = model.model.language_model.layers

    def agg_pre(mod, args, kwargs):
        if not state.active:
            return None
        kwargs["output_attentions"] = True
        return args, kwargs

    def agg_post(mod, args, kwargs, output):
        if not state.active or state.image_positions is None:
            return output
        if not (isinstance(output, tuple) and len(output) > 1) or output[1] is None:
            raise RuntimeError(
                "SID needs attention weights at AGG_LAYER but got none -- is the "
                "model loaded with attn_implementation='eager'? SDPA never "
                "materialises the attention matrix.")
        attn = output[1]                      # (batch, heads, q_len, kv_len)
        last = attn.mean(dim=1)[0, -1]        # mean over heads, current query row
        img_pos = state.image_positions
        scores = last[img_pos]
        keep_n = max(1, int(round(keep_frac * img_pos.numel())))
        keep_idx = scores.topk(keep_n, largest=False).indices   # LEAST attended
        keep = torch.zeros(img_pos.numel(), dtype=torch.bool, device=img_pos.device)
        keep[keep_idx] = True
        state.masked_cols = img_pos[~keep]
        return output

    def later_pre(mod, args, kwargs):
        if not state.active or state.masked_cols is None or state.masked_cols.numel() == 0:
            return None
        am = kwargs.get("attention_mask")
        if am is None:
            return None
        am2 = am.clone()
        am2[..., state.masked_cols] = torch.finfo(am2.dtype).min
        kwargs["attention_mask"] = am2
        return args, kwargs

    layers[agg_layer].self_attn.register_forward_pre_hook(agg_pre, with_kwargs=True)
    layers[agg_layer].self_attn.register_forward_hook(agg_post, with_kwargs=True)
    for i in range(agg_layer + 1, len(layers)):
        layers[i].self_attn.register_forward_pre_hook(later_pre, with_kwargs=True)


def load_model_and_processor(model_path):
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()
    processor = AutoProcessor.from_pretrained(
        model_path, crop_to_patches=True, min_patches=1, max_patches=12)
    ip = processor.image_processor
    ip.crop_to_patches, ip.min_patches, ip.max_patches = True, 1, 12
    if not getattr(ip, "crop_to_patches", False):
        raise RuntimeError(
            "crop_to_patches is still disabled -- InternVL would run on a single "
            "squashed tile (256 visual tokens instead of up to 3328), silently "
            "ruining the comparison. Check the transformers image-processor API.")
    return model, processor


def build_inputs(processor, model, image, question):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=image, text=[text], return_tensors="pt").to(model.device)
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    return inputs


class Branch:
    """One decoding branch with its own KV cache. InternVL's Qwen2.5 backbone
    uses ordinary 1D RoPE, so nothing beyond the cache needs tracking."""

    def __init__(self, model, inputs, device):
        self.model, self.device = model, device
        input_ids = inputs["input_ids"]
        self.attn_mask = inputs["attention_mask"].clone()
        self.image_positions = (input_ids[0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=self.attn_mask,
                        pixel_values=inputs["pixel_values"],
                        use_cache=True, return_dict=True)
        self.cache = out.past_key_values
        self.logits = out.logits[0, -1].float()

    def step(self, token_id):
        new_tok = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        self.attn_mask = torch.cat(
            [self.attn_mask, torch.ones(1, 1, dtype=self.attn_mask.dtype, device=self.device)],
            dim=-1)
        with torch.no_grad():
            out = self.model(input_ids=new_tok, attention_mask=self.attn_mask,
                             past_key_values=self.cache, use_cache=True, return_dict=True)
        self.cache = out.past_key_values
        self.logits = out.logits[0, -1].float()


def select_token(scored, decode):
    if decode == "greedy":
        return int(scored.argmax().item())
    return int(torch.multinomial(torch.softmax(scored, dim=-1), num_samples=1).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sample", "sample_star", "capture"], required=True)
    ap.add_argument("--method", choices=["vcd", "sid"], help="required for --mode capture")
    ap.add_argument("--decode", choices=["greedy", "sample"], default="sample")
    ap.add_argument("--cd-beta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--model-path", default=MODEL_PATH)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.mode == "capture" and not args.method:
        ap.error("--mode capture requires --method {vcd,sid}")

    log_beta = math.log(args.cd_beta) if args.cd_beta > 0 else NEG_INF
    two_branch = args.mode == "capture"
    set_seed(args.seed)
    print(f"[internvl] mode={args.mode} method={args.method} decode={args.decode} "
          f"cd_beta={args.cd_beta} seed={args.seed} n={args.n_images}", flush=True)

    model, processor = load_model_and_processor(args.model_path)
    device = model.device
    eos_id = processor.tokenizer.eos_token_id

    sid_state = SIDState()
    if two_branch and args.method == "sid":
        install_sid_hooks(model, sid_state)

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
        inp = build_inputs(processor, model, img, PROMPT)

        expert = Branch(model, inp, device)
        amateur = None
        if two_branch:
            if args.method == "vcd":
                inp_cd = dict(inp)
                inp_cd["pixel_values"] = add_diffusion_noise(
                    inp["pixel_values"].float()).to(inp["pixel_values"].dtype)
                amateur = Branch(model, inp_cd, device)
            else:
                sid_state.active = True
                sid_state.image_positions = expert.image_positions
                sid_state.masked_cols = None
                amateur = Branch(model, inp, device)
                sid_state.active = False

        chosen_ids, steps_out = [], []
        for step in range(MAX_NEW_TOKENS):
            E = expert.logits
            cutoff = log_beta + E.max().item()
            mask = E < cutoff

            if two_branch:
                if args.method == "sid":
                    sid_state.active = True
                A = amateur.logits
                if args.method == "sid":
                    sid_state.active = False
                cd_pre = (1 + CD_ALPHA) * E - CD_ALPHA * A
                scored = cd_pre.clone(); scored[mask] = NEG_INF
            elif args.mode == "sample_star":
                scored = E.clone(); scored[mask] = NEG_INF
            else:                       # plain sampling baseline, no APC
                scored = E.clone()

            chosen_id = select_token(scored, args.decode)

            ev, ei = torch.topk(E, TOPK)
            rec = {"step": step, "chosen_id": chosen_id,
                   "expert_top_ids": ei.tolist(),
                   "expert_top_logits": [round(v, 4) for v in ev.tolist()]}
            if args.mode != "sample":
                rec["apc_cutoff"] = round(cutoff, 4)
            if two_branch:
                av, ai = torch.topk(A, TOPK)
                cv, ci = torch.topk(cd_pre, TOPK)
                rec.update({
                    "amateur_top_ids": ai.tolist(),
                    "amateur_top_logits": [round(v, 4) for v in av.tolist()],
                    "cd_top_ids": ci.tolist(),
                    "cd_top_pre_apc_logits": [round(v, 4) for v in cv.tolist()],
                    "cd_top_survives_apc": (~mask[ci]).tolist(),
                })
            elif args.mode == "sample_star":
                rec["expert_top_survives_apc"] = (~mask[ei]).tolist()
            steps_out.append(rec)

            chosen_ids.append(chosen_id)
            if chosen_id == eos_id:
                break

            expert.step(chosen_id)
            if two_branch:
                if args.method == "sid":
                    sid_state.active = True
                    sid_state.image_positions = expert.image_positions
                    sid_state.masked_cols = None
                amateur.step(chosen_id)
                if args.method == "sid":
                    sid_state.active = False

        out_f.write(json.dumps({
            "image_id": image_id,
            "method": args.method or args.mode,
            "decode": args.decode,
            "cd_beta": args.cd_beta,
            "caption": processor.tokenizer.decode(chosen_ids, skip_special_tokens=True).strip(),
            "steps": steps_out,
        }) + "\n")
        out_f.flush()
        n += 1
        if n % 10 == 0:
            el = time.time() - t0
            print(f"  [{n}] {el:.0f}s ({el/n:.1f}s/img)", flush=True)

    out_f.close()
    print(f"[internvl] DONE {args.mode}/{args.method or '-'}/{args.decode} "
          f"beta={args.cd_beta} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
