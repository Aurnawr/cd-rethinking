"""
generate_qwen.py -- generation for Qwen2.5-VL-7B-Instruct, direct-sampling
edition, WITH a correctly KV-cached amateur branch (unlike the original
CHAIR_analysis/common/generate_qwen.py, which recomputed the amateur branch
cache-less at every step "for numerical correctness" -- see the design note
below for what that correctness issue actually was and how this script
avoids it instead of working around it).

  --mode sample_star                   single branch (expert only), APC mask
                                        applied to the raw expert distribution
                                        (no amateur branch, no contrastive term)
                                        -- "Sample*"/"Sample-dagger" from the
                                        Mirage of Performance Gains paper
  --mode capture --method {vcd,sid}    two-branch capture, same schema as
                                        generate_llava.py
  --decode {greedy,sample}             applies to both

Same hyperparameters as generate_llava.py's convention: cd_alpha=1.0,
cd_beta=0.1 (Mirage paper's Table 6 convention, NOT the sibling
CHAIR_sampling+ICD package's 0.2 -- see generate_llava.py's module
docstring), max_new_tokens=256, prompt="Describe this image in detail.",
same 500 images.

-------------------------------------------------------------------------
DESIGN NOTE -- why the amateur branch can be safely cached here:

Qwen2.5-VL uses multimodal RoPE (3D position ids: temporal/height/width for
image tokens, standard 1D for text). The model computes this once at the
first forward call of a generation and caches the result in a MUTABLE
ATTRIBUTE on the model object itself: `model.model.rope_deltas`. Every
SUBSEQUENT call in that generation reuses `self.model.rope_deltas` to cheaply
advance positions instead of recomputing the full multimodal index.

That attribute is shared by the WHOLE model object. If you run two branches
(expert, amateur) by alternating calls into the same live model -- which is
exactly what a hand-rolled two-branch loop needs to do -- the second
branch's first call overwrites `rope_deltas` with ITS OWN value, silently
corrupting the other branch's position ids on every later step. For VCD/SID
this happens to be harmless (both branches have identical token/image
structure, so they'd compute the same delta anyway) -- but for ICD, whose
amateur branch has a DIFFERENT (disturbance-prefixed) prompt length, the two
branches' correct deltas genuinely differ, and sharing the attribute would
silently feed one branch the other's position offset. This is almost
certainly what "a hand-rolled mRoPE second-branch cache diverged from the
exact computation" (the original script's docstring) was actually hitting.

The fix here: NEVER let the model compute `rope_deltas` for us. Each branch
calls `model.model.get_rope_index(...)` directly, ONCE, at its own prefill,
and stores the resulting per-branch delta in a plain local Python variable
(`ExpertBranch`/`AmateurBranch` below) -- not on the model. Every subsequent
step for that branch builds its own `position_ids` from that private delta
and passes it explicitly, so the model's `compute_3d_position_ids` auto-path
(the one that reads/writes the shared attribute) is never invoked at all.
Verified: a `[1, bsz, seq]` position_ids tensor (this script's continuation
shape) is exactly what Qwen2_5_VLRotaryEmbedding expects for pure-text
continuation -- it broadcasts to all 3 mrope dimensions, which is the
mathematically correct thing once you're past the image prefix.

SID needs one more piece: attention weights at AGG_LAYER to decide which
image tokens are least-attended (paper mechanism, not random). The model is
loaded with `attn_implementation="eager"` (required -- SDPA does not
materialize attention weights at all, cached or not), and forward
pre/post-hooks on that one layer's self-attention module request and
capture its weights, then later layers' pre-hooks clone the FRAMEWORK-BUILT
causal mask (not a hand-rolled one) and additionally block the pruned
image-token columns. This reuses the framework's own causal/padding mask
construction and only perturbs the columns SID actually needs to change.

Do not trust this without checking outputs/qwen_cache_verification.json
(written by verify_qwen_cache.py) -- it compares this cached path's logits
against the original cache-less computation on a few steps and must show
near-exact agreement before any real run.
-------------------------------------------------------------------------
"""
import argparse, json, math, os, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, DynamicCache
from qwen_vl_utils import process_vision_info

MODEL_PATH = str(REPO / "models" / "Qwen2.5-VL-7B-Instruct")
IMAGE_DIR = REPO / "data" / "coco" / "val2017"
IMAGE_IDS = json.load(open(REPO / "image_ids_500.json"))

IMAGE_TOKEN_ID = 151655
AGG_LAYER = 2
SID_KEEP_FRAC = 0.10  # paper: keep the least-attended 10% of image tokens
MAX_NEW_TOKENS = 256
CD_ALPHA, CD_BETA = 1.0, 0.1  # Mirage paper's Table 6 convention -- see module docstring
LOG_BETA = math.log(CD_BETA)
TOPK = 10
PROMPT = "Describe this image in detail."
NEG_INF = float("-inf")


def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_diffusion_noise(x, t=500):
    betas = torch.sigmoid(torch.linspace(-6, 6, 1000)) * (0.5e-2 - 1e-5) + 1e-5
    ap = torch.cumprod(1 - betas, 0)
    return torch.sqrt(ap[t]) * x + torch.sqrt(1 - ap[t]) * torch.randn_like(x)


class SIDState:
    """Shared mutable state read/written by the hooks below. `active` gates
    everything so the hooks are no-ops for the expert branch and for VCD's
    amateur branch -- only SID's amateur forward calls set active=True."""
    def __init__(self):
        self.active = False
        self.image_positions = None   # LongTensor, absolute column indices of image tokens (fixed once known)
        self.masked_cols = None       # LongTensor, this step's pruned columns (recomputed every call)


def install_sid_hooks(model, state, agg_layer=AGG_LAYER, keep_frac=SID_KEEP_FRAC):
    layers = model.model.language_model.layers

    def agg_pre_hook(module, args, kwargs):
        if not state.active:
            return None
        kwargs["output_attentions"] = True
        return args, kwargs

    def agg_post_hook(module, args, kwargs, output):
        if not state.active or state.image_positions is None:
            return output
        attn_output, attn_weights = output
        if attn_weights is None:
            raise RuntimeError("SID needs real attention weights at AGG_LAYER but got None -- "
                                "is the model loaded with attn_implementation='eager'?")
        avg = attn_weights.mean(dim=1)[0]          # (q_len, kv_len), averaged over heads
        last = avg[-1]                              # attention from the current/last query token
        img_pos = state.image_positions
        img_scores = last[img_pos]
        keep_n = max(1, int(round(keep_frac * img_pos.numel())))
        least_idx = img_scores.topk(keep_n, largest=False).indices
        keep_abs = set(img_pos[least_idx].tolist())
        masked = [p for p in img_pos.tolist() if p not in keep_abs]
        state.masked_cols = torch.tensor(masked, device=attn_weights.device, dtype=torch.long)
        return output

    def later_pre_hook(module, args, kwargs):
        if not state.active or state.masked_cols is None or state.masked_cols.numel() == 0:
            return None
        am = kwargs.get("attention_mask")
        if am is None:
            return None
        am2 = am.clone()
        am2[..., state.masked_cols] = torch.finfo(am2.dtype).min
        kwargs["attention_mask"] = am2
        return args, kwargs

    layers[agg_layer].self_attn.register_forward_pre_hook(agg_pre_hook, with_kwargs=True)
    layers[agg_layer].self_attn.register_forward_hook(agg_post_hook, with_kwargs=True)
    for i in range(agg_layer + 1, len(layers)):
        layers[i].self_attn.register_forward_pre_hook(later_pre_hook, with_kwargs=True)


def build_inputs(processor, img, text, device):
    m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": text}]}]
    chat_text = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    ii, vi = process_vision_info(m)
    return processor(text=[chat_text], images=ii, videos=vi, padding=True, return_tensors="pt").to(device)


class Branch:
    """One decoding branch (expert or amateur). Owns its own KV cache and
    its own mrope delta -- NEVER touches model.model.rope_deltas, so two
    branches sharing one model object cannot corrupt each other (see the
    module docstring)."""
    def __init__(self, model, inputs, device):
        self.model = model
        self.device = device
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        image_grid_thw = inputs.get("image_grid_thw")
        mm_token_type_ids = inputs["mm_token_type_ids"]

        position_ids, mrope_deltas = model.model.get_rope_index(
            input_ids, mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw, attention_mask=attention_mask,
        )
        self.mrope_deltas = mrope_deltas  # (bsz, 1), private to this branch

        img_pos = (input_ids[0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
        self.image_positions = img_pos

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                        pixel_values=inputs.get("pixel_values"), image_grid_thw=image_grid_thw,
                        use_cache=True, return_dict=True)
        self.cache = out.past_key_values
        self.logits = out.logits[0, -1].float()
        self.seq_len = input_ids.shape[1]
        self.attn_mask = attention_mask.clone()

    def step(self, new_token_id):
        new_tok = torch.tensor([[new_token_id]], dtype=torch.long, device=self.device)
        self.attn_mask = torch.cat(
            [self.attn_mask, torch.ones(1, 1, device=self.device, dtype=self.attn_mask.dtype)], dim=-1)
        pos = torch.full((1, self.attn_mask.shape[0], 1), self.seq_len, device=self.device, dtype=torch.long) \
              + self.mrope_deltas.view(1, -1, 1)
        with torch.no_grad():
            out = self.model(input_ids=new_tok, attention_mask=self.attn_mask, position_ids=pos,
                              past_key_values=self.cache, use_cache=True, return_dict=True)
        self.cache = out.past_key_values
        self.logits = out.logits[0, -1].float()
        self.seq_len += 1


def select_token(scored, decode):
    if decode == "greedy":
        return int(scored.argmax().item())
    probs = torch.softmax(scored, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sample_star", "capture"], required=True)
    ap.add_argument("--method", choices=["vcd", "sid"], help="required for --mode capture")
    ap.add_argument("--decode", choices=["greedy", "sample"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.mode == "capture" and not args.method:
        ap.error("--mode capture requires --method {vcd,sid}")

    set_seed(args.seed)
    print(f"[qwen] mode={args.mode} method={args.method} decode={args.decode} "
          f"seed={args.seed} n={args.n_images}", flush=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager").eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    device = model.device

    sid_state = SIDState()
    install_sid_hooks(model, sid_state)
    two_branch = args.mode == "capture"

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
        inp = build_inputs(processor, img, PROMPT, device)

        expert = Branch(model, inp, device)
        amateur = None
        if two_branch:
            if args.method == "vcd":
                inp_cd = dict(inp)
                inp_cd["pixel_values"] = add_diffusion_noise(inp["pixel_values"].float(), 500).to(inp["pixel_values"].dtype)
                amateur = Branch(model, inp_cd, device)
            else:
                sid_state.active = True
                sid_state.image_positions = expert.image_positions
                sid_state.masked_cols = None
                amateur = Branch(model, inp, device)
                sid_state.active = False

        chosen_ids, steps_out = [], []
        eos_id = processor.tokenizer.eos_token_id

        for step in range(MAX_NEW_TOKENS):
            E = expert.logits
            if two_branch:
                if args.method == "sid":
                    sid_state.active = True
                A = amateur.logits
                if args.method == "sid":
                    sid_state.active = False
                cutoff = LOG_BETA + E.max().item()
                mask = E < cutoff
                scored = (1 + CD_ALPHA) * E - CD_ALPHA * A
                scored = scored.clone(); scored[mask] = NEG_INF
            else:
                # Sample*: APC mask applied directly to the expert
                # distribution, no amateur branch, no contrastive term.
                cutoff = LOG_BETA + E.max().item()
                mask = E < cutoff
                scored = E.clone()
                scored[mask] = NEG_INF

            chosen_id = select_token(scored, args.decode)

            if two_branch:
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
            else:
                ev, ei = torch.topk(E, TOPK)
                steps_out.append({
                    "step": step, "chosen_id": chosen_id, "apc_cutoff": round(cutoff, 4),
                    "expert_top_ids": ei.tolist(), "expert_top_logits": [round(v, 4) for v in ev.tolist()],
                    "expert_top_survives_apc": (~mask[ei]).tolist(),
                })

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

        rec = {"image_id": image_id, "method": args.method or "sample_star", "decode": args.decode,
               "caption": processor.tokenizer.decode(chosen_ids, skip_special_tokens=True).strip(),
               "steps": steps_out}
        out_f.write(json.dumps(rec) + "\n"); out_f.flush()
        n += 1
        if n % 10 == 0:
            el = time.time() - t0
            print(f"  [{n}] {el:.0f}s ({el/n:.1f}s/img)", flush=True)

    out_f.close()
    print(f"[qwen] DONE {args.mode}/{args.method or 'sample_star'}/{args.decode} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
