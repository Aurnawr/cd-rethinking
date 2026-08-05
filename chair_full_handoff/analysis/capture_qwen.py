"""
run_qwen_capture.py -- CHAIR captioning + per-step top-30 logit capture for
Qwen2.5-VL-7B-Instruct, using model.generate() (KV-cached main branch) plus a
LogitsProcessor that computes the amateur branch and records the top-30 of
FOUR stages each step: expert, amateur, after-subtraction (CD), after-APC-mask.

Methods:
  greedy : plain generate(), captions only (baseline)  -- fast
  vcd    : amateur = diffusion-noised image
  sid    : amateur = OFFICIAL Self-Introspective Decoding recipe (Mirage repo
           custom_modeling_llama.py `use_sid`): layers 0..AGG_LAYER-1 see the
           full image; from AGG_LAYER onward only a RANDOM 72/576 (=12.5%)
           subset of image tokens is kept, the rest masked. Implemented here
           with per-layer attention-mask injection under sdpa (eager attention
           is broken for Qwen2.5-VL in transformers 5.x). Verified to (a) change
           the output and (b) differ from all-layer masking (layers 0-1 stay full).

Contrastive (alpha=1): cd = 2*E - A ; APC cutoff = log(beta)+max(E).
Saves ONLY ids+logits per step (labelling done later at the caption level, to
avoid the subword-fragment bug). Persistent + resumable + dedup-safe.
"""
import argparse, json, math, os, time
from pathlib import Path
import torch
from PIL import Image
from transformers import (Qwen2_5_VLForConditionalGeneration, AutoProcessor,
                          LogitsProcessor, LogitsProcessorList)
from qwen_vl_utils import process_vision_info

HERE = Path(__file__).resolve().parent
REPO = Path("/teamspace/studios/this_studio/cd-rethinking")
MODEL_PATH = str(HERE / "models" / "Qwen2.5-VL-7B-Instruct")
IMAGE_DIR = REPO / "data" / "coco" / "val2017"
IMAGE_ID_FILE = REPO / "outputs" / "chair" / "llava-7b-greedy" / "captions.jsonl"
OUT_DIR = HERE / "outputs_full_topk"; OUT_DIR.mkdir(exist_ok=True)

IMAGE_TOKEN_ID = 151655
AGG_LAYER = 2
SID_KEEP_FRAC = 72.0 / 576.0
TOPK = 30
MAX_NEW_TOKENS = 256
CD_ALPHA, CD_BETA = 1.0, 0.2
LOG_BETA = math.log(CD_BETA)
PROMPT = "Describe this image in detail."
NEG_INF = float("-inf")


def add_diffusion_noise(x, t=500):
    betas = torch.sigmoid(torch.linspace(-6, 6, 1000)) * (0.5e-2 - 1e-5) + 1e-5
    ap = torch.cumprod(1 - betas, 0)
    return torch.sqrt(ap[t]) * x + torch.sqrt(1 - ap[t]) * torch.randn_like(x)


class SIDState:
    def __init__(self):
        self.active = False
        self.masked_cols = None  # image-token positions to MASK (all but kept subset)


def install_sid_hooks(model, state):
    """Per-layer attention-mask injection: from AGG_LAYER onward, block the
    masked image columns (official SID; layers 0..AGG_LAYER-1 stay full)."""
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


class CaptureCDProcessor(LogitsProcessor):
    def __init__(self, model, method, pv_cd, thw, sid_state=None):
        self.model, self.method = model, method
        self.pv_cd, self.thw, self.sid = pv_cd, thw, sid_state
        self.steps = []

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

        cd_pre = (1 + CD_ALPHA) * E - CD_ALPHA * A
        cutoff = LOG_BETA + E.max().item()
        mask = E < cutoff
        cd_post = cd_pre.clone(); cd_post[mask] = NEG_INF
        chosen = int(cd_post.argmax().item())

        ev, ei = torch.topk(E, TOPK); av, ai = torch.topk(A, TOPK)
        cv, ci = torch.topk(cd_pre, TOPK); pv, pi = torch.topk(cd_post, TOPK)
        self.steps.append({
            "step": len(self.steps), "chosen_id": chosen, "apc_cutoff": round(cutoff, 4),
            "expert_top_ids": ei.tolist(), "expert_top_logits": [round(v, 4) for v in ev.tolist()],
            "amateur_top_ids": ai.tolist(), "amateur_top_logits": [round(v, 4) for v in av.tolist()],
            "cd_top_ids": ci.tolist(), "cd_top_pre_apc_logits": [round(v, 4) for v in cv.tolist()],
            "cd_top_survives_apc": (~mask[ci]).tolist(),
            "post_apc_top_ids": pi.tolist(),
            "post_apc_top_logits": [round(v, 4) if v > -1e30 else None for v in pv.tolist()],
        })
        return cd_post.unsqueeze(0).to(scores.dtype)


def build_inputs(processor, image, device):
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(messages)
    return processor(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(device)


def load_done(p):
    return {json.loads(l)["image_id"] for l in open(p)} if os.path.exists(p) else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["greedy", "vcd", "sid"], required=True)
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--min-pixels", type=int, default=256*28*28)
    ap.add_argument("--max-pixels", type=int, default=1280*28*28)
    args = ap.parse_args()

    print(f"loading Qwen2.5-VL-7B (attn=sdpa) method={args.method}...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda", attn_implementation="sdpa")
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    device = model.device
    sid_state = SIDState()
    if args.method == "sid":
        install_sid_hooks(model, sid_state)

    image_ids = [json.loads(l)["image_id"] for l in open(IMAGE_ID_FILE)][: args.n_images]
    out_path = OUT_DIR / (f"captions_greedy.jsonl" if args.method=="greedy" else f"captions_topk_{args.method}.jsonl")
    done = load_done(out_path)
    print(f"method={args.method} images={len(image_ids)} done={len(done)}", flush=True)

    t0 = time.time(); n = 0
    for idx, iid in enumerate(image_ids):
        if iid in done: continue
        img = Image.open(IMAGE_DIR / f"{iid:012d}.jpg").convert("RGB")
        inp = build_inputs(processor, img, device)
        rec = {"image_id": iid}
        if args.method == "greedy":
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
            gen = out[0, inp.input_ids.shape[1]:]
            rec["caption"] = processor.tokenizer.decode(gen, skip_special_tokens=True).strip()
        else:
            if args.method == "vcd":
                pv_cd = add_diffusion_noise(inp["pixel_values"].float(), 500).to(inp["pixel_values"].dtype)
            else:  # sid: amateur uses the SAME clean image; degradation is via attention masking
                pv_cd = inp["pixel_values"]
                pos = (inp.input_ids[0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
                keep = max(1, int(round(SID_KEEP_FRAC * pos.numel())))
                perm = torch.randperm(pos.numel(), device=device)
                sid_state.masked_cols = pos[perm[keep:]]
            proc = CaptureCDProcessor(model, args.method, pv_cd, inp["image_grid_thw"], sid_state)
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True,
                                     logits_processor=LogitsProcessorList([proc]))
            gen = out[0, inp.input_ids.shape[1]:]
            rec["caption"] = processor.tokenizer.decode(gen, skip_special_tokens=True).strip()
            rec["steps"] = proc.steps
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        n += 1; el = time.time()-t0
        ns = len(rec.get("steps", [])) or len(gen)
        print(f"[{idx+1}/{len(image_ids)}] img={iid} steps={ns} ({el:.1f}s, {el/n:.1f}s/img)", flush=True)
    print(f"ALL DONE {args.method} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
