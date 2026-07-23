"""
Qwen proxy A/B: expert-only generation + magnitude-matched random noise added to
object-category token logits (NO amateur model, NO vision contrast at all), then
the same APC cutoff. Tests whether pure noise of the right size reproduces the
real VCD/SID CHAIR degradation.

  A_vcd/A_sid : bonus ~ Normal(pooled_mean, pooled_std)  of the real VCD/SID d
  B_vcd/B_sid : bonus = bootstrap draw from the real empirical d values

Noise stats come from proxy_stats_qwen.json (Qwen's own measured d = E - A, which
is negative -> the proxy SUPPRESSES object tokens, matching real Qwen CD).
Resumable. Greedy decoding, 500 images.
"""
import argparse, json, math, os, time, random
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
OUT_DIR = HERE / "outputs_proxy"; OUT_DIR.mkdir(exist_ok=True)
CD_BETA = 0.2; LOG_BETA = math.log(CD_BETA); MAX_NEW_TOKENS = 256
PROMPT = "Describe this image in detail."
VARIANTS = ["A_vcd", "B_vcd"]  # budget-limited: both proxy types vs real VCD (add A_sid/B_sid later)


class ProxyProcessor(LogitsProcessor):
    def __init__(self, variant, stats, obj_ids, device):
        self.variant = variant; self.base = "vcd" if "vcd" in variant else "sid"
        self.s = stats[self.base]; self.obj = obj_ids; self.device = device
        self.emp = self.s["empirical_samples"] if variant.startswith("B_") else None

    def __call__(self, input_ids, scores):
        E = scores[0].float()
        n = self.obj.numel()
        if self.variant.startswith("A_"):
            bonus = torch.randn(n, device=self.device) * self.s["pooled_std"] + self.s["pooled_mean"]
        else:
            bonus = torch.tensor(random.choices(self.emp, k=n), device=self.device, dtype=torch.float32)
        scored = E.clone()
        scored[self.obj] += bonus
        cutoff = LOG_BETA + E.max().item()
        scored[E < cutoff] = float("-inf")
        return scored.unsqueeze(0).to(scores.dtype)


def build(proc, img, dev):
    m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    t = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    ii, vi = process_vision_info(m)
    return proc(text=[t], images=ii, videos=vi, return_tensors="pt").to(dev)


def load_done(p):
    return {json.loads(l)["image_id"] for l in open(p)} if os.path.exists(p) else set()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-images", type=int, default=500); args = ap.parse_args()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda", attn_implementation="sdpa").eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=256*28*28, max_pixels=1280*28*28)
    dev = model.device
    stats = json.load(open(HERE/"proxy_stats_qwen.json"))
    obj_ids = torch.tensor(json.load(open(HERE/"obj_token_ids_qwen.json")), device=dev)
    image_ids = [json.loads(l)["image_id"] for l in open(IMAGE_ID_FILE)][: args.n_images]
    print(f"images={len(image_ids)} obj_tokens={obj_ids.numel()} variants={VARIANTS}", flush=True)

    cap_paths = {v: OUT_DIR/f"captions_{v}.jsonl" for v in VARIANTS}
    done = {v: load_done(cap_paths[v]) for v in VARIANTS}
    t0 = time.time(); n = 0
    for idx, iid in enumerate(image_ids):
        need = [v for v in VARIANTS if iid not in done[v]]
        if not need: continue
        img = Image.open(IMAGE_DIR/f"{iid:012d}.jpg").convert("RGB")
        inp = build(processor, img, dev)
        for v in need:
            proc = ProxyProcessor(v, stats, obj_ids, dev)
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True,
                                     logits_processor=LogitsProcessorList([proc]))
            cap = processor.tokenizer.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
            with open(cap_paths[v], "a") as f:
                f.write(json.dumps({"image_id": iid, "caption": cap}) + "\n")
        n += 1; el = time.time()-t0
        print(f"[{idx+1}/{len(image_ids)}] img={iid} ({el:.1f}s, {el/n:.1f}s/img)", flush=True)
    print(f"ALL PROXY DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
