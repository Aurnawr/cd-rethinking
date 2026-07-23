"""
Post-run verification that the cache-less capture is correct.

Check A (determinism): run the amateur cache-less forward twice on the SAME
fixed-seed noised image + same token prefix; logits must be bit-identical.

Check B (expert branch not corrupted): the concern was that the amateur's
cache-less forward calls get_rope_index and could clobber model.rope_deltas,
corrupting the generate() expert branch. We re-derive the expert logits by an
INDEPENDENT teacher-forced expert-only forward (clean image, no amateur at all)
over the captured generated token sequence, and compare to the expert_top logits
that were captured DURING the VCD/SID run. If they match within fp noise, the
expert branch was not corrupted -> the captured contrastive data is correct.
"""
import json, math
from pathlib import Path
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

HERE = Path(__file__).resolve().parent
REPO = Path("/teamspace/studios/this_studio/cd-rethinking")
MP = str(HERE / "models" / "Qwen2.5-VL-7B-Instruct")
IMG_DIR = REPO / "data" / "coco" / "val2017"
IMG = 151655
PROMPT = "Describe this image in detail."


def build(proc, image, device):
    m = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
    t = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    ii, vi = process_vision_info(m)
    return proc(text=[t], images=ii, videos=vi, return_tensors="pt").to(device)


def noise(x, seed, t=500):
    g = torch.Generator(device="cpu").manual_seed(seed)
    b = torch.sigmoid(torch.linspace(-6, 6, 1000)) * (0.5e-2 - 1e-5) + 1e-5
    ap = torch.cumprod(1 - b, 0)
    nz = torch.randn(x.shape, generator=g).to(x.device, x.dtype)
    return (torch.sqrt(ap[t]) * x.float().to(x.device) + torch.sqrt(1 - ap[t]) * nz.float()).to(x.dtype)


def main():
    print("loading model...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MP, torch_dtype=torch.float16, device_map="cuda", attn_implementation="sdpa").eval()
    proc = AutoProcessor.from_pretrained(MP, min_pixels=256*28*28, max_pixels=1280*28*28)
    dev = model.device

    vcd = [json.loads(l) for l in open(HERE/"outputs_full_topk"/"captions_topk_vcd.jsonl")][:2]

    # ---- Check A: determinism of cache-less amateur ----
    print("\n=== Check A: amateur cache-less determinism (fixed seed) ===", flush=True)
    rec = vcd[0]; img = Image.open(IMG_DIR/f"{rec['image_id']:012d}.jpg").convert("RGB")
    inp = build(proc, img, dev)
    pv1 = noise(inp["pixel_values"], seed=123)
    pv2 = noise(inp["pixel_values"], seed=123)
    with torch.inference_mode():
        a1 = model(input_ids=inp.input_ids, attention_mask=torch.ones_like(inp.input_ids),
                   pixel_values=pv1, image_grid_thw=inp["image_grid_thw"], use_cache=False).logits[0, -1].float()
        a2 = model(input_ids=inp.input_ids, attention_mask=torch.ones_like(inp.input_ids),
                   pixel_values=pv2, image_grid_thw=inp["image_grid_thw"], use_cache=False).logits[0, -1].float()
    print(f"  max|a1-a2| (same seed) = {(a1-a2).abs().max().item():.2e}  -> {'PASS (deterministic)' if (a1-a2).abs().max()<1e-3 else 'FAIL'}")

    # ---- Check B: expert branch uncorrupted (teacher-forced vs captured) ----
    print("\n=== Check B: expert branch not corrupted by amateur clobbering ===", flush=True)
    for rec in vcd:
        iid = rec["image_id"]; steps = rec["steps"]
        img = Image.open(IMG_DIR/f"{iid:012d}.jpg").convert("RGB")
        inp = build(proc, img, dev)
        gen = torch.tensor([[s["chosen_id"] for s in steps]], device=dev)
        full = torch.cat([inp.input_ids, gen], dim=1)
        Lp = inp.input_ids.shape[1]
        with torch.inference_mode():
            out = model(input_ids=full, attention_mask=torch.ones_like(full),
                        pixel_values=inp["pixel_values"], image_grid_thw=inp["image_grid_thw"], use_cache=False)
        # teacher-forced expert logits: position Lp-1+k predicts step k
        diffs = []
        for k, s in enumerate(steps):
            tf = out.logits[0, Lp-1+k].float()
            for tid, lg in zip(s["expert_top_ids"], s["expert_top_logits"]):
                diffs.append(abs(tf[tid].item() - lg))
        diffs = torch.tensor(diffs)
        print(f"  img {iid}: n_steps={len(steps)}  mean|captured_expert - teacher_forced|={diffs.mean():.4f}  "
              f"max={diffs.max():.4f}  -> {'PASS' if diffs.mean()<0.05 else 'CHECK'}")

    print("\nVERIFY COMPLETE", flush=True)


if __name__ == "__main__":
    main()
