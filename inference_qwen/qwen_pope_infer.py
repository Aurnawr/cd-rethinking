"""
POPE inference for Qwen2.5-VL-7B-Instruct with the same decoding methods as the
LLaVA pipeline: baseline `sample`, VCD, ICD, SID, and APC (sample-dagger).

Self-contained (transformers >= 4.49 env `qwen_cd`); does NOT touch the LLaVA
repo code. Uses a cache-less full-recompute contrastive loop: because POPE
answers are a single word, only a few decode steps are needed, so re-running the
full forward each step (no KV cache) keeps the two-branch contrastive logic
simple and avoids Qwen mrope/cache-coupling issues.

Methods (contrastive form:  logits = (1+a)*main - a*cd, with APC plausibility cutoff):
  - sample   : main only (standard sampling)
  - apc      : main only + APC plausibility cutoff  (sample-dagger)
  - vcd      : cd = diffusion-noised image
  - icd      : cd = negative-instruction system prompt
  - sid      : cd = same inputs but image tokens (except a random subset) masked
               from layer AGG_LAYER onward (FastV-style), per the repo's SID.
"""
import argparse
import json
import math
import os
import random

import torch
import torch.nn.functional as F
from PIL import Image
import shortuuid

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

IMAGE_TOKEN_ID = 151655
AGG_LAYER = 2
SID_KEEP_FRAC = 72.0 / 576.0  # fraction of image tokens kept in the SID cd branch

ICD_PROMPTS = [
    "You are an object detector to recognize every different objects.",
    "You are an object detector to recognize every different objects by focusing the shapes, colors and relationships of objects.",
    "I want you avoid any specific identification or categorization of the objects depicted.",
    "You are a confused objects detector to provide a fuzzy overview or impression of the image.",
    "You are an object detector to provide a general overview or impression of the image.",
]


def split_list(lst, n):
    chunk = math.ceil(len(lst) / n)
    return [lst[i:i + chunk] for i in range(0, len(lst), chunk)]


def add_diffusion_noise(image_tensor, noise_step=500):
    num_steps = 1000
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
    t = int(noise_step)
    noise = torch.randn_like(image_tensor)
    return alphas_bar_sqrt[t] * image_tensor + one_minus_alphas_bar_sqrt[t] * noise


# --------------------------------------------------------------------------- #
# SID: mask image tokens (except a random subset) from layer AGG_LAYER onward. #
# Implemented with per-layer forward pre-hooks that edit the 4D attention mask.#
# --------------------------------------------------------------------------- #
class SIDState:
    def __init__(self):
        self.active = False
        self.masked_cols = None  # 1D LongTensor of kv positions to mask


def install_sid_hooks(model, state):
    layers = model.model.layers
    min_val = torch.finfo(model.dtype).min

    def make_hook(idx):
        def hook(module, args, kwargs):
            if not state.active or idx < AGG_LAYER or state.masked_cols is None:
                return None
            am = kwargs.get("attention_mask", None)
            if am is None or am.dim() != 4:
                return None
            am = am.clone()
            am[:, :, :, state.masked_cols] = min_val
            kwargs["attention_mask"] = am
            return args, kwargs
        return hook

    for i, layer in enumerate(layers):
        layer.register_forward_pre_hook(make_hook(i), with_kwargs=True)


def build_inputs(processor, image, question, system=None):
    content = [{"type": "image", "image": image},
               {"type": "text", "text": question + " Answer the question using a single word or phrase."}]
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt")
    return inputs


@torch.inference_mode()
def generate_answer(model, processor, image, question, args, sid_state):
    device = model.device
    method = args.method

    main = build_inputs(processor, image, question).to(device)
    main_ids = main.input_ids
    main_pv = main.get("pixel_values")
    main_thw = main.get("image_grid_thw")

    cd_ids = cd_pv = cd_thw = None
    if method == "vcd":
        cd_ids, cd_thw = main_ids, main_thw
        cd_pv = add_diffusion_noise(main_pv.float(), args.noise_step).to(main_pv.dtype)
    elif method == "icd":
        cd = build_inputs(processor, image, question, system=random.choice(ICD_PROMPTS)).to(device)
        cd_ids, cd_pv, cd_thw = cd.input_ids, cd.get("pixel_values"), cd.get("image_grid_thw")
    elif method == "sid":
        cd_ids, cd_pv, cd_thw = main_ids, main_pv, main_thw
        img_pos = (main_ids[0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
        n = img_pos.numel()
        keep = max(1, int(round(SID_KEEP_FRAC * n)))
        perm = torch.randperm(n, device=device)
        masked = img_pos[perm[keep:]]  # mask all but `keep`
        sid_state.masked_cols = masked

    eos_ids = set(processor.tokenizer.eos_token_id if isinstance(processor.tokenizer.eos_token_id, list)
                  else [processor.tokenizer.eos_token_id])
    im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None:
        eos_ids.add(im_end)

    gen = []
    use_cd = method in ("vcd", "icd", "sid", "apc")
    cd_alpha, cd_beta = args.cd_alpha, args.cd_beta

    for _ in range(args.max_new_tokens):
        cur_main = main_ids if not gen else torch.cat(
            [main_ids, torch.tensor([gen], device=device)], dim=1)
        out = model(input_ids=cur_main, attention_mask=torch.ones_like(cur_main),
                    pixel_values=main_pv, image_grid_thw=main_thw, use_cache=False)
        logits = out.logits[:, -1, :]

        if method in ("vcd", "icd", "sid"):
            cd_base = cd_ids if not gen else torch.cat(
                [cd_ids, torch.tensor([gen], device=device)], dim=1)
            if method == "sid":
                sid_state.active = True
            out_cd = model(input_ids=cd_base, attention_mask=torch.ones_like(cd_base),
                           pixel_values=cd_pv, image_grid_thw=cd_thw, use_cache=False)
            if method == "sid":
                sid_state.active = False
            logits_cd = out_cd.logits[:, -1, :]
            cutoff = math.log(cd_beta) + logits.max(dim=-1, keepdim=True).values
            scores = (1 + cd_alpha) * logits - cd_alpha * logits_cd
            scores = scores.masked_fill(logits < cutoff, -float("inf"))
        elif method == "apc":
            cutoff = math.log(cd_beta) + logits.max(dim=-1, keepdim=True).values
            scores = logits.masked_fill(logits < cutoff, -float("inf"))
        else:  # sample baseline
            scores = logits

        if args.temperature > 0:
            probs = F.softmax(scores / args.temperature, dim=-1)
            if args.top_p is not None and args.top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cum = torch.cumsum(sorted_probs, dim=-1)
                mask = cum - sorted_probs > args.top_p
                sorted_probs[mask] = 0.0
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                nxt = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
            else:
                nxt = torch.multinomial(probs, 1)
            token = int(nxt.item())
        else:
            token = int(scores.argmax(dim=-1).item())

        if token in eos_ids:
            break
        gen.append(token)

    text = processor.tokenizer.decode(gen, skip_special_tokens=True).strip()
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--answers-file", required=True)
    ap.add_argument("--method", choices=["sample", "vcd", "icd", "sid", "apc"], default="sample")
    ap.add_argument("--num-chunks", type=int, default=1)
    ap.add_argument("--chunk-idx", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--cd-alpha", type=float, default=1.0)
    ap.add_argument("--cd-beta", type=float, default=0.2)
    ap.add_argument("--noise-step", type=int, default=500)
    args = ap.parse_args()

    # SID needs eager attention (its mask hook edits the 4D attention mask that
    # eager passes to each decoder layer); other methods use the faster sdpa.
    attn_impl = "eager" if args.method == "sid" else "sdpa"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation=attn_impl)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    sid_state = SIDState()
    install_sid_hooks(model, sid_state)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file))]
    questions = split_list(questions, args.num_chunks)[args.chunk_idx]
    os.makedirs(os.path.dirname(os.path.expanduser(args.answers_file)), exist_ok=True)
    ans = open(os.path.expanduser(args.answers_file), "w")

    from tqdm import tqdm
    for line in tqdm(questions, desc=os.path.basename(args.answers_file)):
        image = Image.open(os.path.join(args.image_folder, line["image"])).convert("RGB")
        text = generate_answer(model, processor, image, line["text"], args, sid_state)
        ans.write(json.dumps({"question_id": line["question_id"], "prompt": line["text"],
                              "text": text, "answer_id": shortuuid.uuid(),
                              "model_id": "Qwen2.5-VL-7B-Instruct", "metadata": {}}) + "\n")
        ans.flush()
    ans.close()


if __name__ == "__main__":
    main()
