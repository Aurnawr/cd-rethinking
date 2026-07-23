"""
run_eda.py  --  LLaVA-1.5-7B per-token logit EDA for VCD and SID decoding
=============================================================================

For each of the 10 CHAIR benchmark images (first 10 from the 500-image sample),
runs GREEDY, VCD, and SID decoding step by step WITHOUT using the repo's
monkeypatches. Instead, a manual generation loop intercepts every forward pass
so we can capture the full logit distributions at 4 stages:

  Stage 1  expert_logit      raw logit from the visual model (full/original image)
  Stage 2  amateur_logit     raw logit from the "amateur" forward pass
                               VCD: same model, diffusion-noised image (t=500)
                               SID: same model, original image but 504/576 vision
                                    tokens masked at layers 2+ (only 72 visible)
  Stage 3  cd_logit_pre_apc  contrastive diff before APC:
                               (1+alpha)*expert - alpha*amateur   [alpha=1 default]
  Stage 4  cd_logit_post_apc cd_logit_pre_apc with entries masked to -inf wherever
                               expert_logit < log(beta) + max(expert_logits) [beta=0.2]

For every decoding step, the top-100 tokens by expert logit are saved with all
4 stage values, expert/amateur/cd probabilities, APC survival flag, and ranks.

OUTPUTS (written to outputs/ next to this script):
  outputs/greedy/img_{id}_eda.json
  outputs/vcd/img_{id}_eda.json
  outputs/sid/img_{id}_eda.json
  outputs/summary.csv                   one row per (image, method)

USAGE (run from inside llava_logit_eda/):
  python run_eda.py --model-path liuhaotian/llava-v1.5-7b

  # Or point at a local checkpoint:
  python run_eda.py --model-path /path/to/llava-v1.5-7b

  # Skip a method:
  python run_eda.py --model-path liuhaotian/llava-v1.5-7b --methods greedy vcd

PLACEMENT:
  This file must live inside the cd-rethinking repo so that
  sys.path manipulation below resolves the llava and cd_utils packages.
  Expected layout:
    cd-rethinking/
      llava/
      inference/cd_utils/vcd_utils.py
      llava_logit_eda/run_eda.py   <-- here
"""

import argparse
import csv
import json
import math
import os
import sys
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup: add cd-rethinking repo root and inference/ to sys.path so that
# `llava` and `cd_utils` import without needing a pip install -e step.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent          # cd-rethinking/
_INFERENCE  = _REPO_ROOT / "inference"

for _p in [str(_REPO_ROOT), str(_INFERENCE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Suppress the transformers "llava" AutoConfig collision warning
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    if "llava" in CONFIG_MAPPING._mapping:
        del CONFIG_MAPPING._mapping["llava"]
except Exception:
    pass

# llava/model/__init__.py imports LlavaMPTForCausalLM, which pulls in a
# vendored MPT model that depends on _expand_mask from
# transformers.models.bloom.modeling_bloom — a private API removed in
# transformers 4.36+.  We only need LLaMA, so stub the MPT module before
# any llava import so Python never loads the real broken chain.
import types as _t
_mpt = _t.ModuleType("llava.model.language_model.llava_mpt")
_mpt.LlavaMPTForCausalLM = type("LlavaMPTForCausalLM", (), {})
_mpt.LlavaMPTConfig      = type("LlavaMPTConfig", (), {})
sys.modules["llava.model.language_model.llava_mpt"] = _mpt
del _t, _mpt

from llava.model.builder import load_pretrained_model          # noqa: E402
from llava.mm_utils import (                                    # noqa: E402
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.constants import (                                   # noqa: E402
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle   # noqa: E402

# NOTE: attn_implementation="eager" and rope_scaling nulling (both needed for
# this vendored transformers~4.31-era model under transformers>=5) are now
# handled inside llava.model.builder.load_pretrained_model itself, which
# also loads the checkpoint's state dict manually — transformers>=5's
# automatic from_pretrained weight-loading silently left every LLaMA-backbone
# parameter at its random init value for this custom architecture, which was
# the actual cause of the "state state state" degenerate repetition.


# ---------------------------------------------------------------------------
# add_diffusion_noise — inlined from inference/cd_utils/vcd_utils.py so we
# do NOT have to import that module (which imports GreedySearchOutput from
# transformers.generation.utils — removed in transformers >= 4.36).
# ---------------------------------------------------------------------------
def add_diffusion_noise(image_tensor, noise_step):
    """Add diffusion noise at step `noise_step` (0-999) to an image tensor."""
    num_steps = 1000
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0, t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return alphas_t * x_0 + alphas_1_m_t * noise

    return q_x(image_tensor.clone(), noise_step)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TOP_K        = 100    # how many tokens to save per step (ranked by expert logit)
CD_ALPHA     = 1.0    # contrastive weight: diffs = (1+alpha)*expert - alpha*amateur
CD_BETA      = 0.2    # APC threshold: keep tokens where expert_logit >= log(beta)+max
NOISE_STEP   = 500    # VCD diffusion noise step (0-999)
CONV_MODE    = "vicuna_v1"
QUESTION     = "Describe this image in detail."
MAX_NEW_TOKENS = 256

# SID internal constants (from custom_modeling_llama.py)
SID_SYS_LENGTH        = 35
SID_IMAGE_TOKEN_LEN   = 576
SID_ATTENTION_RANK    = 72
SID_AGG_LAYER         = 2

# COCO val2017 image URL template
_COCO_URL = "http://images.cocodataset.org/val2017/{:012d}.jpg"

_LOG_BETA = math.log(CD_BETA)   # ~ -1.609


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _download_image(image_id: int, cache_dir: Path) -> Path:
    """Download COCO val2017 image if not already cached. Returns local path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / f"{image_id:012d}.jpg"
    if not dst.exists():
        url = _COCO_URL.format(image_id)
        print(f"    Downloading {url} ...")
        urllib.request.urlretrieve(url, dst)
    return dst


def _build_prompt(model, tokenizer) -> torch.LongTensor:
    """Build the tokenised prompt tensor with IMAGE_TOKEN_INDEX placeholder."""
    qs = QUESTION
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0)   # [1, L]
    return input_ids


def _safe_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Softmax that handles -inf entries (they get prob 0)."""
    finite = logits.isfinite()
    out = torch.zeros_like(logits)
    if finite.any():
        tmp = logits.clone()
        tmp[~finite] = float("-inf")
        out = F.softmax(tmp, dim=-1)
    return out


# ---------------------------------------------------------------------------
# Per-step logit capture
# ---------------------------------------------------------------------------

def _capture_step(
    expert_logits: torch.Tensor,    # [vocab_size]  float32
    amateur_logits: torch.Tensor,   # [vocab_size]  float32
    tokenizer,
    method: str,
) -> dict:
    """
    Given expert and amateur raw logits for one decoding step, compute all
    4 stages and return a structured step record.
    """
    V = expert_logits.shape[0]

    # -- Stage 3: raw CD diff ------------------------------------------------
    cd_pre_apc = (1 + CD_ALPHA) * expert_logits - CD_ALPHA * amateur_logits

    # -- Stage 4: APC mask ---------------------------------------------------
    cutoff = _LOG_BETA + expert_logits.max().item()
    cd_post_apc = cd_pre_apc.clone()
    cd_post_apc[expert_logits < cutoff] = float("-inf")

    # -- Probabilities -------------------------------------------------------
    expert_probs  = F.softmax(expert_logits, dim=-1)
    amateur_probs = F.softmax(amateur_logits, dim=-1)
    cd_probs      = _safe_softmax(cd_post_apc)

    # -- Top-K by expert logit AND top-K by post-subtraction (pre-APC) logit --
    # Ranking only by expert logit would hide a token that had a LOW expert
    # logit but got amplified far up the ranking by the subtraction itself
    # (exactly the "amplified from nowhere" case that matters for this
    # analysis) -- so we track both rankings and keep the union.
    k = min(TOP_K, V)
    expert_topk_vals, expert_topk_ids = torch.topk(expert_logits, k)
    cd_topk_vals, cd_topk_ids = torch.topk(cd_pre_apc, k)

    expert_rank = {tid: r + 1 for r, tid in enumerate(expert_topk_ids.tolist())}
    cd_rank     = {tid: r + 1 for r, tid in enumerate(cd_topk_ids.tolist())}
    union_ids = list(dict.fromkeys(expert_topk_ids.tolist() + cd_topk_ids.tolist()))
    union_ids.sort(key=lambda tid: -expert_logits[tid].item())

    top_k_records = []
    for tid in union_ids:
        above = bool(expert_logits[tid].item() >= cutoff)
        cd_post_val = cd_post_apc[tid].item()
        top_k_records.append({
            "rank_by_expert":     expert_rank.get(tid),
            "rank_by_cd_pre_apc": cd_rank.get(tid),
            "in_expert_topk":     tid in expert_rank,
            "in_cd_topk":         tid in cd_rank,
            "token_id":           tid,
            "token_str":          tokenizer.decode([tid]),
            # Stage 1
            "expert_logit":       float(expert_logits[tid]),
            "expert_prob":        float(expert_probs[tid]),
            # Stage 2
            "amateur_logit":      float(amateur_logits[tid]),
            "amateur_prob":       float(amateur_probs[tid]),
            # Stage 3
            "cd_logit_pre_apc":   float(cd_pre_apc[tid]),
            # APC
            "above_apc":          above,
            "apc_cutoff":         cutoff,
            # Stage 4
            "cd_logit_post_apc":  cd_post_val if above else None,
            "cd_prob_post_apc":   float(cd_probs[tid]),
        })

    # Greedy choice from post-APC CD logits
    chosen_id  = int(cd_post_apc.argmax().item())
    chosen_str = tokenizer.decode([chosen_id])

    return {
        "apc_cutoff":          cutoff,
        "apc_beta":            CD_BETA,
        "n_tokens_above_apc":  int((expert_logits >= cutoff).sum().item()),
        "n_vocab":             V,
        "chosen_token_id":     chosen_id,
        "chosen_token_str":    chosen_str,
        "chosen_in_top_k":     chosen_id in union_ids,
        "top_k":               top_k_records,
    }


def _capture_step_greedy(expert_logits: torch.Tensor, tokenizer) -> dict:
    """Simpler capture for baseline greedy (no amateur, no APC)."""
    k = min(TOP_K, expert_logits.shape[0])
    expert_probs = F.softmax(expert_logits, dim=-1)
    topk_vals, topk_ids = torch.topk(expert_logits, k)
    records = [
        {
            "rank_by_expert": r + 1,
            "token_id":       int(tid),
            "token_str":      tokenizer.decode([int(tid)]),
            "expert_logit":   float(v),
            "expert_prob":    float(expert_probs[tid]),
        }
        for r, (v, tid) in enumerate(zip(topk_vals.tolist(), topk_ids.tolist()))
    ]
    chosen_id  = int(expert_logits.argmax().item())
    return {
        "chosen_token_id":  chosen_id,
        "chosen_token_str": tokenizer.decode([chosen_id]),
        "chosen_in_top_k":  chosen_id in topk_ids.tolist(),
        "top_k":            records,
    }


# ---------------------------------------------------------------------------
# Manual generation loop
# ---------------------------------------------------------------------------

def generate_with_eda(
    model,
    tokenizer,
    prompt_ids: torch.LongTensor,
    image_tensor: torch.Tensor,   # [1, C, H, W] float16 on CUDA
    method: str,                  # "greedy" | "vcd" | "sid"
    verbose: bool = False,        # print token-by-token progress
) -> dict:
    """
    Run one complete greedy-decoded caption, capturing all 4 logit stages
    at every decoding step.

    Returns a dict ready to be serialised to JSON.
    """
    # Use no_grad (not inference_mode) — inference_mode creates special tensors
    # that can conflict with the 4.31-era custom LlamaModel under PyTorch 2.8.
    with torch.no_grad():
        return _generate_with_eda_inner(
            model, tokenizer, prompt_ids, image_tensor, method, verbose
        )


def _generate_with_eda_inner(
    model,
    tokenizer,
    prompt_ids: torch.LongTensor,
    image_tensor: torch.Tensor,
    method: str,
    verbose: bool,
) -> dict:
    device = image_tensor.device
    dtype = image_tensor.dtype
    prompt_ids = prompt_ids.to(device)

    # -- Prepare amateur image (VCD only) ------------------------------------
    if method == "vcd":
        noisy = add_diffusion_noise(
            image_tensor.squeeze(0).float(), NOISE_STEP
        ).unsqueeze(0).to(dtype).to(device)
    else:
        noisy = None

    # -- Initialise KV caches ------------------------------------------------
    expert_pkv  = None
    amateur_pkv = None

    # We'll track the attention mask manually.  After the first forward pass
    # the image tokens expand (1 placeholder -> 576 feature vectors), so we
    # read the actual expanded length from the returned KV cache shape.
    attention_mask_expert  = None
    attention_mask_amateur = None

    generated_ids = []
    steps = []

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        # Fallback: LLaMA/Vicuna EOS is always token 2
        eos_id = 2

    for step in range(MAX_NEW_TOKENS):

        # ---- Build model inputs --------------------------------------------
        if step == 0:
            # First step: full prompt + image
            expert_inp = dict(
                input_ids=prompt_ids,
                images=image_tensor,
                use_cache=True,
                return_dict=True,
            )
            if method == "vcd":
                amateur_inp = dict(
                    input_ids=prompt_ids,
                    images=noisy,
                    use_cache=True,
                    return_dict=True,
                )
            elif method == "sid":
                amateur_inp = dict(
                    input_ids=prompt_ids,
                    images=image_tensor,
                    use_sid=True,
                    use_cache=True,
                    return_dict=True,
                )
        else:
            # Subsequent steps: single new token + KV cache
            new_tok = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device=device)

            expert_inp = dict(
                input_ids=new_tok,
                past_key_values=expert_pkv,
                attention_mask=attention_mask_expert,
                use_cache=True,
                return_dict=True,
            )
            if method in ("vcd", "sid"):
                amateur_inp = dict(
                    input_ids=new_tok,
                    past_key_values=amateur_pkv,
                    attention_mask=attention_mask_amateur,
                    use_cache=True,
                    return_dict=True,
                )
                if method == "sid":
                    # use_sid on subsequent steps has no practical effect
                    # (no image tokens in current input) but we keep parity.
                    amateur_inp["use_sid"] = True

        # ---- Forward passes ------------------------------------------------
        expert_out = model(**expert_inp)

        if method in ("vcd", "sid"):
            amateur_out = model(**amateur_inp)

        # ---- Extract logits ------------------------------------------------
        expert_logits = expert_out.logits[0, -1].float()   # [V]

        if method in ("vcd", "sid"):
            amateur_logits = amateur_out.logits[0, -1].float()

        # ---- Debug: print top-5 tokens at step 0 ---------------------------
        if step == 0:
            top5_vals, top5_ids = torch.topk(expert_logits, 5)
            top5_str = ", ".join(
                f"{tid}:'{tokenizer.decode([tid])}'({v:.2f})"
                for tid, v in zip(top5_ids.tolist(), top5_vals.tolist())
            )
            print(f"    [DBG] step0 logit range [{expert_logits.min():.2f}, {expert_logits.max():.2f}]", flush=True)
            print(f"    [DBG] top5 by expert: {top5_str}", flush=True)

        # ---- Update KV cache -----------------------------------------------
        expert_pkv = expert_out.past_key_values

        # After step 0, read the TRUE expanded sequence length from KV shape
        # (image tokens replace the single placeholder, so len > prompt_ids.shape[1])
        if step == 0:
            # past_key_values[0] = (key, value) for layer 0
            # key shape: [batch, n_heads, seq_len, head_dim]
            expanded_len = expert_pkv[0][0].shape[2]
            print(f"    [DBG] expanded_len (prompt+image tokens) = {expanded_len}", flush=True)
            # Build attention masks of the correct expanded length + 1 generated token
            attention_mask_expert  = torch.ones(1, expanded_len + 1, device=device, dtype=torch.long)
            if method in ("vcd", "sid"):
                amateur_pkv = amateur_out.past_key_values
                attention_mask_amateur = torch.ones(1, expanded_len + 1, device=device, dtype=torch.long)
        else:
            # Extend by 1 for the new token
            attention_mask_expert  = torch.cat(
                [attention_mask_expert,  torch.ones(1, 1, device=device, dtype=torch.long)], dim=-1
            )
            if method in ("vcd", "sid"):
                amateur_pkv = amateur_out.past_key_values
                attention_mask_amateur = torch.cat(
                    [attention_mask_amateur, torch.ones(1, 1, device=device, dtype=torch.long)], dim=-1
                )

        # ---- Capture step data ---------------------------------------------
        if method == "greedy":
            step_record = _capture_step_greedy(expert_logits, tokenizer)
        else:
            step_record = _capture_step(
                expert_logits, amateur_logits, tokenizer, method
            )

        step_record["step_idx"] = step
        steps.append(step_record)

        chosen_id = step_record["chosen_token_id"]
        generated_ids.append(chosen_id)

        # Print progress every step if verbose, or every 20 steps otherwise
        tok_str = tokenizer.decode([chosen_id])
        if verbose or step < 5 or step % 20 == 0:
            print(f"    [DBG] step {step:3d} → id={chosen_id} '{tok_str}'", flush=True)

        if chosen_id == eos_id:
            print(f"    [DBG] EOS at step {step}", flush=True)
            break

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"    [DBG] raw ids[:10]: {generated_ids[:10]}", flush=True)
    print(f"    [DBG] generated ({len(generated_ids)} toks): {generated_text[:120]}", flush=True)

    return {
        "method":         method,
        "cd_alpha":       CD_ALPHA if method != "greedy" else None,
        "cd_beta":        CD_BETA  if method != "greedy" else None,
        "noise_step":     NOISE_STEP if method == "vcd" else None,
        "sid_params":     {
            "sys_length":      SID_SYS_LENGTH,
            "image_tok_len":   SID_IMAGE_TOKEN_LEN,
            "attention_rank":  SID_ATTENTION_RANK,
            "agg_layer":       SID_AGG_LAYER,
        } if method == "sid" else None,
        "generated_text": generated_text,
        "n_steps":        len(steps),
        "steps":          steps,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global TOP_K, MAX_NEW_TOKENS  # declared first so we can both read and reassign it below

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True,
                    help="HF model id or local path, e.g. liuhaotian/llava-v1.5-7b")
    ap.add_argument("--model-base", default=None,
                    help="Optional base model path (for LoRA checkpoints)")
    ap.add_argument("--image-ids-file",
                    default=str(_THIS_DIR / "image_ids.json"),
                    help="JSON list of COCO val2017 image ids to process")
    ap.add_argument("--image-cache-dir",
                    default=str(_THIS_DIR / "data"),
                    help="Where to cache downloaded COCO images")
    ap.add_argument("--out-dir",
                    default=str(_THIS_DIR / "outputs"),
                    help="Root output directory")
    ap.add_argument("--methods", nargs="+", default=["greedy", "vcd", "sid"],
                    choices=["greedy", "vcd", "sid"],
                    help="Which methods to run")
    ap.add_argument("--top-k", type=int, default=TOP_K,
                    help=f"Top-K tokens to record per step (default {TOP_K})")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                    help=f"Max decoding steps per caption (default {MAX_NEW_TOKENS})")
    args = ap.parse_args()

    TOP_K = args.top_k
    MAX_NEW_TOKENS = args.max_new_tokens

    # -- Load image ids -------------------------------------------------------
    with open(args.image_ids_file) as f:
        image_ids = json.load(f)
    print(f"Image ids: {image_ids}")

    # -- Load model -----------------------------------------------------------
    print(f"\nLoading model from '{args.model_path}' ...")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    model.eval()
    device = next(model.parameters()).device
    dtype = model.lm_head.weight.dtype
    print(f"  Model on device: {device}  |  context_len: {context_len}")
    print(f"  lm_head  device: {model.lm_head.weight.device}  dtype: {model.lm_head.weight.dtype}")
    print(f"  EOS token id: {tokenizer.eos_token_id}  |  BOS: {tokenizer.bos_token_id}", flush=True)

    # -- Build prompt (shared across all images) ----------------------------
    prompt_ids = _build_prompt(model, tokenizer)
    print(f"  Prompt token length (before image expansion): {prompt_ids.shape[1]}")

    # -- Output dirs ---------------------------------------------------------
    out_root = Path(args.out_dir)
    for m in args.methods:
        (out_root / m).mkdir(parents=True, exist_ok=True)

    image_cache = Path(args.image_cache_dir)

    # -- CSV summary ---------------------------------------------------------
    csv_path = out_root / "summary.csv"
    csv_fields = [
        "image_id", "method", "n_steps", "generated_text",
        "avg_n_above_apc",        # avg tokens surviving APC per step
        "avg_expert_entropy",     # avg entropy of expert distribution
        "avg_amateur_entropy",    # avg entropy of amateur distribution
        "avg_cd_prob_shift",      # avg sum-abs-diff between expert and cd post-apc probs
        "avg_logit_diff_top1",    # avg (expert_logit - amateur_logit) for chosen token
    ]
    csv_rows = []

    # -- Main loop -----------------------------------------------------------
    for image_id in image_ids:
        print(f"\n{'='*60}")
        print(f"Image {image_id}")
        print(f"{'='*60}")

        # Download / load image
        img_path = _download_image(image_id, image_cache)
        img = Image.open(img_path).convert("RGB")
        img_tensor = image_processor.preprocess(img, return_tensors="pt")["pixel_values"]
        img_tensor = img_tensor.to(dtype).to(device)   # [1, C, H, W]
        print(f"  Image size: {img.size}  tensor shape: {img_tensor.shape}  "
              f"dtype: {img_tensor.dtype}  "
              f"min/max: {img_tensor.min():.3f}/{img_tensor.max():.3f}", flush=True)

        for method in args.methods:
            print(f"\n  --- {method.upper()} ---", flush=True)

            result = generate_with_eda(
                model, tokenizer, prompt_ids, img_tensor, method=method,
                verbose=False,   # set True for token-by-token output
            )
            result["image_id"] = image_id

            # Save per-image JSON
            out_path = out_root / method / f"img_{image_id}_eda.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"    Saved {out_path.name}  ({result['n_steps']} steps)")
            print(f"    Text: {result['generated_text'][:120]} ...")

            # Compute summary stats over all steps
            steps = result["steps"]
            if method == "greedy":
                avg_n_apc        = None
                avg_expert_ent   = float(_avg_entropy([s["top_k"] for s in steps], "expert_prob"))
                avg_amateur_ent  = None
                avg_cd_shift     = None
                avg_logit_diff   = None
            else:
                avg_n_apc       = float(sum(s["n_tokens_above_apc"] for s in steps) / len(steps))
                avg_expert_ent  = float(_avg_entropy([s["top_k"] for s in steps], "expert_prob"))
                avg_amateur_ent = float(_avg_entropy([s["top_k"] for s in steps], "amateur_prob"))
                avg_cd_shift    = float(_avg_prob_shift([s["top_k"] for s in steps]))
                avg_logit_diff  = float(_avg_chosen_logit_diff(steps))

            csv_rows.append({
                "image_id":          image_id,
                "method":            method,
                "n_steps":           result["n_steps"],
                "generated_text":    result["generated_text"][:200],
                "avg_n_above_apc":   avg_n_apc,
                "avg_expert_entropy": avg_expert_ent,
                "avg_amateur_entropy": avg_amateur_ent,
                "avg_cd_prob_shift": avg_cd_shift,
                "avg_logit_diff_top1": avg_logit_diff,
            })

    # -- Write summary CSV ---------------------------------------------------
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nSummary CSV written to {csv_path}")
    print("Done.")


# ---------------------------------------------------------------------------
# Summary stat helpers
# ---------------------------------------------------------------------------

def _avg_entropy(step_top_k_lists, prob_key):
    """Mean token-level entropy over steps from top-K records."""
    entropies = []
    for records in step_top_k_lists:
        probs = [r[prob_key] for r in records if r.get(prob_key) is not None]
        if not probs:
            continue
        # Approximate entropy from top-K (sum may be < 1 if K < V)
        total = sum(probs)
        if total <= 0:
            continue
        ent = -sum(p / total * math.log(p / total + 1e-12) for p in probs)
        entropies.append(ent)
    return sum(entropies) / len(entropies) if entropies else 0.0


def _avg_prob_shift(step_top_k_lists):
    """
    Average sum-absolute-difference between expert_prob and cd_prob_post_apc
    over the top-K tokens per step.  Measures how much APC + contrastive
    diff redistriubtes probability mass.
    """
    shifts = []
    for records in step_top_k_lists:
        shift = sum(
            abs(r.get("expert_prob", 0) - r.get("cd_prob_post_apc", 0))
            for r in records
        )
        shifts.append(shift)
    return sum(shifts) / len(shifts) if shifts else 0.0


def _avg_chosen_logit_diff(steps):
    """
    For each step, look at the chosen token and return
    expert_logit - amateur_logit for that token (shows how much the
    contrastive step amplifies / suppresses the chosen token).
    """
    diffs = []
    for s in steps:
        chosen_id = s["chosen_token_id"]
        for r in s["top_k"]:
            if r["token_id"] == chosen_id:
                diff = r["expert_logit"] - r.get("amateur_logit", r["expert_logit"])
                diffs.append(diff)
                break
    return sum(diffs) / len(diffs) if diffs else 0.0


if __name__ == "__main__":
    main()
