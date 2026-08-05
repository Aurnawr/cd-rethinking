"""
run_full_topk_capture.py -- full top-30 candidate capture for VCD and SID,
every decoding step, all 500 images. This is the data that was previously
MISSING: the earlier 500-image run only saved the object-word subset of a
top-50-by-expert union top-50-by-CD-score list, with no independent amateur
ranking and no full post-subtraction candidate list.

For every step of every VCD/SID caption, this saves:
  - top 30 tokens by EXPERT logit (independent ranking)
  - top 30 tokens by AMATEUR logit (independent ranking -- genuinely new,
    never computed anywhere else in this project)
  - top 30 tokens by CD (post-subtraction, PRE-apc) score -- i.e. the top 30
    survivors of "(1+alpha)*E - alpha*A", independent of what's in the
    expert's or amateur's own top-30
  - APC cutoff for that step, and a per-cd-candidate survives-apc flag
  - the actually chosen token id

Greedy captions are NOT regenerated -- reused as-is from outputs_500_clean/
captions_greedy.jsonl, since greedy has no amateur/subtraction step to add
anything new to capture.

Resumable: one line per image per method in captions_topk_{vcd,sid}.jsonl;
images already present are skipped on rerun.
"""
import sys, json, types, time, math, os
from pathlib import Path
_REPO_ROOT = Path("/teamspace/studios/this_studio/cd-rethinking/llava_logit_eda/CHAIR-llava-detailed")
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "inference"))
sys.path.insert(0, str(_REPO_ROOT / "llava_logit_eda"))

from transformers.models.auto.configuration_auto import CONFIG_MAPPING
if "llava" in CONFIG_MAPPING._mapping:
    del CONFIG_MAPPING._mapping["llava"]
_mpt = types.ModuleType("llava.model.language_model.llava_mpt")
_mpt.LlavaMPTForCausalLM = type("LlavaMPTForCausalLM", (), {})
_mpt.LlavaMPTConfig = type("LlavaMPTConfig", (), {})
sys.modules["llava.model.language_model.llava_mpt"] = _mpt

import torch
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from object_mentions import load_chair, COLOR_WORDS, NUMBER_WORDS

REPO_ROOT = _REPO_ROOT
IMAGE_DIR = REPO_ROOT / "data" / "coco" / "val2017"
COCO_ANNOT_DIR = REPO_ROOT / "data" / "coco" / "annotations"
MODEL_PATH = str(REPO_ROOT / "llava_logit_eda" / "models" / "llava-v1.5-7b")
OUT_DIR = Path(__file__).resolve().parent / "outputs_full_topk"
OUT_DIR.mkdir(exist_ok=True)

MAX_NEW_TOKENS = 256
CD_ALPHA = 1.0
CD_BETA = 0.2
NOISE_STEP = 500
LOG_BETA = math.log(CD_BETA)
TOPK = 30


def add_diffusion_noise(image_tensor, noise_step):
    num_steps = 1000
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
    def q_x(x_0, t):
        noise = torch.randn_like(x_0)
        return alphas_bar_sqrt[t] * x_0 + one_minus_alphas_bar_sqrt[t] * noise
    return q_x(image_tensor.clone(), noise_step)


def append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def load_done_ids(path):
    if not os.path.exists(path):
        return set()
    return {json.loads(l)["image_id"] for l in open(path)}


def main():
    print("Loading model...", flush=True)
    model_name = get_model_name_from_path(MODEL_PATH)
    tokenizer, model, image_processor, context_len = load_pretrained_model(MODEL_PATH, None, model_name)
    model.eval()
    device = next(model.parameters()).device
    dtype = model.lm_head.weight.dtype
    print(f"device={device} dtype={dtype}", flush=True)

    image_ids = [json.loads(l)["image_id"] for l in open(REPO_ROOT / "outputs" / "chair" / "llava-7b-greedy" / "captions.jsonl")]
    print(f"n images: {len(image_ids)}", flush=True)

    chair = load_chair(image_ids, str(COCO_ANNOT_DIR))
    vocab_size = model.config.vocab_size
    print("Precomputing token category lookup table...", flush=True)
    token_text = [tokenizer.decode([i]).strip().lower() for i in range(vocab_size)]
    mscoco_words = chair.mscoco_objects
    inv_syn = chair.inverse_synonym_dict
    is_object = [t in mscoco_words for t in token_text]
    canonical = [inv_syn.get(t) for t in token_text]
    is_color = [t in COLOR_WORDS for t in token_text]
    is_number = [t in NUMBER_WORDS for t in token_text]

    def categorize(tid, gt_objects):
        if is_object[tid]:
            return "real_object" if canonical[tid] in gt_objects else "hallucinated_object"
        if is_color[tid]:
            return "color"
        if is_number[tid]:
            return "number"
        return "filler"

    qs = DEFAULT_IMAGE_TOKEN + "\n" + "Describe this image in detail."
    conv_base = conv_templates["vicuna_v1"].copy()
    conv_base.append_message(conv_base.roles[0], qs)
    conv_base.append_message(conv_base.roles[1], None)
    prompt = conv_base.get_prompt()
    prompt_ids_base = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)

    cap_paths = {m: OUT_DIR / f"captions_topk_{m}.jsonl" for m in ["vcd", "sid"]}
    done_ids = load_done_ids(cap_paths["sid"])
    print(f"already done: {len(done_ids)} images", flush=True)

    t0 = time.time()
    n_this_run = 0

    for idx, image_id in enumerate(image_ids):
        if image_id in done_ids:
            continue

        img_path = IMAGE_DIR / f"{image_id:012d}.jpg"
        img = Image.open(img_path).convert("RGB")
        img_tensor = image_processor.preprocess(img, return_tensors="pt")["pixel_values"].to(dtype).to(device)
        gt_objects = chair.imid_to_objects.get(image_id, set())

        for method in ["vcd", "sid"]:
            prompt_ids = prompt_ids_base.to(device)
            noisy = None
            if method == "vcd":
                noisy = add_diffusion_noise(img_tensor.squeeze(0).float(), NOISE_STEP).unsqueeze(0).to(dtype).to(device)

            expert_pkv = amateur_pkv = None
            attn_expert = attn_amateur = None
            chosen_ids = []
            steps_out = []
            eos_id = tokenizer.eos_token_id or 2

            for step in range(MAX_NEW_TOKENS):
                if step == 0:
                    expert_inp = dict(input_ids=prompt_ids, images=img_tensor, use_cache=True, return_dict=True)
                    if method == "vcd":
                        amateur_inp = dict(input_ids=prompt_ids, images=noisy, use_cache=True, return_dict=True)
                    else:
                        amateur_inp = dict(input_ids=prompt_ids, images=img_tensor, use_sid=True, use_cache=True, return_dict=True)
                else:
                    new_tok = torch.tensor([[chosen_ids[-1]]], dtype=torch.long, device=device)
                    expert_inp = dict(input_ids=new_tok, past_key_values=expert_pkv, attention_mask=attn_expert,
                                       use_cache=True, return_dict=True)
                    amateur_inp = dict(input_ids=new_tok, past_key_values=amateur_pkv, attention_mask=attn_amateur,
                                        use_cache=True, return_dict=True)
                    if method == "sid":
                        amateur_inp["use_sid"] = True

                with torch.no_grad():
                    expert_out = model(**expert_inp)
                    amateur_out = model(**amateur_inp)

                expert_logits = expert_out.logits[0, -1].float()
                amateur_logits = amateur_out.logits[0, -1].float()
                cd_pre_apc = (1 + CD_ALPHA) * expert_logits - CD_ALPHA * amateur_logits
                cutoff = LOG_BETA + expert_logits.max().item()
                mask = expert_logits < cutoff
                cd_post_apc = cd_pre_apc.clone()
                cd_post_apc[mask] = float("-inf")
                chosen_id = int(cd_post_apc.argmax().item())

                e_vals, e_ids = torch.topk(expert_logits, TOPK)
                a_vals, a_ids = torch.topk(amateur_logits, TOPK)
                c_vals, c_ids = torch.topk(cd_pre_apc, TOPK)
                c_survives = (~mask[c_ids]).tolist()

                e_ids_l = e_ids.tolist()
                a_ids_l = a_ids.tolist()
                c_ids_l = c_ids.tolist()
                steps_out.append({
                    "step": step,
                    "chosen_id": chosen_id,
                    "chosen_category": categorize(chosen_id, gt_objects),
                    "apc_cutoff": round(cutoff, 4),
                    "expert_top_ids": e_ids_l,
                    "expert_top_logits": [round(v, 4) for v in e_vals.tolist()],
                    "expert_top_category": [categorize(t, gt_objects) for t in e_ids_l],
                    "amateur_top_ids": a_ids_l,
                    "amateur_top_logits": [round(v, 4) for v in a_vals.tolist()],
                    "amateur_top_category": [categorize(t, gt_objects) for t in a_ids_l],
                    "cd_top_ids": c_ids_l,
                    "cd_top_pre_apc_logits": [round(v, 4) for v in c_vals.tolist()],
                    "cd_top_survives_apc": c_survives,
                    "cd_top_category": [categorize(t, gt_objects) for t in c_ids_l],
                })

                pkv = expert_out.past_key_values
                expert_pkv = pkv
                amateur_pkv = amateur_out.past_key_values
                if step == 0:
                    expanded_len = expert_pkv[0][0].shape[2]
                    attn_expert = torch.ones(1, expanded_len + 1, device=device, dtype=torch.long)
                    attn_amateur = torch.ones(1, expanded_len + 1, device=device, dtype=torch.long)
                else:
                    attn_expert = torch.cat([attn_expert, torch.ones(1, 1, device=device, dtype=torch.long)], dim=-1)
                    attn_amateur = torch.cat([attn_amateur, torch.ones(1, 1, device=device, dtype=torch.long)], dim=-1)

                chosen_ids.append(chosen_id)
                if chosen_id == eos_id:
                    break

            caption = tokenizer.decode(chosen_ids, skip_special_tokens=True)
            append_jsonl(cap_paths[method], {"image_id": image_id, "caption": caption, "steps": steps_out})

        n_this_run += 1
        elapsed = time.time() - t0
        rate = n_this_run / elapsed if elapsed > 0 else 0
        remaining = len(image_ids) - len(done_ids) - n_this_run
        eta_min = remaining / rate / 60 if rate > 0 else float("inf")
        print(f"[{len(done_ids)+n_this_run}/{len(image_ids)}] img={image_id} done "
              f"({elapsed:.1f}s elapsed, {rate:.3f} img/s, ETA {eta_min:.1f} min)", flush=True)

    print(f"\nALL DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
