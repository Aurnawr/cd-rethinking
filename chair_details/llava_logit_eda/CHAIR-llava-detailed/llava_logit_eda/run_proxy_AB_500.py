"""
run_proxy_designs_ABCDEF.py -- runs all 14 proxy variants (Designs A-F) over
N images. Single expert-only forward pass per step (no amateur model at all)
for every variant. APC survival is always based on the RAW expert logit only
(exactly matching the real VCD/SID formula), so it cannot be short-circuited
regardless of how large a design's random bonus draw is -- only the SOURCE
of the per-token bonus added on top of survivors differs by design:

  A_vcd / A_sid   : bonus ~ Normal(pooled_mean, pooled_std), object words only,
                    no truth-awareness at all (pure noise, right shape)
  B_vcd / B_sid   : bonus = bootstrap-resampled real empirical d-value,
                    object words only, no truth-awareness (non-parametric A)
  C_vcd / C_sid   : bonus ~ Normal(real_mean,real_std) if the word IS actually
                    present in this image's ground truth, else
                    Normal(hall_mean,hall_std) -- oracle-informed upper bound,
                    NOT a fair deployable proxy
  D_vcd / D_sid   : bonus = fixed per-WORD average (no randomness, no image
                    dependence at all -- e.g. "sandwich" always gets its own
                    precomputed constant)
  E_vcd / E_sid   : same noise as A, but applied to EVERY vocabulary token,
                    not just object words (specificity control)
  F_vcd_realonly / F_sid_realonly   : oracle noise, but ONLY on real-object
                    candidates (hallucinatable candidates get 0 bonus)
  F_vcd_hallonly / F_sid_hallonly   : oracle noise, but ONLY on hallucinatable
                    candidates (real candidates get 0 bonus)

Resumable: captions appended per (image_id, variant); skips already-done
(image_id, variant) pairs on rerun.
"""
import sys, json, types, time, math, random, argparse
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
from object_mentions import load_chair

REPO_ROOT = _REPO_ROOT
IMAGE_DIR = REPO_ROOT / "data" / "coco" / "val2017"
COCO_ANNOT_DIR = REPO_ROOT / "data" / "coco" / "annotations"
MODEL_PATH = str(REPO_ROOT / "llava_logit_eda" / "models" / "llava-v1.5-7b")
OUT_DIR = Path(__file__).resolve().parent / "outputs_proxy_AB_500"
OUT_DIR.mkdir(exist_ok=True)

MAX_NEW_TOKENS = 256
CD_BETA = 0.2
LOG_BETA = math.log(CD_BETA)

VARIANTS = [
    "A_vcd", "A_sid",
    "B_vcd", "B_sid",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=500)
    args = ap.parse_args()

    print("Loading model...", flush=True)
    model_name = get_model_name_from_path(MODEL_PATH)
    tokenizer, model, image_processor, context_len = load_pretrained_model(MODEL_PATH, None, model_name)
    model.eval()
    device = next(model.parameters()).device
    dtype = model.lm_head.weight.dtype
    vocab_size = model.config.vocab_size

    all_500_ids = [json.loads(l)["image_id"] for l in open(REPO_ROOT / "outputs" / "chair" / "llava-7b-greedy" / "captions.jsonl")]
    image_ids = all_500_ids[: args.n_images]
    print(f"n images: {len(image_ids)}", flush=True)

    chair = load_chair(image_ids, str(COCO_ANNOT_DIR))
    stats = json.load(open(Path(__file__).resolve().parent / "proxy_stats.json"))

    print("Precomputing token->word lookup and object-token index...", flush=True)
    token_text = [tokenizer.decode([i]).strip().lower() for i in range(vocab_size)]
    mscoco_words = chair.mscoco_objects
    inv_syn = chair.inverse_synonym_dict
    is_object = torch.tensor([t in mscoco_words for t in token_text], dtype=torch.bool, device=device)
    obj_token_ids = torch.nonzero(is_object, as_tuple=True)[0]  # LongTensor of object-token ids
    obj_canonical = [inv_syn.get(token_text[i.item()]) for i in obj_token_ids]
    n_obj = len(obj_token_ids)
    print(f"  vocab={vocab_size}  n_object_tokens={n_obj}", flush=True)

    # per-word fixed bonus lookup tensors (Design D), aligned to obj_token_ids order
    d_lookup_vec = {}
    for m in ["vcd", "sid"]:
        pw = stats[m]["per_word_bonus"]
        pooled_mean = stats[m]["pooled_mean"]
        vals = [pw.get(w, pooled_mean) for w in obj_canonical]
        d_lookup_vec[m] = torch.tensor(vals, dtype=torch.float32, device=device)

    empirical = {m: stats[m]["empirical_samples"] for m in ["vcd", "sid"]}

    qs = DEFAULT_IMAGE_TOKEN + "\n" + "Describe this image in detail."
    conv_base = conv_templates["vicuna_v1"].copy()
    conv_base.append_message(conv_base.roles[0], qs)
    conv_base.append_message(conv_base.roles[1], None)
    prompt = conv_base.get_prompt()
    prompt_ids_base = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)

    cap_paths = {v: OUT_DIR / f"captions_{v}.jsonl" for v in VARIANTS}
    done = {v: set() for v in VARIANTS}
    for v in VARIANTS:
        if cap_paths[v].exists():
            for line in open(cap_paths[v]):
                done[v].add(json.loads(line)["image_id"])

    def get_bonus(variant, obj_is_real_t):
        """Returns a length-n_obj tensor of bonuses to add to obj_token_ids' logits."""
        base_method = "vcd" if "vcd" in variant else "sid"
        s = stats[base_method]

        if variant.startswith("A_"):
            return torch.randn(n_obj, device=device) * s["pooled_std"] + s["pooled_mean"]
        if variant.startswith("B_"):
            samples = random.choices(empirical[base_method], k=n_obj)
            return torch.tensor(samples, dtype=torch.float32, device=device)
        if variant.startswith("C_"):
            real_noise = torch.randn(n_obj, device=device) * s["real_std"] + s["real_mean"]
            hall_noise = torch.randn(n_obj, device=device) * s["hall_std"] + s["hall_mean"]
            return torch.where(obj_is_real_t, real_noise, hall_noise)
        if variant.startswith("D_"):
            return d_lookup_vec[base_method]
        if variant.startswith("F_") and variant.endswith("realonly"):
            real_noise = torch.randn(n_obj, device=device) * s["real_std"] + s["real_mean"]
            return torch.where(obj_is_real_t, real_noise, torch.zeros(n_obj, device=device))
        if variant.startswith("F_") and variant.endswith("hallonly"):
            hall_noise = torch.randn(n_obj, device=device) * s["hall_std"] + s["hall_mean"]
            return torch.where(obj_is_real_t, torch.zeros(n_obj, device=device), hall_noise)
        raise ValueError(variant)

    t0 = time.time()
    n_img_done_this_run = 0
    for idx, image_id in enumerate(image_ids):
        needed = [v for v in VARIANTS if image_id not in done[v] and not v.startswith("E_")]
        needed_e = [v for v in VARIANTS if image_id not in done[v] and v.startswith("E_")]
        if not needed and not needed_e:
            continue

        img_path = IMAGE_DIR / f"{image_id:012d}.jpg"
        img = Image.open(img_path).convert("RGB")
        img_tensor = image_processor.preprocess(img, return_tensors="pt")["pixel_values"].to(dtype).to(device)
        gt_objects = chair.imid_to_objects.get(image_id, set())
        obj_is_real_t = torch.tensor([c in gt_objects for c in obj_canonical], dtype=torch.bool, device=device)

        for variant in needed + needed_e:
            prompt_ids = prompt_ids_base.to(device)
            pkv = None
            attn = None
            chosen_ids = []
            eos_id = tokenizer.eos_token_id or 2

            for step in range(MAX_NEW_TOKENS):
                if step == 0:
                    inp = dict(input_ids=prompt_ids, images=img_tensor, use_cache=True, return_dict=True)
                else:
                    new_tok = torch.tensor([[chosen_ids[-1]]], dtype=torch.long, device=device)
                    inp = dict(input_ids=new_tok, past_key_values=pkv, attention_mask=attn,
                               use_cache=True, return_dict=True)

                with torch.no_grad():
                    out = model(**inp)

                expert_logits = out.logits[0, -1].float()
                cutoff = LOG_BETA + expert_logits.max().item()
                mask = expert_logits < cutoff  # survival: RAW expert logit only, always

                if variant.startswith("E_"):
                    base_method = "vcd" if "vcd" in variant else "sid"
                    s = stats[base_method]
                    full_bonus = torch.randn(vocab_size, device=device) * s["pooled_std"] + s["pooled_mean"]
                    scored = expert_logits + full_bonus
                else:
                    bonus = get_bonus(variant, obj_is_real_t)
                    full_bonus = torch.zeros(vocab_size, device=device)
                    full_bonus[obj_token_ids] = bonus
                    scored = expert_logits + full_bonus

                scored[mask] = float("-inf")
                chosen_id = int(scored.argmax().item())

                pkv = out.past_key_values
                if step == 0:
                    expanded_len = pkv[0][0].shape[2]
                    attn = torch.ones(1, expanded_len + 1, device=device, dtype=torch.long)
                else:
                    attn = torch.cat([attn, torch.ones(1, 1, device=device, dtype=torch.long)], dim=-1)

                chosen_ids.append(chosen_id)
                if chosen_id == eos_id:
                    break

            caption = tokenizer.decode(chosen_ids, skip_special_tokens=True)
            with open(cap_paths[variant], "a") as f:
                f.write(json.dumps({"image_id": image_id, "caption": caption}) + "\n")
            done[variant].add(image_id)

        n_img_done_this_run += 1
        elapsed = time.time() - t0
        print(f"[{idx+1}/{len(image_ids)}] img={image_id} all variants done "
              f"({elapsed:.1f}s elapsed, {elapsed/n_img_done_this_run:.1f}s/image avg this run)", flush=True)

    print(f"\nALL DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
