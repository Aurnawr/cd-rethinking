"""
run_proxy_experiment.py -- tests the "flat constant boost" proxy for VCD/SID:
expert model ONLY (no amateur, no second forward pass at all), same APC rule
(survival based on raw expert logit), but instead of a per-token amateur-based
bonus, every COCO-object-category word gets a FLAT constant added to its
score if it survives APC. +0.2 for the VCD-proxy, +0.3 for the SID-proxy
(matching the average object-candidate bonus actually measured for real
VCD/SID on the 500-image run).

Usage: python run_proxy_experiment.py --n-images 20
"""
import sys, json, types, time, math, argparse
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
OUT_DIR = Path(__file__).resolve().parent / "outputs_proxy_v2"
OUT_DIR.mkdir(exist_ok=True)

MAX_NEW_TOKENS = 256
CD_BETA = 0.2
LOG_BETA = math.log(CD_BETA)
BONUS = {"proxy_vcd": 0.2, "proxy_sid": 0.3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=20)
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
    print(f"n images: {len(image_ids)} -> {image_ids}", flush=True)

    chair = load_chair(image_ids, str(COCO_ANNOT_DIR))

    print("Precomputing object-word bonus vectors...", flush=True)
    token_text = [tokenizer.decode([i]).strip().lower() for i in range(vocab_size)]
    mscoco_words = chair.mscoco_objects
    is_object = torch.tensor([t in mscoco_words for t in token_text], dtype=torch.bool, device=device)
    bonus_vectors = {
        name: (is_object.float() * b) for name, b in BONUS.items()
    }

    qs = DEFAULT_IMAGE_TOKEN + "\n" + "Describe this image in detail."
    conv_base = conv_templates["vicuna_v1"].copy()
    conv_base.append_message(conv_base.roles[0], qs)
    conv_base.append_message(conv_base.roles[1], None)
    prompt = conv_base.get_prompt()
    prompt_ids_base = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)

    cap_paths = {name: OUT_DIR / f"captions_{name}.jsonl" for name in BONUS}
    for p in cap_paths.values():
        open(p, "w").close()  # fresh run each time (small n, no need to resume)

    t0 = time.time()
    for idx, image_id in enumerate(image_ids):
        img_path = IMAGE_DIR / f"{image_id:012d}.jpg"
        img = Image.open(img_path).convert("RGB")
        img_tensor = image_processor.preprocess(img, return_tensors="pt")["pixel_values"].to(dtype).to(device)

        for proxy_name, bonus_vec in bonus_vectors.items():
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
                boosted = expert_logits + bonus_vec  # apply boost BEFORE computing cutoff/mask
                cutoff = LOG_BETA + boosted.max().item()  # cutoff now based on BOOSTED scores
                mask = boosted < cutoff  # survival now based on BOOSTED scores
                boosted[mask] = float("-inf")
                chosen_id = int(boosted.argmax().item())

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
            with open(cap_paths[proxy_name], "a") as f:
                f.write(json.dumps({"image_id": image_id, "caption": caption}) + "\n")

        elapsed = time.time() - t0
        print(f"[{idx+1}/{len(image_ids)}] img={image_id} done ({elapsed:.1f}s elapsed, "
              f"{elapsed/(idx+1):.1f}s/image avg)", flush=True)

    print(f"\nALL DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
