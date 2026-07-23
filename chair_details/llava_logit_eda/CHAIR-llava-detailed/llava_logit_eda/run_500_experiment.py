"""
run_500_experiment.py -- full-scale (500 image) replication of the Claim 0/1/2/3/4
logit-level statistics from NOVEL_FINDINGS.md, using greedy decoding for VCD/SID
(matching the 10-image pilot methodology, not the sampling-based chair_generate.py
methodology -- confirmed with the user).

Resumable: captions are appended to jsonl files and skipped on rerun; aggregate
stats are checkpointed to JSON every CHECKPOINT_EVERY images.
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
from object_mentions import load_chair, all_mentions

OUT_DIR = Path(__file__).resolve().parent / "outputs_500"
OUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_EVERY = 10
MAX_NEW_TOKENS = 256
CD_ALPHA = 1.0
CD_BETA = 0.2
NOISE_STEP = 500
LOG_BETA = math.log(CD_BETA)

REPO_ROOT = _REPO_ROOT
IMAGE_DIR = REPO_ROOT / "data" / "coco" / "val2017"
COCO_ANNOT_DIR = REPO_ROOT / "data" / "coco" / "annotations"
MODEL_PATH = str(REPO_ROOT / "llava_logit_eda" / "models" / "llava-v1.5-7b")


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


def load_json(p, default):
    return json.load(open(p)) if os.path.exists(p) else default


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
    vocab_size = model.config.vocab_size
    print(f"device={device} dtype={dtype} vocab={vocab_size}", flush=True)

    image_ids = json.load(open(REPO_ROOT / "llava_logit_eda" / "image_ids.json")) if False else None
    image_ids = [json.loads(l)["image_id"] for l in open(REPO_ROOT / "outputs" / "chair" / "llava-7b-greedy" / "captions.jsonl")]
    print(f"n images: {len(image_ids)}", flush=True)

    chair = load_chair(image_ids, str(COCO_ANNOT_DIR))

    # -- precompute per-token-id lookup tables once (avoids millions of decode() calls) --
    print("Precomputing token->word lookup table...", flush=True)
    token_text = [tokenizer.decode([i]).strip().lower() for i in range(vocab_size)]
    mscoco_words = chair.mscoco_objects
    inv_syn = chair.inverse_synonym_dict
    is_object = [t in mscoco_words for t in token_text]
    canonical = [inv_syn.get(t) for t in token_text]

    qs = DEFAULT_IMAGE_TOKEN + "\n" + "Describe this image in detail."
    conv_base = conv_templates["vicuna_v1"].copy()
    conv_base.append_message(conv_base.roles[0], qs)
    conv_base.append_message(conv_base.roles[1], None)
    prompt = conv_base.get_prompt()
    prompt_ids_base = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)

    # -- resumable caption files --
    cap_paths = {m: OUT_DIR / f"captions_{m}.jsonl" for m in ["greedy", "vcd", "sid"]}
    done_ids = load_done_ids(cap_paths["sid"])  # sid written last per image => safe resume marker
    print(f"already done: {len(done_ids)} images", flush=True)

    # -- full raw per-object-candidate log (every single object-word candidate,
    #    every decoding step, every image -- for auditing Claim 0 directly,
    #    not just trusting the running-sum aggregates below) --
    raw_object_candidates_path = OUT_DIR / "raw_object_candidates.jsonl"

    # -- aggregate stats (checkpointed) --
    stats_path = OUT_DIR / "aggregate_stats.json"
    stats = load_json(stats_path, {
        "vcd": {"obj_d_sum": 0.0, "obj_d_n": 0, "nonobj_d_sum": 0.0, "nonobj_d_n": 0,
                "real_d_sum": 0.0, "real_d_n": 0, "hall_d_sum": 0.0, "hall_d_n": 0,
                "real_E_sum": 0.0, "real_E_n": 0, "hall_E_sum": 0.0, "hall_E_n": 0,
                "n_above_apc_sum": 0, "n_above_apc_n": 0, "n_above_apc_le2": 0,
                "flip_by_cat": {}, "steps_by_cat": {}},
        "sid": {"obj_d_sum": 0.0, "obj_d_n": 0, "nonobj_d_sum": 0.0, "nonobj_d_n": 0,
                "real_d_sum": 0.0, "real_d_n": 0, "hall_d_sum": 0.0, "hall_d_n": 0,
                "real_E_sum": 0.0, "real_E_n": 0, "hall_E_sum": 0.0, "hall_E_n": 0,
                "n_above_apc_sum": 0, "n_above_apc_n": 0, "n_above_apc_le2": 0,
                "flip_by_cat": {}, "steps_by_cat": {}},
        "images_done": 0,
    })
    mention_sets_path = OUT_DIR / "mention_sets.jsonl"

    def bump(d, key, val=1):
        d[key] = d.get(key, 0) + val

    t0 = time.time()
    n_processed_this_run = 0

    for idx, image_id in enumerate(image_ids):
        if image_id in done_ids:
            continue

        img_path = IMAGE_DIR / f"{image_id:012d}.jpg"
        img = Image.open(img_path).convert("RGB")
        img_tensor = image_processor.preprocess(img, return_tensors="pt")["pixel_values"].to(dtype).to(device)
        gt_objects = chair.imid_to_objects.get(image_id, set())

        per_method_mentions = {}

        for method in ["greedy", "vcd", "sid"]:
            prompt_ids = prompt_ids_base.to(device)
            noisy = None
            if method == "vcd":
                noisy = add_diffusion_noise(img_tensor.squeeze(0).float(), NOISE_STEP).unsqueeze(0).to(dtype).to(device)

            expert_pkv = amateur_pkv = None
            attn_expert = attn_amateur = None
            chosen_ids = []
            per_step_minimal = []  # (expert_top1_id, chosen_id, n_above_apc or None)

            eos_id = tokenizer.eos_token_id or 2

            for step in range(MAX_NEW_TOKENS):
                if step == 0:
                    expert_inp = dict(input_ids=prompt_ids, images=img_tensor, use_cache=True, return_dict=True)
                    if method == "vcd":
                        amateur_inp = dict(input_ids=prompt_ids, images=noisy, use_cache=True, return_dict=True)
                    elif method == "sid":
                        amateur_inp = dict(input_ids=prompt_ids, images=img_tensor, use_sid=True, use_cache=True, return_dict=True)
                else:
                    new_tok = torch.tensor([[chosen_ids[-1]]], dtype=torch.long, device=device)
                    expert_inp = dict(input_ids=new_tok, past_key_values=expert_pkv, attention_mask=attn_expert,
                                       use_cache=True, return_dict=True)
                    if method in ("vcd", "sid"):
                        amateur_inp = dict(input_ids=new_tok, past_key_values=amateur_pkv, attention_mask=attn_amateur,
                                            use_cache=True, return_dict=True)
                        if method == "sid":
                            amateur_inp["use_sid"] = True

                with torch.no_grad():
                    expert_out = model(**expert_inp)
                    if method in ("vcd", "sid"):
                        amateur_out = model(**amateur_inp)

                expert_logits = expert_out.logits[0, -1].float()
                expert_top1_id = int(expert_logits.argmax().item())

                if method == "greedy":
                    chosen_id = expert_top1_id
                    per_step_minimal.append((expert_top1_id, chosen_id, None))
                else:
                    amateur_logits = amateur_out.logits[0, -1].float()
                    cd_pre_apc = (1 + CD_ALPHA) * expert_logits - CD_ALPHA * amateur_logits
                    cutoff = LOG_BETA + expert_logits.max().item()
                    mask = expert_logits < cutoff
                    n_above = int((~mask).sum().item())
                    cd_post_apc = cd_pre_apc.clone()
                    cd_post_apc[mask] = float("-inf")
                    chosen_id = int(cd_post_apc.argmax().item())
                    per_step_minimal.append((expert_top1_id, chosen_id, n_above))

                    # -- candidate-level accumulation (Claim 0/1), independent of final text --
                    k = 50
                    expert_topk_ids = torch.topk(expert_logits, k).indices.tolist()
                    cd_topk_ids = torch.topk(cd_pre_apc, k).indices.tolist()
                    cand_ids = set(expert_topk_ids) | set(cd_topk_ids)
                    s = stats[method]
                    for tid in cand_ids:
                        e = float(expert_logits[tid]); a = float(amateur_logits[tid]); d = e - a
                        if is_object[tid]:
                            s["obj_d_sum"] += d; s["obj_d_n"] += 1
                            node = canonical[tid]
                            is_real = node in gt_objects
                            if is_real:
                                s["real_d_sum"] += d; s["real_d_n"] += 1
                                s["real_E_sum"] += e; s["real_E_n"] += 1
                            else:
                                s["hall_d_sum"] += d; s["hall_d_n"] += 1
                                s["hall_E_sum"] += e; s["hall_E_n"] += 1
                            # -- full raw record for EVERY object-candidate token, every step --
                            # (not aggregated -- for auditing the Claim 0 numbers directly)
                            append_jsonl(raw_object_candidates_path, {
                                "image_id": image_id, "method": method, "step": step,
                                "token_id": tid, "token_str": token_text[tid],
                                "canonical": node, "is_real": is_real,
                                "expert_logit": round(e, 4), "amateur_logit": round(a, 4),
                                "d_expert_minus_amateur": round(d, 4),
                                "chosen_this_step": (tid == chosen_id),
                                "n_above_apc_this_step": n_above,
                            })
                        else:
                            s["nonobj_d_sum"] += d; s["nonobj_d_n"] += 1
                    s["n_above_apc_sum"] += n_above; s["n_above_apc_n"] += 1
                    if n_above <= 2:
                        s["n_above_apc_le2"] += 1

                # -- KV cache bookkeeping --
                expert_pkv = expert_out.past_key_values
                if step == 0:
                    expanded_len = expert_pkv[0][0].shape[2]
                    attn_expert = torch.ones(1, expanded_len + 1, device=device, dtype=torch.long)
                    if method in ("vcd", "sid"):
                        amateur_pkv = amateur_out.past_key_values
                        attn_amateur = torch.ones(1, expanded_len + 1, device=device, dtype=torch.long)
                else:
                    attn_expert = torch.cat([attn_expert, torch.ones(1, 1, device=device, dtype=torch.long)], dim=-1)
                    if method in ("vcd", "sid"):
                        amateur_pkv = amateur_out.past_key_values
                        attn_amateur = torch.cat([attn_amateur, torch.ones(1, 1, device=device, dtype=torch.long)], dim=-1)

                chosen_ids.append(chosen_id)
                if chosen_id == eos_id:
                    break

            caption = tokenizer.decode(chosen_ids, skip_special_tokens=True)
            append_jsonl(cap_paths[method], {"image_id": image_id, "caption": caption})

            # -- mention categorization (cheap, CPU-only) + flip-rate-by-category update --
            mentions = all_mentions(chair, tokenizer, image_id, chosen_ids)
            per_method_mentions[method] = mentions
            if method in ("vcd", "sid"):
                step_category = {}
                for m in mentions:
                    for si in m["step_indices"]:
                        cat = "hallucinated" if (m["category"] == "object" and m["hallucinated"]) else \
                              "real" if m["category"] == "object" else m["category"]
                        step_category.setdefault(si, cat)
                s = stats[method]
                for si, (etop1, chosen, n_above) in enumerate(per_step_minimal):
                    cat = step_category.get(si, "neither")
                    bump(s["steps_by_cat"], cat)
                    if chosen != etop1:
                        bump(s["flip_by_cat"], cat)

        # per-image mention sets for density / whack-a-mole
        rec = {"image_id": image_id}
        for method in ["greedy", "vcd", "sid"]:
            obj = [m for m in per_method_mentions[method] if m["category"] == "object"]
            rec[f"{method}_real"] = sorted({m["node_word"] for m in obj if not m["hallucinated"]})
            rec[f"{method}_hall"] = sorted({m["node_word"] for m in obj if m["hallucinated"]})
            rec[f"{method}_n_tokens"] = len(chosen_ids) if method == "sid" else rec.get(f"{method}_n_tokens", 0)
        append_jsonl(mention_sets_path, rec)

        n_processed_this_run += 1
        stats["images_done"] += 1
        elapsed = time.time() - t0
        rate = n_processed_this_run / elapsed if elapsed > 0 else 0
        remaining = len(image_ids) - stats["images_done"]
        eta_min = remaining / rate / 60 if rate > 0 else float("inf")
        print(f"[{stats['images_done']}/{len(image_ids)}] img={image_id} done "
              f"({elapsed:.1f}s elapsed this run, {rate:.3f} img/s, ETA {eta_min:.1f} min)", flush=True)

        if n_processed_this_run % CHECKPOINT_EVERY == 0:
            json.dump(stats, open(stats_path, "w"))
            print(f"  [checkpoint saved at {stats['images_done']} images]", flush=True)

    json.dump(stats, open(stats_path, "w"))
    print(f"\nALL DONE. {stats['images_done']} images processed total.", flush=True)

    # -- self-audit: recompute the headline Claim-0 numbers directly from the raw
    #    per-token log and cross-check against the running sums above. If these
    #    don't match, the running sums have a bug and should not be trusted. --
    print("\n=== SELF-AUDIT: recomputing from raw_object_candidates.jsonl ===", flush=True)
    for method in ["vcd", "sid"]:
        real_d, hall_d, real_e, hall_e = [], [], [], []
        for line in open(raw_object_candidates_path):
            rec = json.loads(line)
            if rec["method"] != method:
                continue
            if rec["is_real"]:
                real_d.append(rec["d_expert_minus_amateur"]); real_e.append(rec["expert_logit"])
            else:
                hall_d.append(rec["d_expert_minus_amateur"]); hall_e.append(rec["expert_logit"])
        s = stats[method]
        audit_real_d = sum(real_d) / len(real_d) if real_d else float("nan")
        audit_hall_d = sum(hall_d) / len(hall_d) if hall_d else float("nan")
        audit_real_e = sum(real_e) / len(real_e) if real_e else float("nan")
        audit_hall_e = sum(hall_e) / len(hall_e) if hall_e else float("nan")
        running_real_d = s["real_d_sum"] / s["real_d_n"] if s["real_d_n"] else float("nan")
        running_hall_d = s["hall_d_sum"] / s["hall_d_n"] if s["hall_d_n"] else float("nan")
        print(f"{method}: n_real={len(real_d)} (running said {s['real_d_n']})  "
              f"n_hall={len(hall_d)} (running said {s['hall_d_n']})")
        print(f"  audit avg(d) real={audit_real_d:.4f} vs running={running_real_d:.4f}  "
              f"(match: {abs(audit_real_d-running_real_d) < 1e-6})")
        print(f"  audit avg(d) hall={audit_hall_d:.4f} vs running={running_hall_d:.4f}  "
              f"(match: {abs(audit_hall_d-running_hall_d) < 1e-6})")
        print(f"  audit avg(E) real={audit_real_e:.4f}  audit avg(E) hall={audit_hall_e:.4f}", flush=True)


if __name__ == "__main__":
    main()
