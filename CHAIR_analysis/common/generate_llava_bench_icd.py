"""
generate_llava_bench_icd.py -- ICD (Instruction Contrastive Decoding,
arXiv:2403.18715) on the 60-question LLaVA-Bench-in-the-Wild set.

Faithful to the official reference implementation (github.com/p1k0pan/ICD,
experiments/gen_scripts/icd_llava_bench_ib.py): the 5 disturbance prompts
there are used as 5 SEPARATE full passes over the dataset (one prompt per
run), not randomly mixed per-question within one run -- this script takes
the disturbance prompt as a fixed CLI argument for exactly that reason.
cd_alpha=1, cd_beta=0.1 (matches both the paper text and the official code's
hardcoded default).

Uses a manual per-step decoding loop (same pattern as generate_llava.py),
not HF's generate()/GenerationMixin monkey-patching -- the vendored
icd_utils.py's monkey-patch approach targets an older transformers API
(GreedySearchOutput etc., removed in transformers 5.x) and isn't worth
resurrecting when this project's own proven pattern already does the same
job directly against the model.

Generation is sample-based (temperature=1.0), matching how VCD/APC/greedy
were already run for the existing Jaccard analysis in jaccard_experiment/.

Output schema matches jaccard_experiment's existing llava_bench_infer_*.py
answer files: {question_id, prompt, text, answer_id, model_id, metadata}.
"""
import argparse, json, os, sys, types, random
from pathlib import Path

HERE = Path(__file__).resolve().parent          # common/
REPO = HERE.parent                              # CHAIR_analysis/
sys.path.insert(0, str(HERE))
import torch
import torch.nn.functional as F
from PIL import Image

from transformers.models.auto.configuration_auto import CONFIG_MAPPING
if "llava" in CONFIG_MAPPING._mapping:
    del CONFIG_MAPPING._mapping["llava"]
_mpt = types.ModuleType("llava.model.language_model.llava_mpt")
_mpt.LlavaMPTForCausalLM = type("LlavaMPTForCausalLM", (), {})
_mpt.LlavaMPTConfig = type("LlavaMPTConfig", (), {})
sys.modules["llava.model.language_model.llava_mpt"] = _mpt

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates

MODEL_PATH = str(REPO / "models" / "llava-v1.5-7b")

MAX_NEW_TOKENS = 512
CD_ALPHA = 1.0
CD_BETA = 0.1  # paper + official code value
LOG_BETA = torch.log(torch.tensor(CD_BETA)).item()

# Verbatim from the official ICD repo's icd_llava_bench_ib.py (byte-identical
# to this project's own icd_utils.get_random_icd_prompt() pool).
DISTURBANCE_PROMPTS = {
    "p1": "You are an object detector to recognize every different objects.",
    "p2": "You are an object detector to recognize every different objects by focusing the shapes, colors and relationships of objects.",
    "n1": "I want you avoid any specific identification or categorization of the objects depicted.",
    "n2": "You are a confused objects detector to provide a fuzzy overview or impression of the image.",
    "p3": "You are an object detector to provide a general overview or impression of the image.",
}


def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_prompt_ids(tokenizer, model, qs, conv_mode="vicuna_v1"):
    if model.config.mm_use_im_start_end:
        qs_full = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs_full = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs_full)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-key", required=True, choices=list(DISTURBANCE_PROMPTS.keys()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--question-file", default=str(REPO / "llava_bench" / "questions.jsonl"))
    ap.add_argument("--image-folder", default=str(REPO / "llava_bench" / "images"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="only process the first N questions (smoke test)")
    args = ap.parse_args()

    set_seed(args.seed)
    disturbance_text = DISTURBANCE_PROMPTS[args.prompt_key]
    print(f"[icd] prompt_key={args.prompt_key} disturbance={disturbance_text!r} cd_alpha={CD_ALPHA} cd_beta={CD_BETA}", flush=True)

    model_name = get_model_name_from_path(MODEL_PATH)
    tokenizer, model, image_processor, _ = load_pretrained_model(MODEL_PATH, None, model_name)
    model.eval()
    device = next(model.parameters()).device
    dtype = model.lm_head.weight.dtype
    eos_id = tokenizer.eos_token_id or 2

    questions = [json.loads(l) for l in open(args.question_file)]
    if args.limit:
        questions = questions[: args.limit]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = {json.loads(l)["question_id"] for l in open(args.out)} if os.path.exists(args.out) else set()
    out_f = open(args.out, "a")

    import time; t0 = time.time(); n = 0
    for q in questions:
        qid = q["question_id"]
        if qid in done:
            continue
        image_file, qs = q["image"], q["text"]

        iseed = (args.seed * 1_000_003 + qid) % (2**31 - 1)
        torch.manual_seed(iseed)

        prompt_ids = build_prompt_ids(tokenizer, model, qs).to(device)
        prompt_ids_cd = build_prompt_ids(tokenizer, model, disturbance_text + " " + qs).to(device)

        image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        img_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].to(dtype).to(device)

        expert_pkv = amateur_pkv = attn_e = attn_a = None
        chosen_ids = []

        for step in range(MAX_NEW_TOKENS):
            if step == 0:
                expert_inp = dict(input_ids=prompt_ids, images=img_tensor, use_cache=True, return_dict=True)
                amateur_inp = dict(input_ids=prompt_ids_cd, images=img_tensor, use_cache=True, return_dict=True)
            else:
                new_tok = torch.tensor([[chosen_ids[-1]]], dtype=torch.long, device=device)
                expert_inp = dict(input_ids=new_tok, past_key_values=expert_pkv, attention_mask=attn_e,
                                   use_cache=True, return_dict=True)
                amateur_inp = dict(input_ids=new_tok, past_key_values=amateur_pkv, attention_mask=attn_a,
                                    use_cache=True, return_dict=True)

            with torch.no_grad():
                expert_out = model(**expert_inp)
                E = expert_out.logits[0, -1].float()
                amateur_out = model(**amateur_inp)
                A = amateur_out.logits[0, -1].float()

            cutoff = LOG_BETA + E.max().item()
            mask = E < cutoff
            scored = (1 + CD_ALPHA) * E - CD_ALPHA * A
            scored = scored.clone(); scored[mask] = float("-inf")

            probs = F.softmax(scored, dim=-1)
            chosen_id = int(torch.multinomial(probs, num_samples=1).item())

            expert_pkv = expert_out.past_key_values
            amateur_pkv = amateur_out.past_key_values
            if step == 0:
                attn_e = torch.ones(1, expert_pkv[0][0].shape[2] + 1, device=device, dtype=torch.long)
                attn_a = torch.ones(1, amateur_pkv[0][0].shape[2] + 1, device=device, dtype=torch.long)
            else:
                attn_e = torch.cat([attn_e, torch.ones(1, 1, device=device, dtype=torch.long)], dim=-1)
                attn_a = torch.cat([attn_a, torch.ones(1, 1, device=device, dtype=torch.long)], dim=-1)

            chosen_ids.append(chosen_id)
            if chosen_id == eos_id:
                break

        outputs = tokenizer.decode(chosen_ids, skip_special_tokens=True).strip()

        out_f.write(json.dumps({
            "question_id": qid, "prompt": qs, "text": outputs,
            "answer_id": f"icd-{args.prompt_key}-{qid}", "model_id": "llava-v1.5-7b-icd",
            "metadata": {"prompt_key": args.prompt_key, "disturbance": disturbance_text},
        }) + "\n")
        out_f.flush()
        n += 1
        if n % 10 == 0:
            el = time.time() - t0
            print(f"  [{n}] {el:.0f}s ({el/n:.1f}s/q)", flush=True)

    out_f.close()
    print(f"[icd] DONE {args.prompt_key} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
