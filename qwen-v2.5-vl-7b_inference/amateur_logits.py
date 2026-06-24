import argparse
import torch
import os
import json
from tqdm import tqdm

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from llava.utils import disable_torch_init

import math

# Only need the amateur-construction helpers, not the LogitsProcessors
# themselves -- this script replicates the *first forward pass* each
# processor would do at generation step 0, since POPE answers are
# single-token Yes/No and only step 0 ever matters for that.
from cd_utils.vcd_utils import add_diffusion_noise
from cd_utils.icd_utils import get_random_icd_prompt


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def resolve_yes_no_ids(processor):
    """Resolve the token ids used for 'Yes' and 'No' as the *first* token of
    a generated continuation. Tries the bare and leading-space variants and
    picks whichever the tokenizer encodes as a single token, since this is
    sensitive to the chat template's assistant-turn boundary and shouldn't
    be hardcoded.
    """
    tok = processor.tokenizer
    candidates = {"Yes": ["Yes", " Yes"], "No": ["No", " No"]}
    resolved = {}
    for word, variants in candidates.items():
        chosen = None
        for v in variants:
            ids = tok.encode(v, add_special_tokens=False)
            if len(ids) == 1:
                chosen = ids[0]
                break
        if chosen is None:
            # Fall back to first token of the multi-token encoding rather
            # than crashing -- print loudly so it's visible in logs.
            ids = tok.encode(variants[0], add_special_tokens=False)
            chosen = ids[0]
            print(f"[warn] '{word}' did not encode to a single token "
                  f"({[tok.decode([i]) for i in ids]}); using first token id={chosen}")
        resolved[word] = chosen
    print(f"[info] Resolved token ids -> Yes: {resolved['Yes']} "
          f"({tok.decode([resolved['Yes']])!r}), "
          f"No: {resolved['No']} ({tok.decode([resolved['No']])!r})")
    return resolved["Yes"], resolved["No"]


def audit(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    yes_id, no_id = resolve_yes_no_ids(processor)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]

        qs = qs + " Answer the question using a single word or phrase."
        full_image_path = os.path.join(args.image_folder, image_file)

        # Build the SAME "main" inputs infer_cd.py builds -- needed as the
        # base prompt/image for VCD (same prompt, noised pixels) and as the
        # source of input_token_len bookkeeping for ICD (not needed here
        # since we only ever do step 0, but kept for clarity/parity).
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": full_image_path},
                    {"type": "text", "text": qs}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(model.device)

        if args.use_vcd:
            pixel_values_cd = add_diffusion_noise(inputs["pixel_values"], args.noise_step)
            pixel_values_cd = pixel_values_cd.to(device=model.device, dtype=model.dtype)
            amateur_input_ids = inputs["input_ids"]
            amateur_pixel_values = pixel_values_cd
            amateur_image_grid_thw = inputs["image_grid_thw"]

        elif args.use_icd:
            icd_prompt = get_random_icd_prompt()
            messages_cd = [
                {"role": "system", "content": icd_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": full_image_path},
                        {"type": "text", "text": qs}
                    ]
                }
            ]
            text_cd = processor.apply_chat_template(messages_cd, tokenize=False, add_generation_prompt=True)
            inputs_cd = processor(
                text=[text_cd], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
            ).to(model.device)
            amateur_input_ids = inputs_cd["input_ids"]
            amateur_pixel_values = inputs["pixel_values"]
            amateur_image_grid_thw = inputs["image_grid_thw"]

        else:
            raise ValueError("Specify exactly one of --use-vcd or --use-icd")

        attention_mask = torch.ones_like(amateur_input_ids)
        cache_position = torch.arange(amateur_input_ids.shape[1], device=amateur_input_ids.device)

        with torch.inference_mode():
            amateur_out = model(
                input_ids=amateur_input_ids,
                attention_mask=attention_mask,
                pixel_values=amateur_pixel_values,
                image_grid_thw=amateur_image_grid_thw,
                cache_position=cache_position,
                use_cache=False,
            )
        amateur_logits = amateur_out.logits[:, -1, :].float()

        logit_yes = amateur_logits[0, yes_id].item()
        logit_no = amateur_logits[0, no_id].item()
        delta_amateur = logit_yes - logit_no

        ans_file.write(json.dumps({
            "question_id": idx,
            "logit_amateur_yes": logit_yes,
            "logit_amateur_no": logit_no,
            "delta_amateur": delta_amateur,
        }) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to Qwen2.5-VL checkpoint")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    parser.add_argument("--use-vcd", action='store_true', default=False)
    parser.add_argument("--use-icd", action='store_true', default=False)
    parser.add_argument("--noise-step", type=int, default=999)
    args = parser.parse_args()

    assert args.use_vcd != args.use_icd, "Specify exactly one of --use-vcd or --use-icd"

    audit(args)