"""MME contrastive-decoding inference for Qwen2.5-VL-7B-Instruct (VCD / ICD / SID).

Qwen counterpart of ``mme_infer_cd.py``. The LLaVA version monkeypatches
``GenerationMixin.greedy_search`` / ``sample`` (removed/renamed in the
Transformers releases required by Qwen2.5-VL) and relies on LLaVA-only
``prepare_inputs_for_generation_{vcd,sid}`` helpers. Neither exists for Qwen.

Instead this script runs a self-contained two-stream autoregressive loop. The
main stream sees the real input; the contrastive stream sees a distorted variant:

  * vcd: diffusion-noised image pixels (``add_diffusion_noise``, reused verbatim).
  * icd: a negative system prompt (``get_random_icd_prompt``, reused verbatim).
  * sid: an image-free (text-only) prompt, removing visual grounding.

The CD logit combination is identical to the LLaVA utilities
(``cd_alpha=1.0``, ``cd_beta=0.2``):

    cutoff   = log(cd_beta) + max(logits)
    cd_logits= ((1+alpha)*logits - alpha*logits_cd).masked_fill(logits < cutoff, -inf)

Both greedy (temperature 0) and sampling (temperature 1) paths are supported.
Multimodal RoPE position ids are computed explicitly per stream (via the model's
own ``get_rope_index``) so the two streams never corrupt each other's cached
``rope_deltas``. Exactly one of ``--use-vcd`` / ``--use-icd`` / ``--use-sid`` may
be set (``validate_cd_flags`` rejects conflicts); with none set this falls back
to plain generation.
"""

import argparse
import os
import json

import torch
import shortuuid
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from cd_utils.vcd_utils import add_diffusion_noise
from cd_utils.icd_utils import get_random_icd_prompt

from mme_infer_common import (
    split_list,
    get_chunk,
    derive_do_sample,
    build_answer_record,
    validate_cd_flags,
)

INSTRUCTION_SUFFIX = " Answer the question using a single word or phrase."
DEFAULT_QWEN_PATH = "/teamspace/studios/this_studio/models/Qwen2.5-VL-7B-Instruct"

CD_ALPHA = 1.0
CD_BETA = 0.2


def load_qwen(args):
    model_path = os.path.expanduser(args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    model_name = os.path.basename(model_path.rstrip("/"))
    return model, processor, model_name


def _process(model, processor, messages, with_image, image=None):
    """Run the processor for a chat-message list, returning model-ready inputs."""
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if with_image:
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    else:
        inputs = processor(text=[prompt], return_tensors="pt")
    inputs = inputs.to(model.device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    return inputs


def build_main_inputs(model, processor, image_path, question_text):
    image = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": question_text},
        ],
    }]
    return _process(model, processor, messages, with_image=True, image=image), image


def build_contrastive_inputs(model, processor, args, main_inputs, image, image_path, question_text):
    """Build the contrastive-stream inputs for the selected CD method."""
    if args.use_vcd:
        # Same tokens/grid as the main stream; only the pixels are noised.
        inputs_cd = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in main_inputs.items()}
        inputs_cd["pixel_values"] = add_diffusion_noise(
            main_inputs["pixel_values"], args.noise_step
        ).to(model.dtype)
        return inputs_cd
    if args.use_icd:
        messages = [
            {"role": "system", "content": get_random_icd_prompt()},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question_text},
            ]},
        ]
        return _process(model, processor, messages, with_image=True, image=image)
    if args.use_sid:
        # Image-free prompt: removes visual grounding (text-only contrastive stream).
        messages = [{"role": "user", "content": [{"type": "text", "text": question_text}]}]
        return _process(model, processor, messages, with_image=False)
    return None


def _eos_ids(model):
    eos = model.generation_config.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {eos}
    return set(eos)


def _init_stream(model, inputs):
    """Initialize a decode-stream state with explicit multimodal RoPE position ids."""
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    image_grid_thw = inputs.get("image_grid_thw")
    position_ids, rope_deltas = model.get_rope_index(
        input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": inputs.get("pixel_values"),
        "image_grid_thw": image_grid_thw,
        "position_ids": position_ids,
        "rope_deltas": rope_deltas,
        "past_key_values": None,
        "cur_len": input_ids.shape[1],
        "first": True,
    }


def _forward_stream(model, stream):
    """Advance one stream by a single forward pass; return the last-position logits."""
    if stream["first"]:
        model_inputs = {
            "input_ids": stream["input_ids"],
            "attention_mask": stream["attention_mask"],
            "position_ids": stream["position_ids"],
            "pixel_values": stream["pixel_values"],
            "image_grid_thw": stream["image_grid_thw"],
            "past_key_values": stream["past_key_values"],
            "use_cache": True,
        }
    else:
        last_id = stream["input_ids"][:, -1:]
        # Absolute index of the new token = cur_len - 1; mrope offset = + rope_deltas.
        delta = (stream["cur_len"] - 1) + stream["rope_deltas"]
        pos = torch.arange(1, device=last_id.device).view(1, -1) + delta
        position_ids = pos.unsqueeze(0).expand(3, -1, -1)
        model_inputs = {
            "input_ids": last_id,
            "attention_mask": stream["attention_mask"],
            "position_ids": position_ids,
            "past_key_values": stream["past_key_values"],
            "use_cache": True,
        }
    outputs = model(**{k: v for k, v in model_inputs.items() if v is not None}, return_dict=True)
    stream["past_key_values"] = outputs.past_key_values
    stream["first"] = False
    return outputs.logits[:, -1, :]


def _append_token(stream, token):
    stream["input_ids"] = torch.cat([stream["input_ids"], token[:, None]], dim=-1)
    if stream["attention_mask"] is not None:
        ones = torch.ones(
            (stream["attention_mask"].shape[0], 1),
            dtype=stream["attention_mask"].dtype,
            device=stream["attention_mask"].device,
        )
        stream["attention_mask"] = torch.cat([stream["attention_mask"], ones], dim=-1)
    stream["cur_len"] += 1


def _combine_cd(logits, logits_cd):
    cutoff = torch.log(torch.tensor(CD_BETA, device=logits.device)) + \
        logits.max(dim=-1, keepdim=True).values
    diffs = (1 + CD_ALPHA) * logits - CD_ALPHA * logits_cd
    return diffs.masked_fill(logits < cutoff, -float("inf"))


def _top_p_filter(scores, top_p):
    sorted_logits, sorted_idx = torch.sort(scores, descending=True, dim=-1)
    cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cum_probs > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    to_remove = remove.scatter(-1, sorted_idx, remove)
    return scores.masked_fill(to_remove, -float("inf"))


def _select_token(scores, do_sample, temperature, top_p):
    if not do_sample:
        return torch.argmax(scores, dim=-1)
    if temperature and temperature > 0 and temperature != 1.0:
        scores = scores / temperature
    if top_p is not None:
        scores = _top_p_filter(scores, top_p)
    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)


def contrastive_generate(model, inputs_main, inputs_cd, args, do_sample, max_new_tokens=1024):
    """Two-stream contrastive autoregressive decode; returns the full main input_ids."""
    eos_ids = _eos_ids(model)
    stream = _init_stream(model, inputs_main)
    stream_cd = _init_stream(model, inputs_cd) if inputs_cd is not None else None

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = _forward_stream(model, stream)
            if stream_cd is not None:
                logits_cd = _forward_stream(model, stream_cd)
                scores = _combine_cd(logits, logits_cd)
            else:
                scores = logits
            next_token = _select_token(scores, do_sample, args.temperature, args.top_p)
            _append_token(stream, next_token)
            if stream_cd is not None:
                _append_token(stream_cd, next_token)
            if next_token.item() in eos_ids:
                break
    return stream["input_ids"]


def decode_trimmed(processor, prompt_len, generated_ids):
    trimmed = generated_ids[:, prompt_len:]
    text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return text.strip()


def eval_model(args):
    validate_cd_flags(args)

    model, processor, model_name = load_qwen(args)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    do_sample = derive_do_sample(args.temperature)
    with open(answers_file, "w") as ans_file:
        for line in tqdm(questions):
            image_path = os.path.join(args.image_folder, line["image"])
            qs = line["text"] + INSTRUCTION_SUFFIX
            inputs_main, image = build_main_inputs(model, processor, image_path, qs)
            inputs_cd = build_contrastive_inputs(
                model, processor, args, inputs_main, image, image_path, qs
            )
            prompt_len = inputs_main["input_ids"].shape[1]
            output_ids = contrastive_generate(model, inputs_main, inputs_cd, args, do_sample)
            output_text = decode_trimmed(processor, prompt_len, output_ids)
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps(build_answer_record(line, output_text, ans_id, model_name)) + "\n")
            ans_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=DEFAULT_QWEN_PATH)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    # cd parameters
    parser.add_argument("--use-icd", action="store_true", default=False)
    parser.add_argument("--use-vcd", action="store_true", default=False)
    parser.add_argument("--use-sid", action="store_true", default=False)
    parser.add_argument("--noise-step", type=int, default=900)
    args = parser.parse_args()

    eval_model(args)
