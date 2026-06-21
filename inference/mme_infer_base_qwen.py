"""MME base inference for Qwen2.5-VL-7B-Instruct.

Qwen counterpart of ``mme_infer_base.py``. Uses the Hugging Face
``Qwen2_5_VLForConditionalGeneration`` + ``AutoProcessor`` contract with
chat-template messages instead of the LLaVA ``IMAGE_TOKEN_INDEX`` /
``conv_templates`` / ``tokenizer_image_token`` machinery.

Output is written to the same per-method directory as the LLaVA scripts but with
the ``qwen25-7b-mme-`` filename prefix, in the identical answer-record schema, so
``eval/mme_eval.py`` consumes it unchanged.
"""

import argparse
import os
import json

import torch
import shortuuid
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from mme_infer_common import (
    split_list,
    get_chunk,
    derive_do_sample,
    build_answer_record,
)

INSTRUCTION_SUFFIX = " Answer the question using a single word or phrase."
DEFAULT_QWEN_PATH = "/teamspace/studios/this_studio/models/Qwen2.5-VL-7B-Instruct"


def load_qwen(args):
    """Load the Qwen2.5-VL model + processor and derive the model name."""
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


def build_inputs(model, processor, image_path, question_text):
    """Build Qwen chat-template model inputs for a single image+question."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": question_text},
        ],
    }]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    inputs = inputs.to(model.device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    return inputs


def decode_trimmed(processor, inputs, generated_ids):
    """Drop the prompt tokens and decode only the newly generated text."""
    trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return text.strip()


def run_method(model, processor, inputs, args, do_sample):
    """Stock Qwen generation (greedy at temperature 0, sampling at temperature 1)."""
    gen_kwargs = dict(
        max_new_tokens=1024,
        use_cache=True,
        do_sample=do_sample,
        num_beams=args.num_beams,
    )
    if do_sample:
        gen_kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            gen_kwargs["top_p"] = args.top_p
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    return decode_trimmed(processor, inputs, generated_ids)


def eval_model(args):
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
            inputs = build_inputs(model, processor, image_path, qs)
            output_text = run_method(model, processor, inputs, args, do_sample)
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
    args = parser.parse_args()

    eval_model(args)
