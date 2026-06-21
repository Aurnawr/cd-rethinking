"""MME OLM inference for Qwen2.5-VL-7B-Instruct.

Qwen counterpart of ``mme_infer_olm.py``. OLM is a greedy-only intervention that
nudges the next-token decision toward "yes" when the yes/no mass is high and the
two are close. The LLaVA version monkeypatches ``GenerationMixin.greedy_search``
and hardcodes Llama yes/no token ids (3869/1939), which are wrong for Qwen.

Here the same decision logic is expressed as a ``LogitsProcessor`` handed to the
stock ``model.generate`` greedy path, so Qwen handles all of its own multimodal
RoPE / cache plumbing. The yes/no token ids are resolved from the Qwen tokenizer
at load time rather than hardcoded. Run greedy (temperature 0) only.
"""

import argparse
import os
import json

import torch
import shortuuid
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import LogitsProcessor, LogitsProcessorList

from mme_infer_common import (
    split_list,
    get_chunk,
    derive_do_sample,
    build_answer_record,
)

INSTRUCTION_SUFFIX = " Answer the question using a single word or phrase."
DEFAULT_QWEN_PATH = "/teamspace/studios/this_studio/models/Qwen2.5-VL-7B-Instruct"


class OLMLogitsProcessor(LogitsProcessor):
    """Reproduce the OLM greedy yes/no surgery as a logits processor.

    Mirrors ``spurious_utils/olm_utils.py``: on the (post-processor) scores,
    if P(yes)+P(no) > 0.2 and |P(yes)-P(no)| < 0.5, force the next greedy token
    to be "yes"; otherwise leave the scores unchanged so the stock argmax wins.
    Forcing is done by lifting the "yes" logit above the row max, which makes it
    the argmax under greedy decoding.
    """

    def __init__(self, yes_id, no_id):
        self.yes_id = yes_id
        self.no_id = no_id

    def __call__(self, input_ids, scores):
        probs = torch.softmax(scores, dim=-1)
        sum_prob = probs[:, self.yes_id] + probs[:, self.no_id]
        abs_diff = torch.abs(probs[:, self.yes_id] - probs[:, self.no_id])
        force_yes = (sum_prob > 0.2) & (abs_diff < 0.5)
        if force_yes.any():
            scores = scores.clone()
            row_max = scores.max(dim=-1, keepdim=True).values
            for i in range(scores.shape[0]):
                if force_yes[i]:
                    scores[i, self.yes_id] = row_max[i] + 1.0
        return scores


def resolve_yes_no_ids(processor):
    """Resolve the first sub-token ids of "Yes"/"No" from the Qwen tokenizer."""
    tok = processor.tokenizer
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    return yes_id, no_id


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


def build_inputs(model, processor, image_path, question_text):
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
    trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return text.strip()


def eval_model(args):
    model, processor, model_name = load_qwen(args)

    logits_processor = None
    if args.use_olm:
        yes_id, no_id = resolve_yes_no_ids(processor)
        logits_processor = LogitsProcessorList([OLMLogitsProcessor(yes_id, no_id)])

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
            gen_kwargs = dict(
                max_new_tokens=1024,
                use_cache=True,
                do_sample=do_sample,
                num_beams=args.num_beams,
            )
            if logits_processor is not None:
                gen_kwargs["logits_processor"] = logits_processor
            if do_sample:
                gen_kwargs["temperature"] = args.temperature
                if args.top_p is not None:
                    gen_kwargs["top_p"] = args.top_p
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, **gen_kwargs)
            output_text = decode_trimmed(processor, inputs, generated_ids)
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
    # olm parameter
    parser.add_argument("--use-olm", action="store_true", default=False)
    args = parser.parse_args()

    eval_model(args)
