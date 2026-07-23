"""
LLaVA-Bench inference with contrastive decoding methods (VCD / SID).
Adapted from inference/pope_infer_cd.py but:
  - No " Answer the question using a single word or phrase." suffix
    (bench requires free-form, detailed answers)
  - max_new_tokens=1024 (full response)
  - temperature=0  →  greedy decoding  (matches Table 4 "greedy" column)

Usage – VCD:
    python eval/llava_bench_infer_cd.py \
        --model-path ./pretrained_models/llava-v1.5-7b \
        --image-folder ./data/llava_bench/images \
        --question-file ./data/llava_bench/questions.jsonl \
        --answers-file ./outputs/llava_bench/answers_vcd.jsonl \
        --conv-mode vicuna_v1 --temperature 0 --use-vcd

Usage – SID:
    python eval/llava_bench_infer_cd.py \
        --model-path ./pretrained_models/llava-v1.5-7b \
        --image-folder ./data/llava_bench/images \
        --question-file ./data/llava_bench/questions.jsonl \
        --answers-file ./outputs/llava_bench/answers_sid.jsonl \
        --conv-mode vicuna_v1 --temperature 0 --use-sid
"""
import argparse
import os
import json
from tqdm import tqdm
import sys

import torch
from PIL import Image
import shortuuid

from llava.constants import (IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
                             DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (tokenizer_image_token, get_model_name_from_path,
                             KeywordsStoppingCriteria)

# cd_utils lives under inference/; add it to the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference"))
from cd_utils.vcd_utils import add_diffusion_noise, evolve_vcd_greedy_search, evolve_vcd_sampling
from cd_utils.sid_utils import evolve_sid_greedy_search, evolve_sid_sampling


def eval_model(args):
    if args.use_vcd:
        evolve_vcd_greedy_search()
        evolve_vcd_sampling()
    if args.use_sid:
        evolve_sid_greedy_search()
        evolve_sid_sampling()

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file))]
    os.makedirs(os.path.dirname(os.path.expanduser(args.answers_file)), exist_ok=True)
    ans_file = open(os.path.expanduser(args.answers_file), "w")

    for line in tqdm(questions, desc=os.path.basename(args.answers_file)):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]           # no single-word suffix for bench
        cur_prompt = qs

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        # VCD: add diffusion noise to produce the contrastive image branch
        image_tensor_cd = None
        if args.use_vcd:
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        use_sid = True if args.use_sid else None

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True,
                images_cd=(image_tensor_cd.unsqueeze(0).half().cuda()
                           if image_tensor_cd is not None else None),
                input_ids_cd=None,   # ICD not used for bench
                use_sid=use_sid,
            )

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)].strip()

        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": shortuuid.uuid(),
            "model_id": model_name,
            "metadata": {},
        }) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="./data/llava_bench/images")
    parser.add_argument("--question-file", type=str, default="./data/llava_bench/questions.jsonl")
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--use-vcd", action="store_true", default=False)
    parser.add_argument("--use-sid", action="store_true", default=False)
    parser.add_argument("--noise-step", type=int, default=900)
    args = parser.parse_args()
    eval_model(args)
