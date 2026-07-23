"""
Unified contrastive-decoding inference for LLaVA-Bench-in-the-Wild.

Covers the rows missing from the Jaccard experiment that Table 4 / Table 6 of
"The Mirage of Performance Gains" (2504.10020v4) require:

  * VCD  under greedy base   ->  --use-vcd --temperature 0        (Table 4)
  * SID  under greedy base   ->  --use-sid --temperature 0        (Table 4)
  * SID  under sampling base ->  --use-sid --temperature 1.0      (Table 6)
  * (VCD under sampling base already exists as outputs/llava/vcd_sample.jsonl)

Mirrors the proven POPE script (LLava1.5-7B/inference/pope_infer_cd.py) but with
LLaVA-Bench question/answer I/O identical to llava_bench_infer_greedy.py.

Uses the repo-root `llava` package, which ships custom_modeling_llama.py (SID support).
"""
import sys
import os
# Use the exact self-contained LLaVA package (transformers==4.31 era, SID-capable)
# that produced the original Jaccard outputs: llava-bench/LLava1.5-7B/llava
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LLava1.5-7B"))

import argparse
import torch
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math

from cd_utils.vcd_utils import add_diffusion_noise, evolve_vcd_greedy_search, evolve_vcd_sampling
from cd_utils.sid_utils import evolve_sid_greedy_search, evolve_sid_sampling


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


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
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step) if args.use_vcd else None

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
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                images_cd=(image_tensor_cd.unsqueeze(0).half().cuda() if image_tensor_cd is not None else None),
                input_ids_cd=None,
                cd_alpha=args.cd_alpha,
                cd_beta=args.cd_beta,
                use_sid=use_sid,
            )

        input_token_len = input_ids.shape[1]
        n_diff = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff > 0:
            print(f'[Warning] {n_diff} output_ids differ from input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

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
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--model-base", type=str, default=None)
    p.add_argument("--image-folder", type=str, default="")
    p.add_argument("--question-file", type=str, required=True)
    p.add_argument("--answers-file", type=str, required=True)
    p.add_argument("--conv-mode", type=str, default="vicuna_v1")
    p.add_argument("--num-chunks", type=int, default=1)
    p.add_argument("--chunk-idx", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0, help="0 -> greedy (Table 4); 1.0 -> sampling (Table 6)")
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--use-vcd", action="store_true", default=False)
    p.add_argument("--use-sid", action="store_true", default=False)
    p.add_argument("--noise-step", type=int, default=900)
    p.add_argument("--cd-alpha", type=float, default=1.0)
    p.add_argument("--cd-beta", type=float, default=0.1)
    args = p.parse_args()

    eval_model(args)
