import argparse
import torch
import os
import json
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
import transformers

from cd_utils.vcd_utils import add_diffusion_noise
from cd_utils.icd_utils import get_random_icd_prompt


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def resolve_yes_no_ids(tokenizer):
    """Resolve single-token ids for 'Yes' and 'No', trying bare and
    leading-space variants. Prints once for sanity-checking in logs.
    """
    candidates = {"Yes": ["Yes", " Yes"], "No": ["No", " No"]}
    resolved = {}
    for word, variants in candidates.items():
        chosen = None
        for v in variants:
            ids = tokenizer.encode(v, add_special_tokens=False)
            if len(ids) == 1:
                chosen = ids[0]
                break
        if chosen is None:
            ids = tokenizer.encode(variants[0], add_special_tokens=False)
            chosen = ids[0]
            print(f"[warn] '{word}' did not encode to a single token "
                  f"({[tokenizer.decode([i]) for i in ids]}); using first token id={chosen}")
        resolved[word] = chosen
    print(f"[info] Resolved token ids -> Yes: {resolved['Yes']} "
          f"({tokenizer.decode([resolved['Yes']])!r}), "
          f"No: {resolved['No']} ({tokenizer.decode([resolved['No']])!r})")
    return resolved["Yes"], resolved["No"]


def audit(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )
    model.eval()

    yes_id, no_id = resolve_yes_no_ids(tokenizer)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]

        # Build prompt -- exactly mirrors infer_cd.py's prompt construction
        if model.config.mm_use_im_start_end:
            qs_with_img = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs_with_img = DEFAULT_IMAGE_TOKEN + '\n' + qs
        qs_with_img = qs_with_img + " Answer the question using a single word or phrase."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs_with_img)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensor_cuda = image_tensor.unsqueeze(0).half().cuda()

        # Build the model_kwargs that prepare_inputs_for_generation expects,
        # same as what model.generate() would pass in on its first step.
        model_kwargs = {
            "images": image_tensor_cuda,
        }

        if args.use_vcd:
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
            image_tensor_cd_cuda = image_tensor_cd.unsqueeze(0).half().cuda()

            # model_kwargs_cd mirrors the copy made at the top of the
            # monkey-patched greedy_search before any generation runs.
            model_kwargs_cd = model_kwargs.copy()
            model_kwargs_cd["images_cd"] = image_tensor_cd_cuda

            with torch.inference_mode():
                # Same call as in vcd_utils greedy_search VCD (2/3):
                # prepare_inputs_for_generation_vcd swaps images → images_cd
                model_inputs_cd = model.prepare_inputs_for_generation_vcd(
                    input_ids, **model_kwargs_cd
                )
                outputs_cd = model(
                    **model_inputs_cd,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
            amateur_logits = outputs_cd.logits[:, -1, :].float()

        elif args.use_icd:
            icd_prompt = get_random_icd_prompt()
            conv_cd = conv_templates[args.conv_mode].copy()
            conv_cd.system = icd_prompt
            conv_cd.append_message(conv.roles[0], qs_with_img)
            conv_cd.append_message(conv.roles[1], None)
            prompt_cd = conv_cd.get_prompt()
            input_ids_cd = tokenizer_image_token(
                prompt_cd, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).cuda()

            # Same attention_mask rebuild as in icd_utils greedy_search ICD (2/3)
            model_kwargs_cd = model_kwargs.copy()
            model_kwargs_cd["input_ids_cd"] = input_ids_cd
            generation_mixin = transformers.generation.utils.GenerationMixin()
            model_kwargs_cd["attention_mask"] = (
                generation_mixin._prepare_attention_mask_for_generation(
                    input_ids_cd,
                    model.generation_config.pad_token_id,
                    model.generation_config.eos_token_id,
                )
            )

            with torch.inference_mode():
                # Same call as in icd_utils greedy_search ICD (2/3)
                model_inputs_cd = model.prepare_inputs_for_generation(
                    input_ids_cd, **model_kwargs_cd
                )
                outputs_cd = model(
                    **model_inputs_cd,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
            amateur_logits = outputs_cd.logits[:, -1, :].float()

        else:
            raise ValueError("Specify exactly one of --use-vcd or --use-icd")

        logit_yes = amateur_logits[0, yes_id].item()
        logit_no  = amateur_logits[0, no_id].item()
        delta_amateur = logit_yes - logit_no

        ans_file.write(json.dumps({
            "question_id": idx,
            "logit_amateur_yes": logit_yes,
            "logit_amateur_no":  logit_no,
            "delta_amateur":     delta_amateur,
        }) + "\n")
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",    type=str, required=True)
    parser.add_argument("--model-base",    type=str, default=None)
    parser.add_argument("--image-folder",  type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file",  type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode",     type=str, default="llava_v1")
    parser.add_argument("--num-chunks",    type=int, default=1)
    parser.add_argument("--chunk-idx",     type=int, default=0)

    parser.add_argument("--use-vcd",    action='store_true', default=False)
    parser.add_argument("--use-icd",    action='store_true', default=False)
    parser.add_argument("--noise-step", type=int, default=900)

    args = parser.parse_args()
    assert args.use_vcd != args.use_icd, "Specify exactly one of --use-vcd or --use-icd"

    audit(args)