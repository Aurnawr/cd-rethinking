import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from PIL import Image
import math

from cd_utils.vcd_utils import add_diffusion_noise, evolve_vcd_greedy_search, evolve_vcd_sampling
from cd_utils.icd_utils import get_random_icd_prompt, evolve_icd_greedy_search, evolve_icd_sampling
from cd_utils.sid_utils import evolve_sid_greedy_search, evolve_sid_sampling
from cd_utils.attn_logger import compute_attention_masses, get_expanded_image_span, get_expanded_image_span_dynamic

from llava.model.language_model.custom_modeling_llama import LlamaAttention as LlavaLlamaAttention

original_attn_forward = LlavaLlamaAttention.forward

def patched_attn_forward(self, hidden_states, attention_mask=None,
                          position_ids=None, past_key_value=None,
                          output_attentions=False, use_cache=False):
    return original_attn_forward(
        self,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=True,
        use_cache=use_cache,
    )

LlavaLlamaAttention.forward = patched_attn_forward


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):

    if args.use_vcd:
        evolve_vcd_greedy_search()
        evolve_vcd_sampling()
    if args.use_icd:
        evolve_icd_greedy_search()
        evolve_icd_sampling()
    if args.use_sid:
        evolve_sid_greedy_search()
        evolve_sid_sampling()

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    # ---------- register forward hooks ----------
    captured_attentions = []

    def make_attn_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                captured_attentions.append(output[1].detach().cpu())
        return hook

    hooks = []
    for idx_layer, layer in enumerate(model.model.layers):
        h = layer.self_attn.register_forward_hook(make_attn_hook(idx_layer))
        hooks.append(h)

    # ---------- sanity check ----------
    captured_attentions.clear()
    dummy_ids = torch.randint(0, 100, (1, 10)).cuda()
    with torch.inference_mode():
        _ = model.model(input_ids=dummy_ids, use_cache=False, return_dict=True)
    if len(captured_attentions) == 0:
        raise RuntimeError("Hooks still not firing after monkey-patch.")
    print(f"[Sanity check passed] {len(captured_attentions)} layers captured, "
          f"shape={captured_attentions[0].shape}")
    captured_attentions.clear()

    # ---------- setup ----------
    num_image_patches = model.get_vision_tower().num_patches
    yes_token_id = tokenizer(" Yes", add_special_tokens=False).input_ids[-1]
    no_token_id  = tokenizer(" No",  add_special_tokens=False).input_ids[-1]

    questions    = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions    = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    attn_file    = os.path.expanduser(args.attn_file)
    warn_file    = os.path.expanduser(args.warnings_file)

    for path in [answers_file, attn_file, warn_file]:
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

    ans_f  = open(answers_file, "w")
    attn_f = open(attn_file,    "w")
    warn_f = open(warn_file,    "w")

    # ---------- main loop ----------
    for line in tqdm(questions):
        idx        = line["question_id"]
        image_file = line["image"]
        qs         = line["text"]
        cur_prompt = qs

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        qs = qs + " Answer the question using a single word or phrase."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()

        if args.use_icd:
            icd_prompt = get_random_icd_prompt()
            conv_cd = conv_templates[args.conv_mode].copy()
            conv_cd.system = icd_prompt
            conv_cd.append_message(conv.roles[0], qs)
            conv_cd.append_message(conv.roles[1], None)
            prompt_cd    = conv_cd.get_prompt()
            input_ids_cd = tokenizer_image_token(
                prompt_cd, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).cuda()
        else:
            input_ids_cd = None

        image        = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = process_images([image], image_processor, model.config)[0]
        image_sizes  = [image.size]

        if args.use_vcd:
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        else:
            image_tensor_cd = None

        stop_str          = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords          = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        use_sid           = True if args.use_sid else None

        # AnyRes: the number of image tokens is dynamic (base + hi-res crops +
        # newlines), so the image-token span is computed after each forward from
        # the attention key-length rather than the static num_image_patches.
        img_start, img_end = None, None

        # init all values
        exp_visual_mass  = None
        exp_total_mass   = None
        ama_visual_mass  = None
        ama_total_mass   = None
        fraction_expert  = None
        fraction_cd      = None
        expert_yes_no_gap = None
        cd_yes_no_gap    = None
        expert_token_id  = None
        cd_token_id      = None
        cd_flipped       = None

        with torch.inference_mode():

            # ---- EXPERT forward pass ----
            captured_attentions.clear()

            expert_out = model(
                input_ids=input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=image_sizes,
                output_attentions=False,
                return_dict=True,
                use_cache=False,
            )

            expert_logits = expert_out.logits[:, -1, :]
            expert_attn   = tuple(captured_attentions)

            if len(expert_attn) == 0:
                warn_f.write(f"[Warning] question_id={idx}: expert_attn hooks returned nothing\n")
                warn_f.flush()
            else:
                actual_seq = expert_attn[0].shape[-1]
                img_start, img_end = get_expanded_image_span_dynamic(
                    input_ids[0], IMAGE_TOKEN_INDEX, actual_seq
                )
                if img_end is not None and img_end <= actual_seq:
                    exp_visual_mass, exp_total_mass = compute_attention_masses(
                        expert_attn, img_start, img_end,
                        query_position=-1
                    )
                else:
                    warn_f.write(
                        f"[Warning] question_id={idx}: img_end={img_end} > "
                        f"actual_seq={actual_seq}, skipping expert masses\n"
                    )
                    warn_f.flush()

            expert_token_id   = int(torch.argmax(expert_logits, dim=-1).item())
            expert_yes_no_gap = float(
                (expert_logits[0, yes_token_id] - expert_logits[0, no_token_id]).item()
            )

            # ---- AMATEUR (CD) forward pass ----
            if args.use_vcd:
                cd_images    = image_tensor_cd.unsqueeze(0).half().cuda()
                cd_input_ids = input_ids
            elif args.use_icd:
                cd_images    = image_tensor.unsqueeze(0).half().cuda()
                cd_input_ids = input_ids_cd
            elif args.use_sid:
                cd_images    = None
                cd_input_ids = input_ids
            else:
                cd_images    = None
                cd_input_ids = None

            if cd_input_ids is not None:
                captured_attentions.clear()

                cd_out = model(
                    input_ids=cd_input_ids,
                    images=cd_images,
                    image_sizes=image_sizes,
                    output_attentions=False,
                    return_dict=True,
                    use_cache=False,
                )

                cd_logits_raw = cd_out.logits[:, -1, :]
                cd_attn       = tuple(captured_attentions)

                if len(cd_attn) == 0:
                    warn_f.write(f"[Warning] question_id={idx}: cd_attn hooks returned nothing\n")
                    warn_f.flush()
                else:
                    cd_actual_seq = cd_attn[0].shape[-1]

                    cd_img_start, cd_img_end = get_expanded_image_span_dynamic(
                        cd_input_ids[0], IMAGE_TOKEN_INDEX, cd_actual_seq
                    )

                    if cd_img_end is not None and cd_img_end <= cd_actual_seq:
                        ama_visual_mass, ama_total_mass = compute_attention_masses(
                            cd_attn, cd_img_start, cd_img_end,
                            query_position=-1
                        )
                    else:
                        warn_f.write(
                            f"[Warning] question_id={idx}: cd_img_end={cd_img_end} > "
                            f"cd_actual_seq={cd_actual_seq}, skipping amateur masses\n"
                        )
                        warn_f.flush()

                # correct CD formula: (1+alpha)*expert - alpha*amateur
                cd_alpha      = args.cd_alpha
                cd_combined   = (1 + cd_alpha) * expert_logits - cd_alpha * cd_logits_raw
                cd_token_id   = int(torch.argmax(cd_combined, dim=-1).item())
                cd_yes_no_gap = float(
                    (cd_combined[0, yes_token_id] - cd_combined[0, no_token_id]).item()
                )

                cd_flipped = bool(
                    expert_token_id == yes_token_id and cd_token_id == no_token_id
                )

                # visual fractions — the core comparison
                # fraction_expert = s_visual_exp / s_final_exp
                # fraction_cd     = ((1+alpha)*s_visual_exp - alpha*s_visual_ama)
                #                 / ((1+alpha)*s_final_exp  - alpha*s_final_ama)
                # claim: fraction_expert ≈ fraction_cd
                if exp_visual_mass is not None and ama_visual_mass is not None:
                    fraction_expert = exp_visual_mass / (exp_total_mass + 1e-8)

                    cd_visual   = (1 + cd_alpha) * exp_visual_mass - cd_alpha * ama_visual_mass
                    cd_total    = (1 + cd_alpha) * exp_total_mass  - cd_alpha * ama_total_mass
                    fraction_cd = cd_visual / (cd_total + 1e-8)

        # ------------------------------------------------------------------
        # FULL GENERATION
        # ------------------------------------------------------------------
        captured_attentions.clear()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True,
                images_cd=(image_tensor_cd.unsqueeze(0).half().cuda() if image_tensor_cd is not None else None),
                image_sizes_cd=image_sizes,
                input_ids_cd=input_ids_cd,
                use_sid=use_sid
            )

        captured_attentions.clear()

        input_token_len = input_ids.shape[1]
        n_diff = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff > 0:
            warn_f.write(
                f"[Warning] question_id={idx}: {n_diff} output_ids differ from input_ids\n"
            )
            warn_f.flush()

        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        # ------------------------------------------------------------------
        # WRITE OUTPUTS
        # ------------------------------------------------------------------
        ans_id = shortuuid.uuid()

        ans_f.write(json.dumps({
            "question_id": idx,
            "prompt":      cur_prompt,
            "text":        outputs,
            "answer_id":   ans_id,
            "model_id":    model_name,
            "metadata": {
                "expert_token_id":         expert_token_id,
                "cd_token_id":             cd_token_id,
                "cd_flipped_yes_to_no":    cd_flipped,
                "fraction_expert":         fraction_expert,
                "fraction_cd":             fraction_cd,
                "expert_yes_no_logit_gap": expert_yes_no_gap,
                "cd_yes_no_logit_gap":     cd_yes_no_gap,
            }
        }) + "\n")
        ans_f.flush()

        attn_f.write(json.dumps({
            "question_id":             idx,
            "image":                   image_file,
            "exp_visual_mass":         exp_visual_mass,
            "exp_total_mass":          exp_total_mass,
            "ama_visual_mass":         ama_visual_mass,
            "ama_total_mass":          ama_total_mass,
            "fraction_expert":         fraction_expert,
            "fraction_cd":             fraction_cd,
            "expert_yes_no_logit_gap": expert_yes_no_gap,
            "cd_yes_no_logit_gap":     cd_yes_no_gap,
            "expert_token_id":         expert_token_id,
            "cd_token_id":             cd_token_id,
            "cd_flipped_yes_to_no":    cd_flipped,
            "num_image_patches":       num_image_patches,
            "img_token_span":          [img_start, img_end],
            "cd_alpha":                args.cd_alpha,
        }) + "\n")
        attn_f.flush()

    # ---------- cleanup ----------
    for h in hooks:
        h.remove()

    ans_f.close()
    attn_f.close()
    warn_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",    type=str,   default="facebook/opt-350m")
    parser.add_argument("--model-base",    type=str,   default=None)
    parser.add_argument("--image-folder",  type=str,   default="")
    parser.add_argument("--question-file", type=str,   default="tables/question.jsonl")
    parser.add_argument("--answers-file",  type=str,   default="answer.jsonl")
    parser.add_argument("--attn-file",     type=str,   default="attn_grounding.jsonl")
    parser.add_argument("--warnings-file", type=str,   default="warnings.log")
    parser.add_argument("--conv-mode",     type=str,   default="llava_v1")
    parser.add_argument("--num-chunks",    type=int,   default=1)
    parser.add_argument("--chunk-idx",     type=int,   default=0)
    parser.add_argument("--temperature",   type=float, default=0.2)
    parser.add_argument("--top_p",         type=float, default=None)
    parser.add_argument("--num_beams",     type=int,   default=1)
    parser.add_argument("--cd-alpha",      type=float, default=1.0)
    parser.add_argument("--use-icd",   action='store_true', default=False)
    parser.add_argument("--use-vcd",   action='store_true', default=False)
    parser.add_argument("--use-sid",   action='store_true', default=False)
    parser.add_argument("--noise-step", type=int, default=900)
    args = parser.parse_args()

    eval_model(args)