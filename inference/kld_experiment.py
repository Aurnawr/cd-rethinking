import argparse
import torch
import os
import json
import random
from tqdm import tqdm
from torch import nn

from datasets import load_dataset

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from PIL import Image
import io

from cd_utils.vcd_utils import add_diffusion_noise
from cd_utils.icd_utils import get_random_icd_prompt


# ---------------------------------------------------------------------------
# KL divergence helper
# ---------------------------------------------------------------------------

def kl_divergence(logits_expert: torch.Tensor, logits_amateur: torch.Tensor) -> float:
    """
    KL( P_expert || P_amateur )
    Both inputs are raw logits of shape (1, vocab_size).
    torch.nn.functional.kl_div(log_Q, P) = sum( P * (log P - log Q) ) = KL(P||Q)
    """
    kld = nn.functional.kl_div(
        nn.functional.log_softmax(logits_amateur, dim=-1),  # log Q  (amateur)
        nn.functional.softmax(logits_expert,      dim=-1),  # P      (expert)
        reduction="sum",
        log_target=False,
    )
    return kld.item()


# ---------------------------------------------------------------------------
# Per-method forward pass for the amateur model
# ---------------------------------------------------------------------------

def get_amateur_logits_vcd(model, input_ids, model_kwargs_cd,
                            output_attentions, output_hidden_states):
    """VCD amateur: same input_ids, but noisy image via prepare_inputs_for_generation_vcd."""
    model_inputs_cd = model.prepare_inputs_for_generation_vcd(input_ids, **model_kwargs_cd)
    with torch.inference_mode():
        outputs_cd = model(
            **model_inputs_cd,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
    return outputs_cd.logits[:, -1, :], outputs_cd


def get_amateur_logits_icd(model, input_ids, input_ids_cd, model_kwargs_cd,
                            pad_token_id, eos_token_id,
                            output_attentions, output_hidden_states):
    """ICD amateur: different prompt (system prompt perturbed), same generated tokens appended."""
    import transformers
    generationMixin = transformers.generation.utils.GenerationMixin()
    model_kwargs_cd["attention_mask"] = generationMixin._prepare_attention_mask_for_generation(
        input_ids_cd, pad_token_id, eos_token_id
    )
    model_inputs_cd = model.prepare_inputs_for_generation(input_ids_cd, **model_kwargs_cd)
    with torch.inference_mode():
        outputs_cd = model(
            **model_inputs_cd,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
    return outputs_cd.logits[:, -1, :], outputs_cd


def get_amateur_logits_sid(model, input_ids, model_kwargs_cd,
                            output_attentions, output_hidden_states):
    """SID amateur: same input_ids, but no image via prepare_inputs_for_generation_sid."""
    model_inputs_cd = model.prepare_inputs_for_generation_sid(input_ids, **model_kwargs_cd)
    with torch.inference_mode():
        outputs_cd = model(
            **model_inputs_cd,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
    return outputs_cd.logits[:, -1, :], outputs_cd


# ---------------------------------------------------------------------------
# CD next-token selection (mirrors the logic in *_utils exactly)
# ---------------------------------------------------------------------------

def cd_next_token(logits_expert, logits_amateur, cd_alpha=1.0, cd_beta=0.2, do_sample=False):
    """
    Apply the CD formula and return the chosen next token.
    do_sample=False -> greedy (argmax)
    do_sample=True  -> multinomial sampling from the CD distribution
                       (mirrors the sample() path in vcd/icd/sid_utils)
    NOTE: logits passed in should already be temperature-scaled by the caller.
    """
    cutoff    = torch.log(torch.tensor(cd_beta)) + logits_expert.max(dim=-1, keepdim=True).values
    diffs     = (1 + cd_alpha) * logits_expert - cd_alpha * logits_amateur
    cd_logits = diffs.masked_fill(logits_expert < cutoff, -float("inf"))

    if do_sample:
        cd_probs   = nn.functional.softmax(cd_logits, dim=-1)
        next_token = torch.multinomial(cd_probs, num_samples=1).squeeze(1)
    else:
        next_token = torch.argmax(cd_logits, dim=-1)

    return next_token, cd_logits


# ---------------------------------------------------------------------------
# Main generation loop for one question
# ---------------------------------------------------------------------------

def run_cd_kld_loop(
    model, tokenizer, image_processor,
    input_ids,          # (1, seq_len)  — expert prompt tokens
    image_tensor,       # (C, H, W)     — clean image
    method,             # "vcd" | "icd" | "sid"
    image_sizes=None,   # AnyRes: [(W, H)] original size for spatial unpad
    # method-specific extras
    image_tensor_cd=None,   # VCD: noisy image tensor
    input_ids_cd=None,      # ICD: perturbed prompt token ids (1, seq_len_cd)
    # CD hyper-params
    cd_alpha=1.0,
    cd_beta=0.2,
    noise_step=900,
    max_new_tokens=128,
    temperature=0.0,    # 0 = greedy (matches --temperature 0 in your shell scripts)
    # stopping
    stop_str=None,
):
    """
    Runs the CD generation loop for one sample.
    At every timestep t:
      1. Expert forward pass  → P_expert(· | context_t)
      2. Amateur forward pass → P_amateur(· | context_t)   (context_t = CD-chosen tokens so far)
      3. KLD  = KL( P_expert || P_amateur )  recorded
      4. Next token chosen via the CD mechanism → appended to context

    Returns:
        kld_values : list of float, one per generated token
        generated_text : str
    """
    device = input_ids.device

    # ---- initialise model kwargs for expert ----
    model_kwargs = {
        "images": image_tensor.unsqueeze(0).half().to(device),
        "image_sizes": image_sizes,
        "attention_mask": torch.ones(input_ids.shape, dtype=torch.long, device=device),
        "use_cache": True,
    }

    # ---- initialise model kwargs for amateur ----
    if method == "vcd":
        assert image_tensor_cd is not None
        model_kwargs_cd = {
            "images_cd": image_tensor_cd.unsqueeze(0).half().to(device),
            "image_sizes_cd": image_sizes,
            "attention_mask": torch.ones(input_ids.shape, dtype=torch.long, device=device),
            "use_cache": True,
        }
    elif method == "icd":
        assert input_ids_cd is not None
        model_kwargs_cd = {
            "input_ids_cd": input_ids_cd.clone(),
            "images": image_tensor.unsqueeze(0).half().to(device),  # ICD uses same clean image
            "image_sizes": image_sizes,
            "attention_mask": torch.ones(input_ids_cd.shape, dtype=torch.long, device=device),
            "use_cache": True,
        }
    elif method == "sid":
        model_kwargs_cd = {
            "use_sid": True,
            "images": image_tensor.unsqueeze(0).half().to(device),
            "image_sizes": image_sizes,
            "attention_mask": torch.ones(input_ids.shape, dtype=torch.long, device=device),
            "use_cache": True,
        }
    else:
        raise ValueError(f"Unknown method: {method}")

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(device)

    kld_values     = []
    generated_ids  = []

    # shared growing sequence driven by the CD mechanism
    current_input_ids = input_ids.clone()  # (1, seq_len)

    # ICD: we also grow a parallel cd sequence (different prefix, same new tokens)
    if method == "icd":
        current_input_ids_cd = input_ids_cd.clone()

    for step in range(max_new_tokens):
        # ----------------------------------------------------------------
        # 1. Expert forward pass
        # ----------------------------------------------------------------
        model_inputs = model.prepare_inputs_for_generation(current_input_ids, **model_kwargs)
        with torch.inference_mode():
            outputs = model(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
        logits_expert = outputs.logits[:, -1, :]   # (1, vocab)

        # ----------------------------------------------------------------
        # 2. Amateur forward pass  (same context = current_input_ids)
        # ----------------------------------------------------------------
        if method == "vcd":
            logits_amateur, outputs_cd = get_amateur_logits_vcd(
                model, current_input_ids, model_kwargs_cd,
                output_attentions=False, output_hidden_states=False,
            )
        elif method == "icd":
            logits_amateur, outputs_cd = get_amateur_logits_icd(
                model, current_input_ids, current_input_ids_cd, model_kwargs_cd,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_attentions=False, output_hidden_states=False,
            )
        elif method == "sid":
            logits_amateur, outputs_cd = get_amateur_logits_sid(
                model, current_input_ids, model_kwargs_cd,
                output_attentions=False, output_hidden_states=False,
            )

        # ----------------------------------------------------------------
        # 3. KL divergence  — always on raw (unscaled) logits
        # ----------------------------------------------------------------
        kld = kl_divergence(logits_expert, logits_amateur)
        kld_values.append(kld)

        # ----------------------------------------------------------------
        # 4. Pick next token via CD mechanism
        #    Apply temperature scaling here (after KLD) if temperature > 0.
        #    temperature=0 → greedy (argmax), matching --temperature 0 in your scripts.
        # ----------------------------------------------------------------
        if temperature > 0:
            logits_expert_scaled  = logits_expert  / temperature
            logits_amateur_scaled = logits_amateur / temperature
        else:
            logits_expert_scaled  = logits_expert
            logits_amateur_scaled = logits_amateur

        next_token, _ = cd_next_token(logits_expert_scaled, logits_amateur_scaled, cd_alpha, cd_beta, do_sample=(temperature > 0))
        # next_token: (1,)

        generated_ids.append(next_token.item())

        # ----------------------------------------------------------------
        # 5. Check stopping conditions
        # ----------------------------------------------------------------
        if eos_token_id_tensor is not None:
            if next_token.item() in eos_token_id:
                break

        decoded_so_far = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if stop_str and decoded_so_far.endswith(stop_str):
            break

        # ----------------------------------------------------------------
        # 6. Append token and update KV-cache kwargs
        # ----------------------------------------------------------------
        current_input_ids = torch.cat([current_input_ids, next_token[:, None]], dim=-1)

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False
        )

        if method == "vcd":
            model_kwargs_cd = model._update_model_kwargs_for_generation(
                outputs_cd, model_kwargs_cd, is_encoder_decoder=False
            )
        elif method == "icd":
            # ICD: append the same new token to the cd sequence too
            current_input_ids_cd = torch.cat([current_input_ids_cd, next_token[:, None]], dim=-1)
            model_kwargs_cd["input_ids_cd"] = current_input_ids_cd
            model_kwargs_cd = model._update_model_kwargs_for_generation(
                outputs_cd, model_kwargs_cd, is_encoder_decoder=False
            )
        elif method == "sid":
            model_kwargs_cd = model._update_model_kwargs_for_generation(
                outputs_cd, model_kwargs_cd, is_encoder_decoder=False
            )

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return kld_values, generated_text


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_llava_bench():
    ds = load_dataset("mlfoundations/VisIT-Bench", split="test", verification_mode="no_checks")
    samples = []
    for i, item in enumerate(ds):
        samples.append({
            "question_id": i,
            "image": item["image"],        # PIL.Image
            "question": item["instruction"],  # field name is different
        })
    print(f"Loaded {len(samples)} samples from VisIT-Bench")
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def eval_model(args):
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    samples = load_llava_bench()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    results = []

    for sample in tqdm(samples, desc=f"[{args.method.upper()}] KLD experiment"):
        question_id = sample["question_id"]
        pil_image   = sample["image"]
        question    = sample["question"]

        # ---- build prompt (identical to inference_cd.py) ----
        qs = question
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()

        # ---- preprocess image ----
        image_tensor = process_images([pil_image], image_processor, model.config)[0]
        image_sizes = [pil_image.size]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        # ---- method-specific setup ----
        image_tensor_cd = None
        input_ids_cd    = None

        if args.method == "vcd":
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)

        elif args.method == "icd":
            icd_prompt = get_random_icd_prompt()
            conv_cd = conv_templates[args.conv_mode].copy()
            conv_cd.system = icd_prompt
            conv_cd.append_message(conv.roles[0], qs)
            conv_cd.append_message(conv.roles[1], None)
            prompt_cd = conv_cd.get_prompt()
            input_ids_cd = tokenizer_image_token(
                prompt_cd, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).cuda()

        # SID needs no extra setup

        # ---- run loop ----
        kld_values, generated_text = run_cd_kld_loop(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            input_ids=input_ids,
            image_tensor=image_tensor.cuda(),
            image_sizes=image_sizes,
            method=args.method,
            image_tensor_cd=image_tensor_cd.cuda() if image_tensor_cd is not None else None,
            input_ids_cd=input_ids_cd,
            cd_alpha=args.cd_alpha,
            cd_beta=args.cd_beta,
            noise_step=args.noise_step,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            stop_str=stop_str,
        )

        results.append({
            "question_id":   question_id,
            "question":      question,
            "generated":     generated_text,
            "kld_per_token": kld_values,       # list[float], index = timestep
            "num_tokens":    len(kld_values),
        })

        # flush after every sample so we don't lose progress
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone. Results saved to {args.output_file}")
    print(f"Total samples: {len(results)}")
    print(f"Avg generated length: {sum(r['num_tokens'] for r in results) / len(results):.1f} tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",      type=str,   required=True)
    parser.add_argument("--model-base",      type=str,   default=None)
    parser.add_argument("--conv-mode",       type=str,   default="vicuna_v1")
    parser.add_argument("--method",          type=str,   required=True,
                        choices=["vcd", "icd", "sid"],
                        help="Which CD method to run")
    parser.add_argument("--output-file",     type=str,   required=True,
                        help="Path to save JSON results (e.g. outputs/vcd_kld.json)")
    # CD hyper-params
    parser.add_argument("--cd-alpha",        type=float, default=1.0)
    parser.add_argument("--cd-beta",         type=float, default=0.2)
    parser.add_argument("--noise-step",      type=int,   default=900,
                        help="VCD only: diffusion noise step")
    # generation — temperature 0 means greedy (matches your other shell scripts)
    parser.add_argument("--temperature",     type=float, default=0.0)
    parser.add_argument("--max-new-tokens",  type=int,   default=128)

    args = parser.parse_args()
    eval_model(args)