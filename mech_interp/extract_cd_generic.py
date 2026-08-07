#!/usr/bin/env python3
"""Method-agnostic logit-lens extractor for VCD / ICD / SID on POPE-COCO.

For every POPE question we run two single forward passes (T=1, no generation loop),
using THIS repo's own LLaVA machinery so the representations match the real pipeline:

  * expert : model(images=clean_image)                     -- the amateur-free branch
  * amateur: method-specific contrastive branch (amateur_branches.build_amateur_inputs)

At the final prompt-token position we apply the model's OWN frozen head as a logit lens
(final RMSNorm `model.model.norm` + `model.lm_head`) to each of the 33 hidden layers,
restrict to the Yes/No token rows, and record the decision margin

    m_l = max_Yes z_l - max_No z_l        (positive when the model would answer "Yes")

for BOTH branches. Since every CD method here forms z_CD = (1+a)z_expert - a*z_amateur
and the margin is linear in logits, the per-unit-alpha shift is Delta_l = m^E_l - m^A_l,
recovered offline from the two saved margin arrays.

Outputs (per method, under --out-dir):
  logit_lens_per_sample_<method>.npz
     clean_margin   [N,33] f32   expert Yes-No margin per layer
     amateur_margin [N,33] f32   amateur Yes-No margin per layer
     gt   [N] i8   1=object present, 0=absent
     pred [N] i8   baseline greedy prediction from expert logits (1=Yes)
     split[N] U12
  hiddens_ht_<method>.npz   (only H u T, i.e. pred==Yes -- the analysis population)
     expert_hidden   [n,33,4096] f16
     amateur_hidden  [n,33,4096] f16
     gt   [n] i8
     idx  [n] i32   row index back into the per-sample arrays
  extract_summary_<method>.json

The lens is verified to reproduce the true final logits (layer 32) to a small MAE.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "inference"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.mm_utils import get_model_name_from_path  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402

# Reuse the exact prompt / token / io helpers from the original extractor -- do NOT
# duplicate them, so the expert branch is byte-identical to the validated pipeline.
from extract_activations import (  # noqa: E402
    ANSWER_SUFFIX,  # noqa: F401  (kept for parity / debugging)
    build_input_ids,
    label_to_int,
    load_questions,
    yes_no_token_ids,
)
from amateur_branches import METHODS, build_amateur_inputs  # noqa: E402
from sid_correct import (  # noqa: E402
    SID_AGG_LAYER,
    SID_ATTENTION_RANK,
    install_correct_sid,
    set_sid_key_position,
)

SPLITS = ("random", "popular", "adversarial")


@torch.inference_mode()
def forward_hiddens(model, input_ids, image_tensor, use_sid=None):
    """One forward pass. Return (hs [33,4096] cuda model-dtype, last_logits [V] f32 cuda)."""
    out = model(
        input_ids,
        images=image_tensor.unsqueeze(0).half().cuda(),
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
        use_sid=use_sid,
    )
    hs = torch.stack([h[0, -1, :] for h in out.hidden_states], dim=0)  # [33,4096]
    last_logits = out.logits[0, -1, :].float()
    return hs, last_logits


@torch.inference_mode()
def lens_margins(model, hs, yes_ids, no_ids):
    """Logit-lens Yes-No margin per layer. hs:[33,4096] cuda -> margins [33] np.float32.

    This repo's custom LlamaModel appends the FINAL hidden state AFTER `self.norm`
    (custom_modeling_llama.py L751->755), so hidden_states[32] is already post-norm and
    equals the lm_head input. Layers 0..31 are pre-norm residuals: apply the frozen final
    RMSNorm to them; leave layer 32 as-is so the readout margin is exactly the true margin.
    """
    W = model.lm_head.weight  # [V,4096]
    h = hs.to(W.dtype).clone()
    h[:-1] = model.model.norm(h[:-1])  # layers 0..31 (pre-norm) -> through final norm
    # h[-1] is hidden_states[32], already normed -> use directly
    logits = h @ W.t()  # [33, V]
    yes = logits[:, yes_ids].max(dim=1).values
    no = logits[:, no_ids].max(dim=1).values
    return (yes - no).float().cpu().numpy()


def extract_method(args, tokenizer, model, image_processor):
    method = args.method
    yes_ids, no_ids = yes_no_token_ids(tokenizer)
    yes_t = torch.tensor(yes_ids, device="cuda")
    no_t = torch.tensor(no_ids, device="cuda")

    pope_dir = os.path.join(args.data_dir, "pope", "coco")
    images_dir = os.path.join(pope_dir, "images")
    os.makedirs(args.out_dir, exist_ok=True)

    clean_margin, amateur_margin = [], []
    gt_all, pred_all, split_all = [], [], []
    exp_h, ama_h, ht_gt, ht_idx = [], [], [], []
    lens_mae_sum, lens_mae_n = 0.0, 0
    row = 0

    per_sample = os.path.join(args.out_dir, f"logit_lens_per_sample_{method}.npz")
    ht_path = os.path.join(args.out_dir, f"hiddens_ht_{method}.npz")

    def flush(done_splits):
        """Checkpoint everything accumulated so far (robust to mid-run kills).

        Written atomically-ish after each split so a disconnect/preemption costs at
        most the split in flight, never the whole run.
        """
        np.savez_compressed(
            per_sample,
            clean_margin=np.asarray(clean_margin, np.float32),
            amateur_margin=np.asarray(amateur_margin, np.float32),
            gt=np.asarray(gt_all, np.int8),
            pred=np.asarray(pred_all, np.int8),
            split=np.asarray(split_all, dtype="U12"),
        )
        np.savez_compressed(
            ht_path,
            expert_hidden=np.stack(exp_h) if exp_h else np.zeros((0, 33, 4096), np.float16),
            amateur_hidden=np.stack(ama_h) if ama_h else np.zeros((0, 33, 4096), np.float16),
            gt=np.asarray(ht_gt, np.int8),
            idx=np.asarray(ht_idx, np.int32),
        )
        n = len(gt_all)
        n_h = sum(1 for g, p in zip(gt_all, pred_all) if g == 0 and p == 1)
        n_t = sum(1 for g, p in zip(gt_all, pred_all) if g == 1 and p == 1)
        lens_mae = lens_mae_sum / max(lens_mae_n, 1)
        summary = {
            "method": method, "n": n, "n_H": n_h, "n_T": n_t,
            "lens_final_layer_MAE": lens_mae,
            "splits_requested": list(args.splits), "splits_done": list(done_splits),
            "per_sample": per_sample, "hiddens_ht": ht_path,
        }
        with open(os.path.join(args.out_dir, f"extract_summary_{method}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[{method}] checkpoint after {done_splits}: N={n} H={n_h} T={n_t} "
              f"lens MAE={lens_mae:.2e}", flush=True)
        return summary

    done_splits = []
    summary = None
    for split in args.splits:
        questions = load_questions(pope_dir, split)
        if args.max_samples > 0:
            questions = questions[: args.max_samples]
        for i, line in enumerate(tqdm(questions, desc=f"{method}:{split}")):
            try:
                image = Image.open(os.path.join(images_dir, line["image"])).convert("RGB")
            except (FileNotFoundError, OSError) as e:
                print(f"[skip] {line['image']}: {e}")
                continue

            input_ids = build_input_ids(tokenizer, model, line["text"], args.conv_mode)
            clean_img = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

            # deterministic amateur (VCD diffusion noise reproduces on re-run;
            # corrected SID is attention-deterministic, no RNG)
            torch.manual_seed(args.noise_seed + row)
            am = build_amateur_inputs(
                method, tokenizer, model, input_ids, line["text"], clean_img,
                args.conv_mode, args.noise_step,
            )
            # SID: record the image-band start for THIS sample's amateur pass
            # (must be read from the pre-multimodal-merge input_ids).
            if method == "sid":
                set_sid_key_position(model, am["input_ids"])

            hs_e, logits_e = forward_hiddens(model, input_ids, clean_img, use_sid=None)
            hs_a, _ = forward_hiddens(
                model, am["input_ids"], am["image_tensor"], **am["fwd_kwargs"]
            )

            m_e = lens_margins(model, hs_e, yes_t, no_t)
            m_a = lens_margins(model, hs_a, yes_t, no_t)

            # lens fidelity check on the final layer (should match true logits' margin)
            true_margin = float(logits_e[yes_t].max() - logits_e[no_t].max())
            lens_mae_sum += abs(m_e[-1] - true_margin)
            lens_mae_n += 1

            pred_id = int(torch.argmax(logits_e).item())
            pred_yes = 1 if tokenizer.decode([pred_id]).strip().lower().startswith("yes") else 0
            gt = label_to_int(line["label"])

            clean_margin.append(m_e)
            amateur_margin.append(m_a)
            gt_all.append(gt)
            pred_all.append(pred_yes)
            split_all.append(split)

            if pred_yes == 1:  # H u T -- the population every Delta analysis restricts to
                exp_h.append(hs_e.float().cpu().numpy().astype(np.float16))
                ama_h.append(hs_a.float().cpu().numpy().astype(np.float16))
                ht_gt.append(gt)
                ht_idx.append(row)
            row += 1

        done_splits.append(split)
        summary = flush(done_splits)  # checkpoint after every split

    print(f"[{method}] DONE  wrote {per_sample}\n[{method}] wrote {ht_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-base", default=None)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--conv-mode", default="vicuna_v1")
    ap.add_argument("--noise-step", type=int, default=900)
    ap.add_argument("--noise-seed", type=int, default=1234)
    ap.add_argument("--max-samples", type=int, default=0, help="0=all; else cap per split")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--sid-attention-rank", type=int, default=SID_ATTENTION_RANK,
                    help="SID: number of LEAST-attended vision tokens kept (SID default 100)")
    ap.add_argument("--sid-agg-layer", type=int, default=SID_AGG_LAYER,
                    help="SID: decoder layer at which vision tokens are ranked (default 2)")
    args = ap.parse_args()

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    print(f"[model] {model_name}  num_hidden_layers={model.config.num_hidden_layers}")

    if args.method == "sid":
        # Replace the repo's RANDOM vision-token selection with faithful SID (CT2S):
        # attention-ranked least-important tokens. No repo file is edited.
        install_correct_sid(model, args.sid_agg_layer, args.sid_attention_rank)
        print(f"[sid] faithful CT2S installed  agg_layer={args.sid_agg_layer}  "
              f"attention_rank={args.sid_attention_rank} (kept, least-attended of {576})")

    extract_method(args, tokenizer, model, image_processor)


if __name__ == "__main__":
    main()
