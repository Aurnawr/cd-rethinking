#!/usr/bin/env python3
"""Fail-fast validation of the Qwen2.5-VL CD pipeline before any multi-hour job.

Every check here guards a failure mode that would otherwise produce *plausible but wrong* figures
rather than an error. Ordered cheapest-first; the first failure exits non-zero.

  1. layer count      L+1 hidden states with L == config.num_hidden_layers
  2. lens fidelity    |lens_margin[L] - true_margin| ~ 0. A wrong norm/head contract (e.g. if a
                      transformers bump stops appending the post-norm state) silently corrupts
                      every margin, so this is the single most important check.
  3. generate vs logit  model.generate(max_new_tokens=5)'s first token agrees with the forward
                      argmax used as `pred` -- the same sanity check the LLaVA pipeline runs
                      (mech_interp/extract_activations.py:207).
  4. Yes/No coverage  the greedy first token is actually Yes-ish or No-ish. If Qwen answers with
                      prose, the prompt is wrong and the Yes-No margin is meaningless.
  5. attention capture  the CT2S hook receives real attention weights (i.e. eager attention is
                      active); sdpa silently yields None.
  6. SID bites        exactly n_vision - round(ratio*n_vision) vision columns are masked, and the
                      SID amateur's margin actually differs from the expert's.
  7. amateurs differ  VCD and ICD margins differ from the expert too (catches a no-op amateur).

Usage:
    python selfcheck.py --model-path <weights> --data-dir <data> [--n 8]
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from amateur_branches import METHODS, amateur_context, build_amateur_inputs  # noqa: E402
from qwen_runtime import (  # noqa: E402
    MODEL_ID, build_inputs, decode_is_yes, forward_hiddens, image_token_id, label_to_int,
    lens_margins, load_model, load_questions, n_hidden_layers, pick_device, vision_positions,
    yes_no_token_ids,
)
from sid_ct2s import SID_KEEP_RATIO, SID_RANK_LAYER, SidCT2S  # noqa: E402


class CheckFailed(Exception):
    pass


def report(ok, name, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        raise CheckFailed(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=MODEL_ID)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="random")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-impl", default="eager")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--noise-step", type=int, default=900)
    ap.add_argument("--sid-rank-layer", type=int, default=SID_RANK_LAYER)
    ap.add_argument("--sid-keep-ratio", type=float, default=SID_KEEP_RATIO)
    ap.add_argument("--sid-keep-tokens", type=int, default=None)
    ap.add_argument("--lens-tol", type=float, default=1e-2)
    ap.add_argument("--min-yesno", type=float, default=0.75,
                    help="min fraction of samples whose greedy first token is Yes/No")
    args = ap.parse_args()

    print("=" * 70)
    print(f"selfcheck: {args.model_path}  dtype={args.dtype}  attn={args.attn_impl}")
    print("=" * 70)

    device = pick_device(args.device)
    processor, model = load_model(args.model_path, dtype=args.dtype, attn_impl=args.attn_impl,
                                  device=device)
    tok = processor.tokenizer
    yes_ids, no_ids = yes_no_token_ids(tok)
    yes_t = torch.tensor(yes_ids, device=device)
    no_t = torch.tensor(no_ids, device=device)
    img_tok = image_token_id(model, processor)
    n_layers = n_hidden_layers(model)
    sid = SidCT2S(model, rank_layer=args.sid_rank_layer, keep_ratio=args.sid_keep_ratio,
                  keep_tokens=args.sid_keep_tokens)
    print(f"model: {n_layers} decoder layers, image_token_id={img_tok}, "
          f"yes_ids={yes_ids} no_ids={no_ids}")
    print(f"sid:   {sid.describe()}")

    pope_dir = os.path.join(args.data_dir, "pope", "coco")
    images_dir = os.path.join(pope_dir, "images")
    questions = load_questions(pope_dir, args.split)[: args.n]
    if not questions:
        raise SystemExit(f"no questions in {pope_dir} for split '{args.split}'")

    lens_err, gen_match, yesno_hits, correct = [], 0, 0, 0
    delta_seen = {m: [] for m in METHODS}
    sid_stats = []
    attn_captured = False
    n_states = set()

    for line in questions:
        image = Image.open(os.path.join(images_dir, line["image"])).convert("RGB")
        inputs = build_inputs(processor, image, line["text"], device=device)
        hs_e, logits_e = forward_hiddens(model, inputs)
        m_e = lens_margins(model, hs_e, yes_t, no_t)

        n_states.add(int(hs_e.shape[0]))
        true_margin = float(logits_e[yes_t].max() - logits_e[no_t].max())
        lens_err.append(abs(float(m_e[-1]) - true_margin))

        pred_id = int(torch.argmax(logits_e).item())
        pred_tok = tok.decode([pred_id]).strip()
        pred_yes = decode_is_yes(tok, pred_id)
        gt = label_to_int(line["label"])
        correct += int(pred_yes == gt)
        yesno_hits += int(pred_tok.lower().startswith(("yes", "no")))

        with torch.inference_mode():
            gen_ids = model.generate(**inputs, do_sample=False, max_new_tokens=5)
        gen = tok.decode(gen_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        gen_match += int(bool(pred_tok) and gen.lower().startswith(pred_tok.lower()))

        for method in METHODS:
            torch.manual_seed(1234)
            am_inputs = build_amateur_inputs(method, processor, inputs, image, line["text"],
                                             args.noise_step, device=device)
            with amateur_context(method, am_inputs, sid=sid, img_token_id=img_tok):
                hs_a, _ = forward_hiddens(model, am_inputs)
            m_a = lens_margins(model, hs_a, yes_t, no_t)
            delta_seen[method].append(abs(float(m_e[-1] - m_a[-1])))
            if method == "sid":
                attn_captured = attn_captured or torch.is_tensor(sid.attn_weights)
                n_vis = int(vision_positions(am_inputs["input_ids"], img_tok).numel())
                sid_stats.append((n_vis, sid.n_vision, sid.n_keep, sid.n_dropped))

        print(f"  gt={line['label']:>3}  pred='{pred_tok}'  generate='{gen}'  "
              f"m_expert={true_margin:+.2f}  lens={m_e[-1]:+.2f}", flush=True)

    n = len(questions)
    print("\nchecks:")
    report(n_states == {n_layers + 1}, "layer count",
           f"hidden-state counts seen: {sorted(n_states)} (expected {n_layers + 1} for L={n_layers})")

    mae = float(np.mean(lens_err))
    report(mae < args.lens_tol, "lens fidelity",
           f"mean |lens[L] - true| = {mae:.2e} (tol {args.lens_tol:g})")

    report(gen_match >= n - 1, "generate vs logit-argmax", f"{gen_match}/{n} agree")

    report(yesno_hits / n >= args.min_yesno, "Yes/No coverage",
           f"{yesno_hits}/{n} greedy tokens are Yes/No")

    report(attn_captured, "attention capture",
           f"CT2S hook saw attention weights at layer {args.sid_rank_layer} "
           f"(needs attn_implementation='eager')")

    n_vis, sid_vis, keep, dropped = sid_stats[0]
    expected_keep = (args.sid_keep_tokens if args.sid_keep_tokens is not None
                     else max(1, min(int(round(args.sid_keep_ratio * n_vis)), n_vis)))
    report(sid_vis == n_vis and keep == expected_keep and dropped == n_vis - expected_keep,
           "SID token selection",
           f"{keep}/{n_vis} vision tokens kept, {dropped} masked (expected keep {expected_keep})")

    for method in METHODS:
        mean_delta = float(np.mean(delta_seen[method]))
        report(mean_delta > 1e-4, f"{method} amateur differs from expert",
               f"mean |Delta| at readout = {mean_delta:.4f}")

    print(f"\nbaseline greedy accuracy on {n} '{args.split}' samples: {correct / n:.3f}  "
          f"(near 0.5 on a real run means the prompt is wrong -- stop and fix)")
    print("selfcheck: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CheckFailed as e:
        print(f"\nselfcheck FAILED at: {e}", file=sys.stderr)
        raise SystemExit(1)
