#!/usr/bin/env python3
"""One-pass logit-lens extraction for VCD / ICD / SID on Qwen2.5-VL over POPE-COCO.

For every POPE question we run a single forward pass per branch (T=1, no generation loop):

  * expert  : the clean, amateur-free pass -- shared by all three methods
  * amateur : one per method (noised pixels / adversarial system prompt / CT2S masking)

The LLaVA pipeline this ports from ran `extract_activations.py` (2 passes) and then
`extract_cd_generic.py` once per method (2 passes each) = 8 forwards/sample, recomputing the
identical expert pass four times. Here the expert pass is computed once and shared:
**4 forwards/sample**, one resumable job, same artifacts.

At the final prompt-token position we apply the model's OWN frozen head as a logit lens (final
RMSNorm + lm_head) to each of the L+1 hidden layers, restrict to the Yes/No token rows, and record
the decision margin

    m_l = max_Yes z_l - max_No z_l          (positive when the model would answer "Yes")

for the expert and for each amateur. Since every CD method forms
z_CD = (1+a) z_expert - a z_amateur and the margin is linear in logits, the per-unit-alpha shift
is Delta_l = m^E_l - m^A_l, recovered offline from the two saved margin arrays.

Outputs
-------
<results-dir>/logit_lens_per_sample_<method>.npz
    clean_margin   [N,L+1] f32   expert Yes-No margin per layer
    amateur_margin [N,L+1] f32   amateur Yes-No margin per layer
    gt   [N] i8    1=object present, 0=absent
    pred [N] i8    baseline greedy prediction from expert logits (1=Yes)
    split[N] U12
<results-dir>/extract_summary_<method>.json
<act-dir>/<split>_shard####.npz
    clean_hidden      [n,L+1,d] f16   expert residuals -> the per-layer probes
    clean_norm        [n,L+1]   f32   ||h_expert[l]||               (the clean-norm confound)
    cd_change_<m>     [n,L+1]   f32   ||h_expert[l] - h_amateur[l]|| per method
    gt, pred          [n]       i8
<act-dir>/<split>_shard####.meta.jsonl

Residual-magnitude stats (resid_stats.py) are derived from `clean_norm` / `cd_change_*` rather
than from stored hidden tensors, so no multi-GB H-union-T hidden dumps are needed and the norms
are computed in fp32 before any fp16 rounding.

Checkpointing: every split flushes all per-method npz + summaries, so a disconnect costs at most
the split in flight. `--resume` picks up from the last completed split.
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
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from amateur_branches import METHODS, amateur_context, build_amateur_inputs  # noqa: E402
from qwen_runtime import (  # noqa: E402
    MODEL_ID, build_inputs, decode_is_yes, forward_hiddens, image_token_id, label_to_int,
    lens_margins, load_model, load_questions, n_hidden_layers, pick_device, yes_no_token_ids,
)
from sid_ct2s import SID_KEEP_RATIO, SID_RANK_LAYER, SidCT2S  # noqa: E402

SPLITS = ("random", "popular", "adversarial")


class Accumulator:
    """Per-sample arrays for every method, plus the shard buffer for the probe features."""

    def __init__(self, methods):
        self.methods = list(methods)
        self.clean_margin = []
        self.amateur_margin = {m: [] for m in self.methods}
        self.gt, self.pred, self.split = [], [], []
        self.lens_mae_sum, self.lens_mae_n = 0.0, 0
        self.done_splits = []

    @property
    def n(self):
        return len(self.gt)

    def restore(self, results_dir):
        """Reload a previous run's per-method npz. Returns the common completed-split list."""
        done = None
        for m in self.methods:
            path = os.path.join(results_dir, f"logit_lens_per_sample_{m}.npz")
            summ = os.path.join(results_dir, f"extract_summary_{m}.json")
            if not (os.path.exists(path) and os.path.exists(summ)):
                return []
            with open(summ) as f:
                meta = json.load(f)
            ds = list(meta.get("splits_done", []))
            if done is None:
                done = ds
            elif done != ds:
                print(f"[resume] method npz disagree on completed splits ({done} vs {ds}); "
                      f"restarting from scratch")
                return []
        for i, m in enumerate(self.methods):
            z = np.load(os.path.join(results_dir, f"logit_lens_per_sample_{m}.npz"),
                        allow_pickle=True)
            self.amateur_margin[m] = list(z["amateur_margin"].astype(np.float32))
            if i == 0:
                self.clean_margin = list(z["clean_margin"].astype(np.float32))
                self.gt = list(z["gt"].astype(np.int8))
                self.pred = list(z["pred"].astype(np.int8))
                self.split = list(z["split"])
        self.done_splits = list(done or [])
        return self.done_splits


def save_shard(act_dir, split, shard_idx, buf, methods):
    npz_path = os.path.join(act_dir, f"{split}_shard{shard_idx:04d}.npz")
    payload = {
        "clean_hidden": np.stack(buf["clean_hidden"]).astype(np.float16),
        "clean_norm": np.stack(buf["clean_norm"]).astype(np.float32),
        "gt": np.asarray(buf["gt"], dtype=np.int8),
        "pred": np.asarray(buf["pred"], dtype=np.int8),
    }
    for m in methods:
        payload[f"cd_change_{m}"] = np.stack(buf[f"cd_change_{m}"]).astype(np.float32)
    np.savez_compressed(npz_path, **payload)
    with open(os.path.join(act_dir, f"{split}_shard{shard_idx:04d}.meta.jsonl"), "w") as f:
        for meta in buf["meta"]:
            f.write(json.dumps(meta) + "\n")


def new_buffer(methods):
    keys = ["clean_hidden", "clean_norm", "gt", "pred", "meta"] + [f"cd_change_{m}" for m in methods]
    return {k: [] for k in keys}


def flush_checkpoint(acc, args, extra):
    """Write every per-method npz + summary. Called after each completed split."""
    clean = np.asarray(acc.clean_margin, np.float32)
    gt = np.asarray(acc.gt, np.int8)
    pred = np.asarray(acc.pred, np.int8)
    split = np.asarray(acc.split, dtype="U12")
    lens_mae = acc.lens_mae_sum / max(acc.lens_mae_n, 1)
    n_h = int(((gt == 0) & (pred == 1)).sum())
    n_t = int(((gt == 1) & (pred == 1)).sum())
    for m in acc.methods:
        np.savez_compressed(
            os.path.join(args.results_dir, f"logit_lens_per_sample_{m}.npz"),
            clean_margin=clean,
            amateur_margin=np.asarray(acc.amateur_margin[m], np.float32),
            gt=gt, pred=pred, split=split,
        )
        summary = {
            "method": m, "model": args.model_path, "n": int(len(gt)), "n_H": n_h, "n_T": n_t,
            "lens_final_layer_MAE": lens_mae,
            "splits_requested": list(args.splits), "splits_done": list(acc.done_splits),
            **extra,
        }
        with open(os.path.join(args.results_dir, f"extract_summary_{m}.json"), "w") as f:
            json.dump(summary, f, indent=2)
    print(f"[checkpoint] splits_done={acc.done_splits} N={len(gt)} H={n_h} T={n_t} "
          f"lens MAE={lens_mae:.2e}", flush=True)


def run(args):
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.act_dir, exist_ok=True)

    device = pick_device(args.device)
    processor, model = load_model(args.model_path, dtype=args.dtype, attn_impl=args.attn_impl,
                                  min_pixels=args.min_pixels, max_pixels=args.max_pixels,
                                  device=device)
    n_layers = n_hidden_layers(model)
    print(f"[model] {args.model_path}  num_hidden_layers={n_layers} "
          f"-> {n_layers + 1} hidden states  dtype={args.dtype}  attn={args.attn_impl}")

    yes_ids, no_ids = yes_no_token_ids(processor.tokenizer)
    yes_t = torch.tensor(yes_ids, device=device)
    no_t = torch.tensor(no_ids, device=device)
    img_tok = image_token_id(model, processor)
    sid = SidCT2S(model, rank_layer=args.sid_rank_layer, keep_ratio=args.sid_keep_ratio,
                  keep_tokens=args.sid_keep_tokens)
    print(f"[sid] {sid.describe()}")

    methods = list(args.methods)
    acc = Accumulator(methods)
    if args.resume:
        done = acc.restore(args.results_dir)
        if done:
            print(f"[resume] reusing {acc.n} samples from splits {done}")

    pope_dir = os.path.join(args.data_dir, "pope", "coco")
    images_dir = os.path.join(pope_dir, "images")
    extra = {
        "sid_rank_layer": args.sid_rank_layer,
        "sid_keep_ratio": args.sid_keep_ratio,
        "sid_keep_tokens": args.sid_keep_tokens,
        "noise_step": args.noise_step,
        "n_hidden_layers": n_layers,
        "act_dir": args.act_dir,
    }

    for split in args.splits:
        if split in acc.done_splits:
            print(f"[skip] split '{split}' already complete")
            continue
        questions = load_questions(pope_dir, split)
        if args.max_samples > 0:
            questions = questions[: args.max_samples]

        buf = new_buffer(methods)
        shard_idx, n_correct, n_total = 0, 0, 0
        for line in tqdm(questions, desc=f"extract:{split}"):
            try:
                image = Image.open(os.path.join(images_dir, line["image"])).convert("RGB")
            except (FileNotFoundError, OSError) as e:
                print(f"[skip] {line['image']}: {e}")
                continue

            expert_inputs = build_inputs(processor, image, line["text"], device=device)
            hs_e, logits_e = forward_hiddens(model, expert_inputs)
            m_e = lens_margins(model, hs_e, yes_t, no_t)

            true_margin = float(logits_e[yes_t].max() - logits_e[no_t].max())
            acc.lens_mae_sum += abs(float(m_e[-1]) - true_margin)
            acc.lens_mae_n += 1

            pred_yes = decode_is_yes(processor.tokenizer, int(torch.argmax(logits_e).item()))
            gt = label_to_int(line["label"])
            n_correct += int(pred_yes == gt)
            n_total += 1

            hs_e32 = hs_e.float()
            clean_norm = torch.linalg.vector_norm(hs_e32, dim=1).cpu().numpy()

            for method in methods:
                # deterministic amateur: VCD's diffusion noise reproduces on re-run/resume;
                # ICD and corrected SID carry no RNG at all.
                torch.manual_seed(args.noise_seed + acc.n)
                am_inputs = build_amateur_inputs(method, processor, expert_inputs, image,
                                                 line["text"], args.noise_step, device=device)
                with amateur_context(method, am_inputs, sid=sid, img_token_id=img_tok):
                    hs_a, _ = forward_hiddens(model, am_inputs)
                acc.amateur_margin[method].append(lens_margins(model, hs_a, yes_t, no_t))
                buf[f"cd_change_{method}"].append(
                    torch.linalg.vector_norm(hs_e32 - hs_a.float(), dim=1).cpu().numpy()
                )

            acc.clean_margin.append(m_e)
            acc.gt.append(gt)
            acc.pred.append(pred_yes)
            acc.split.append(split)

            buf["clean_hidden"].append(hs_e32.cpu().numpy().astype(np.float16))
            buf["clean_norm"].append(clean_norm)
            buf["gt"].append(gt)
            buf["pred"].append(pred_yes)
            buf["meta"].append({
                "split": split, "question_id": line.get("question_id"), "image": line["image"],
                "text": line["text"], "gt": gt, "pred_yes": pred_yes,
                "expert_margin": true_margin, "n_vision": int(sid.n_vision),
            })

            if len(buf["gt"]) >= args.shard_size:
                save_shard(args.act_dir, split, shard_idx, buf, methods)
                shard_idx += 1
                buf = new_buffer(methods)

        if buf["gt"]:
            save_shard(args.act_dir, split, shard_idx, buf, methods)

        acc.done_splits.append(split)
        print(f"[extract:{split}] samples={n_total}  baseline greedy accuracy="
              f"{n_correct / max(n_total, 1):.4f}")
        flush_checkpoint(acc, args, extra)

    print(f"[done] wrote per-method npz to {args.results_dir}, activations to {args.act_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=MODEL_ID)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--act-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--attn-impl", default="eager",
                    help="must be 'eager' for SID: sdpa/flash do not return attention weights")
    ap.add_argument("--min-pixels", type=int, default=None)
    ap.add_argument("--max-pixels", type=int, default=None,
                    help="bound the vision-token count; None = Qwen defaults (~390 tokens on COCO)")
    ap.add_argument("--noise-step", type=int, default=900, help="VCD diffusion step")
    ap.add_argument("--noise-seed", type=int, default=1234)
    ap.add_argument("--sid-rank-layer", type=int, default=SID_RANK_LAYER,
                    help="0-indexed decoder layer whose attention ranks vision tokens "
                         "(SID paper Layer i=3 -> 2)")
    ap.add_argument("--sid-keep-ratio", type=float, default=SID_KEEP_RATIO,
                    help="fraction of LEAST-attended vision tokens kept (SID paper: 0.10)")
    ap.add_argument("--sid-keep-tokens", type=int, default=None,
                    help="absolute override for --sid-keep-ratio (released-code parity: 100)")
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all; else cap per split")
    ap.add_argument("--resume", action="store_true", help="skip splits already completed")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
