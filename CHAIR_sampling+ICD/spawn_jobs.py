"""
spawn_jobs.py -- submits the full CHAIR_sampling+ICD run plan to the deployed
Modal app, asynchronously (each job keeps running on Modal even if this
script/terminal disconnects right after submission).

Prerequisites (see README.md):
  1. modal setup                       (one-time auth)
  2. modal run modal_app.py --action setup     (one-time asset download)
  3. modal deploy modal_app.py                 (deploy the persistent app)

Then:
  python spawn_jobs.py --smoke          # tiny sanity run, 3 images, 1 seed, 1 ICD prompt
  python spawn_jobs.py --full           # the real run: 500 images, seed 0,
                                         # baseline + VCD + SID + all 5 ICD prompts
                                         # (8 jobs total, ~15 GPU-hr on an L4)

Everything uses --decode sample (this package's whole point: direct sampling
as the decoding rule, instead of the greedy baseline used in the original
CHAIR_analysis experiment). LLaVA only -- this is the "firstly" leg; Qwen is
a separate, later addition (see GUIDE.md).

Seed choice: 1 seed on LLaVA, paired with 1 seed on Qwen (once that leg is
added), covers two models instead of spending the same GPU-hours on extra
seeds of one model -- a deliberate choice given the fixed compute budget.
"""
import argparse
import json

import modal


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true", help="3 images, seed 0 only, 1 ICD prompt only")
    g.add_argument("--full", action="store_true", help="500 images, seeds 0+1, all 5 ICD prompts")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                     help="override the default seed list")
    args = ap.parse_args()

    if args.smoke:
        n_images = 3
        seeds = args.seeds or [0]
        icd_prompts = ["n2"]
        call_log_path = "smoke_call_ids.json"
    else:
        n_images = 500
        seeds = args.seeds or [0]
        icd_prompts = ["p1", "p2", "n1", "n2", "p3"]
        call_log_path = "full_call_ids.json"

    gen = modal.Function.from_name("cd-rethink-sampling-icd", "generate")
    gen_icd = modal.Function.from_name("cd-rethink-sampling-icd", "generate_icd")

    calls = {}

    for seed in seeds:
        out = f"captures/llava_baseline_seed{seed}.jsonl"
        call = gen.spawn(mode="baseline", method="", decode="sample", seed=seed,
                          out_name=out, n_images=n_images)
        calls[f"baseline_seed{seed}"] = call.object_id
        print(f"baseline seed{seed} -> {call.object_id}")

    for method in ("vcd", "sid"):
        for seed in seeds:
            out = f"captures/llava_{method}_seed{seed}.jsonl"
            call = gen.spawn(mode="capture", method=method, decode="sample", seed=seed,
                              out_name=out, n_images=n_images)
            calls[f"{method}_seed{seed}"] = call.object_id
            print(f"{method} seed{seed} -> {call.object_id}")

    for prompt_key in icd_prompts:
        for seed in seeds:
            out = f"captures/llava_icd_{prompt_key}_seed{seed}.jsonl"
            call = gen_icd.spawn(prompt_key=prompt_key, decode="sample", seed=seed,
                                  out_name=out, n_images=n_images)
            calls[f"icd_{prompt_key}_seed{seed}"] = call.object_id
            print(f"icd_{prompt_key} seed{seed} -> {call.object_id}")

    with open(call_log_path, "w") as f:
        json.dump(calls, f, indent=2)
    print(f"\n{len(calls)} jobs submitted. Call IDs saved to {call_log_path}.")
    print("Check progress with: modal app logs cd-rethink-sampling-icd")


if __name__ == "__main__":
    main()
