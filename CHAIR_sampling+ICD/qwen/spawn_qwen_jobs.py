"""
spawn_qwen_jobs.py -- submits the Qwen run plan to the deployed Modal app,
asynchronously (each job keeps running on Modal even if this script/terminal
disconnects right after submission). Mirrors ../spawn_jobs.py exactly, for
Qwen2.5-VL-7B-Instruct instead of LLaVA-1.5-7B.

Prerequisites (see README.md in this folder):
  1. modal setup                                    (one-time auth)
  2. modal run ../modal_app.py --action setup       (downloads BOTH models -- safe
                                                       to re-run, skips what's there)
  3. modal deploy ../modal_app.py                    (deploy the persistent app)
  4. python verify_qwen_cache_run.py                 (MUST pass before --full)

Then, from this qwen/ directory:
  python spawn_qwen_jobs.py --smoke     # 3 images, 1 seed, 1 ICD prompt
  python spawn_qwen_jobs.py --full      # 500 images, seed 0, baseline+VCD+SID+5 ICD prompts
                                         # (8 jobs total)

Everything uses --decode sample, matching the LLaVA leg's convention.
"""
import argparse
import json

import modal


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true", help="3 images, seed 0 only, 1 ICD prompt only")
    g.add_argument("--full", action="store_true", help="500 images, seed 0, all 5 ICD prompts")
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="override the default seed list")
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

    gen = modal.Function.from_name("cd-rethink-sampling-icd", "generate_qwen")
    gen_icd = modal.Function.from_name("cd-rethink-sampling-icd", "generate_qwen_icd")

    calls = {}

    for seed in seeds:
        out = f"captures/qwen_baseline_seed{seed}.jsonl"
        call = gen.spawn(mode="baseline", method="", decode="sample", seed=seed, out_name=out, n_images=n_images)
        calls[f"baseline_seed{seed}"] = call.object_id
        print(f"baseline seed{seed} -> {call.object_id}")

    for method in ("vcd", "sid"):
        for seed in seeds:
            out = f"captures/qwen_{method}_seed{seed}.jsonl"
            call = gen.spawn(mode="capture", method=method, decode="sample", seed=seed,
                              out_name=out, n_images=n_images)
            calls[f"{method}_seed{seed}"] = call.object_id
            print(f"{method} seed{seed} -> {call.object_id}")

    for prompt_key in icd_prompts:
        for seed in seeds:
            out = f"captures/qwen_icd_{prompt_key}_seed{seed}.jsonl"
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
