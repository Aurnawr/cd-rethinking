"""
Compute BPE-token Jaccard similarity between ICD (Instruction Contrastive
Decoding, arXiv:2403.18715) and the existing greedy / direct-sampling / VCD /
APC(beta=0.1) outputs on LLaVA-Bench-in-the-Wild (60 questions, LLaVA-1.5-7B).

ICD is run as 5 SEPARATE full passes (one per disturbance prompt: p1, p2,
n1, n2, p3), matching the official reference implementation
(github.com/p1k0pan/ICD) rather than randomly mixing prompts within one run.
Each pass is compared individually against the 4 baseline methods, plus the
mean across the 5 passes; a same-model ICD-vs-ICD cross table is also
reported to show how much the disturbance-prompt choice itself moves the
output relative to the CD-vs-baseline gaps.
"""

import os
import json
import numpy as np
from transformers import AutoTokenizer

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT  = os.path.dirname(SCRIPT_DIR)          # jaccard_experiment/
REPO        = os.path.dirname(EXPERIMENT)          # Guneesh_CDrethink/
OUTPUTS     = os.path.join(EXPERIMENT, "outputs", "llava_bench")
ICD_DIR     = os.path.join(REPO, "CHAIR_analysis", "icd_llava_bench")
MODEL_PATH  = os.path.join(REPO, "CHAIR_analysis", "models", "llava-v1.5-7b")

GREEDY_FILE = os.path.join(OUTPUTS, "greedy", "llava-7b-llava_bench-greedy.jsonl")
SAMPLE_FILE = os.path.join(OUTPUTS, "sample", "llava-7b-llava_bench-sample.jsonl")
VCD_FILE    = os.path.join(OUTPUTS, "vcd",    "llava-7b-llava_bench-vcd.jsonl")
APC01_FILE  = os.path.join(OUTPUTS, "apc", "beta_0.100", "llava-7b-llava_bench-apc-beta0.100.jsonl")

ICD_KEYS = ["p1", "p2", "n1", "n2", "p3"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return {d["question_id"]: d["text"] for d in (json.loads(l) for l in f)}


def bpe_token_set(tokenizer, text):
    return set(tokenizer.encode(text, add_special_tokens=False))


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def compare(tokenizer, base, other, base_name, other_name, token_cache):
    common_ids = sorted(set(base) & set(other))
    scores = []
    for qid in common_ids:
        if (base_name, qid) not in token_cache:
            token_cache[(base_name, qid)] = bpe_token_set(tokenizer, base[qid])
        if (other_name, qid) not in token_cache:
            token_cache[(other_name, qid)] = bpe_token_set(tokenizer, other[qid])
        scores.append(jaccard(token_cache[(base_name, qid)], token_cache[(other_name, qid)]))
    return np.mean(scores), np.std(scores), len(scores), scores


def main():
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

    print("Loading output files ...")
    baselines = {
        "Greedy": load_jsonl(GREEDY_FILE),
        "Direct Sampling": load_jsonl(SAMPLE_FILE),
        "VCD": load_jsonl(VCD_FILE),
        "APC (beta=0.1)": load_jsonl(APC01_FILE),
    }
    icd = {k: load_jsonl(os.path.join(ICD_DIR, f"{k}.jsonl")) for k in ICD_KEYS}
    for k, d in icd.items():
        assert len(d) == 60, f"ICD {k}: expected 60 questions, got {len(d)}"

    token_cache = {}

    print("\n== ICD (per disturbance prompt) vs baselines ==")
    table = {}  # (icd_key, baseline_name) -> (mean, std, n)
    for k in ICD_KEYS:
        for bname, bdata in baselines.items():
            m, s, n, _ = compare(tokenizer, icd[k], bdata, f"icd_{k}", bname, token_cache)
            table[(k, bname)] = (m, s, n)
            print(f"  ICD[{k}] vs {bname:16s}  Jaccard = {m:.4f} +/- {s:.4f}  (n={n})")

    print("\n== ICD mean-across-5-prompts vs each baseline ==")
    mean_row = {}
    for bname in baselines:
        vals = [table[(k, bname)][0] for k in ICD_KEYS]
        mean_row[bname] = (np.mean(vals), np.std(vals))
        print(f"  ICD[mean of 5] vs {bname:16s}  Jaccard = {np.mean(vals):.4f} (std across prompts = {np.std(vals):.4f})")

    print("\n== ICD-vs-ICD cross table (effect of disturbance-prompt choice alone) ==")
    cross = {}
    header = "        " + "".join(f"{k:>8s}" for k in ICD_KEYS)
    print(header)
    for k1 in ICD_KEYS:
        row = f"  {k1:5s}"
        for k2 in ICD_KEYS:
            if k1 == k2:
                m = 1.0
            else:
                m, _, _, _ = compare(tokenizer, icd[k1], icd[k2], f"icd_{k1}", f"icd_{k2}", token_cache)
            cross[(k1, k2)] = m
            row += f"{m:8.4f}"
        print(row)

    off_diag = [cross[(k1, k2)] for k1 in ICD_KEYS for k2 in ICD_KEYS if k1 != k2]
    print(f"\n  ICD-vs-ICD off-diagonal mean Jaccard = {np.mean(off_diag):.4f} +/- {np.std(off_diag):.4f}")

    # save CSV
    csv_path = os.path.join(EXPERIMENT, "analysis", "jaccard_icd_comparisons.csv")
    with open(csv_path, "w") as f:
        f.write("icd_prompt_key,baseline,mean_jaccard,std_jaccard,n\n")
        for k in ICD_KEYS:
            for bname in baselines:
                m, s, n = table[(k, bname)]
                f.write(f"{k},{bname},{m:.6f},{s:.6f},{n}\n")
        for bname in baselines:
            m, s = mean_row[bname]
            f.write(f"mean_of_5,{bname},{m:.6f},{s:.6f},60\n")

    cross_csv = os.path.join(EXPERIMENT, "analysis", "jaccard_icd_cross.csv")
    with open(cross_csv, "w") as f:
        f.write("icd_key_1,icd_key_2,mean_jaccard\n")
        for k1 in ICD_KEYS:
            for k2 in ICD_KEYS:
                f.write(f"{k1},{k2},{cross[(k1, k2)]:.6f}\n")

    print(f"\nSaved -> {csv_path}")
    print(f"Saved -> {cross_csv}")


if __name__ == "__main__":
    main()
