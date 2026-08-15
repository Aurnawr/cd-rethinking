"""
jaccard_internvl_apc_sweep.py -- same "APC vs. greedy / vs. direct sampling"
Jaccard analysis as main_resolved.tex (Section ssec:jaccard-results,
tab:apc-greedy-both / tab:apc-both-refs / fig:apc-greedy-curve), applied to
the InternVL3-8B run pushed by the teammate at
github.com/Aurnawr/Rethinking-CD/tree/master/intern/jaccard_experiment/outputs/llava_bench
(same 60-question LLaVA-Bench-in-the-Wild set, same file convention --
just downloaded locally into outputs/llava_bench_internvl/).

For every APC beta in the sweep: mean token-wise (BPE) Jaccard similarity
between APC-augmented sampling and (a) greedy search, (b) plain direct
sampling, each with a PAIRED bootstrap 95% CI (10,000 resamples over the
60 questions, resampling question indices -- not independent per-condition
resampling, since vs-greedy and vs-sample scores for the same question
share that question's APC output).

Usage:
  python jaccard_internvl_apc_sweep.py --B 10000
"""
import argparse
import glob
import json
import os

import numpy as np
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT = os.path.dirname(SCRIPT_DIR)
OUTPUTS = os.path.join(EXPERIMENT, "outputs", "llava_bench_internvl")
MODEL_PATH = os.path.join(EXPERIMENT, "models", "internvl3-8b")

GREEDY_FILE = os.path.join(OUTPUTS, "greedy", "internvl3-8b-llava_bench-greedy.jsonl")
SAMPLE_FILE = os.path.join(OUTPUTS, "sample", "internvl3-8b-llava_bench-sample.jsonl")
APC_GLOB = os.path.join(OUTPUTS, "apc", "beta_*", "*.jsonl")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return {d["question_id"]: d["text"] for d in (json.loads(l) for l in f)}


def bpe_token_set(tokenizer, text):
    return set(tokenizer.encode(text, add_special_tokens=False))


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def paired_scores(tokenizer, a_texts, b_texts, cache):
    qids = sorted(set(a_texts) & set(b_texts))
    scores = []
    for qid in qids:
        ta = cache.setdefault(("a", id(a_texts), qid), bpe_token_set(tokenizer, a_texts[qid]))
        tb = cache.setdefault(("b", id(b_texts), qid), bpe_token_set(tokenizer, b_texts[qid]))
        scores.append(jaccard(ta, tb))
    return np.array(scores)


def bootstrap_mean_ci(arr, B, rng):
    n = len(arr)
    boot = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        boot[b] = arr[idx].mean()
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=10000)
    args = ap.parse_args()

    print("Loading InternVL3-8B tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    greedy = load_jsonl(GREEDY_FILE)
    sample = load_jsonl(SAMPLE_FILE)

    apc_files = sorted(glob.glob(APC_GLOB))
    betas = []
    for path in apc_files:
        beta_str = os.path.basename(os.path.dirname(path)).replace("beta_", "")
        betas.append((float(beta_str), path))
    betas.sort()

    cache = {}
    rows = []
    print("\n== InternVL3-8B: APC vs. Greedy and vs. Direct Sampling ==")
    for beta, path in betas:
        apc = load_jsonl(path)
        vs_greedy = paired_scores(tokenizer, apc, greedy, cache)
        vs_sample = paired_scores(tokenizer, apc, sample, cache)
        diff = vs_greedy - vs_sample  # paired per-question difference

        rng1 = np.random.default_rng(0)
        g_lo, g_hi = bootstrap_mean_ci(vs_greedy, args.B, rng1)
        rng2 = np.random.default_rng(0)
        s_lo, s_hi = bootstrap_mean_ci(vs_sample, args.B, rng2)
        rng3 = np.random.default_rng(0)
        d_lo, d_hi = bootstrap_mean_ci(diff, args.B, rng3)

        row = {
            "beta": beta, "n": len(vs_greedy),
            "vs_greedy": float(vs_greedy.mean()), "vs_greedy_ci": [g_lo, g_hi],
            "vs_sample": float(vs_sample.mean()), "vs_sample_ci": [s_lo, s_hi],
            "diff": float(diff.mean()), "diff_ci": [d_lo, d_hi],
        }
        rows.append(row)
        print(f"  beta={beta:.3f}  vs.Greedy={row['vs_greedy']:.4f} [{g_lo:.4f},{g_hi:.4f}]   "
              f"vs.Sampling={row['vs_sample']:.4f} [{s_lo:.4f},{s_hi:.4f}]   "
              f"Diff={row['diff']:+.4f} [{d_lo:+.4f},{d_hi:+.4f}]  (n={row['n']})")

    out_json = os.path.join(SCRIPT_DIR, "jaccard_internvl_apc_sweep.json")
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved -> {out_json}")

    csv_path = os.path.join(SCRIPT_DIR, "jaccard_internvl_apc_sweep.csv")
    with open(csv_path, "w") as f:
        f.write("beta,vs_greedy,vs_greedy_lo,vs_greedy_hi,vs_sample,vs_sample_lo,vs_sample_hi,diff,diff_lo,diff_hi,n\n")
        for r in rows:
            f.write(f"{r['beta']:.3f},{r['vs_greedy']:.6f},{r['vs_greedy_ci'][0]:.6f},{r['vs_greedy_ci'][1]:.6f},"
                    f"{r['vs_sample']:.6f},{r['vs_sample_ci'][0]:.6f},{r['vs_sample_ci'][1]:.6f},"
                    f"{r['diff']:.6f},{r['diff_ci'][0]:.6f},{r['diff_ci'][1]:.6f},{r['n']}\n")
    print(f"Saved -> {csv_path}")


if __name__ == "__main__":
    main()
