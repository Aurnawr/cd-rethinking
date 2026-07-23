"""Precompute all statistics needed for proxy Designs A-F from the already-saved
500-image raw per-token log. Pure CPU, no model/GPU needed."""
import json
from collections import defaultdict

OUT = {}

for method in ["vcd", "sid"]:
    all_d = []
    real_d, hall_d = [], []
    word_d = defaultdict(list)
    for line in open("outputs_500_clean/raw_object_candidates.jsonl"):
        rec = json.loads(line)
        if rec["method"] != method:
            continue
        d = rec["d_expert_minus_amateur"]
        all_d.append(d)
        word_d[rec["canonical"]].append(d)
        (real_d if rec["is_real"] else hall_d).append(d)

    def mean_std(xs):
        m = sum(xs) / len(xs)
        s = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
        return m, s

    pooled_mean, pooled_std = mean_std(all_d)
    real_mean, real_std = mean_std(real_d)
    hall_mean, hall_std = mean_std(hall_d)

    # per-word lookup: use the word's own mean if it has >=20 instances,
    # else fall back to the pooled mean (avoids overfitting to noisy small-n words)
    per_word = {}
    for w, vals in word_d.items():
        per_word[w] = (sum(vals) / len(vals)) if len(vals) >= 20 else pooled_mean

    OUT[method] = {
        "pooled_mean": pooled_mean, "pooled_std": pooled_std,
        "real_mean": real_mean, "real_std": real_std,
        "hall_mean": hall_mean, "hall_std": hall_std,
        "per_word_bonus": per_word,
        "empirical_samples": all_d,  # for Design B bootstrap resampling
    }
    print(f"{method}: pooled mean={pooled_mean:.4f} std={pooled_std:.4f} | "
          f"real mean={real_mean:.4f} std={real_std:.4f} | "
          f"hall mean={hall_mean:.4f} std={hall_std:.4f} | "
          f"n_words_in_lookup={len(per_word)}")

json.dump(OUT, open("proxy_stats.json", "w"))
print("\nSaved to proxy_stats.json")
