"""
chair_bootstrap_sampling.py -- CHAIR-S/CHAIR-I with 95% image-level
bootstrap CIs (B=10,000) for the direct-sampling leg (baseline, VCD, SID,
ICD pooled over its 5 disturbance-prompt passes), for both LLaVA and Qwen.
Same methodology as the paper's Table 16 (tab:proxyA-noise): resample
IMAGES with replacement, not individual caption records -- for pooled ICD,
each resampled image carries ALL of its 5 disturbance-prompt captions along
together (they're correlated, conditioned on the same image), never
resampled independently.

Usage:
  python chair_bootstrap_sampling.py --B 10000 --out chair_bootstrap_sampling.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from eval.chair import CHAIR

COCO = REPO / "data" / "coco" / "annotations"
CAPTURES = REPO / "outputs" / "captures"

ICD_KEYS = ["p1", "p2", "n1", "n2", "p3"]


def load_records(model, method):
    """Returns list of {"image_id","caption"} for one condition. ICD pools
    all 5 disturbance-prompt files together (multiple records per image)."""
    if method == "icd":
        recs = []
        for k in ICD_KEYS:
            f = CAPTURES / f"{model}_icd_{k}_seed0.jsonl"
            if not f.exists():
                print(f"[warn] missing {f}")
                continue
            recs.extend(json.loads(l) for l in open(f, encoding="utf-8"))
        return recs
    f = CAPTURES / f"{model}_{method}_seed0.jsonl"
    if not f.exists():
        return None
    return [json.loads(l) for l in open(f, encoding="utf-8")]


def chair_with_ci(chair_obj, records, B, rng):
    # Build per-caption RAW (non-deduplicated) mention counts directly via
    # caption_to_words -- compute_chair()'s "sentences" output deduplicates
    # objects per caption (list(set(node_words))), which would silently
    # change the CHAIR-I ratio (mentions with duplicates matter: "a dog...
    # another dog" is 2 mentions, not 1) and make the bootstrap CI
    # inconsistent with the point estimate computed the correct (raw) way.
    caps = [{"image_id": r["image_id"], "caption": r["caption"]} for r in records]
    res = chair_obj.compute_chair(caps)  # point estimate only, uses raw counts internally

    by_image = {}
    for cap in caps:
        imid = cap["image_id"]
        if imid not in chair_obj.imid_to_objects:
            continue
        words, node_words, idxs, raw_words = chair_obj.caption_to_words(cap["caption"])
        gt = chair_obj.imid_to_objects[imid]
        n_hall = sum(1 for w in node_words if w not in gt)
        n_total = len(node_words)
        by_image.setdefault(imid, []).append((1 if n_hall > 0 else 0, n_hall, n_total))
    image_ids = list(by_image.keys())
    n_img = len(image_ids)

    # Internal consistency check: the by_image aggregation must reproduce
    # compute_chair's own point estimate exactly (same underlying counts,
    # just grouped differently) -- if not, something is still wrong.
    all_recs = [rec for recs in by_image.values() for rec in recs]
    check_s = sum(r[0] for r in all_recs) / len(all_recs)
    check_i = sum(r[1] for r in all_recs) / sum(r[2] for r in all_recs)
    assert abs(check_s - res["CHAIR_s"]) < 1e-9, f"CHAIR_s mismatch: {check_s} vs {res['CHAIR_s']}"
    assert abs(check_i - res["CHAIR_i"]) < 1e-9, f"CHAIR_i mismatch: {check_i} vs {res['CHAIR_i']}"

    boot_s = np.empty(B)
    boot_i = np.empty(B)
    for b in range(B):
        sampled = rng.choice(image_ids, size=n_img, replace=True)
        n_caps = n_hall_caps = 0
        n_hall_words = n_total_words = 0
        for iid in sampled:
            for is_hall, n_h, n_t in by_image[iid]:
                n_caps += 1
                n_hall_caps += is_hall
                n_hall_words += n_h
                n_total_words += n_t
        boot_s[b] = n_hall_caps / n_caps if n_caps else np.nan
        boot_i[b] = n_hall_words / n_total_words if n_total_words else np.nan

    return {
        "num_captions": res["num_captions"],
        "num_images": n_img,
        "CHAIR_s": res["CHAIR_s"],
        "CHAIR_s_ci": [float(np.nanpercentile(boot_s, 2.5)), float(np.nanpercentile(boot_s, 97.5))],
        "CHAIR_i": res["CHAIR_i"],
        "CHAIR_i_ci": [float(np.nanpercentile(boot_i, 2.5)), float(np.nanpercentile(boot_i, 97.5))],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--out", default=str(HERE.parent / "CHAIR_ANALYSIS_ICDSAMPLING" / "chair_bootstrap_sampling.json"))
    args = ap.parse_args()

    results = {}
    for model in ("llava", "qwen"):
        image_ids = set()
        for method in ("baseline", "vcd", "sid", "icd"):
            recs = load_records(model, method)
            if recs:
                image_ids |= {r["image_id"] for r in recs}
        chair_obj = CHAIR(image_ids, str(COCO))
        chair_obj.get_annotations()

        for method in ("baseline", "vcd", "sid", "icd"):
            recs = load_records(model, method)
            if not recs:
                print(f"[skip] {model} {method}: no data")
                continue
            rng = np.random.default_rng(0)
            r = chair_with_ci(chair_obj, recs, args.B, rng)
            key = f"{model}_{method}"
            results[key] = r
            print(f"{model:6} {method:10} CHAIR-S {r['CHAIR_s']*100:5.1f} "
                  f"[{r['CHAIR_s_ci'][0]*100:.1f},{r['CHAIR_s_ci'][1]*100:.1f}]   "
                  f"CHAIR-I {r['CHAIR_i']*100:5.1f} [{r['CHAIR_i_ci'][0]*100:.1f},{r['CHAIR_i_ci'][1]*100:.1f}]   "
                  f"(n_captions={r['num_captions']}, n_images={r['num_images']})")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
