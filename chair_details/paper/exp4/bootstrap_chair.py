"""
bootstrap_chair.py -- image-level bootstrap 95% CIs for CHAIR-S and CHAIR-I on
the real methods and the noise proxies, so the proxy-versus-real comparison has
variance rather than eyeballed point estimates.

Per image we record (has_hallucination, n_hallucinated_instances, n_total_
instances) using the standard CHAIR scorer (non-deduped mentions), then resample
the images with replacement (B=10000). CHAIR-S = sum(has_hall)/n images;
CHAIR-I = sum(hall_instances)/sum(total_instances). This captures image-sampling
variance for a single noise draw; it does NOT capture noise-draw variance (that
needs re-running the proxy, which needs a GPU).

Output: bootstrap_chair.json (+ printed table)
"""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path("/teamspace/studios/this_studio/cd-rethinking")
LLAVA_ROOT = REPO / "llava_logit_eda" / "CHAIR-llava-detailed"
LL = LLAVA_ROOT / "llava_logit_eda"
QD = REPO / "CHAIR-qwen-detailed"
HERE = Path(__file__).resolve().parent
COCO = LLAVA_ROOT / "data" / "coco" / "annotations"

FILES = {
    "LLaVA": {
        "Greedy":      LL / "outputs_500_clean" / "captions_greedy.jsonl",
        "VCD (real)":  LL / "outputs_full_topk" / "captions_topk_vcd.jsonl",
        "SID (real)":  LL / "outputs_full_topk" / "captions_topk_sid.jsonl",
        "VCD proxy A": LL / "outputs_proxy_AB_500" / "captions_A_vcd.jsonl",
        "VCD proxy B": LL / "outputs_proxy_AB_500" / "captions_B_vcd.jsonl",
        "SID proxy A": LL / "outputs_proxy_AB_500" / "captions_A_sid.jsonl",
        "SID proxy B": LL / "outputs_proxy_AB_500" / "captions_B_sid.jsonl",
    },
    "Qwen": {
        "Greedy":      QD / "outputs_full_topk" / "captions_greedy.jsonl",
        "VCD (real)":  QD / "outputs_full_topk" / "captions_topk_vcd.jsonl",
        "SID (real)":  QD / "outputs_full_topk" / "captions_topk_sid.jsonl",
        "VCD proxy A": QD / "outputs_proxy" / "captions_A_vcd.jsonl",
        "VCD proxy B": QD / "outputs_proxy" / "captions_B_vcd.jsonl",
    },
}


def per_image(cap_file, chair):
    rows = []
    for line in open(cap_file):
        r = json.loads(line)
        _, node_words, _, _ = chair.caption_to_words(r["caption"])
        gt = chair.imid_to_objects.get(int(r["image_id"]), set())
        hall = [w for w in node_words if w not in gt]
        rows.append((1.0 if hall else 0.0, float(len(hall)), float(len(node_words))))
    return np.array(rows)  # columns: has_hall, n_hall, n_total


def boot(arr, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(arr)
    def stat(idx):
        a = arr[idx]
        cs = a[:, 0].sum() / n
        ci = a[:, 1].sum() / a[:, 2].sum() if a[:, 2].sum() else float("nan")
        return cs, ci
    p_cs, p_ci = stat(np.arange(n))
    cs_d = np.empty(B); ci_d = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        cs_d[b], ci_d[b] = stat(idx)
    def ci(pt, d):
        return {"pt": 100 * pt, "lo": 100 * float(np.percentile(d, 2.5)),
                "hi": 100 * float(np.percentile(d, 97.5))}
    return {"n": n, "CHAIR_s": ci(p_cs, cs_d), "CHAIR_i": ci(p_ci, ci_d)}


def main():
    sys.path.insert(0, str(LLAVA_ROOT))
    from eval.chair import CHAIR
    out = {}
    for model, files in FILES.items():
        # one CHAIR object per model (image ids identical across its files)
        ids = [int(json.loads(l)["image_id"]) for l in open(next(iter(files.values())))]
        chair = CHAIR(imids=ids, coco_path=str(COCO)); chair.get_annotations()
        out[model] = {}
        for label, path in files.items():
            arr = per_image(path, chair)
            out[model][label] = boot(arr)
            r = out[model][label]
            print(f"{model:6} {label:12} CHAIR-S {r['CHAIR_s']['pt']:.1f} "
                  f"[{r['CHAIR_s']['lo']:.1f},{r['CHAIR_s']['hi']:.1f}]  "
                  f"CHAIR-I {r['CHAIR_i']['pt']:.1f} "
                  f"[{r['CHAIR_i']['lo']:.1f},{r['CHAIR_i']['hi']:.1f}]", flush=True)
    json.dump(out, open(HERE / "bootstrap_chair.json", "w"), indent=2)


if __name__ == "__main__":
    main()
