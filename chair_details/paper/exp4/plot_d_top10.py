"""
plot_d_top10.py -- regenerate the top-10 d = E - A bar plots from the SAME
computation that produces d_stats.json (compute_d_stats.py), so the figures and
the results table are guaranteed to agree. Overwrites figs/{model}_top10_{m}_{g}.png.
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/teamspace/studios/this_studio/cd-rethinking")
LLAVA_ROOT = REPO / "llava_logit_eda" / "CHAIR-llava-detailed"
HERE = Path(__file__).resolve().parent
COCO = LLAVA_ROOT / "data" / "coco" / "annotations"
FIG = HERE / "figs"; FIG.mkdir(exist_ok=True)
K = 10

CFG = {
    "llava": {"topk": LLAVA_ROOT / "llava_logit_eda" / "outputs_full_topk",
              "tok": LLAVA_ROOT / "llava_logit_eda" / "models" / "llava-v1.5-7b", "slow": True,
              "name": "LLaVA-1.5-7B"},
    "qwen":  {"topk": REPO / "CHAIR-qwen-detailed" / "outputs_full_topk",
              "tok": REPO / "CHAIR-qwen-detailed" / "models" / "Qwen2.5-VL-7B-Instruct", "slow": False,
              "name": "Qwen2.5-VL-7B"},
}


def collect(model, method, tok, chair_mod, singularize):
    cfg = CFG[model]
    recs = [json.loads(l) for l in open(cfg["topk"] / f"captions_topk_{method}.jsonl")]
    chair = chair_mod.load_chair([r["image_id"] for r in recs], str(COCO))
    real, hall = [], []
    for r in recs:
        gt = chair.imid_to_objects.get(r["image_id"], set())
        for s in r["steps"]:
            am = dict(zip(s["amateur_top_ids"], s["amateur_top_logits"]))
            for tid, E in zip(s["expert_top_ids"][:K], s["expert_top_logits"][:K]):
                txt = tok.decode([tid])
                if not (txt.startswith(" ") or txt.startswith("▁")):
                    continue
                w = "".join(c for c in txt.lower() if c.isalpha())
                if not w:
                    continue
                w = singularize(w)
                if w not in chair.mscoco_objects:
                    continue
                A = am.get(tid)
                if A is None:
                    continue
                (hall if chair.inverse_synonym_dict[w] not in gt else real).append(E - A)
    return real, hall


def plot(vals, title, path):
    n = len(vals)
    mean = sum(vals) / n
    pos = 100.0 * sum(1 for v in vals if v > 0) / n
    colors = ["#c0392b" if v > 0 else "#2c5f9e" for v in vals]
    # same canvas as the emitted figures (7x3.4in) so fonts match on the page;
    # density/solid look comes from high dpi, not a wider canvas.
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(range(n), vals, color=colors, width=1.0, linewidth=0)
    ax.set_xlim(-0.5, n - 0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(mean, color="black", lw=1.2, ls="--")
    # this figure is physically wide (13in), so fonts are scaled up so they stay
    # large after the figure is shrunk into the 2x2 grid in the paper
    # identical font sizes to the emitted figures (same canvas -> same on-page size)
    ax.set_xlabel("index (n)", fontsize=11)
    ax.set_ylabel("d = expert - amateur logit", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.text(0.99, 0.98, f"n = {n}\nmean = {mean:+.3f}\n%pos = {pos:.0f}%   %neg = {100-pos:.0f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round", fc="0.95", ec="0.6"))
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return n, mean, pos


def main():
    sys.path.insert(0, str(LLAVA_ROOT))
    import llava_logit_eda.object_mentions as chair_mod
    from eval.chair import singularize
    from transformers import AutoTokenizer
    for model in ["llava", "qwen"]:
        cfg = CFG[model]
        tok = AutoTokenizer.from_pretrained(str(cfg["tok"]), use_fast=not cfg["slow"])
        for method in ["vcd", "sid"]:
            real, hall = collect(model, method, tok, chair_mod, singularize)
            for grp, vals in (("real", real), ("hall", hall)):
                lbl = "Real Objects" if grp == "real" else "Hallucinated Objects"
                title = f"{cfg['name']}  {method.upper()}  top-10\n{lbl}"
                n, m, p = plot(vals, title, FIG / f"{model}_top10_{method}_{grp}.png")
                print(f"{model} {method} {grp}: n={n} mean={m:+.3f} %pos={p:.1f}", flush=True)


if __name__ == "__main__":
    main()
