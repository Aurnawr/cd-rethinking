"""
plot_d_emitted.py -- emitted-word d = E - A bar plots as THIN single vertical
spikes (one line per emitted object word), not solid filled bars. Numbers match
compute_d_stats.py (emitted population, caption-level labels, first-token E/A).
Writes figs/{model}_emit_{method}_{group}.png.
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

CFG = {
    "llava": {"topk": LLAVA_ROOT / "llava_logit_eda" / "outputs_full_topk",
              "tok": LLAVA_ROOT / "llava_logit_eda" / "models" / "llava-v1.5-7b", "slow": True,
              "name": "LLaVA-1.5-7B"},
    "qwen":  {"topk": REPO / "CHAIR-qwen-detailed" / "outputs_full_topk",
              "tok": REPO / "CHAIR-qwen-detailed" / "models" / "Qwen2.5-VL-7B-Instruct", "slow": False,
              "name": "Qwen2.5-VL-7B"},
}


def collect(model, method, tok, chair_mod):
    cfg = CFG[model]
    recs = [json.loads(l) for l in open(cfg["topk"] / f"captions_topk_{method}.jsonl")]
    chair = chair_mod.load_chair([r["image_id"] for r in recs], str(COCO))
    real, hall = [], []
    for r in recs:
        steps = r["steps"]
        chosen = [s["chosen_id"] for s in steps]
        emap = {s["step"]: dict(zip(s["expert_top_ids"], s["expert_top_logits"])) for s in steps}
        amap = {s["step"]: dict(zip(s["amateur_top_ids"], s["amateur_top_logits"])) for s in steps}
        for m in chair_mod.all_mentions(chair, tok, r["image_id"], chosen):
            if m["category"] != "object":
                continue
            si = m["step_indices"][0]
            if si >= len(steps):
                continue
            tid = steps[si]["chosen_id"]
            E, A = emap[si].get(tid), amap[si].get(tid)
            if E is None or A is None:
                continue
            (hall if m["hallucinated"] else real).append(E - A)
    return real, hall


def plot(vals, title, path):
    n = len(vals)
    mean = sum(vals) / n
    pos = 100.0 * sum(1 for v in vals if v > 0) / n
    colors = ["#c0392b" if v > 0 else "#2c5f9e" for v in vals]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    # one THIN vertical spike per emitted word, no fill
    ax.vlines(range(n), 0, vals, colors=colors, linewidth=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(mean, color="black", lw=1.0, ls="--")
    ax.set_xlim(-1, n)
    ax.set_xlabel("emitted object word index", fontsize=11)
    ax.set_ylabel("d = expert - amateur logit", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.text(0.99, 0.98, f"n = {n}\nmean = {mean:+.3f}\n%pos = {pos:.0f}%   %neg = {100-pos:.0f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round", fc="0.95", ec="0.6"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return n, mean, pos


def main():
    sys.path.insert(0, str(LLAVA_ROOT))
    import llava_logit_eda.object_mentions as chair_mod
    from transformers import AutoTokenizer
    for model in ["llava", "qwen"]:
        cfg = CFG[model]
        tok = AutoTokenizer.from_pretrained(str(cfg["tok"]), use_fast=not cfg["slow"])
        for method in ["vcd", "sid"]:
            real, hall = collect(model, method, tok, chair_mod)
            for grp, vals in (("real", real), ("hall", hall)):
                lbl = "Real Objects" if grp == "real" else "Hallucinated Objects"
                title = f"{cfg['name']}  {method.upper()}  emitted\n{lbl}"
                n, m, p = plot(vals, title, FIG / f"{model}_emit_{method}_{grp}.png")
                print(f"{model} {method} {grp}: n={n} mean={m:+.3f} %pos={p:.1f}", flush=True)


if __name__ == "__main__":
    main()
