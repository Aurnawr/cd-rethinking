"""
plot_jaccard_internvl.py -- renders the InternVL3-8B APC-beta-sweep Jaccard
results as a figure (matching main_resolved.tex's fig:apc-greedy-curve style)
and a table image (matching tab:apc-greedy-both / tab:apc-both-refs style).

Usage:
  python plot_jaccard_internvl.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = json.load(open(HERE / "jaccard_internvl_apc_sweep.json"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

betas = np.array([r["beta"] for r in rows])
vg = np.array([r["vs_greedy"] for r in rows])
vg_lo = np.array([r["vs_greedy_ci"][0] for r in rows])
vg_hi = np.array([r["vs_greedy_ci"][1] for r in rows])
vs = np.array([r["vs_sample"] for r in rows])
vs_lo = np.array([r["vs_sample_ci"][0] for r in rows])
vs_hi = np.array([r["vs_sample_ci"][1] for r in rows])

# ---------- Figure: beta vs Jaccard, with CI bands ----------
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(betas, vg, color="red", marker="o", markersize=4, linewidth=1.6, label="vs. Greedy")
ax.fill_between(betas, vg_lo, vg_hi, color="red", alpha=0.15)
ax.plot(betas, vs, color="blue", marker="o", markersize=4, linewidth=1.6, label="vs. Direct Sampling")
ax.fill_between(betas, vs_lo, vs_hi, color="blue", alpha=0.15)
ax.axvline(0.1, color="black", linestyle=":", linewidth=1.2, label="VCD default beta=0.1")

ax.set_xlabel("APC beta")
ax.set_ylabel("mean token-wise Jaccard similarity")
ax.set_title("InternVL3-8B: APC beta vs. Jaccard similarity\n(LLaVA-Bench-in-the-Wild, 60 questions, 95% bootstrap CI bands)")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(HERE / "apc_vs_greedy_jaccard_internvl.png", dpi=150)
plt.close(fig)
print(f"[plot] {HERE / 'apc_vs_greedy_jaccard_internvl.png'}")

# ---------- Table image ----------
def fmt(pt, lo, hi, signed=False):
    if signed:
        return f"{pt:+.4f}\n[{lo:+.4f},{hi:+.4f}]"
    return f"{pt:.4f}\n[{lo:.4f},{hi:.4f}]"

col_labels = ["APC beta", "vs. Greedy", "vs. Direct Sampling", "Diff (Greedy-Sampling)"]
cell_text = []
for r in rows:
    cell_text.append([
        f"{r['beta']:.3f}",
        fmt(r["vs_greedy"], *r["vs_greedy_ci"]),
        fmt(r["vs_sample"], *r["vs_sample_ci"]),
        fmt(r["diff"], *r["diff_ci"], signed=True),
    ])

n_rows = len(rows) + 1
fig2, ax2 = plt.subplots(figsize=(9, 0.42 * n_rows))
ax2.axis("off")
table = ax2.table(cellText=cell_text, colLabels=col_labels, cellLoc="center",
                   colWidths=[0.15, 0.28, 0.28, 0.29], bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(9)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("0.7")
    if r == 0:
        cell.set_text_props(fontweight="bold")
        cell.set_facecolor("0.85")

fig2.subplots_adjust(top=0.90, bottom=0.03, left=0.02, right=0.98)
fig2.suptitle("InternVL3-8B: APC vs. Greedy and vs. Direct Sampling (paired bootstrap 95% CI, B=10,000)",
              fontsize=11, fontweight="bold", y=0.97)
out_path = HERE / "jaccard_internvl_table.png"
fig2.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig2)
print(f"[table] {out_path}")
