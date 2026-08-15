"""
plot_table16_extended.py -- renders an extended version of the paper's
Table 16 (main_resolved.tex, tab:proxyA-noise): the original greedy-leg
rows (Greedy / VCD-greedy(real) / SID-greedy(real) / Proxy) verbatim from
the paper, plus new rows from this package's direct-sampling leg
(Direct-Sampling baseline / VCD-sampling(real) / SID-sampling(real) /
ICD-sampling(real), the last pooled over its 5 disturbance-prompt passes).

Sampling-leg numbers come from chair_bootstrap_sampling.json (same
methodology: 95% image-level bootstrap CI, B=10,000).

Usage:
  python plot_table16_extended.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUTDIR = REPO / "CHAIR_ANALYSIS_ICDSAMPLING"

# Verbatim from main_resolved.tex, tab:proxyA-noise (the paper's original
# greedy-leg Table 16).
GREEDY_LEG = {
    "llava": [
        ("Greedy (baseline)",       50.0, (45.6, 54.4), 13.9, (12.5, 15.3)),
        ("VCD-greedy (real)",       55.0, (50.6, 59.4), 15.6, (14.1, 17.0)),
        ("SID-greedy (real)",       54.2, (49.8, 58.6), 15.1, (13.7, 16.6)),
        ("Proxy (greedy)",          52.4, (48.0, 56.6), 14.8, (13.3, 16.3)),
    ],
    "qwen": [
        ("Greedy (baseline)",       30.8, (26.7, 34.8),  7.7, (6.5, 8.9)),
        ("VCD-greedy (real)",       36.4, (32.4, 40.6), 10.5, (8.9, 12.1)),
        ("SID-greedy (real)",       38.4, (34.2, 42.6), 10.5, (9.1, 12.0)),
        ("Proxy (greedy)",          30.8, (26.8, 34.8),  8.2, (6.9, 9.5)),
    ],
}

SAMPLING_LABELS = {
    "baseline": "Direct-Sampling (baseline)",
    "vcd": "VCD-sampling (real)",
    "sid": "SID-sampling (real)",
    "icd": "ICD-sampling (real)",
}
MODEL_TITLE = {"llava": "LLaVA-1.5-7B", "qwen": "Qwen2.5-VL-7B"}


def fmt(pt, ci):
    return f"{pt:.1f}\n[{ci[0]:.1f},{ci[1]:.1f}]"


def main():
    sampling = json.load(open(OUTDIR / "chair_bootstrap_sampling.json"))

    rows = []  # (model_label_or_blank, setting, chair_s_str, chair_i_str, is_new)
    for model in ("llava", "qwen"):
        first = True
        for setting, s, s_ci, i, i_ci in GREEDY_LEG[model]:
            rows.append((MODEL_TITLE[model] if first else "", setting, fmt(s, s_ci), fmt(i, i_ci), False))
            first = False
        for method in ("baseline", "vcd", "sid", "icd"):
            key = f"{model}_{method}"
            if key not in sampling:
                continue
            r = sampling[key]
            s, s_ci = r["CHAIR_s"] * 100, [x * 100 for x in r["CHAIR_s_ci"]]
            i, i_ci = r["CHAIR_i"] * 100, [x * 100 for x in r["CHAIR_i_ci"]]
            rows.append(("", SAMPLING_LABELS[method], fmt(s, s_ci), fmt(i, i_ci), True))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(rows) + 1  # +1 header
    fig, ax = plt.subplots(figsize=(9, 0.34 * n_rows))
    ax.axis("off")

    col_labels = ["Model", "Setting", "CHAIR-S ↓", "CHAIR-I ↓"]
    cell_text = [[r[0], r[1], r[2], r[3]] for r in rows]

    table = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center",
                      colWidths=[0.16, 0.34, 0.25, 0.25], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("0.7")
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("0.85")
        else:
            row = rows[r - 1]
            if row[3] and rows[r - 1][4]:  # new sampling-leg row
                cell.set_facecolor("#eaf3ff")
            if row[0]:  # model-name row (first of a block) -> slightly bolder top border
                cell.set_edgecolor("0.3")

    fig.subplots_adjust(top=0.93, bottom=0.05, left=0.02, right=0.98)
    fig.suptitle("Extended Table 16: CHAIR-S / CHAIR-I, greedy-leg (paper) vs. direct-sampling-leg (this package)",
                 fontsize=11, fontweight="bold", y=0.985)
    fig.text(0.5, 0.012, "Blue rows = new direct-sampling-leg results (this package). "
                         "White rows = original paper values (greedy leg), verbatim. "
                         "Brackets are 95% image-level bootstrap CIs (B=10,000).",
             ha="center", fontsize=8, style="italic")

    out_path = OUTDIR / "table16_extended.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[table] {out_path}")


if __name__ == "__main__":
    main()
