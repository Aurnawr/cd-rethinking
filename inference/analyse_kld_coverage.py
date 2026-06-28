"""
analyze_coverage.py  —  Show how many sequences are still active at each token timestep
Saves a formatted table image instead of a line plot.

Usage:
    python analyze_coverage.py \
        --vcd outputs/kld_experiment/vcd_kld.json \
        --icd outputs/kld_experiment/icd_kld.json \
        --sid outputs/kld_experiment/sid_kld.json \
        --out outputs/kld_experiment/coverage_table.png
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TIMESTEPS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 100, 110, 120, 128]


def load_results(path):
    with open(path) as f:
        return json.load(f)


def get_stats(results):
    lengths = np.array([r["num_tokens"] for r in results])
    total   = len(lengths)
    rows    = []
    for t in TIMESTEPS:
        count = int(np.sum(lengths >= t))
        pct   = 100 * count / total
        rows.append((t, count, total, pct))
    summary = {
        "min":    int(lengths.min()),
        "max":    int(lengths.max()),
        "mean":   float(lengths.mean()),
        "median": float(np.median(lengths)),
        "total":  total,
    }
    return rows, summary


def save_table_image(all_rows, all_summaries, all_labels, out_path):
    n_methods = len(all_labels)
    n_rows    = len(TIMESTEPS)

    # ---- build column headers and cell data ----
    col_headers = ["Token"]
    for label, summary in zip(all_labels, all_summaries):
        col_headers.append(f"{label}\ncount / {summary['total']}")
        col_headers.append(f"{label}\n%")

    table_data = []
    for i, t in enumerate(TIMESTEPS):
        row = [str(t)]
        for j in range(n_methods):
            count, total, pct = all_rows[j][i][1], all_rows[j][i][2], all_rows[j][i][3]
            row.append(f"{count}/{total}")
            row.append(f"{pct:.1f}%")
        table_data.append(row)

    n_cols = len(col_headers)

    # ---- figure sizing ----
    fig_w = 2.2 + n_methods * 2.8
    fig_h = 0.8 + n_rows * 0.38 + 1.4   # rows + summary block
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # ---- colour cells by coverage % ----
    cell_colors = []
    for i, t in enumerate(TIMESTEPS):
        row_colors = ["#F5F5F5"]   # token column — neutral
        for j in range(n_methods):
            pct = all_rows[j][i][3]
            # count cell
            if pct >= 75:   bg = "#C8E6C9"   # green
            elif pct >= 50: bg = "#FFF9C4"   # yellow
            elif pct >= 25: bg = "#FFE0B2"   # orange
            else:           bg = "#FFCDD2"   # red
            row_colors.append(bg)
            row_colors.append(bg)
        cell_colors.append(row_colors)

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_headers,
        cellColours=cell_colors,
        cellLoc="center",
        loc="upper center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)

    # bold the token 85 row
    for col in range(n_cols):
        cell = tbl[TIMESTEPS.index(85) + 1, col]   # +1 for header row
        cell.set_text_props(fontweight="bold")

    # ---- summary stats below the table ----
    summary_lines = []
    for label, summary in zip(all_labels, all_summaries):
        summary_lines.append(
            f"{label}:  min={summary['min']}  max={summary['max']}  "
            f"mean={summary['mean']:.1f}  median={summary['median']:.1f} tokens"
        )
    summary_text = "Sequence length summary\n" + "\n".join(summary_lines)
    fig.text(0.5, 0.01, summary_text, ha="center", va="bottom", fontsize=9,
             family="monospace", color="#333333")

    ax.set_title("Sample Coverage at Each Token Timestep\n"
                 "(green ≥75%  |  yellow ≥50%  |  orange ≥25%  |  red <25%  |  bold = plot cutoff)",
                 fontsize=11, pad=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Coverage table saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcd", type=str, default=None)
    parser.add_argument("--icd", type=str, default=None)
    parser.add_argument("--sid", type=str, default=None)
    parser.add_argument("--out", type=str, default="outputs/kld_experiment/coverage_table.png")
    args = parser.parse_args()

    method_paths = {"VCD": args.vcd, "ICD": args.icd, "SID": args.sid}

    all_rows      = []
    all_summaries = []
    all_labels    = []

    for label, path in method_paths.items():
        if path is None:
            continue
        results       = load_results(path)
        rows, summary = get_stats(results)
        all_rows.append(rows)
        all_summaries.append(summary)
        all_labels.append(label)
        # still print to terminal too
        print(f"\n{label} ({summary['total']} samples)  "
              f"min={summary['min']} max={summary['max']} "
              f"mean={summary['mean']:.1f} median={summary['median']:.1f}")
        for t, count, total, pct in rows:
            marker = " <-- plot cutoff" if t == 85 else ""
            print(f"  token {t:>3}: {count:>3}/{total}  ({pct:5.1f}%){marker}")

    if all_rows:
        save_table_image(all_rows, all_summaries, all_labels, args.out)