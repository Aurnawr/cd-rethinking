"""
plot_kld.py  —  Plot KL divergence vs token timestep for VCD / ICD / SID

Usage:
    python plot_kld.py \
        --vcd  outputs/vcd_kld.json  \
        --icd  outputs/icd_kld.json  \
        --sid  outputs/sid_kld.json  \
        --out  outputs/kld_plot.png

You can pass any subset of the three flags; only the available ones will be plotted.
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_results(path: str):
    with open(path) as f:
        return json.load(f)


def align_and_average(results, max_len=None):
    """
    Given a list of per-sample result dicts (each with 'kld_per_token'),
    truncate / pad to a common length and average across samples.

    Returns:
        timesteps : np.array  shape (T,)
        mean_kld  : np.array  shape (T,)
        std_kld   : np.array  shape (T,)
    """
    sequences = [r["kld_per_token"] for r in results]

    if max_len is None:
        # use the actual maximum length — coverage filter below handles sparse tail
        max_len = max(len(s) for s in sequences)

    # truncate everything to max_len; skip sequences shorter than 10 tokens
    truncated = [s[:max_len] for s in sequences if len(s) >= 10]

    if not truncated:
        raise ValueError("No sequences with >=2 tokens found.")

    # pad shorter sequences with NaN and take nanmean
    # clamp negatives to 0 (float precision artifact when distributions are very close)
    padded = np.full((len(truncated), max_len), np.nan)
    for i, s in enumerate(truncated):
        padded[i, : len(s)] = np.maximum(s, 1e-10)  # clamp to small epsilon so log scale works

    mean_kld = np.nanmean(padded, axis=0)
    std_kld  = np.nanstd(padded,  axis=0)
    timesteps = np.arange(1, max_len + 1)

    # drop trailing positions with < 10 % coverage (too few samples reached there)
    coverage = np.sum(~np.isnan(padded), axis=0) / len(truncated)
    valid    = coverage >= 0.10
    return timesteps[valid], mean_kld[valid], std_kld[valid]


def smooth(arr, window=5):
    """Simple moving-average smoother using valid convolution (no zero-padding at edges)."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="valid")
    # valid mode shortens the array by (window-1), so trim the timesteps to match in the caller
    return smoothed


# ---------------------------------------------------------------------------
# main plot
# ---------------------------------------------------------------------------

def plot(args):
    method_paths = {
        "VCD": args.vcd,
        "ICD": args.icd,
        "SID": args.sid,
    }

    colors = {
        "VCD": "#E07B39",
        "ICD": "#4A90D9",
        "SID": "#5BAD6F",
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    any_plotted = False
    for label, path in method_paths.items():
        if path is None or not os.path.isfile(path):
            if path is not None:
                print(f"[warn] file not found for {label}: {path}")
            continue

        results = load_results(path)
        print(f"{label}: {len(results)} samples loaded from {path}")

        timesteps, mean_kld, std_kld = align_and_average(results, max_len=args.max_tokens)

        # optional smoothing — valid convolution trims (window-1) points from edges
        if args.smooth > 1:
            mean_kld  = smooth(mean_kld, args.smooth)
            trim      = args.smooth - 1
            timesteps = timesteps[trim // 2 : len(timesteps) - (trim - trim // 2)]

        c = colors[label]
        ax.plot(timesteps, mean_kld, label=label, color=c, linewidth=2)
        any_plotted = True

    if not any_plotted:
        print("No data files found. Nothing to plot.")
        return

    ax.set_xlabel("Token Timestep", fontsize=13)
    ax.set_ylabel("KL Divergence  KL( P_expert ‖ P_amateur )  [log scale]", fontsize=13)
    ax.set_title("KL Divergence Between Expert and Amateur Distributions\nAcross Decoding Timesteps", fontsize=13)
    ax.set_yscale("log")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(left=1, right=85)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"\nPlot saved to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcd",         type=str, default=None, help="JSON output from kld_experiment.py --method vcd")
    parser.add_argument("--icd",         type=str, default=None, help="JSON output from kld_experiment.py --method icd")
    parser.add_argument("--sid",         type=str, default=None, help="JSON output from kld_experiment.py --method sid")
    parser.add_argument("--out",         type=str, default="outputs/kld_plot.png")
    parser.add_argument("--max-tokens",  type=int, default=None,
                        help="Truncate all sequences to this many tokens before averaging. "
                             "Default: median sequence length.")
    parser.add_argument("--smooth",      type=int, default=1,
                        help="Moving-average window for smoothing (1 = no smoothing)")
    args = parser.parse_args()
    plot(args)