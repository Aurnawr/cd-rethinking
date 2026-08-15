#!/usr/bin/env python3
"""The four per-method logit-lens figures, for Qwen2.5-VL.

Port of `mech_interp/logit_lens_figures.py` with the layer count read from the data instead of a
hardcoded 33, so it renders LLaVA-1.5 (33) and Qwen2.5-VL-7B (29) identically.

  1. halluc_auc_vs_cd_effect_<m>.png  hallucination decodability vs CD's mean shift on H
  2. layerwise_battery_<m>.png        4-panel selectivity battery
  3. logit_lens_curves_<m>.png        directional margins on H + where Delta lands
  4. cd_normalized_curves_<m>.png     raw vs de-confounded residual-magnitude curves

Everything margin-based is recomputed per method from the per-sample npz (clean_margin /
amateur_margin / gt / pred). The probe curves (probe_truth_auc = presence Probe A,
probe_halluc_auc = hallucination Probe B) come from --probe-csv: they are read off the
clean/expert pass, which is shared by all three CD methods, so the same curves apply to
VCD/ICD/SID. Residual-magnitude curves come from --resid-csv.

Convention: Delta_l = m_clean_l - m_amateur_l ; the CD margin shift is +alpha*Delta.
Delta>0 reinforces "Yes"; Delta<0 suppresses it (the correcting direction on a hallucination).
"""
import argparse
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_TAG = "Qwen2.5-VL"


# --------------------------------------------------------------------------- utils
def rank_auc(scores, labels):
    """AUC that positive-class `scores` rank above negatives (ties = 0.5)."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels).astype(bool)
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ss = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks[labels].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def wilson(k, n, z=1.96):
    """95% Wilson interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def norm01(x):
    x = np.asarray(x, float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def read_csv_cols(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0].keys()}


def first_cross(curve, thr=0.95):
    idx = np.where(np.asarray(curve, float) >= thr)[0]
    return int(idx[0]) if len(idx) else None


def layer_axis_label(n_states):
    return f"Layer (0 = embeddings, 1..{n_states - 1} = transformer blocks)"


def load_margins(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    cm = z["clean_margin"].astype(float)
    am = (z["amateur_margin"] if "amateur_margin" in z.files else z["noisy_margin"]).astype(float)
    return cm, am, z["gt"].astype(int), z["pred"].astype(int)


# --------------------------------------------------------------------------- figures
def fig_halluc_auc_vs_cd(cm, am, gt, pred, probe, method, out):
    H = (gt == 0) & (pred == 1)
    meanD_H = (cm - am)[H].mean(axis=0)
    L = cm.shape[1]
    x = np.arange(L)
    p_h = probe["probe_halluc_auc"]
    l_dec = first_cross(p_h, 0.95)
    l_peak = int(np.argmax(meanD_H))

    fig, ax = plt.subplots(figsize=(14, 7))
    ax2 = ax.twinx()
    ax.plot(x, p_h, "-s", color="darkorange", label="Probe: hallucination decodable (AUC)")
    ax2.plot(x, meanD_H, "-^", color="green",
             label="CD effect: mean $\\Delta_\\ell$ on hallucinations (log-odds)")
    ax.axhline(0.5, color="0.5", ls=":", lw=1)
    ax.text(0.3, 0.505, "chance", color="0.5", fontsize=9)
    if l_dec is not None:
        ax.axvline(l_dec, color="darkorange", ls="--", alpha=0.7)
        ax.text(l_dec + 0.2, 0.6, f"hallucination\ndecodable ($\\ell={l_dec}$)",
                color="darkorange", fontsize=9)
    ax2.axvline(l_peak, color="green", ls="--", alpha=0.6)
    ax2.annotate(f"CD effect\npeaks ($\\ell={l_peak}$)", (l_peak, meanD_H[l_peak]),
                 color="green", fontsize=9)
    ax.set_xlabel(layer_axis_label(L))
    ax.set_ylabel("hallucination decodability  (AUC)", color="darkorange")
    ax2.set_ylabel("CD shift $\\Delta_\\ell$ on hallucinated samples  (log-odds)", color="green")
    ax.set_title(f"[{MODEL_TAG} / {method.upper()}] Where the hallucination is decided "
                 f"vs where CD's effect lands")
    lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
    ax.legend(lines, [l.get_label() for l in lines], loc="center left", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def fig_layerwise_battery(cm, am, gt, pred, probe, method, out):
    H = (gt == 0) & (pred == 1)
    T = (gt == 1) & (pred == 1)
    pos = H | T
    yH = H[pos].astype(int)
    present_all = (gt == 1).astype(int)
    d = cm - am
    L = cm.shape[1]
    x = np.arange(L)
    sel = np.array([rank_auc(-d[pos, l], yH) for l in range(L)])
    auc_clean = np.array([rank_auc(cm[:, l], present_all) for l in range(L)])
    auc_amateur = np.array([rank_auc(am[:, l], present_all) for l in range(L)])
    n_H = int(H.sum())
    frac = np.array([(d[H, l] > 0).mean() for l in range(L)])
    lo = np.array([wilson(int((d[H, l] > 0).sum()), n_H)[0] for l in range(L)])
    hi = np.array([wilson(int((d[H, l] > 0).sum()), n_H)[1] for l in range(L)])
    r2 = np.array([np.corrcoef(d[pos, l], yH)[0, 1] ** 2 if np.std(d[pos, l]) > 0 else np.nan
                   for l in range(L)])
    p_truth = probe["probe_truth_auc"]
    l_pres = first_cross(p_truth, 0.95)

    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(20, 13))
    def vl(ax):
        if l_pres is not None:
            ax.axvline(l_pres, color="steelblue", ls="--", alpha=0.6)

    a1.plot(x, p_truth, "-s", color="tab:blue",
            label="information AVAILABLE:\nProbe A object-presence AUC (H vs T = absent vs present)")
    a1.plot(x, sel, "-o", color="tab:green",
            label="information USED by CD:\nselectivity AUC of $-\\Delta_\\ell$ (H vs T)")
    a1.axhline(0.5, color="0.5", ls=":")
    a1.text(max(L - 4, 0), 0.505, "chance", color="0.5", fontsize=9)
    vl(a1)
    if l_pres is not None:
        a1.text(l_pres + 0.2, 0.6, f"presence formed\n$\\ell={l_pres}$", color="steelblue", fontsize=9)
    a1.set_ylabel("AUC  [probability]")
    a1.legend(fontsize=9, loc="center")
    a1.set_title("1. Presence is decodable (so H vs T is separable) - CD's shift can't separate them")

    a2.plot(x, auc_clean, "-o", color="tab:blue",
            label="expert pass: AUC($m_{clean}\\to$present), all samples")
    a2.plot(x, auc_amateur, "-^", color="tab:red",
            label="amateur pass: AUC($m_{amateur}\\to$present), all samples")
    a2.axhline(0.5, color="0.5", ls=":")
    vl(a2)
    a2.set_ylabel("AUC  [probability]")
    a2.legend(fontsize=9, loc="lower right")
    a2.set_title("2. What CD subtracts: does the amateur pass form the presence signal?")

    a3.plot(x, frac, "-o", color="tab:red")
    a3.fill_between(x, lo, hi, color="tab:red", alpha=0.2, label="95% Wilson CI")
    a3.axhline(0.5, color="0.5", ls=":")
    a3.text(0.3, 0.52, "50% (undirected)", color="0.5", fontsize=9)
    vl(a3)
    a3.set_ylabel("fraction of $\\mathcal{H}$ with $\\Delta_\\ell>0$\n(pushed toward Yes)")
    a3.set_xlabel(layer_axis_label(L))
    a3.legend(fontsize=9, loc="lower right")
    a3.set_title("3. Depth where CD's anti-correction becomes systematic")

    a4.plot(x, r2, "-o", color="tab:purple")
    a4.set_ylim(0, max(0.05, float(np.nanmax(r2)) * 1.2 if np.isfinite(np.nanmax(r2)) else 0.25))
    vl(a4)
    a4.set_ylabel("$R^2$ of $\\Delta_\\ell$ explained by\nhallucination status (H vs T)")
    a4.set_xlabel(layer_axis_label(L))
    a4.set_title("4. How much of CD's shift is 'about' hallucination at all?")

    for ax in (a1, a2, a3, a4):
        ax.grid(alpha=0.25)
    fig.suptitle(f"[{MODEL_TAG} / {method.upper()}] Layer-wise test: CD's shift never separates "
                 f"hallucinated from correct Yes-answers, at any depth", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def fig_logit_lens_curves(cm, am, gt, pred, probe, resid, method, out):
    H = (gt == 0) & (pred == 1)
    d = cm - am
    clean_H, amateur_H, delta_H = cm[H].mean(axis=0), am[H].mean(axis=0), d[H].mean(axis=0)
    L = cm.shape[1]
    x = np.arange(L)
    l_peak = int(np.argmax(delta_H))

    fig, (aL, aR) = plt.subplots(1, 2, figsize=(20, 7))
    aL.plot(x, clean_H, "-o", color="tab:blue", label="expert Yes-No margin")
    aL.plot(x, amateur_H, "-s", color="tab:red", label="amateur Yes-No margin")
    aL.plot(x, delta_H, "-^", color="tab:green", label="delta = expert - amateur (CD shift)")
    aL.axhline(0, color="0.4", lw=1)
    aL.axvline(l_peak, color="tab:green", ls="--", alpha=0.6)
    aL.set_xlabel(layer_axis_label(L))
    aL.set_ylabel("Yes-No log-odds")
    aL.set_title(f"[{MODEL_TAG} / {method.upper()}] Directional logit-lens on hallucinated GT-No samples")
    aL.legend(fontsize=10)
    aL.grid(alpha=0.25)

    aR.plot(x, norm01(np.abs(delta_H)), "-^", color="tab:green", label="|delta| halluc (norm.)")
    aR.plot(x, probe["probe_halluc_auc"], "-s", color="darkorange", label="Probe B: hallucination AUC")
    if resid is not None:
        aR.plot(x, norm01(resid["cd_relative_mean"]), "-x", color="0.5",
                label="CD relative magnitude (norm.)")
    aR.axvline(l_peak, color="tab:green", ls="--", alpha=0.6)
    aR.set_xlabel(layer_axis_label(L))
    aR.set_ylabel("normalized")
    aR.set_title(f"[{MODEL_TAG} / {method.upper()}] Where the DIRECTIONAL correction lands "
                 f"vs hallucination representation")
    aR.legend(fontsize=10, loc="lower right")
    aR.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def fig_cd_normalized(probe, resid, method, out):
    if resid is None:
        print(f"[skip] cd_normalized_curves_{method}: no resid csv")
        return
    cd_raw = resid["cd_raw_mean"]
    clean_norm = resid["clean_state_norm_mean"]
    cd_rel = resid["cd_relative_mean"]
    p_truth, p_halluc = probe["probe_truth_auc"], probe["probe_halluc_auc"]
    L = len(cd_raw)
    x = np.arange(L)
    l_raw, l_rel = int(np.argmax(cd_raw)), int(np.argmax(cd_rel))
    l_truth, l_halluc = first_cross(p_truth, 0.95), first_cross(p_halluc, 0.95)

    fig, (aL, aR) = plt.subplots(1, 2, figsize=(20, 7))
    aL.plot(x, norm01(cd_raw), "-^", color="tab:red", label="CD raw L2 (norm.) - CONFOUNDED")
    aL.plot(x, norm01(clean_norm), "-x", color="0.5",
            label="$||h_{clean}[\\ell]||$ (norm.) - the artifact")
    aL.plot(x, norm01(cd_rel), "-o", color="tab:green", label="CD relative (norm.) - de-confounded")
    aL.axvline(l_rel, color="tab:green", ls="--", alpha=0.6)
    aL.axvline(l_raw, color="tab:red", ls=":", alpha=0.6)
    aL.set_xlabel(layer_axis_label(L))
    aL.set_ylabel("normalized")
    aL.set_title(f"[{MODEL_TAG} / {method.upper()}] Raw vs relative residual change")
    aL.legend(fontsize=10)
    aL.grid(alpha=0.25)

    aR.plot(x, p_truth, "-o", color="tab:blue", label="Probe A: truth AUC")
    aR.plot(x, p_halluc, "-s", color="darkorange", label="Probe B: hallucination AUC")
    aR.plot(x, norm01(cd_rel), "-^", color="tab:green", label="CD relative (norm.)")
    aR.axhline(0.95, color="0.6", ls=":")
    aR.axvline(l_rel, color="tab:green", ls="--", alpha=0.6)
    if l_truth is not None:
        aR.axvline(l_truth, color="tab:blue", ls="--", alpha=0.5)
    if l_halluc is not None:
        aR.axvline(l_halluc, color="darkorange", ls="--", alpha=0.5)
    aR.set_xlabel(layer_axis_label(L))
    aR.set_ylabel("AUC / normalized")
    aR.set_title(f"[{MODEL_TAG} / {method.upper()}] De-confounded CD vs where truth/hallucination "
                 f"is decodable")
    aR.legend(fontsize=10, loc="lower right")
    aR.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="logit_lens_per_sample_<method>.npz")
    ap.add_argument("--method", required=True)
    ap.add_argument("--probe-csv", required=True,
                    help="per_layer_normalized.csv (shared, method-independent)")
    ap.add_argument("--resid-csv", default=None, help="per_layer_resid_<method>.csv")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cm, am, gt, pred = load_margins(args.npz)
    probe = read_csv_cols(args.probe_csv)
    if probe is None:
        raise SystemExit(f"probe csv missing or empty: {args.probe_csv}")
    if len(probe["probe_truth_auc"]) != cm.shape[1]:
        raise SystemExit(f"probe csv has {len(probe['probe_truth_auc'])} layers but the margins "
                         f"have {cm.shape[1]} -- these came from different runs")
    resid = read_csv_cols(args.resid_csv)
    m = args.method
    n_H = int(((gt == 0) & (pred == 1)).sum())
    n_T = int(((gt == 1) & (pred == 1)).sum())
    print(f"[{m}] N={len(gt)} H={n_H} T={n_T} layers={cm.shape[1]}")
    if n_H == 0:
        raise SystemExit(f"[{m}] no hallucinations in this run -- every figure would be empty")

    o = args.out_dir
    fig_halluc_auc_vs_cd(cm, am, gt, pred, probe, m, os.path.join(o, f"halluc_auc_vs_cd_effect_{m}.png"))
    fig_layerwise_battery(cm, am, gt, pred, probe, m, os.path.join(o, f"layerwise_battery_{m}.png"))
    fig_logit_lens_curves(cm, am, gt, pred, probe, resid, m, os.path.join(o, f"logit_lens_curves_{m}.png"))
    fig_cd_normalized(probe, resid, m, os.path.join(o, f"cd_normalized_curves_{m}.png"))


if __name__ == "__main__":
    main()
