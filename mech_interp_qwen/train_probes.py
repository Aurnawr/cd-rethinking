#!/usr/bin/env python3
"""Per-layer linear probes + residual-change curves for the Qwen2.5-VL POPE-COCO battery.

Consumes the shards written by extract_cd.py and produces the two curves every downstream figure
needs, plus the classic single-method summary:

  Probe A (truth)        : hidden_l -> ground-truth object presence   (all samples)
  Probe B (hallucination): hidden_l -> baseline false-positive 'Yes', restricted to
                           ground-truth-No samples only (it predicts the ERROR, not the label)
  CD curve               : mean_l ||h_expert[l] - h_amateur[l]||_2 for the primary method

Outputs (under --results-dir)
  per_layer.csv             layer, probe_truth_{acc,auc,bal_acc}, probe_halluc_{acc,auc,bal_acc},
                            cd_change_mean, cd_change_norm
  per_layer_normalized.csv  layer, cd_raw_mean, clean_state_norm_mean, cd_relative_mean,
                            cd_raw_norm, cd_relative_norm, probe_truth_auc, probe_halluc_auc
                            (this is the --probe-csv every logit_lens_figures run reads)
  layerwise_curves.png      the three normalized curves with l* markers
  summary.{txt,json}        l*_truth, l*_halluc, l*_CD, balances, and the coincide verdict

Two deliberate differences from `mech_interp/train_probes.py`
  * Layer count and hidden dim come from the data, not hardcoded 33/4096 (Qwen2.5-VL-7B is 29/3584).
  * One `cross_val_predict(..., method="predict_proba")` per probe instead of three separate CV
    passes (cross_val_score + predict_proba + predict). Accuracy/balanced-accuracy are derived
    from the same out-of-fold probabilities -- identical numbers for logistic regression, since
    `predict` is exactly `proba >= 0.5` -- at a third of the fits.

Features are standardized per fold (StandardScaler inside the pipeline), a scale-only transform
that leaks no labels across folds. Peak selection uses AUC, not raw accuracy, because POPE's
gt-No hallucination population is class-imbalanced.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METHODS = ("vcd", "icd", "sid")


def load_shards(act_dir: str, splits, method: str):
    """Concatenate all shards. clean_hidden stays fp16 (cast per layer) to bound memory."""
    ch, cd, cn, gt, pred, split_of = [], [], [], [], [], []
    for split in splits:
        files = sorted(glob.glob(os.path.join(act_dir, f"{split}_shard*.npz")))
        if not files:
            print(f"[warn] no shards for split '{split}' in {act_dir}")
        for fp in files:
            z = np.load(fp)
            ch.append(z["clean_hidden"])                       # fp16 [n,L+1,d]
            cd.append(z[f"cd_change_{method}"].astype(np.float32))
            cn.append(z["clean_norm"].astype(np.float32))
            gt.append(z["gt"].astype(np.int64))
            pred.append(z["pred"].astype(np.int64))
            split_of.append(np.array([split] * len(z["gt"])))
    if not ch:
        raise SystemExit(f"No shards found in {act_dir} for splits {splits}")
    return (np.concatenate(ch), np.concatenate(cd), np.concatenate(cn),
            np.concatenate(gt), np.concatenate(pred), np.concatenate(split_of))


def probe_layer(X, y, cv, n_jobs):
    """5-fold out-of-fold accuracy + AUC + balanced accuracy for one layer's features."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
    try:
        proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba", n_jobs=n_jobs)[:, 1]
    except (ValueError, IndexError):
        return float("nan"), float("nan"), float("nan")
    yhat = (proba >= 0.5).astype(int)  # == LogisticRegression.predict for the binary case
    try:
        auc = roc_auc_score(y, proba)
    except ValueError:
        auc = float("nan")
    return float((yhat == y).mean()), float(auc), float(balanced_accuracy_score(y, yhat))


def norm01(x):
    x = np.asarray(x, float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def run(args):
    os.makedirs(args.results_dir, exist_ok=True)
    ch, cd, cn, gt, pred, split_of = load_shards(args.act_dir, args.splits, args.method)
    n, n_layers, dim = ch.shape
    print(f"[data] N={n}  layers={n_layers}  dim={dim}  primary CD method={args.method}")

    print(f"[balance] ground truth: present(Yes)={int(gt.sum())} absent(No)={int((gt == 0).sum())}")
    for split in args.splits:
        m = split_of == split
        if m.sum():
            print(f"[baseline] {split:>12}: n={int(m.sum()):>5}  greedy acc={(pred[m] == gt[m]).mean():.4f}")

    no_mask = gt == 0
    y_hall = pred[no_mask]  # 1 = false-positive Yes = a hallucination
    n_no, n_hall = int(no_mask.sum()), int(y_hall.sum())
    hall_rate = n_hall / max(n_no, 1)
    print(f"[halluc] ground-truth-No samples={n_no}  hallucinated(Yes)={n_hall}  rate={hall_rate:.4f}")
    do_probe_b = n_no >= args.min_probe_b and 0 < n_hall < n_no
    if not do_probe_b:
        print("[halluc] Probe B skipped (too few No samples or degenerate class balance).")

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    rows = []
    for l in range(n_layers):
        Xl = ch[:, l, :].astype(np.float32)
        a_acc, a_auc, a_bal = probe_layer(Xl, gt, cv, args.n_jobs)
        if do_probe_b:
            b_acc, b_auc, b_bal = probe_layer(Xl[no_mask], y_hall, cv, args.n_jobs)
        else:
            b_acc = b_auc = b_bal = float("nan")
        rows.append({
            "layer": l,
            "probe_truth_acc": a_acc, "probe_truth_auc": a_auc, "probe_truth_bal_acc": a_bal,
            "probe_halluc_acc": b_acc, "probe_halluc_auc": b_auc, "probe_halluc_bal_acc": b_bal,
            "cd_change_mean": float(cd[:, l].mean()),
        })
        print(f"  L{l:>2}  truthAcc={a_acc:.3f} truthAUC={a_auc:.3f}  "
              f"hallAcc={b_acc:.3f} hallAUC={b_auc:.3f}  cd={cd[:, l].mean():.3f}", flush=True)

    df = pd.DataFrame(rows)
    cd_mean = df["cd_change_mean"].to_numpy()
    df["cd_change_norm"] = (cd_mean - cd_mean.min()) / (np.ptp(cd_mean) + 1e-12)
    csv_path = os.path.join(args.results_dir, "per_layer.csv")
    df.to_csv(csv_path, index=False)

    # ---- per_layer_normalized.csv: residual magnitudes on H + the two probe curves ----
    H = (gt == 0) & (pred == 1)
    if H.sum() == 0:
        raise SystemExit("[probes] no hallucinations (gt=No, pred=Yes) -- nothing to normalize over")
    cd_raw = cd[H].mean(axis=0)
    clean_state = cn[H].mean(axis=0)
    cd_rel = (cd[H] / np.clip(cn[H], 1e-6, None)).mean(axis=0)
    norm_df = pd.DataFrame({
        "layer": np.arange(n_layers),
        "cd_raw_mean": cd_raw,
        "clean_state_norm_mean": clean_state,
        "cd_relative_mean": cd_rel,
        "cd_raw_norm": norm01(cd_raw),
        "cd_relative_norm": norm01(cd_rel),
        "probe_truth_auc": df["probe_truth_auc"].to_numpy(),
        "probe_halluc_auc": df["probe_halluc_auc"].to_numpy(),
    })
    norm_path = os.path.join(args.results_dir, "per_layer_normalized.csv")
    norm_df.to_csv(norm_path, index=False)

    # ---- peaks + plot ----
    l_truth = int(df["probe_truth_auc"].idxmax())
    l_cd = int(df["cd_change_mean"].idxmax())
    l_hall = int(df["probe_halluc_auc"].idxmax()) if do_probe_b else None

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["layer"], df["probe_truth_auc"], "-o", ms=3, label="Probe A: ground-truth AUC")
    if do_probe_b:
        ax.plot(df["layer"], df["probe_halluc_auc"], "-s", ms=3, label="Probe B: hallucination AUC")
    ax.plot(df["layer"], df["cd_change_norm"], "-^", ms=3,
            label=f"{args.method.upper()} residual change (norm.)")
    for lyr, c in [(l_truth, "C0"), (l_cd, "C2")] + ([(l_hall, "C1")] if do_probe_b else []):
        ax.axvline(lyr, color=c, ls="--", alpha=0.5)
    ax.set_xlabel(f"Layer index (0 = embeddings, 1..{n_layers - 1} = transformer blocks)")
    ax.set_ylabel("AUC  /  normalized residual change")
    ax.set_title("POPE-COCO (Qwen2.5-VL): where truth is decodable vs where CD perturbs the residual stream")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png_path = os.path.join(args.results_dir, "layerwise_curves.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    gap_truth = abs(l_cd - l_truth)
    verdict = (
        f"l*_truth={l_truth}, l*_CD={l_cd} (gap {gap_truth} layers). "
        + ("COINCIDE" if gap_truth <= args.coincide_tol else "DIFFERENT DEPTH")
        + " -> "
        + (f"{args.method.upper()}'s perturbation peaks near the layer that encodes object presence."
           if gap_truth <= args.coincide_tol else
           f"{args.method.upper()}'s largest residual effect is NOT at the layer encoding object "
           f"presence -- direct evidence its correction does not operate on the presence representation.")
    )
    if do_probe_b:
        gap_h = abs(l_cd - l_hall)
        verdict += (f"\nl*_halluc={l_hall}, l*_CD={l_cd} (gap {gap_h} layers). "
                    + ("COINCIDE" if gap_h <= args.coincide_tol else "DIFFERENT DEPTH") + ".")

    summary = {
        "model_family": "qwen2.5-vl", "primary_method": args.method,
        "n_samples": int(n), "n_layers": int(n_layers), "hidden_dim": int(dim),
        "gt_present": int(gt.sum()), "gt_absent": int((gt == 0).sum()),
        "probe_b_n_no_samples": n_no, "probe_b_hallucinated": n_hall,
        "probe_b_hallucination_rate": hall_rate, "probe_b_evaluated": do_probe_b,
        "l_star_truth": l_truth, "l_star_truth_auc": float(df.loc[l_truth, "probe_truth_auc"]),
        "l_star_halluc": l_hall,
        "l_star_halluc_auc": (float(df.loc[l_hall, "probe_halluc_auc"]) if do_probe_b else None),
        "l_star_CD": l_cd, "cd_at_peak": float(df.loc[l_cd, "cd_change_mean"]),
        "cd_at_truth_peak": float(df.loc[l_truth, "cd_change_mean"]),
        "verdict": verdict,
    }
    with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.results_dir, "summary.txt"), "w") as f:
        f.write(verdict + "\n\n" + json.dumps(summary, indent=2) + "\n")

    print("\n==== SUMMARY ====")
    print(verdict)
    print(f"\ncsv : {csv_path}\nnorm: {norm_path}\nplot: {png_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--method", default="vcd", choices=METHODS,
                    help="which method's residual-change curve goes in per_layer.csv")
    ap.add_argument("--splits", nargs="+", default=["random", "popular", "adversarial"])
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=-1, help="parallel CV folds")
    ap.add_argument("--min-probe-b", type=int, default=50, help="min GT-No samples to run Probe B")
    ap.add_argument("--coincide-tol", type=int, default=2, help="|peak gap| <= tol counts as coinciding")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
