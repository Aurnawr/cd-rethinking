#!/usr/bin/env python3
"""Experiment 3: oracle calibration of the selectivity metric.

A null selectivity AUC admits two readings: either CD's shift is genuinely non-selective, or
the metric is too insensitive to see selectivity even when it is there. This experiment rules
out the second by pushing a shift that is selective *by construction* through the identical
pipeline.

Part A - oracle ceiling.
  Fit Probe A (object presence) on the per-layer logit-lens margin vector of the H u T
  population, out-of-fold. With p+ the predicted presence probability and a = 1 - p+ the
  predicted absence, the oracle shift is

      Delta_oracle = -a

  i.e. suppress "Yes" in proportion to predicted absence. Its selectivity AUC through the same
  `rank_auc(-Delta, H)` used for CD is the ceiling the instrument can register on this data.
  Fitting on margins rather than hidden states makes the ceiling conservative: presence is more
  decodable from the residual stream (Probe A AUC 0.99) than from the 1-D-per-layer margins.

Part B - graded oracle (how small a selective component would we catch?).
  Inject a scaled selective component into CD's own shift,

      Delta_s = Delta_CD + s * u_oracle,     u_oracle = standardize(Delta_oracle)

  and sweep s. u_oracle has unit standard deviation, so s is in log-odds. s* is the smallest s
  whose selectivity clears the detection threshold, defined as the upper end of the 95%
  stratified-bootstrap CI of CD's *own* readout selectivity: the level an injected component
  must reach before the metric separates it from what CD alone already produces. (Because CD
  sits at chance this is numerically "chance plus a CI half-width", but the CI-of-the-observed
  form is the one that makes s* a detection limit rather than a significance-vs-0.5 test.)
  Reporting s* against CD's mean |Delta| converts "CD is not selective" into "CD's shift is N
  times larger than the smallest selective component this test would have detected, and still
  registers nothing".

Also writes the supporting per-layer selectivity figure with bootstrap CIs, plus a residualized
selectivity curve (Delta ranked after regressing out the expert margin) that checks the readout
number is not an artifact of H and T sitting at different expert margins to begin with.

    python mech_interp_calib/oracle_calibration.py \
        --npz results/qwen2.5-vl-7b/logit_lens_per_sample_vcd.npz \
        --model-tag qwen2.5-vl-7b --method vcd --out-dir results/calibration
"""
import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from calib_common import (  # noqa: E402
    READOUT, Lens, bootstrap_auc_ci, model_label, probe_oof_proba, rank_auc,
)


def residualized_selectivity(lens, layer):
    """Selectivity of -Delta after removing its linear dependence on the expert margin.

    H and T differ in expert margin by construction, so a shift that is merely a function of
    the margin would score above chance without carrying any hallucination-specific
    information. Residualizing removes that channel.
    """
    d = lens.delta_pos(layer)
    m = lens.margin_pos(layer)
    if np.std(m) == 0:
        return rank_auc(-d, lens.y_H)
    beta = np.cov(d, m, bias=True)[0, 1] / np.var(m)
    return rank_auc(-(d - beta * m), lens.y_H)


def oracle_direction(lens, folds, seed):
    """Out-of-fold presence probe on the per-layer margins -> unit-scale oracle direction."""
    X = lens.clean[lens.pos]
    y_present = 1 - lens.y_H          # T = present = 1, H = absent = 0
    p_present = probe_oof_proba(X, y_present, folds=folds, seed=seed)
    absence = 1.0 - p_present
    delta_oracle = -absence
    u = (delta_oracle - delta_oracle.mean()) / delta_oracle.std()
    return delta_oracle, u, float(rank_auc(p_present, y_present.astype(bool)))


def graded_sweep(delta_cd, u, y_H, s_max, s_step):
    s_grid = np.arange(0.0, s_max + 0.5 * s_step, s_step)
    auc = np.array([rank_auc(-(delta_cd + s * u), y_H) for s in s_grid])
    return s_grid, auc


def plot_graded(res, out_png):
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.plot(res["s_grid"], res["auc_grid"], color="0.15", lw=2.6)
    ax.axhline(0.5, color="0.6", ls=":", lw=1.2)
    ax.axhline(res["detection_threshold"], color="0.45", ls="--", lw=1.4)
    ax.text(res["s_grid"][-1], res["detection_threshold"] + 0.006,
            "detection threshold (CD's 95% CI upper bound)", color="0.45", ha="right",
            fontsize=10)
    ax.plot([0], [res["selectivity_cd"]], "o", ms=11, color="#a01c30", zorder=5)
    ax.annotate(f"CD observed\n{res['selectivity_cd']:.2f}", (0, res["selectivity_cd"]),
                xytext=(10, -26), textcoords="offset points", color="#a01c30", fontsize=11)
    if res["s_star"] is not None:
        ax.axvline(res["s_star"], color="#1f6fb4", lw=1.6)
        ax.plot([res["s_star"]], [res["auc_at_s_star"]], "o", ms=11, color="#1f6fb4", zorder=5)
        ax.annotate(f"$s^*$ = {res['s_star']:.2f} log-odds\n"
                    f"({100 * res['s_star_frac_of_cd']:.1f}% of CD's $|\\Delta|$)",
                    (res["s_star"], res["auc_at_s_star"]),
                    xytext=(14, 16), textcoords="offset points", color="#1f6fb4", fontsize=12)
    ax.set_xlabel("Injected selective shift $s$ (log-odds), "
                  "$\\Delta^s = \\Delta^{CD} + s\\,\\hat u_{\\mathrm{oracle}}$", fontsize=12)
    ax.set_ylabel("Selectivity AUC", fontsize=12)
    ax.set_title(f"{res['headline']}\n{res['model_label']}, {res['method'].upper()}",
                 fontsize=13)
    ax.grid(alpha=0.2)
    fig.text(0.5, 0.035,
             f"By-construction oracle ceiling = {res['oracle_ceiling']:.2f}. "
             f"CD mean $|\\Delta|$ = {res['cd_mean_abs_delta']:.2f} log-odds.",
             ha="center", fontsize=10, color="0.35")
    fig.text(0.5, 0.008,
             f"Mean $\\Delta$ at the readout: $\\mathcal{{H}}$ {res['cd_mean_delta_H']:+.2f}, "
             f"$\\mathcal{{T}}$ {res['cd_mean_delta_T']:+.2f} log-odds "
             f"($\\Delta>0$ = pushes toward Yes).",
             ha="center", fontsize=10, color="0.35")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("wrote", out_png)


def plot_selectivity_ci(res, out_png):
    layers = np.arange(len(res["sel_by_layer"]))
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.axhspan(0.5 - res["ci_halfwidth"], 0.5 + res["ci_halfwidth"], color="0.85", zorder=0)
    ax.fill_between(layers, res["sel_ci_lo"], res["sel_ci_hi"], color="#9ecae1", alpha=0.55,
                    label="95% bootstrap CI (raw)")
    ax.plot(layers, res["sel_by_layer"], color="#1f6fb4", lw=2.4, label="raw  $-\\Delta_\\ell$")
    ax.plot(layers, res["sel_resid_by_layer"], color="#a01c30", lw=2.4,
            label="residualized  $-\\Delta_\\ell \\mid m^E_\\ell$")
    ax.axhline(0.5, color="0.5", ls=":", lw=1.2)
    ax.set_xlabel("Transformer layer", fontsize=12)
    ax.set_ylabel("Selectivity AUC ($-\\Delta_\\ell$ ranks $\\mathcal{H}$ over $\\mathcal{T}$)",
                  fontsize=11)
    ax.set_title(f"Layer-wise selectivity with bootstrap CIs\n"
                 f"{res['model_label']}, {res['method'].upper()}", fontsize=13)
    ax.legend(fontsize=10, loc="best")
    ax.grid(alpha=0.2)
    fig.text(0.5, 0.005,
             f"$|\\mathcal{{H}}|$ = {res['n_H']}, $|\\mathcal{{T}}|$ = {res['n_T']}. "
             f"Grey band = chance $\\pm$ {res['ci_halfwidth']:.3f}. "
             f"AUC > 0.5 = suppresses hallucinations more; readout raw "
             f"{res['selectivity_cd']:.2f}, residualized {res['sel_resid_by_layer'][-1]:.2f}.",
             ha="center", fontsize=10, color="0.35")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("wrote", out_png)


def run(lens, model_tag, method, folds, seed, n_boot, s_max, s_step):
    delta_cd = lens.delta_pos(READOUT)
    sel_cd = float(lens.selectivity(READOUT))

    lo, hi, se = bootstrap_auc_ci(-delta_cd, lens.y_H.astype(bool), n_boot=n_boot, seed=seed)
    half = 1.96 * se
    threshold = hi

    delta_oracle, u, probe_auc = oracle_direction(lens, folds, seed)
    ceiling = float(rank_auc(-delta_oracle, lens.y_H.astype(bool)))

    s_grid, auc_grid = graded_sweep(delta_cd, u, lens.y_H.astype(bool), s_max, s_step)
    above = np.flatnonzero(auc_grid >= threshold)
    s_star = float(s_grid[above[0]]) if above.size else None
    auc_at_s_star = float(auc_grid[above[0]]) if above.size else None

    mean_abs = float(np.abs(delta_cd).mean())
    mean_H = float(lens.delta[lens.H, READOUT].mean())
    mean_T = float(lens.delta[lens.T, READOUT].mean())
    margin_scale = float(np.median(np.abs(lens.margin_pos(READOUT))))
    sel_layers = lens.selectivity_by_layer()
    ci = [bootstrap_auc_ci(-lens.delta_pos(l), lens.y_H.astype(bool),
                           n_boot=max(400, n_boot // 4), seed=seed + l)
          for l in range(lens.n_layers)]

    return {
        "model_tag": model_tag,
        "model_label": model_label(model_tag),
        "method": method,
        "n": int(lens.n),
        "n_layers": int(lens.n_layers),
        "n_H": lens.n_H,
        "n_T": lens.n_T,
        "selectivity_cd": sel_cd,
        "selectivity_cd_ci": [lo, hi],
        "ci_halfwidth": float(half),
        "detection_threshold": float(threshold),
        "detection_threshold_rule": "upper end of the 95% bootstrap CI of CD's readout selectivity",
        "chance_plus_ci_halfwidth": float(0.5 + half),
        "cd_above_chance": bool(lo > 0.5),
        "oracle_ceiling": ceiling,
        "probe_a_margin_auc": probe_auc,
        "cd_mean_abs_delta": mean_abs,
        "cd_mean_delta_H": mean_H,
        "cd_mean_delta_T": mean_T,
        "cd_frac_delta_pos_H": float((lens.delta[lens.H, READOUT] > 0).mean()),
        "cd_frac_delta_pos_T": float((lens.delta[lens.T, READOUT] > 0).mean()),
        "expert_margin_scale": margin_scale,
        "verdict": verdict(sel_cd, lo, mean_H, mean_T, margin_scale),
        "headline": headline(lo, sel_cd, mean_H, (mean_abs / s_star) if s_star else None),
        "s_star": s_star,
        "auc_at_s_star": auc_at_s_star,
        "s_star_frac_of_cd": (s_star / mean_abs) if s_star else None,
        "detection_factor": (mean_abs / s_star) if s_star else None,
        "s_grid": s_grid,
        "auc_grid": auc_grid,
        "sel_by_layer": sel_layers,
        "sel_resid_by_layer": np.array([residualized_selectivity(lens, l)
                                        for l in range(lens.n_layers)]),
        "sel_ci_lo": np.array([c[0] for c in ci]),
        "sel_ci_hi": np.array([c[1] for c in ci]),
        "folds": folds,
        "seed": seed,
        "n_boot": n_boot,
    }


def verdict(sel, ci_lo, mean_H, mean_T, margin_scale):
    """One line separating a real correction from the boost-correct-positives-harder artifact.

    A selectivity AUC above chance only means -Delta ranks H above T. It does *not* mean CD
    suppresses hallucinations: if Delta is positive on both classes and merely larger on T, the
    ranking is right while the intervention is backwards. And even a correctly-signed mean is
    only a correction if it is large enough to move a decision, so it is reported against the
    scale of the margin it is added to (median |m_expert| on H u T).
    """
    if not (ci_lo > 0.5):
        return ("non-selective: the shift's ranking of H over T is indistinguishable from "
                f"chance (AUC {sel:.2f}, CI lower bound {ci_lo:.2f}).")
    if mean_H > 0:
        return (f"selectivity {sel:.2f} is above chance but is the boost-correct-positives "
                f"artifact: the shift is positive on both classes (H {mean_H:+.2f}, "
                f"T {mean_T:+.2f} log-odds), so on a hallucination it still reinforces the "
                f"false Yes, just less than it reinforces a correct one.")
    return (f"selectivity {sel:.2f} above chance and correctly signed on H ({mean_H:+.2f} "
            f"log-odds), but the H-vs-T differential is only {mean_T - mean_H:+.2f} log-odds "
            f"against a margin of median magnitude {margin_scale:.1f} "
            f"({100 * abs(mean_H) / margin_scale:.1f}% of it) - a nudge, not a correction.")


def headline(ci_lo, sel, mean_H, factor):
    """Short figure title matching the verdict branch."""
    if not (ci_lo > 0.5):
        if factor:
            return (f"Selectivity sits at chance, and the test would have caught a selective "
                    f"shift {factor:.0f}x smaller than CD's own")
        return "Selectivity sits at chance"
    if mean_H > 0:
        return (f"Ranks $\\mathcal{{H}}$ over $\\mathcal{{T}}$ at {sel:.2f}, yet still moves "
                f"hallucinations {mean_H:+.2f} log-odds toward Yes")
    return f"Correctly signed on $\\mathcal{{H}}$ ({mean_H:+.2f} log-odds) but tiny"


def to_json(res):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--s-max", type=float, default=3.0)
    ap.add_argument("--s-step", type=float, default=0.01)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    lens = Lens(args.npz)
    if lens.n_H < 10 or lens.n_T < 10:
        raise SystemExit(f"{args.npz}: H={lens.n_H} T={lens.n_T} -- too few to calibrate")

    res = run(lens, args.model_tag, args.method, args.folds, args.seed,
              args.n_boot, args.s_max, args.s_step)
    stem = f"{args.model_tag}_{args.method}"
    plot_graded(res, os.path.join(args.out_dir, f"fig_oracle_graded_{stem}.png"))
    plot_selectivity_ci(res, os.path.join(args.out_dir, f"fig_selectivity_ci_{stem}.png"))
    with open(os.path.join(args.out_dir, f"oracle_{stem}.json"), "w") as f:
        json.dump(to_json(res), f, indent=2)

    factor = res["detection_factor"]
    lo, hi = res["selectivity_cd_ci"]
    print(f"[oracle {stem}] CD={res['selectivity_cd']:.3f} CI=[{lo:.3f},{hi:.3f}] "
          f"ceiling={res['oracle_ceiling']:.3f} s*={res['s_star']} "
          f"mean|D|={res['cd_mean_abs_delta']:.2f} "
          f"factor={f'{factor:.1f}x' if factor else 'n/a'}")
    print(f"    {res['verdict']}")


if __name__ == "__main__":
    main()
