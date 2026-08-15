#!/usr/bin/env python3
"""End-to-end test of the analysis chain on synthetic data - no GPU, no model, no dataset.

Fabricates activation shards and per-method margin npz in exactly the format extract_cd.py
writes, then drives the real resid_stats -> train_probes -> logit_lens_figures -> analyze_all
pipeline and asserts every expected artifact appears with the right shape.

This covers everything downstream of the forward pass: the shard contract, the probe CSVs, all
12 regen figures and the cross-method battery. What it cannot cover (model loading, the logit
lens, CT2S masking) is what selfcheck.py validates on the real model.

    python mech_interp_qwen/test_analysis_synthetic.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = ("vcd", "icd", "sid")
SPLITS = ("random", "popular", "adversarial")
N_PER_SPLIT = 120
N_LAYERS = 29          # Qwen2.5-VL-7B: 28 blocks -> 29 hidden states
DIM = 64               # tiny, so the probes run in seconds
SHARD = 50


def make_split(rng, split, act_dir):
    """One split's shards, with a presence signal that strengthens with depth."""
    n = N_PER_SPLIT
    gt = rng.integers(0, 2, size=n)
    # the model is right most of the time but hallucinates on some gt-No samples
    pred = np.where(gt == 1, (rng.random(n) < 0.85).astype(int), (rng.random(n) < 0.35).astype(int))

    hidden = rng.normal(size=(n, N_LAYERS, DIM)).astype(np.float32)
    # inject a presence direction that only becomes decodable from the middle layers on
    for l in range(N_LAYERS):
        strength = 0.0 if l < 6 else min(1.6, 0.25 * (l - 5))
        hidden[:, l, 0] += strength * (2 * gt - 1)
        hidden[:, l, 1] += 0.6 * strength * (2 * pred - 1)
    # residual norms grow with depth - the confound cd_relative_mean divides out
    hidden *= (1.0 + 0.35 * np.arange(N_LAYERS))[None, :, None]

    clean_norm = np.linalg.norm(hidden, axis=2).astype(np.float32)
    payload_cd = {
        m: (clean_norm * rng.uniform(0.03, 0.09, size=(n, N_LAYERS))).astype(np.float32)
        for m in METHODS
    }

    for start in range(0, n, SHARD):
        sl = slice(start, min(start + SHARD, n))
        arrays = {
            "clean_hidden": hidden[sl].astype(np.float16),
            "clean_norm": clean_norm[sl],
            "gt": gt[sl].astype(np.int8),
            "pred": pred[sl].astype(np.int8),
        }
        for m in METHODS:
            arrays[f"cd_change_{m}"] = payload_cd[m][sl]
        idx = start // SHARD
        np.savez_compressed(os.path.join(act_dir, f"{split}_shard{idx:04d}.npz"), **arrays)
        with open(os.path.join(act_dir, f"{split}_shard{idx:04d}.meta.jsonl"), "w") as f:
            for i in range(sl.start, sl.stop):
                f.write(json.dumps({"split": split, "gt": int(gt[i]), "pred_yes": int(pred[i])}) + "\n")
    return gt, pred


def make_margins(rng, gt, pred, split_names, results_dir):
    """Per-method margin npz: expert shared, amateurs shifted so Delta > 0 on H."""
    n = len(gt)
    depth = np.arange(N_LAYERS)[None, :] / (N_LAYERS - 1)
    clean = (rng.normal(scale=0.5, size=(n, N_LAYERS)) + 6.0 * depth * (2 * gt - 1)[:, None]
             ).astype(np.float32)
    for i, m in enumerate(METHODS):
        # a one-sided, hallucination-blind shift: the very pattern the battery is testing for
        amateur = clean - (0.8 + 0.4 * i) * depth - rng.normal(scale=0.3, size=(n, N_LAYERS))
        np.savez_compressed(
            os.path.join(results_dir, f"logit_lens_per_sample_{m}.npz"),
            clean_margin=clean, amateur_margin=amateur.astype(np.float32),
            gt=gt.astype(np.int8), pred=pred.astype(np.int8),
            split=np.asarray(split_names, dtype="U12"),
        )
        with open(os.path.join(results_dir, f"extract_summary_{m}.json"), "w") as f:
            json.dump({"method": m, "n": int(n), "lens_final_layer_MAE": 0.0,
                       "splits_done": list(SPLITS)}, f)


def run(cmd, env):
    print("  $", " ".join(str(c) for c in cmd[1:]), flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"command failed: {' '.join(str(c) for c in cmd)}")
    return proc.stdout


def main():
    tmp = tempfile.mkdtemp(prefix="qwen_mech_test_")
    act_dir = os.path.join(tmp, "activations")
    results_dir = os.path.join(tmp, "results")
    regen_dir = os.path.join(results_dir, "regen")
    for d in (act_dir, results_dir, regen_dir):
        os.makedirs(d, exist_ok=True)

    try:
        rng = np.random.default_rng(0)
        gts, preds, split_names = [], [], []
        for split in SPLITS:
            g, p = make_split(rng, split, act_dir)
            gts.append(g)
            preds.append(p)
            split_names.extend([split] * len(g))
        gt, pred = np.concatenate(gts), np.concatenate(preds)
        make_margins(rng, gt, pred, split_names, results_dir)
        n_H = int(((gt == 0) & (pred == 1)).sum())
        n_T = int(((gt == 1) & (pred == 1)).sum())
        print(f"[synthetic] N={len(gt)} H={n_H} T={n_T} layers={N_LAYERS} dim={DIM}")
        assert n_H > 20 and n_T > 20, "synthetic data degenerate"

        env = dict(os.environ, MPLBACKEND="Agg")
        py = sys.executable

        print("[1/4] resid_stats")
        for m in METHODS:
            run([py, os.path.join(HERE, "resid_stats.py"), "--act-dir", act_dir,
                 "--results-dir", results_dir, "--method", m, "--splits", *SPLITS], env)

        print("[2/4] train_probes")
        run([py, os.path.join(HERE, "train_probes.py"), "--act-dir", act_dir,
             "--results-dir", results_dir, "--method", "vcd", "--splits", *SPLITS], env)

        print("[3/4] logit_lens_figures")
        for m in METHODS:
            run([py, os.path.join(HERE, "logit_lens_figures.py"),
                 "--npz", os.path.join(results_dir, f"logit_lens_per_sample_{m}.npz"),
                 "--method", m,
                 "--probe-csv", os.path.join(results_dir, "per_layer_normalized.csv"),
                 "--resid-csv", os.path.join(results_dir, f"per_layer_resid_{m}.csv"),
                 "--out-dir", regen_dir], env)

        print("[4/4] analyze_all")
        run([py, os.path.join(HERE, "analyze_all.py"), "--results-dir", results_dir], env)

        # ---- assertions ----
        failures = []
        expected = [os.path.join(results_dir, f) for f in (
            "per_layer.csv", "per_layer_normalized.csv", "summary.txt", "summary.json",
            "layerwise_curves.png", "cross_method_battery.png", "summary_all.json", "summary_all.csv",
        )]
        expected += [os.path.join(results_dir, f"per_layer_resid_{m}.csv") for m in METHODS]
        for fig in ("halluc_auc_vs_cd_effect", "layerwise_battery", "logit_lens_curves",
                    "cd_normalized_curves"):
            expected += [os.path.join(regen_dir, f"{fig}_{m}.png") for m in METHODS]
        for path in expected:
            if not (os.path.exists(path) and os.path.getsize(path) > 0):
                failures.append(f"missing or empty: {os.path.relpath(path, tmp)}")

        import csv as _csv
        with open(os.path.join(results_dir, "per_layer_normalized.csv")) as f:
            rows = list(_csv.DictReader(f))
        if len(rows) != N_LAYERS:
            failures.append(f"per_layer_normalized.csv has {len(rows)} rows, expected {N_LAYERS}")
        for col in ("probe_truth_auc", "probe_halluc_auc", "cd_raw_mean",
                    "clean_state_norm_mean", "cd_relative_mean"):
            if col not in rows[0]:
                failures.append(f"per_layer_normalized.csv missing column {col}")
        truth_auc = np.array([float(r["probe_truth_auc"]) for r in rows])
        if not np.nanmax(truth_auc) > 0.75:
            failures.append(f"Probe A never becomes decodable (max AUC {np.nanmax(truth_auc):.3f}) "
                            f"- the probe path is broken")
        halluc_auc = np.array([float(r["probe_halluc_auc"]) for r in rows])
        if np.all(np.isnan(halluc_auc)):
            failures.append("Probe B produced only NaN - the gt-No restriction is broken")

        n_regen = len([f for f in os.listdir(regen_dir) if f.endswith(".png")])
        if n_regen != 12:
            failures.append(f"{n_regen} regen figures, expected 12")

        if failures:
            print("\nFAILED:")
            for f in failures:
                print("  -", f)
            return 1
        print(f"\nPASSED: 12 regen figures + probe CSVs + cross-method battery, "
              f"Probe A peak AUC {np.nanmax(truth_auc):.3f}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
