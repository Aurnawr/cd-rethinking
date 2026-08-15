#!/usr/bin/env python3
"""Whole-pipeline integration test on a tiny random model and a fake POPE set - CPU only.

Runs the real `run_all_lightning.sh` stage sequence -- extract_cd -> train_probes -> resid_stats
-> logit_lens_figures -> analyze_all -- against a 4-layer randomly-initialised Qwen2.5-VL saved to
disk and a synthetic POPE-COCO directory. It is the only test that exercises extract_cd.py: the
shard contract, the per-method margin npz, per-split checkpointing and `--resume`.

The model is random, so the *numbers* are meaningless; what is being tested is that every stage
runs, agrees on shapes, and produces all 12 regen figures. Its head is biased onto the Yes/No
token rows so the greedy prediction is actually a Yes or a No and the H / T populations are
non-empty -- otherwise every downstream stage would have nothing to analyse.

    python mech_interp_qwen/test_pipeline_tiny.py     # ~2-3 minutes on CPU
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from qwen_runtime import lm_head_weight, yes_no_token_ids  # noqa: E402
from test_model_tiny import PROCESSOR_ID, tiny_model  # noqa: E402

SPLITS = ("random", "popular", "adversarial")
N_PER_SPLIT = 40
METHODS = ("vcd", "icd", "sid")
OBJECTS = ["bicycle", "person", "dog", "car", "bottle", "chair", "cat", "boat"]


def bias_head_to_yes_no(model, yes_ids, no_ids):
    """Make the random head answer Yes/No, and make which one depend on the input.

    Without this, argmax over 152k random logits never lands on a Yes/No token, `pred` is 0 for
    every sample, the hallucination set H is empty and every figure is undefined.
    """
    with torch.no_grad():
        W = lm_head_weight(model)
        g = torch.Generator().manual_seed(7)
        for ids in (yes_ids, no_ids):
            for i in ids:
                W[i] = torch.randn(W.shape[1], generator=g) * 6.0


def make_fake_pope(data_dir, n_images=12):
    """A POPE-COCO directory in exactly the layout download_pope_data.py produces."""
    pope_dir = os.path.join(data_dir, "pope", "coco")
    images_dir = os.path.join(pope_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    rng = np.random.default_rng(1)

    names = []
    for i in range(n_images):
        name = f"COCO_val2014_{i:012d}.jpg"
        Image.fromarray(rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)).save(
            os.path.join(images_dir, name), quality=90
        )
        names.append(name)

    for split in SPLITS:
        with open(os.path.join(pope_dir, f"coco_pope_{split}.json"), "w") as f:
            for q in range(N_PER_SPLIT):
                f.write(json.dumps({
                    "question_id": q,
                    "image": names[q % n_images],
                    "text": f"Is there a {OBJECTS[q % len(OBJECTS)]} in the image?",
                    "label": "yes" if q % 2 == 0 else "no",
                }) + "\n")
    return pope_dir


def run(cmd, env, label):
    print(f"\n--- {label} ---", flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    tail = "\n".join(proc.stdout.strip().splitlines()[-6:])
    if proc.returncode != 0:
        print(proc.stdout[-6000:])
        print(proc.stderr[-6000:], file=sys.stderr)
        raise SystemExit(f"FAILED: {label}")
    print(tail)
    return proc.stdout


def main():
    from transformers import AutoProcessor

    tmp = tempfile.mkdtemp(prefix="qwen_mech_pipeline_")
    model_dir = os.path.join(tmp, "model")
    data_dir = os.path.join(tmp, "data")
    act_dir = os.path.join(tmp, "activations")
    results_dir = os.path.join(tmp, "results")
    regen_dir = os.path.join(results_dir, "regen")
    for d in (act_dir, results_dir, regen_dir):
        os.makedirs(d, exist_ok=True)

    try:
        print("building tiny model + processor ...")
        processor = AutoProcessor.from_pretrained(
            PROCESSOR_ID, min_pixels=4 * 28 * 28, max_pixels=16 * 28 * 28
        )
        model = tiny_model()
        yes_ids, no_ids = yes_no_token_ids(processor.tokenizer)
        bias_head_to_yes_no(model, yes_ids, no_ids)
        model.save_pretrained(model_dir, safe_serialization=True)
        processor.save_pretrained(model_dir)
        make_fake_pope(data_dir)
        print(f"  model -> {model_dir}\n  data  -> {data_dir}")

        env = dict(os.environ, MPLBACKEND="Agg", TOKENIZERS_PARALLELISM="false")
        py = sys.executable
        common = ["--data-dir", data_dir, "--act-dir", act_dir, "--results-dir", results_dir,
                  "--model-path", model_dir, "--device", "cpu", "--dtype", "float32",
                  "--attn-impl", "eager", "--shard-size", "25"]

        run([py, os.path.join(HERE, "selfcheck.py"), "--model-path", model_dir,
             "--data-dir", data_dir, "--device", "cpu", "--dtype", "float32", "--n", "6"],
            env, "selfcheck")

        out = run([py, os.path.join(HERE, "extract_cd.py"), *common], env, "extract_cd")
        if "[checkpoint]" not in out:
            raise SystemExit("extract_cd never checkpointed")

        # --resume must skip every completed split rather than redo it
        out = run([py, os.path.join(HERE, "extract_cd.py"), *common, "--resume"], env,
                  "extract_cd --resume")
        for split in SPLITS:
            if f"[skip] split '{split}' already complete" not in out:
                raise SystemExit(f"--resume did not skip '{split}'")

        for m in METHODS:
            run([py, os.path.join(HERE, "resid_stats.py"), "--act-dir", act_dir,
                 "--results-dir", results_dir, "--method", m], env, f"resid_stats {m}")

        run([py, os.path.join(HERE, "train_probes.py"), "--act-dir", act_dir,
             "--results-dir", results_dir, "--method", "vcd", "--min-probe-b", "10"],
            env, "train_probes")

        for m in METHODS:
            run([py, os.path.join(HERE, "logit_lens_figures.py"),
                 "--npz", os.path.join(results_dir, f"logit_lens_per_sample_{m}.npz"),
                 "--method", m,
                 "--probe-csv", os.path.join(results_dir, "per_layer_normalized.csv"),
                 "--resid-csv", os.path.join(results_dir, f"per_layer_resid_{m}.csv"),
                 "--out-dir", regen_dir], env, f"figures {m}")

        run([py, os.path.join(HERE, "analyze_all.py"), "--results-dir", results_dir],
            env, "analyze_all")

        # ---- assertions ----
        failures = []
        n_expected = N_PER_SPLIT * len(SPLITS)
        n_layers = model.config.num_hidden_layers + 1

        for m in METHODS:
            z = np.load(os.path.join(results_dir, f"logit_lens_per_sample_{m}.npz"),
                        allow_pickle=True)
            if z["clean_margin"].shape != (n_expected, n_layers):
                failures.append(f"{m}: clean_margin {z['clean_margin'].shape}, "
                                f"expected {(n_expected, n_layers)}")
            if np.allclose(z["clean_margin"], z["amateur_margin"]):
                failures.append(f"{m}: amateur margins identical to expert - amateur is a no-op")
            with open(os.path.join(results_dir, f"extract_summary_{m}.json")) as f:
                summ = json.load(f)
            if summ["lens_final_layer_MAE"] > 1e-3:
                failures.append(f"{m}: lens MAE {summ['lens_final_layer_MAE']:.2e} too large")
            if list(summ["splits_done"]) != list(SPLITS):
                failures.append(f"{m}: splits_done {summ['splits_done']}")

        # the three methods must share the expert pass exactly, and differ from each other
        margins = {m: np.load(os.path.join(results_dir, f"logit_lens_per_sample_{m}.npz"))
                   for m in METHODS}
        if not all(np.array_equal(margins["vcd"]["clean_margin"], margins[m]["clean_margin"])
                   for m in METHODS):
            failures.append("expert margins differ across methods - the shared pass is broken")
        for a, b in (("vcd", "icd"), ("vcd", "sid"), ("icd", "sid")):
            if np.allclose(margins[a]["amateur_margin"], margins[b]["amateur_margin"]):
                failures.append(f"{a} and {b} amateurs are identical")

        shards = [f for f in os.listdir(act_dir) if f.endswith(".npz")]
        if len(shards) != len(SPLITS) * (N_PER_SPLIT // 25 + 1):
            failures.append(f"{len(shards)} shards written, expected "
                            f"{len(SPLITS) * (N_PER_SPLIT // 25 + 1)}")
        z = np.load(os.path.join(act_dir, sorted(shards)[0]))
        for key in ("clean_hidden", "clean_norm", "gt", "pred", *[f"cd_change_{m}" for m in METHODS]):
            if key not in z.files:
                failures.append(f"shard missing '{key}'")

        pngs = sorted(f for f in os.listdir(regen_dir) if f.endswith(".png"))
        if len(pngs) != 12:
            failures.append(f"{len(pngs)} regen figures, expected 12: {pngs}")
        for path in ("per_layer.csv", "per_layer_normalized.csv", "summary.json",
                     "cross_method_battery.png", "summary_all.csv", "layerwise_curves.png"):
            p = os.path.join(results_dir, path)
            if not (os.path.exists(p) and os.path.getsize(p) > 0):
                failures.append(f"missing or empty: {path}")

        print()
        if failures:
            print("FAILED:")
            for f in failures:
                print("  -", f)
            return 1
        print(f"PASSED: full pipeline, {n_expected} samples x {len(METHODS)} methods, "
              f"{n_layers} layers, 12 regen figures, resume verified")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
