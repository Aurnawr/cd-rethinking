# Gaussian noise-proxy study — multi-seed CHAIR runs (handoff)

This folder is a self-contained package that runs the **full-vocabulary Gaussian
noise-proxy** experiment across **3 random seeds** on 500 MSCOCO images for two
models (LLaVA-1.5-7B and Qwen2.5-VL-7B), and reports CHAIR-S / CHAIR-I per seed.
It is sized for a **~7 GPU-hour budget on one L4**.

## What this run produces (and what it does NOT)

**Runs here (the new result):** the `proxy` at 3 seeds per model, plus a cheap
deterministic `greedy` baseline. Estimated cost: ~4.5-6.5 h total on one L4.

**Deliberately NOT run here:** `vcd` and `sid`. On Qwen the amateur branch is
computed cache-less (hours per seed), so seeding them would need ~1-2 days of
GPU. Their single-run CHAIR numbers and image-level bootstrap CIs already exist
in the paper and are the reference the proxy is compared against.

## Why this run is needed

The key result is that the "amateur" branch of contrastive decoding (VCD/SID)
can be **replaced by matched random Gaussian noise** added to the expert logits,
with no loss of CHAIR performance — i.e. the amateur branch is not doing anything
a generic perturbation could not. That was previously a **single unseeded run**,
and the proxy is inherently random, so a reviewer can dismiss a one-shot number.
Running 3 seeds shows the result **reproduces across independent noise draws**.

Two things to look for in the results:
1. Does `proxy` land at the greedy / paper-VCD / paper-SID CHAIR level, and stay
   there across the 3 seeds? (Expected: yes, tight spread.)
2. Qwen's proxy previously returned exactly greedy's CHAIR-S (30.8 vs 30.8);
   the seeds show whether that is coincidence or a real "noise rarely flips a
   Qwen argmax" effect. (CHAIR-I already differed, so it is not a pure no-op.)

## How to report it (avoid the n=3 std trap)

With only 3 seeds, report the **individual per-seed values or their range**, not
a standard deviation (a 3-point std is unreliable). Keep the paper's image-level
bootstrap CIs as the primary, uniform uncertainty across all methods; present
these seeds as a supplementary reproducibility check on the proxy.

## What is different from the original proxy

- The proxy here adds Gaussian noise to **every vocabulary logit** (full
  vocabulary), not only to object-category tokens. This is the stronger,
  cleaner control.
- Every method takes a `--seed`; `run_all.sh` sweeps 3 seeds for the proxy.

## Contents

```
run_all.sh                 one command runs smoke test + full runs + aggregation
requirements.txt           exact package versions
image_ids_500.json         the fixed 500 MSCOCO val2017 image ids (same order for all)
proxy_stats_llava.json     measured mean/std of d = E - A (LLaVA), matched by the proxy
proxy_stats_qwen.json      same for Qwen
models/                    llava-v1.5-7b/ and Qwen2.5-VL-7B-Instruct/ (full weights)
data/coco/val2017/         the 500 images
data/coco/annotations/     instances_val2017.json, captions_val2017.json (CHAIR ground truth)
scripts/gen_llava.py       generate captions: greedy | vcd | sid | proxy   (LLaVA)
scripts/gen_qwen.py        generate captions: greedy | vcd | sid | proxy   (Qwen)
scripts/eval/chair_eval.py score a caption file -> CHAIR-S / CHAIR-I
scripts/eval/chair.py      CHAIR implementation
scripts/aggregate_seeds.py mean +/- std across seeds -> outputs/aggregate.json
scripts/llava/             vendored LLaVA package (needed by gen_llava.py)
outputs/                   created at runtime (caps/, aggregate.json)
```

## Runtime target

Built and intended for a **Lightning AI studio on a single NVIDIA L4 (24 GB)** —
the same environment these scripts were developed in. On such a studio the
default Python env already has every dependency (torch 2.8, transformers 5.12.1,
qwen-vl-utils, and the vendored LLaVA), so **no installation is usually needed**;
just run from a terminal in the studio. Both 7B models fit in fp16 on one L4
(LLaVA ~14 GB, Qwen ~16 GB, run one at a time).

## Assets (weights + images) are auto-downloaded

The model weights and the 500 COCO images are **not** in git (too large for
GitHub). You do not need to fetch them manually: **`run_all.sh` checks for them
and runs `download_assets.sh` automatically if they are missing.** So a single
`bash run_all.sh` handles everything, whether you cloned from GitHub or received
the folder with weights already inside.

If you prefer to download them ahead of time (e.g. overnight), run:
```
bash download_assets.sh
```
This fetches `liuhaotian/llava-v1.5-7b` (~13 GB), `Qwen/Qwen2.5-VL-7B-Instruct`
(~16 GB), the COCO val2017 annotations, and the 500 images from
`image_ids_500.json`. Idempotent. First-time download is ~29 GB, so expect
some minutes before the smoke test starts on a fresh clone.

## How to run

1. (Only if not on the original Lightning studio.) Install deps into a
   CUDA-capable env:
   ```
   pip install -r requirements.txt
   ```
   (Keep `transformers==5.12.1`; the vendored LLaVA package is patched for it.)

2. Sanity check on 2 images (fast, catches env/path problems):
   ```
   bash run_all.sh --smoke
   ```

3. Full run (edit `SEEDS`, `MODELS` at the top of `run_all.sh` to control cost):
   ```
   bash run_all.sh
   ```
   If your GPU env uses a specific python, pass it:
   `PY=/path/to/python bash run_all.sh`

4. Read `outputs/aggregate.json` (or the printed table) for CHAIR mean ± std.

## Cost note (please read before launching)

- The `proxy` is a single expert forward per token (KV-cached) plus an
  element-wise noise add -- i.e. **greedy speed**. This is why 3 seeds on both
  models is affordable (~4.5-6.5 h on one L4).
- Rough per-seed estimates (confirm with the smoke test): LLaVA proxy
  ~0.5-0.75 h; Qwen proxy ~1-1.5 h. Greedy baseline ~0.5 h (LLaVA) / ~1 h (Qwen),
  run once.
- If time is tight, set `SEEDS=(0 1)` (2 seeds) or `RUN_GREEDY=0`.
- `vcd`/`sid` are not run here by design (Qwen's cache-less amateur makes them
  cost ~1-2 days across seeds); use the paper's existing single-run numbers +
  bootstrap CIs for those.
- Runs are **resumable**: finished images are skipped, and any run whose
  `_chair.json` already exists is skipped, so you can stop and restart.
- **Run `bash run_all.sh --smoke` first**: it prints per-image time on 2 images;
  multiply by 500 to confirm the true per-seed cost before spending GPU hours.

## Notes

- `greedy` is deterministic; it is run once (seed 0) as the baseline.
- The proxy noise magnitude is taken from `proxy_stats_{model}.json`
  (`pooled_mean`, `pooled_std` of the measured contrastive adjustment
  `d = E - A`), matched to VCD by default (`PROXY_STATS=vcd` in `run_all.sh`;
  set to `sid` to match SID instead).
- Everything uses greedy decoding, `max_new_tokens=256`, alpha=1, beta=0.2,
  prompt "Describe this image in detail." — identical to the paper.
