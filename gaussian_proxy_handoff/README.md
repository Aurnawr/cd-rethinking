# Gaussian noise-proxy study — multi-seed CHAIR runs (handoff)

This folder is a self-contained package to reproduce, **with multiple random
seeds**, the Gaussian noise-proxy experiment for the "why contrastive decoding
fails" paper. It runs on 500 MSCOCO images for two models (LLaVA-1.5-7B and
Qwen2.5-VL-7B) and reports CHAIR-S / CHAIR-I as **mean ± std across seeds**.

## Why this run is needed

The key result is that the "amateur" branch of contrastive decoding (VCD/SID)
can be **replaced by matched random Gaussian noise** added to the expert logits,
with no loss of CHAIR performance — i.e. the amateur branch is not doing anything
a generic perturbation could not. That result was previously a **single unseeded
run**. VCD's noise draw, SID's kept-token subset, and the proxy's Gaussian noise
are all stochastic, so a reviewer can dismiss a one-shot number. This package
runs **N seeds** of `vcd`, `sid`, and `proxy` (plus deterministic `greedy`) so
each number gets an error bar.

Two things to look for in the results:
1. Does `proxy` stay within a seed or two of `vcd`/`sid`/`greedy` CHAIR? (Expected: yes.)
2. Qwen's proxy previously returned exactly greedy's CHAIR-S (30.8 vs 30.8);
   seeds will show whether that is coincidence or a real "noise rarely flips a
   Qwen argmax" effect. (CHAIR-I already differed, so it is not a pure no-op.)

## What is different from the original proxy

- The proxy here adds Gaussian noise to **every vocabulary logit** (full
  vocabulary), not only to object-category tokens. This is the stronger,
  cleaner control.
- Every method takes a `--seed`; `run_all.sh` sweeps several seeds.

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

## If you cloned this from GitHub: get the large assets first

The model weights and the 500 COCO images are **not** in git (too large for
GitHub). Download them into the expected paths with:
```
bash download_assets.sh
```
This fetches `liuhaotian/llava-v1.5-7b`, `Qwen/Qwen2.5-VL-7B-Instruct`, the COCO
val2017 annotations, and the 500 images from `image_ids_500.json`. Idempotent.
(If you received this folder directly with the weights already inside, skip this.)

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

- **LLaVA** runs two cached forward passes per step; ~fast.
- **Qwen `vcd`/`sid`** recompute the amateur branch **cache-less** every step and
  are **slow** (potentially hours per seed for 500 images). This is required for
  numerical correctness (a hand-rolled mRoPE second-branch cache diverged). If
  GPU time is tight, start with `SEEDS=(0 1 2)` and consider fewer seeds for Qwen
  `vcd`/`sid`, or run LLaVA first.
- Runs are **resumable**: finished images are skipped, and any run whose
  `_chair.json` already exists is skipped, so you can stop and restart.

## Notes

- `greedy` is deterministic; it is run once (seed 0) and its std should be ~0.
- The proxy noise magnitude is taken from `proxy_stats_{model}.json`
  (`pooled_mean`, `pooled_std` of the measured contrastive adjustment
  `d = E - A`), matched to VCD by default (`PROXY_STATS=vcd` in `run_all.sh`;
  set to `sid` to match SID instead).
- Everything uses greedy decoding, `max_new_tokens=256`, alpha=1, beta=0.2,
  prompt "Describe this image in detail." — identical to the paper.
