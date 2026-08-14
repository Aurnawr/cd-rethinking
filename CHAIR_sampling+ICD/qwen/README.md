# Qwen2.5-VL-7B-Instruct leg: step-by-step instructions

This folder is the Qwen counterpart of the LLaVA leg one level up
(`CHAIR_sampling+ICD/`). Read `../README.md` first if you haven't run the
LLaVA leg yet -- it explains the overall experiment and does the one-time
Modal account setup. This file only covers what's Qwen-specific.

**Do not skip the cache-correctness verification (Step 2 below).** Qwen's
KV-cache implementation here is nontrivial (see `GUIDE.md` for why) and was
specifically built and verified to be correct -- but "verified on the
machine that built it" isn't the same as "verified on your Modal account,
your GPU allocation, your library versions." Re-run it yourself before
spending real GPU-hours.

## Step 0 -- prerequisites

You should already have, from the LLaVA leg:
- Modal account set up (`modal setup`)
- The app deployed (`modal deploy ../modal_app.py`, run from `qwen/` so the
  relative path resolves, or `modal deploy modal_app.py` from the parent dir)

If you haven't downloaded the Qwen weights yet:

```
modal run ../modal_app.py --action setup
```

This is idempotent -- if you already ran it for LLaVA, it skips those
weights and only fetches the ~16 GB Qwen2.5-VL-7B-Instruct weights (COCO
images/annotations are shared with the LLaVA leg, already there).

## Step 1 -- smoke test (cheap, do this first)

```
python spawn_qwen_jobs.py --smoke
```

Submits 4 tiny jobs (baseline, VCD, SID, ICD prompt `n2`), 3 images each.
Watch with `modal app logs cd-rethink-sampling-icd`, look for 4 `DONE`
lines with no tracebacks.

## Step 2 -- verify the KV-cache implementation (REQUIRED before --full)

```
python verify_qwen_cache_run.py
```

This blocks until done (a few minutes) and prints PASS or FAILED. It
compares `generate_qwen.py`'s cached two-branch loop against a brute-force,
cache-less recomputation on a few images/steps, for both VCD and SID.

**If it prints PASS**, you're clear to proceed to Step 3.

**If it prints FAILED**, stop. Do not run `--full`. The failure output
(saved to `outputs/qwen_cache_verification.json` on the volume) will show
per-step logit differences and argmax mismatches -- capture that and get
help before spending GPU-hours on a possibly-incorrect run. See
`GUIDE.md`'s "What PASS actually means" section for how to read borderline
results (a very small mismatch rate at very late steps is expected bf16
behavior, not necessarily a bug -- the script already accounts for this,
but the reasoning is worth understanding before you override anything).

## Step 3 -- the real run

```
python spawn_qwen_jobs.py --full
```

Submits **8 jobs**, all `--decode sample`, 500 images, seed 0:
baseline, VCD, SID, and ICD x 5 disturbance prompts -- mirroring the LLaVA
leg exactly. Same checkpointing, same resumability, same async
fire-and-forget submission as the LLaVA leg.

Check progress: `modal app logs cd-rethink-sampling-icd`

## Step 4 -- pull results and verify

From the `qwen/` directory:

```
cd ..
modal volume get cd-rethink-sampling-icd-vol outputs/captures/qwen_baseline_seed0.jsonl outputs/captures/qwen_baseline_seed0.jsonl --force
# ...repeat per file, or use a loop -- see ../README.md Step 5 for the pattern, just with qwen_ filenames
python common/verify_outputs.py --n-images 500 --glob "qwen_*.jsonl"
```

## Step 5 -- analysis

From the `CHAIR_sampling+ICD/` root (the shared analysis scripts take
`--model llava` or `--model qwen`):

```
bash download_local_annotations.sh   # if not already done for the LLaVA leg
python common/contrastive_analysis.py --model qwen --methods baseline vcd sid icd_p1 icd_p2 icd_n1 icd_n2 icd_p3 --seeds 0
python common/agreement_analysis.py  --model qwen --methods vcd sid icd_p1 icd_p2 icd_n1 icd_n2 icd_p3 --seeds 0
python common/plot_d_histograms.py   --model qwen --methods vcd sid icd_p1 icd_p2 icd_n1 icd_n2 icd_p3 --seeds 0
```

## Files in this folder

```
qwen/
  README.md                  this file
  GUIDE.md                   why the KV-cache design works, the verification story, hyperparameters
  generate_qwen.py            baseline / VCD / SID generation, KV-cached both branches
  generate_qwen_icd.py        ICD generation (5 disturbance-prompt passes)
  verify_qwen_cache.py        the correctness check itself (also runnable standalone on Modal)
  verify_qwen_cache_run.py    convenience wrapper: submit + block + print PASS/FAIL
  spawn_qwen_jobs.py           submits the full run plan (mirrors ../spawn_jobs.py)
```

`modal_app.py` stays at the `CHAIR_sampling+ICD/` root -- one shared Modal
app/volume for both the LLaVA and Qwen legs, so there's only one thing to
deploy and one volume to manage.
