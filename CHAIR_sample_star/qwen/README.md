# Qwen2.5-VL-7B-Instruct leg: step-by-step instructions

This folder is the Qwen counterpart of the LLaVA leg one level up
(`CHAIR_sample_star/`). Read `../README.md` first -- it explains Sample*,
the beta=0.1 convention, and the shared-volume Modal setup. This file only
covers what's Qwen-specific.

**Do not skip the cache-correctness verification (Step 2 below).** Qwen's
KV-cache implementation here is the exact same code, already fp32-verified
correct, from the sibling `CHAIR_sampling+ICD` package -- the cache math is
entirely beta-independent (a position/decoding-mechanics question, not a
hyperparameter one), so this should pass immediately. Re-run it anyway:
"verified elsewhere" isn't the same as "verified on this deployment, your
Modal account, your GPU allocation."

## Step 0 -- prerequisites

You should already have, from the LLaVA leg / the sibling package:
- Modal account set up (`modal setup`)
- The app deployed (`modal deploy ../modal_app.py`, run from `qwen/` so the
  relative path resolves, or `modal deploy modal_app.py` from the parent dir)
- Model weights on the shared volume (`cd-rethink-sampling-icd-vol`) --
  already there if you've run either the sibling package's setup or this
  package's `modal run ../modal_app.py --action setup`.

## Step 1 -- smoke test (cheap, do this first)

```
python spawn_qwen_jobs.py --smoke
```

Submits 4 tiny jobs (sample_star, VCD, SID, ICD prompt `n2`), 3 images each.
Watch with `modal app logs cd-rethink-sample-star`, look for 4 `DONE`
lines with no tracebacks.

## Step 2 -- verify the KV-cache implementation (REQUIRED before --full)

```
python verify_qwen_cache_run.py
```

This blocks until done (a few minutes) and prints PASS or FAILED. It
compares `generate_qwen.py`'s cached two-branch loop against a brute-force,
cache-less recomputation on a few images/steps, for both VCD and SID.

**If it prints PASS**, you're clear to proceed to Step 3.

**If it prints FAILED**, stop. Do not run `--full`. This would be
surprising given the identical mechanism already passed in the sibling
package (the cache correctness doesn't depend on cd_beta at all) -- a
failure here would suggest something about THIS deployment/environment is
different, not a resurfacing of the original bug. Capture the failure
output (saved to `outputs/sample_star/qwen_cache_verification.json` on the
volume) and investigate before spending any GPU-hours. See the sibling
package's `qwen/GUIDE.md` for the full story of what was checked and how
(fp32 control run, etc.) if you need the background.

## Step 3 -- the real run (DO NOT run without explicit go-ahead)

```
python spawn_qwen_jobs.py --full
```

Submits **8 jobs**, all `--decode sample`, cd_beta=0.1, 500 images, seed 0:
sample_star, VCD, SID, and ICD x 5 disturbance prompts. Ready to go, but
per instruction this package is prepared and smoke-tested only until you
give explicit permission to run the real jobs.

Check progress: `modal app logs cd-rethink-sample-star`

## Step 4 -- pull results and verify

From the `qwen/` directory:

```
cd ..
modal volume get cd-rethink-sampling-icd-vol outputs/sample_star/captures/qwen_sample_star_seed0.jsonl outputs/captures/qwen_sample_star_seed0.jsonl --force
# ...repeat per file, or use a loop -- see ../README.md Step 1 for the pattern, just with qwen_ filenames
python common/verify_outputs.py --n-images 500 --glob "qwen_*.jsonl"
```

## Step 5 -- analysis

From the `CHAIR_sample_star/` root (the shared analysis scripts take
`--model llava` or `--model qwen`):

```
bash download_local_annotations.sh   # if not already done for the LLaVA leg
python common/contrastive_analysis.py --model qwen --methods sample_star vcd sid icd_p1 icd_p2 icd_n1 icd_n2 icd_p3 --seeds 0
python common/agreement_analysis.py  --model qwen --methods vcd sid icd_p1 icd_p2 icd_n1 icd_n2 icd_p3 --seeds 0
python common/plot_d_per_step.py     --model qwen --methods vcd sid icd_p1 icd_p2 icd_n1 icd_n2 icd_p3 --seeds 0
```

## Files in this folder

```
qwen/
  README.md                  this file
  GUIDE.md                    why the KV-cache design works, what's reused vs. new here
  generate_qwen.py            sample_star / VCD / SID generation, KV-cached both branches
  generate_qwen_icd.py        ICD generation (5 disturbance-prompt passes)
  verify_qwen_cache.py        the correctness check itself (also runnable standalone on Modal)
  verify_qwen_cache_run.py    convenience wrapper: submit + block + print PASS/FAIL
  spawn_qwen_jobs.py           submits the full run plan (mirrors ../spawn_jobs.py)
```

`modal_app.py` stays at the `CHAIR_sample_star/` root -- one shared Modal
app for both the LLaVA and Qwen legs of THIS package, pointed at the same
Modal Volume the sibling package uses (see `../README.md` Step 0).
