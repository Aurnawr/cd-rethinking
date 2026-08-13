# CHAIR_sampling+ICD: direct-sampling baseline, VCD, SID, and ICD on CHAIR

This package is self-contained and independent of any other repo/account.
It runs on **your own Modal subscription**. Follow this file top to bottom
and you don't need to read or rely on anything else.

## What this is

The original CHAIR study (see the sibling `CHAIR_analysis/` work) used
**greedy decoding** as the baseline and compared it against VCD/SID under
greedy decoding too. This package repeats the *entire* experiment with
**direct sampling** as the decoding rule instead — for the plain baseline
AND for VCD, SID, and ICD (Instruction Contrastive Decoding). Everything
else (the 500 images, the CHAIR metric, the top-10 logit capture schema,
`cd_alpha=1.0`, `cd_beta=0.2`, VCD's `noise_step=500`) is held identical to
the original greedy-baseline experiment on purpose, so the ONLY variable
that changes is the decoding rule. See `GUIDE.md` for the full rationale.

**Scope of this run: LLaVA-1.5-7B first.** A Qwen2.5-VL-7B-Instruct leg is
added as a second phase (see the note at the bottom of `GUIDE.md`) — do not
start it until told to; the LLaVA leg alone is enough to validate everything
end to end.

## Repository layout

```
CHAIR_sampling+ICD/
  README.md                    this file — step-by-step instructions
  GUIDE.md                     methodology depth: dataset, formulas, schema, budget
  requirements.txt             exact package versions (only needed for local analysis)
  image_ids_500.json           the fixed 500 MSCOCO val2017 image ids (same set/order as CHAIR_analysis)
  modal_app.py                 the Modal app: asset setup + generation, GPU-side
  spawn_jobs.py                submits the full run plan asynchronously
  download_local_annotations.sh   fetches ONLY the small COCO annotation JSONs, for local analysis
  common/
    generate_llava.py          baseline / VCD / SID generation + top-10 capture
    generate_llava_icd.py      ICD generation (5 disturbance-prompt passes) + top-10 capture
    agreement_analysis.py      intervention rate + top-10 overlap, per seed, with bootstrap CIs
    contrastive_analysis.py    d = E-A stats, object-mass buckets, CHAIR-S/I, per seed
    plot_d_histograms.py       simple matplotlib d-histograms (real vs hallucinated)
    verify_outputs.py          structural sanity checks on capture files — RUN THIS FIRST
    object_mentions.py         maps caption text back to CHAIR object mentions
    eval/chair.py               the CHAIR metric implementation
    llava/                      the LLaVA-1.5 model package (includes the paper-correct SID fix)
  outputs/                      (empty until you generate — gitignored)
```

## Step 0 — Modal account setup (one time)

```
pip install modal
python3 -m modal setup
```

This opens a browser to link your Modal account. If you're on a headless
machine, it prints a URL to open manually. You need your **own** Modal
subscription/credits for this — nothing here depends on anyone else's
account.

## Step 1 — Deploy the app (one time, and after any code edit)

From inside `CHAIR_sampling+ICD/`:

```
modal deploy modal_app.py
```

This registers the app under the name `cd-rethink-sampling-icd` on **your**
Modal account, with its own private Volume `cd-rethink-sampling-icd-vol` —
fully separate from any other project's Modal app or volume.

## Step 2 — Download assets to the Modal volume (one time, ~15-25 min)

```
modal run modal_app.py --action setup
```

Downloads LLaVA-1.5-7B (~13 GB) and the 500 COCO val2017 images +
annotations into the volume. Idempotent — safe to re-run if it's
interrupted, it skips anything already there. Watch progress with:

```
modal app logs cd-rethink-sampling-icd
```

Wait for `[setup] assets ready: 500 images total` before moving on.

## Step 3 — Smoke test BEFORE the real run

**Do not skip this.** It costs a couple of minutes of GPU time and catches
environment/code problems before they cost real GPU-hours.

```
python spawn_jobs.py --smoke
```

This submits 4 tiny jobs (baseline, VCD, SID, and one ICD disturbance
prompt — `n2`), each on just 3 images, 1 seed. Watch them with:

```
modal app logs cd-rethink-sampling-icd
```

Look for 4 `DONE` lines with no tracebacks. Then pull the results down and
verify them:

```
modal volume get cd-rethink-sampling-icd-vol outputs/captures ./outputs/captures --force
bash download_local_annotations.sh
python common/verify_outputs.py --n-images 3
```

`verify_outputs.py` checks line counts, no duplicate/out-of-set image ids,
required schema fields present, no empty captions, and that the contrastive
formula (`cd = (1+alpha)E - alpha*A`) is internally consistent. If it
prints `All files passed structural verification.`, you're clear to run the
real thing. If anything fails, stop and investigate (or ask) before
spending real GPU-hours — do not proceed past a failed verification.

## Step 4 — The real run

```
python spawn_jobs.py --full
```

This submits **8 jobs**, all `--decode sample`, 500 images, seed 0:
- 1 baseline job (~1 GPU-hr)
- VCD (~2 GPU-hr)
- SID (~2 GPU-hr)
- ICD × 5 disturbance prompts (~2 GPU-hr each = ~10 GPU-hr)

**Total: ~15 GPU-hr on one L4**, well inside a 35-40 hour budget (the rest
is reserved for the Qwen leg, added later — see `GUIDE.md`).

Jobs run on Modal's infrastructure independently of your laptop — you can
close the terminal right after submission. Progress is checkpointed to the
volume every ~60s, so a killed/restarted job resumes from wherever it left
off (already-finished images are skipped automatically).

Check progress any time with:

```
modal app logs cd-rethink-sampling-icd
```

`full_call_ids.json` (written by `spawn_jobs.py`) records each job's call
ID if you need to check on or cancel a specific one:

```python
import modal
call = modal.FunctionCall.from_id("<call_id>")
call.get(timeout=0)   # raises if still running, returns result if done
# call.cancel()       # only if you need to stop it
```

## Step 5 — Pull results down and verify again

```
modal volume get cd-rethink-sampling-icd-vol outputs/captures ./outputs/captures --force
python common/verify_outputs.py --n-images 500
```

Every file should report `[OK]` with `n=500`. Fix or re-run anything that
doesn't before trusting the analysis below.

## Step 6 — Run the analysis

Needs the local COCO annotations (small, ~50 MB — NOT the images or model
weights, those stay on the Modal volume):

```
bash download_local_annotations.sh
pip install -r requirements.txt      # only transformers/numpy/matplotlib needed locally, no GPU
python common/contrastive_analysis.py --methods baseline vcd sid icd_p1 icd_p2 icd_n1 icd_n2 icd_p3 --seeds 0
python common/agreement_analysis.py  --methods vcd sid icd_p1 icd_p2 icd_n1 icd_n2 icd_p3 --seeds 0
python common/plot_d_histograms.py   --methods vcd sid icd_p1 icd_p2 icd_n1 icd_n2 icd_p3 --seeds 0
```

`contrastive_analysis.py` prints the CHAIR-S/CHAIR-I table (including the
new sampling baseline, replacing the old greedy-baseline row) plus d = E-A
stats and object-mass buckets for every two-branch method. `agreement_analysis.py`
prints intervention rates and top-10 overlap tables. `plot_d_histograms.py`
writes simple matplotlib histograms to `figures/`.

## If something goes wrong

- **A job errors out mid-run**: check `modal app logs cd-rethink-sampling-icd`
  for the traceback. Fix the code, `modal deploy modal_app.py` again, then
  re-spawn just that one job (the partial output file is preserved — it
  will resume, not restart from image 0).
- **Not sure if a run finished**: `verify_outputs.py --n-images 500` will
  tell you exactly which files are short and by how much.
- **Unsure how much GPU time you've used**: check the Usage page on your
  Modal dashboard (modal.com), not this repo.

See `GUIDE.md` for the full methodology, exact hyperparameter rationale,
and the capture-file schema reference.
