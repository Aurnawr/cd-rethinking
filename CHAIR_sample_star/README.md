# CHAIR_sample_star: Sample* leg (APC-masked direct sampling, no CD)

This package is self-contained. Read this file top to bottom and you don't
need to rely on anything else -- except see Step 0 below about the shared
Modal volume with the sibling `CHAIR_sampling+ICD` package.

## What this is

**Sample\*** (also written "Sample-dagger", Sample†) is a baseline from Yin
et al.'s *"Mirage of Performance Gains"*: plain temperature-1.0 sampling
from the expert model's logits, but with the **Adaptive Plausibility
Constraint (APC)** mask applied first (`cutoff = log(beta) + max(E)`,
candidates below cutoff excluded) -- **no amateur branch, no contrastive
term at all**. It isolates how much of contrastive decoding's apparent
effect the plausibility mask alone accounts for, independent of any actual
"amateur model" contrast.

This package repeats the full CHAIR experiment -- Sample* baseline, VCD,
SID, and ICD, all with full per-step top-10 logit capture for the d = E-A
experiment -- exactly like the sibling `CHAIR_sampling+ICD` package, except:

1. **Baseline redefined**: `sample_star` mode applies the APC mask to the
   plain expert distribution before sampling (the sibling package's
   `baseline` mode had NO mask at all -- pure unconstrained sampling).
2. **cd_beta = 0.1 everywhere** (Sample*, VCD, SID, ICD alike) -- matching
   the Mirage paper's own Table 6, which evaluates all four under the same
   beta so the comparison isolates the amateur branch's contribution at a
   fixed mask strength. The sibling package used beta=0.2 to match the
   original greedy-baseline `CHAIR_analysis` convention instead -- a
   different, equally valid choice for a different comparison; this
   package's choice is deliberately different, see `GUIDE.md`.

**Scope**: 1 seed on LLaVA-1.5-7B, 1 seed on Qwen2.5-VL-7B-Instruct (same
seed-budget reasoning as the sibling package). **Smoke-test only until you
have explicit go-ahead to run the real 500-image jobs** -- do not run
`spawn_jobs.py --full` or `qwen/spawn_qwen_jobs.py --full` without that.

## Repository layout

```
CHAIR_sample_star/
  README.md                    this file
  GUIDE.md                     methodology depth: Sample* rationale, schema, budget
  requirements.txt
  image_ids_500.json           same fixed 500 MSCOCO image ids as the sibling package
  modal_app.py                 Modal app -- REUSES the sibling package's volume (see below)
  spawn_jobs.py                LLaVA run plan
  download_local_annotations.sh
  common/
    generate_llava.py          sample_star / VCD / SID generation + top-10 capture
    generate_llava_icd.py      ICD generation (5 disturbance-prompt passes)
    agreement_analysis.py      intervention rate + top-10 overlap, per seed, bootstrap CIs
    contrastive_analysis.py    d = E-A stats, object-mass buckets, CHAIR-S/I, per seed
    chair_bootstrap.py         CHAIR-S/I with image-level bootstrap CIs
    plot_d_histograms.py       d-value histograms (real vs hallucinated)
    plot_d_per_step.py         per-sample d bar plots (matches the paper's actual figure style)
    verify_outputs.py          structural sanity checks -- RUN THIS FIRST after any generation
    object_mentions.py, eval/chair.py, llava/   (vendored, unchanged from the sibling package)
  qwen/
    README.md, GUIDE.md        Qwen-specific instructions (KV-cache design, verification story)
    generate_qwen.py           sample_star / VCD / SID, KV-cached both branches
    generate_qwen_icd.py       ICD generation
    verify_qwen_cache.py, verify_qwen_cache_run.py   cache-correctness check -- already
                                verified in the sibling package (cache math is beta-independent),
                                re-run here as a sanity check on THIS deployment, not a re-derivation
    spawn_qwen_jobs.py          Qwen run plan
```

## Step 0 -- Modal setup (reuses the sibling package's assets)

If you've already run `CHAIR_sampling+ICD`'s setup, **skip straight to Step
1** -- this package's `modal_app.py` points at the exact same Modal Volume
(`cd-rethink-sampling-icd-vol`), which already has both models' weights and
the 500 COCO images. No re-download needed.

If this is a fresh Modal account with neither package set up yet:

```
pip install modal
python3 -m modal setup                        # one-time account link
modal deploy modal_app.py                      # deploy this package's app
modal run modal_app.py --action setup          # downloads both models + COCO assets (~30GB, ~20 min)
```

Either way, deploy this package's own app (distinct from the sibling's):

```
modal deploy modal_app.py
```

## Step 1 -- smoke test (LLaVA)

```
python spawn_jobs.py --smoke
```

Submits 4 tiny jobs (sample_star, VCD, SID, ICD prompt `n2`), 3 images
each. Watch with `modal app logs cd-rethink-sample-star`, look for 4 `DONE`
lines with no tracebacks. Then pull results and verify:

```
modal volume get cd-rethink-sampling-icd-vol outputs/sample_star/captures ./outputs/captures --force
python common/verify_outputs.py --n-images 3
```

`verify_outputs.py` must print `All files passed structural verification.`
before you go any further.

## Step 2 -- Qwen leg

See `qwen/README.md` for the Qwen-specific steps. **Run `verify_qwen_cache_run.py`
there before any Qwen generation** -- it's a quick re-run of the already-proven
cache-correctness check (the cache math itself doesn't depend on cd_beta,
so this should pass immediately, but re-verifying on this exact deployment
before spending GPU time is the same discipline the sibling package used).

## Step 3 -- the real run (DO NOT run without explicit go-ahead)

```
python spawn_jobs.py --full           # LLaVA: 8 jobs, ~15 GPU-hr
python qwen/spawn_qwen_jobs.py --full # Qwen: 8 jobs
```

Both scripts exist and are ready, but per instruction this package is
prepared and smoke-tested only -- the real 500-image jobs are not submitted
until you say so.

## Step 4 -- analysis (once real data exists)

```
bash download_local_annotations.sh
python common/contrastive_analysis.py --model llava --methods sample_star vcd sid icd_p1 icd_p2 icd_n1 icd_n2 icd_p3 --seeds 0
python common/agreement_analysis.py  --model llava --methods vcd sid icd_p1 icd_p2 icd_n1 icd_n2 icd_p3 --seeds 0
python common/plot_d_per_step.py     --model llava --methods vcd sid icd_p1 icd_p2 icd_n1 icd_n2 icd_p3 --seeds 0
```

(swap `--model qwen` for the Qwen leg once its real data exists.)

See `GUIDE.md` for the full methodology, hyperparameter rationale, and
capture-file schema reference.
