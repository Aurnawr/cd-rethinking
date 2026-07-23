# RUN — how to re-run everything

This folder is **self-contained**: it holds the model weights, COCO images +
annotations, code dependencies (`llava/`, `eval/`, `inference/`), all scripts,
and all results. Nothing outside this folder is needed except the Python env
(the `cloudspace` conda env with torch + transformers already installed).

## Layout

```
CHAIR-llava-detailed/
├── llava/  eval/  inference/        # code dependencies (copied from repo)
├── data/coco/{val2017, annotations} # the 500 COCO images + annotations
├── outputs/chair/llava-7b-greedy/captions.jsonl   # image-id ordering reference
└── llava_logit_eda/                 # ALL experiment content lives here
    ├── models/llava-v1.5-7b/        # 13 GB weights
    ├── *.py                         # scripts (paths already point at this folder)
    ├── *.md                         # reports / findings
    ├── outputs_500_clean/           # 500-img greedy/VCD/SID + raw_object_candidates
    ├── outputs_full_topk/           # 500-img expert+amateur+CD top-30 per step
    ├── outputs_proxy_AB_500/        # Design A/B proxy captions (500 img)
    ├── outputs_proxy_ABCDEF/        # Designs A–F (70 img)
    ├── ROC analysis/  amateur_bucket_analysis/  figures/  figures_modified/
    └── proxy_stats.json  image_ids.json
```

The scripts' `_REPO_ROOT` has been repointed to this folder, so this folder now
plays the role the original `cd-rethinking` repo did.

## To re-run (always `cd` into the scripts dir first)

```bash
conda activate cloudspace   # or the env with torch+transformers
cd /teamspace/studios/this_studio/cd-rethinking/llava_logit_eda/CHAIR-llava-detailed/llava_logit_eda
```

**GPU experiments** (need a GPU; ~4 hr each for 500 images):
```bash
python run_full_topk_capture.py            # expert+amateur+CD top-30 capture (→ outputs_full_topk/)
python run_500_experiment.py               # greedy/VCD/SID + object-candidate d log (→ outputs_500/)
python run_proxy_AB_500.py --n-images 500  # Gaussian(A) + bootstrap(B) proxies (→ outputs_proxy_AB_500/)
python run_proxy_designs_ABCDEF.py --n-images 70   # Designs A–F (→ outputs_proxy_ABCDEF/)
```

**CPU-only analyses** (no GPU; seconds–minutes, run on existing data):
```bash
python compute_proxy_stats.py    # rebuild proxy_stats.json from raw_object_candidates
# ROC, amateur-bucket, entropy, overlap analyses were run as inline scripts —
# their outputs are already in ROC analysis/ and amateur_bucket_analysis/.
```

## Notes / caveats
- All runs are **resumable**: caption files are appended and completed
  (image_id, variant) pairs are skipped on restart.
- `run_500_experiment.py` writes to `outputs_500/`; the analyzed, de-duplicated
  version is `outputs_500_clean/` (one duplicate image removed — see
  `CLAIM_VALIDATION_500.md`).
- Decoding is **greedy** for all logit-level captures (deterministic/reproducible).
- After a run finishes, leave the GPU idle so the studio can sleep.

See `README.md` for what each report/result means and the headline numbers.
