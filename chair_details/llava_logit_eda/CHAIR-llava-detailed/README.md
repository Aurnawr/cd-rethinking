# CHAIR-llava-detailed

Self-contained logit-level reproducibility/critique study of contrastive
decoding (VCD / SID) on **LLaVA-1.5-7B**, open-ended COCO captioning, evaluated
with the **CHAIR** protocol. Everything needed to re-run and every result lives
in this one folder (weights, images, code deps, scripts, outputs, figures).

- **How to re-run:** see `RUN.md`.
- Ground truth for "real vs. hallucinated" object = standard CHAIR (COCO
  instance-segmentation categories ∪ objects in the 5 reference captions,
  mapped through the CHAIR synonym dict).
- All logit-level captures use **greedy** decoding (deterministic).

---

## Where things are

Top level = the runtime dependencies (so the folder is portable):
`llava/  eval/  inference/` (code), `data/coco/` (500 images + annotations),
`outputs/chair/.../captions.jsonl` (image-id ordering).

All experiment content is under **`llava_logit_eda/`**:

### Reports (start here) — `llava_logit_eda/*.md`
| file | what it covers |
|---|---|
| `NOVEL_FINDINGS.md` | 10-image pilot: mechanism, Claim 0 (C = E + d), APC collapse, overlap |
| `CLAIM_VALIDATION_500.md` | 500-image validation: d real-vs-hall, APC collapse, whack-a-mole |
| `PROXY_DESIGN_A…F_*.md` | six spurious-proxy designs (noise / bootstrap / oracle / fixed / vocab / one-sided) |
| `TOP10_CANDIDATE_TABLES.md` | per-step top-10 candidate logit tables (pilot) |

### Scripts — `llava_logit_eda/*.py` (paths already repointed to this folder)
`object_mentions.py` (shared dep) · `run_eda.py` + `generate_report.py` (pilot) ·
`run_500_experiment.py` · `run_full_topk_capture.py` · `compute_proxy_stats.py` ·
`run_proxy_AB_500.py` · `run_proxy_designs_ABCDEF.py` ·
`run_proxy_experiment.py` / `_v2.py` (flat-constant baseline, null effect).

### Raw data — `llava_logit_eda/`
- `outputs_500_clean/` — 500-img greedy/VCD/SID captions + `raw_object_candidates.jsonl` (~57k object candidates: E, A, d, is_real, chosen) + `mention_sets.jsonl` + `aggregate_stats.json`
- `outputs_full_topk/` — 500 img, every step: expert/amateur/CD top-30 ids+logits+category, APC cutoff, survival, chosen (source for ROC / amateur-bucket / entropy / overlap)
- `outputs/` — 10-image pilot per-step JSONs + `EDA_report.pdf`
- `outputs_proxy_AB_500/`, `outputs_proxy_ABCDEF/` — proxy captions + CHAIR results
- (`outputs_500/`, `outputs_proxy/`, `outputs_proxy_v2/`, `outputs_smoke/` = intermediate/superseded, kept for completeness)

### Analysis outputs — `llava_logit_eda/`
- `ROC analysis/` — ROC curves (expert vs CD vs contrastive term), with embedded captions
- `amateur_bucket_analysis/` — degraded-branch bucket mass + `bucket_mass_bars.png` + CSVs + `summary.json` (incl. denial/generic word lists)
- `figures/`, `figures_modified/` — d-value line plots (top-10/30, real/hall; latter with per-image bands)
- `proxy_stats.json`, `image_ids.json`

---

## Headline results (500 images unless noted)

1. **APC collapse:** median 2 survivors/step; **47–48% of steps have 1 survivor**
   (output forced = greedy); full pipeline == greedy at **92–94%** of steps.
2. **Contrastive term carries no hallucination signal:** ROC AUC of d = E−A is
   **0.48 (VCD) / 0.53 (SID)** — at/below chance; adding it does not raise the
   expert's own AUC (0.64).
3. **Degraded branch does not hallucinate:** ~97% of amateur mass is non-object
   language; absent-object mass **0.007** < present-object **0.027**; amateur−expert
   on absent objects = **+0.0003** (nothing to subtract).
4. **Degraded branch ≈ clean branch:** same next token **92–94%** of the time;
   near-identical entropy (1.11 vs 1.12) and peakedness.
5. **Noise reproduces CD:** magnitude-matched Gaussian (A) and empirical-resample
   (B) proxies match real VCD/SID whack-a-mole direction and size
   (real +25/+18; proxy +27/+37/+16).
