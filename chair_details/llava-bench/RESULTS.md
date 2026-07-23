# LLaVA-Bench reproduction results (LLaVA-1.5-7B)

Judge: **Claude Opus 4.8**, in-session, single pass, consolidated protocol (reference
scored once per question; all methods calibrated together). Metric = relative score
= mean(model)/mean(reference) × 100 over the 60 questions. Reference mean = 89.7/100.

## Table 4 — greedy base

| Method | Paper | This repro (Claude judge) |
|---|---|---|
| Greedy | 65.7 | **62.1** |
| VCD (greedy) | 64.6 | **61.9** |
| SID (greedy) | 64.9 | **63.4** |

## Table 6 — direct-sampling base

| Method | Paper | This repro (Claude judge) |
|---|---|---|
| Sample | 64.3 | **56.1** |
| Sample† (APC, β=0.1) | 65.3 | **56.7** |
| VCD (sample) | 64.2 | **62.1** |
| SID (sample) | 64.5 | **62.6** |

Per-category (relative): conv / detail / complex available in `_judge/summary` computation.

## What reproduces (matches the paper's thesis)

1. **Under greedy, contrastive decoding does not beat greedy.** VCD-greedy (61.9) ≈ Greedy
   (62.1); SID-greedy (63.4) within noise. This is the paper's core "mirage" claim: the
   contrastive term adds nothing on top of greedy. (VCD-greedy answers were frequently
   *token-identical* to greedy.)
2. **The adaptive plausibility constraint gives a small bump under sampling.** Sample†
   (56.7) > Sample (56.1), same direction and similar magnitude (~+0.6 vs paper +1.0).

## Where it diverges (worth investigating / reporting)

3. **VCD-sample (62.1) and SID-sample (62.6) sit well above plain Sample (56.1)** — a
   ~6-point gap, larger than the paper's ~0. Mechanism is consistent with the Mirage thesis
   (both include the plausibility constraint, which suppresses the low-probability, often
   hallucinated tokens that plain temperature-1.0 sampling emits), **but**:
4. **Standalone APC (Sample†, β=0.1) does NOT recover VCD/SID-sample quality here** (56.7 vs
   62), unlike the paper where Sample† is the *highest* row. Likely β=0.1 is too loose a
   constraint (keeps tokens ≥ 0.1·max prob → still samples noisily). The repo has a full β
   sweep (0.0–1.0); Sample† should climb toward greedy as β tightens. **Recommendation:**
   report Sample† at the β the paper used, or show the β-sweep curve — this is itself a
   clean Mirage-style ablation.

## Caveats (state these in the writeup)

- Claude ≠ gpt-4-0314, so absolute numbers run ~3 pts lower than the paper (stricter judge;
  reference mean 89.7 deflates relatives). **Only the pattern is comparable**, not the
  absolute value.
- Single judge pass on 60 questions → ~1-pt deltas are within noise. For a submission, run
  ≥3 passes and report mean±std (or use gpt-4-0314 for a faithful absolute reproduction).
- Scores stored in `outputs/llava/_judge/scores_running.json` (per-question, all methods).
