# CHAIR contrastive-decoding study — full handoff

This folder is a complete, self-describing snapshot of a study on **why contrastive
decoding (VCD and SID) does not reduce object hallucination in multimodal LLMs**,
run on the **CHAIR** generative-captioning benchmark for two models,
**LLaVA-1.5-7B** and **Qwen2.5-VL-7B-Instruct**, over a fixed set of **500 MSCOCO
val2017 images**.

It contains the raw per-step logit captures, the derived results with confidence
intervals, the analysis code, and the paper sources. A fresh reader (human or a new
Claude Code session) should be able to read this file top to bottom, understand
exactly what was done and what was found, reproduce every number, and continue the
work. Read this file first.

---

## 1. The thesis in one paragraph

VCD and SID both build a "contrastive" next-token score `C = (1+alpha)E - alpha A`
from an **expert** branch `E` (the model on the clean image) and an **amateur**
branch `A` (the model on a corrupted image: diffusion noise for VCD, image-token
dropout for SID), then apply an **adaptive plausibility constraint (APC)**. They are
sold as reducing hallucination. On CHAIR we find they do not: on LLaVA they make
CHAIR slightly worse than plain greedy, on Qwen clearly worse. The mechanistic
experiments explain why, and a noise proxy shows the amateur branch carries no
special signal — replacing it with matched random Gaussian noise reproduces the
same CHAIR outcome.

---

## 2. TL;DR findings (all with 95% image-level bootstrap CIs, B=10000, resampling the 500 images)

**CHAIR (lower is better), from `results/bootstrap_chair.json`:**

| Model | Greedy CHAIR-S | VCD | SID | Proxy A (noise) |
|---|---|---|---|---|
| LLaVA-1.5-7B | 50.0 [45.6, 54.4] | 55.0 [50.6, 59.4] | 54.2 [49.8, 58.6] | 52.4 [48.0, 56.6] |
| Qwen2.5-VL-7B | 30.8 [26.8, 34.8] | 36.4 [32.4, 40.6] | 38.4 [34.2, 42.6] | 30.8 [26.8, 34.8] |

Neither real method beats greedy; both are worse. The Gaussian-noise proxy (no
amateur branch at all) lands in the same band as the real methods. See
`results/proxyA_table_ci.tex` for the full CHAIR-S / CHAIR-I table with CIs.

**Experiment 2 — intervention rate** (`paper/exp2and3/03_results.tex`, Table 1):
the contrastive step is a no-op at most steps. CD's emitted token equals the
expert's own greedy top-1 at 94.0% (LLaVA-VCD) / 91.8% (LLaVA-SID) / 75.3%
(Qwen-VCD) / 75.7% (Qwen-SID) of all steps. Even at steps that emit a hallucinated
object, CD still reproduces the expert's greedy token 75.1% / 73.1% (LLaVA) and
62.2% / 62.5% (Qwen) of the time. It fires more at hallucination steps (gap 17.6–22.1
pp, all CIs exclude zero) but still does not fix them.

**Experiment 3 — candidate overlap** (`03_results.tex`, Table 2): all three
branches deliberate over the same words. Expert top-10 vs amateur top-10 overlap is
~9.3/10 (LLaVA) and ~8.0–8.4/10 (Qwen), and the contrastive top-10 tracks the
expert almost exactly (E∩C ≈ E∩A in every cell). The contrast supplies no new
grounded candidate; it only re-weights a shortlist the expert already fixed.

**Shortlist composition at hallucination steps** (`03_results.tex`, Table 3):
among object candidates in the top-10 at a hallucination step, 82–86% are
themselves hallucinated objects for both models. Contrastive decoding does not
repair this; on Qwen it makes the real share smaller (VCD 16.0% → 13.7%, SID
14.5% → 14.2%).

**The adjustment d = E − A** (`results/d_stats.json`): a working suppressor would
give hallucinated objects a clearly negative d. It does not. On LLaVA the emitted
hallucinated objects have d = +0.79 (real +0.26): the contrast amplifies
hallucinations more than real objects. On Qwen d is mostly negative because the
amateur is over-confident, but it suppresses real objects more than hallucinated
ones (emitted real d = −1.28 vs hall d = −0.10). Either way, d does not
selectively push hallucinations down.

**Bucket / paired analysis** (`analysis/bucket_ci.py`, `analysis/bucket_ci_paired.py`):
normalised probability mass on real vs hallucinated objects among the top-5 at
object steps, per branch, with a paired image-level bootstrap of the
(contrastive − expert) difference. LLaVA: the contrastive step does not
significantly change the real-object mass (paired diff ≈ 0, CI includes 0). Qwen:
the contrastive step significantly *reduces* real-object mass (paired diff
significantly negative), i.e. it moves probability toward hallucinations. Re-run
the paired script to reproduce the exact diffs and p-values.

---

## 3. Setup and decoding configuration

- **Models:** `liuhaotian/llava-v1.5-7b`, `Qwen/Qwen2.5-VL-7B-Instruct`.
- **Benchmark:** CHAIR on 500 MSCOCO val2017 images. The exact ids and order are in
  `data/image_ids_500.json` (verified to match every capture file, same set and
  same order). Prompt: "Describe this image in detail."
- **Decoding:** greedy, `max_new_tokens = 256`.
- **Contrastive score:** `C = (1+alpha)E - alpha A` with `alpha = 1`, so `C = 2E - A`.
- **APC (plausibility):** keep only tokens with expert logit `>= log(beta) + max_w E(w)`,
  `beta = 0.2`; everything else set to `-inf` before the argmax. See section 6.1.
- **VCD amateur:** diffusion-noised image, `noise_step = 500` (of 1000, sigmoid betas).
- **SID amateur:** image-token dropout — keep 72 of 576 image tokens, mask the rest
  from decoder layer `AGG_LAYER = 2` onward. See section 6.2 (the selection is
  random, not attention-based — this matters).
- **Noise proxy (Proxy A):** discard the amateur branch; instead add Gaussian noise
  `N(mu, sigma^2)` to the expert's object-category logits, with `(mu, sigma)` matched
  to the measured distribution of `d = E - A`. LLaVA VCD-matched: mu=+0.199,
  sigma=0.853. Qwen VCD-matched: mu=−0.846, sigma=1.531. Full measured d arrays are
  in `data/{llava,qwen}/proxy_stats_*.json`. There is also a bootstrap variant
  "Proxy B" that resamples empirical d values instead of drawing Gaussian.

---

## 4. The capture data — format and how to use it

`data/captures/` holds the per-step top-30 logit captures, **gzipped** (lossless;
`gunzip` restores byte-for-byte identical files — verified by md5). Naming matches
what the analysis scripts expect: `<model>_<method>.jsonl` where model in
{llava, qwen}, method in {vcd, sid}.

Sizes: llava_vcd 25M, llava_sid 26M, qwen_vcd 43M, qwen_sid 43M (gzipped; ~700M raw).

Each line is one image: `{"image_id", "caption", "steps": [ ... ]}`. Each step is:

```
{
  "step": int,                       # decoding position
  "chosen_id": int,                  # token CD actually emitted at this step
  "chosen_category": str,            # real / hall / non-object (as captured)
  "apc_cutoff": float,               # log(beta)+max_w E(w) at this step
  "expert_top_ids":   [30 ints],     # expert branch top-30 token ids (E)
  "expert_top_logits":[30 floats],
  "amateur_top_ids":  [30 ints],     # amateur branch top-30 (A)
  "amateur_top_logits":[30 floats],
  "cd_top_ids":       [30 ints],     # contrastive score top-30 (2E-A, pre-APC)
  "cd_top_pre_apc_logits":[30 floats],
  "cd_top_survives_apc":[30 bools]   # whether each cd candidate passes the APC cutoff
}
```

To use the captures with the runnable analysis code in the sibling `CHAIR_analysis/`
package:

```
cd chair_full_handoff/data/captures
gunzip -k *.gz                        # -k keeps the .gz too
mkdir -p ../../../CHAIR_analysis/outputs/captures
cp *.jsonl ../../../CHAIR_analysis/outputs/captures/
# then, from CHAIR_analysis/ (see its README for env + assets):
python common/agreement_analysis.py   --model llava --B 10000
python common/contrastive_analysis.py --model qwen  --B 10000
```

The greedy and proxy **caption** outputs (not per-step logits, just the generated
text) are in `data/llava/` and `data/qwen/`; these are what `bootstrap_chair.py`
scores.

---

## 5. What each file is (map)

```
chair_full_handoff/
  HANDOFF.md                         this file
  data/
    image_ids_500.json               the fixed 500 image ids + order (matches all captures)
    captures/                        per-step top-30 logit captures, gzipped
      llava_vcd.jsonl.gz  llava_sid.jsonl.gz
      qwen_vcd.jsonl.gz   qwen_sid.jsonl.gz
    llava/
      captions_greedy.jsonl          greedy captions (for CHAIR scoring)
      proxy/captions_{A,B}_{vcd,sid}.jsonl   noise-proxy captions
      proxy_stats_llava.json         measured d = E-A distribution used by the proxy
    qwen/
      captions_greedy.jsonl
      proxy/captions_{A,B}_vcd.jsonl
      proxy_stats_qwen.json
  results/
    bootstrap_chair.json             CHAIR-S/CHAIR-I with 95% CIs for every setting
    d_stats.json                     mean d and %(d>0) by emitted/top10/top30, real/hall
    proxyA_table_ci.tex              the Proxy-A-vs-real CHAIR table with CIs (paper-ready)
    bucket_table_ci.tex              object-mass bucket table with CIs
    object_steps_LLaVA_vcd_60img.md  human-readable top-30 logit viewer, object steps only
  analysis/                          the scripts (read-and-run; paths may need editing)
    bootstrap_chair.py               image-level bootstrap CIs for CHAIR
    agreement_analysis.py            intervention rate + top-10 overlap (reads captures)
    contrastive_analysis.py          d = E-A, buckets, proxy-vs-greedy CHAIR
    bucket_ci.py / bucket_ci_paired.py   bucket CIs and the paired difference test
    view_object_steps.py             builds the markdown logit viewer
    object_mentions.py               caption-level CHAIR object attribution (shared helper)
    capture_llava.py / capture_qwen.py   the generators that PRODUCED the captures (need GPU)
  paper/
    exp2and3/                        Experiments 2 & 3 (intervention + overlap) tex
    exp4/                            Experiment 4 (d analysis + noise proxy) tex
    appendix_final/                  worked-example appendix (2 examples) + assets + build.py
```

Not included here (intentionally): model weights, COCO images, compiled/preview
PDFs, `__pycache__`, and a superseded 79M per-image EDA dump. Weights and images
are fetched by `CHAIR_analysis/download_assets.sh`.

---

## 6. Two code-vs-paper discrepancies found (IMPORTANT for next steps)

These were verified from the actual inference code in the repo, not from prose.
Both must be resolved before the paper's method descriptions are correct.

### 6.1 APC gates on the EXPERT logit, so the plausible set is the expert's

File: `inference/cd_utils/vcd_utils.py` (identical in `sid_utils.py`,
`spurious_utils/apc_utils.py`, `icd_utils.py`, and the `llava-bench/` mirrors).

```
139  next_token_logits    = outputs.logits[:, -1, :]      # EXPERT (clean image)
162  next_token_logits_cd = outputs_cd.logits[:, -1, :]   # AMATEUR
173  cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(-1, keepdim=True).values
175  diffs  = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd   # 2E - A
176  cd_logits = diffs.masked_fill(next_token_logits < cutoff, -inf)           # gate on EXPERT
```

The keep condition uses the **expert** logits, not the contrastive score. So the
candidate pool APC allows is fixed entirely by the expert distribution; the contrast
can only reorder within it. beta is applied in log space (`log(beta)+maxE`, the
active "version 2"), equivalent to `p(w) >= beta * p_max`. Worked check for image
481582 at the "cars" step: expert max = 21.45, cutoff = log(0.2)+21.45 = 19.84,
`keep.sum() = 1` → the plausible set is the singleton {cars}. (If APC instead gated
the contrastive score, "vehicles" E=19.36 and "other" E=19.33 would also survive;
they do not under the real code.) This is standard VCD/APC; make sure the paper
describes the plausible set as the expert's, and see it as a mechanism for why the
contrast cannot add grounding.

### 6.2 SID selects image tokens at RANDOM, not by attention

File: `llava/model/language_model/custom_modeling_llama.py` (byte-identical in all
five copies in the repo: main `llava/`, `llava-bench/`, `llava_logit_eda/`,
`gaussian_proxy_handoff/`, `CHAIR_analysis/`).

```
701  ATTENTION_RANK = 72     702  AGG_LAYER = 2
714  # randomly select ATTENTION_RANK tokens
715  random_indices = torch.randperm(IMAGE_TOKEN_LENGTH)[:ATTENTION_RANK]   # 72 of 576
716  top_attention_rank_index = random_indices + SYS_LENGTH
720  gen_attention_mask[:, SYS_LENGTH:SYS_LENGTH+576] = False               # mask all image cols
721  gen_attention_mask[:, top_attention_rank_index] = True                 # re-enable 72 random ones
```

Qwen path is the same: `capture_qwen.py` uses `torch.randperm(pos.numel())` to keep
`round(0.125 * N)` image tokens (comment there literally says "RANDOM 72/576").

The names `ATTENTION_RANK`, `top_attention_rank_index`, and the "FastV Token Rerank"
comment are leftover scaffolding; the selection is uniform random and was random from
the first commit (checked with `git log -L`). No attention-based ranking exists
anywhere in the repo. Determinism check: `torch.randperm(576)[:72]` under two
different seeds overlaps ~9/72 (chance); so different seeds retain different tokens
and give different amateur logits — an attention-based selection would be
seed-independent.

**Consequence:** as implemented, "SID" here is a random image-token masking
ablation, not the published attention-based SID. A negative result about this branch
does not automatically transfer to the real method. Before finalising: either (a)
implement attention-ranked selection and re-run to keep the "SID" label faithful, or
(b) relabel every "SID" result as a random-masking ablation and fix the main-text
sentence that calls it attention-based. Cross-check against the official SID repo
first. This is the single most important open issue.

---

## 7. Reproduction recipe

The runnable, self-contained code lives in the sibling `CHAIR_analysis/` package
(same branch), which has a `download_assets.sh` (weights + COCO + the 500 images), a
`requirements.txt`, and per-experiment `run.sh` scripts. The fastest path to
reproduce the numbers in this handoff:

1. Set up `CHAIR_analysis/` per its README (env + `bash download_assets.sh`).
2. Stage the captures from here (section 4) into `CHAIR_analysis/outputs/captures/`.
   The agreement and contrastive analyses will then run without a GPU (they only
   read the captured logits).
3. `bootstrap_chair.py` (here in `analysis/`) reproduces the CHAIR CI table from the
   greedy/proxy caption files in `data/`. Point its `FILES` paths at `data/`.
4. Regenerating the captures themselves (`capture_llava.py`, `capture_qwen.py`)
   needs a GPU. Qwen capture is slow: the amateur branch is recomputed without a KV
   cache each step for numerical correctness (a cached second branch under Qwen's
   mRoPE did not match), so expect hours for 500 images.

Determinism note: VCD's noise draw and SID's kept-token subset are random; they are
seeded per image in the generators, so a rerun is reproducible, but exact values can
shift slightly across hardware/library builds. Aggregate CHAIR over 500 images is
stable.

---

## 8. Known limitations / gotchas

- **The proxy runs did NOT record per-step logits** — only `{image_id, caption}`.
  So the agreement-rate and top-10 overlap analyses (which need per-step top-k)
  **cannot** be computed for the proxy from the stored files. Doing a proxy-vs-VCD/
  SID/greedy agreement/overlap comparison requires re-running the proxy in a
  capture-style mode that logs expert top-k, post-noise proxy top-k, and the chosen
  token. Needs a GPU.
- **CIs here are image-sampling variance for a single noise draw.** They do not
  include noise-draw variance for the proxy or the random SID mask. Multi-seed runs
  (a few seeds) would add that; see `gaussian_proxy_handoff/` (sibling folder), which
  is set up to run the multi-seed Gaussian proxy on a small GPU budget.
- **Coincidence to be aware of:** Qwen greedy and Proxy A have the same CHAIR-S point
  estimate (30.8) and nearly the same CI. This is genuine: the captions differ
  (362/500 differ; 64 images flip hallucination status) but equal numbers flip each
  way, so the count is unchanged. CHAIR-I differs (7.69 vs 8.18). Not a bug; CHAIR-S
  is quantised to steps of 1/500.
- **"SID" = random masking** (section 6.2) — the biggest caveat.

---

## 9. Open next steps (suggested)

1. Resolve the SID random-vs-attention issue (section 6.2): check the official SID
   repo, then either implement attention-ranked selection and re-run, or relabel.
2. Re-run the noise proxy in capture mode so agreement-rate and overlap can be
   computed for the proxy and put side by side with VCD/SID/greedy.
3. Add multi-seed proxy runs for noise-draw CIs (use `gaussian_proxy_handoff/`).
4. Fix any main-text wording so APC (section 6.1) and SID (section 6.2) match the
   code exactly.
5. Finalise the appendix (`paper/appendix_final/`, two worked examples) and the
   Proxy-A CI table into the main paper.

---

## 10. Sibling folders on this branch

- `CHAIR_analysis/` — clean, anonymised, self-contained runnable package (download
  assets + run.sh per experiment). Use this to actually run things.
- `gaussian_proxy_handoff/` — packaged multi-seed Gaussian-proxy run for a small GPU
  budget (weights/data auto-download, 3 seeds).

This handoff folder is the detailed working snapshot (raw captures + results +
narrative); the two siblings are the runnable/portable packages.
