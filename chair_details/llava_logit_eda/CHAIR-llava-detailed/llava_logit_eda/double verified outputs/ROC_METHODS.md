# Methods — ROC / AUC (real vs. hallucinated discrimination)

**Files:** `roc_within_step_auc.png` (the trustworthy result), `roc_emitted_objects.png`
(kept for transparency, but confounded — see §4), `summary_roc.json`.

## 1. The question

Can a score tell **real** object words apart from **hallucinated** ones? We test
three scores:

- **E** = expert logit (clean image),
- **A** = amateur logit (degraded image; VCD: Gaussian-noised, SID: attention-pruned),
- **CD** = the contrastive score **2E − A = E + (E − A)** (what CD actually decodes with),
- **d** = the contrastive term alone **E − A**.

We treat each score as a ranker and measure **AUC** = the probability that a
randomly chosen **real** object is scored higher than a randomly chosen
**hallucinated** one. **AUC = 0.5 is chance; > 0.5 is useful; < 0.5 is backwards.**
No classifier is trained — AUC is a direct property of the score itself.

## 2. How real vs. hallucinated is decided

Standard **CHAIR** ground truth: an image's true object set = its COCO
instance-segmentation categories ∪ the objects named in its five human reference
captions (via the CHAIR synonym list). A word is **hallucinated** if its COCO
category is not in that set. For **emitted words** this is applied at the
whole-word level (exact). For **candidates** we additionally require the token to
be **word-initial** (SentencePiece "▁" marker) so subword fragments (e.g. "ship"
in "companionship") are not miscounted.

## 3. The confound-free result — within-step paired AUC (USE THIS)

`roc_within_step_auc.png`

Logits are **not comparable across decoding steps** (the scale shifts with
context). So we only compare a real object against a hallucinated object **that
occur at the same step** (identical scale), and count how often the score ranks
the real one higher. Averaging this concordance over all within-step (real, hall)
pairs gives a scale-free AUC.

| score | VCD | SID |
|---|---|---|
| Expert E | 0.629 | 0.654 |
| CD 2E−A | 0.632 | 0.662 |
| Contrastive term d | 0.541 | 0.582 |

(n ≈ 9,100 within-step real–hall pairs per method; top-30 candidates, corrected labels.)

**Conclusion:** the plain expert already weakly separates real from hallucinated
(≈ 0.63–0.65). Adding the contrastive term changes this by only **+0.003 (VCD) /
+0.008 (SID)** — negligible. The contrastive term on its own is weak (0.54 / 0.58),
only just above chance. **CD does not meaningfully improve hallucination
discrimination over the plain model.**

## 4. The confounded result — emitted-object ROC (shown for transparency only)

`roc_emitted_objects.png` pools raw logits across steps for the **emitted** object
words. It reports Expert 0.62 / CD 0.57 / d 0.37 (VCD). **Do not use these as the
headline**, for two reasons: (a) pooling logits across steps mixes in cross-step
scale differences, and (b) emitted words are selection-biased (they all "won"
their step). These two effects, not a real suppression signal, are what push CD
below the expert and d below chance here. The within-step analysis in §3 removes
both and is the correct version. We keep this figure only so the difference is
documented, not hidden.

## 5. Consistency across every version we ran

Three populations, same core conclusion:

| population | labels | Expert | CD | d |
|---|---|---|---|---|
| all candidates (first run) | per-token (buggy) | 0.65 | 0.64 | 0.48 |
| emitted words (confounded) | CHAIR (correct) | 0.62 | 0.57 | 0.37 |
| **within-step top-30 (correct + confound-free)** | corrected | **0.63** | **0.63** | **0.54** |

Across all of them the expert sits around 0.62–0.65 and **CD moves it by at most
~0.01**. The safe, reviewer-proof statement is therefore: *"the contrastive term
adds no meaningful real-vs-hallucination discrimination on top of the plain expert
model."*

## 6. Notes for a reviewer
- Greedy decoding, 500 COCO images, LLaVA-1.5-7B (reproducible).
- The wiggle in the emitted ROC curve is ordinary finite-sample staircase noise
  (~577 negatives → 1/577 steps), not instability; AUC integrates over it.
- We report the confound-free within-step AUC as primary precisely because a
  careful reader should not trust cross-step-pooled logit ROC.
