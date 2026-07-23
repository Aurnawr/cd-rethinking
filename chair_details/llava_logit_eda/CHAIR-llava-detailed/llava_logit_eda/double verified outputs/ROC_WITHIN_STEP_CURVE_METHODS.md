# Methods — the within-step ROC CURVE (top-30)

**File:** `roc_within_step_curve.png` (curve) · `roc_within_step_auc.png` (AUC bars)
· numbers in `summary_roc_within_curve.json`.

This explains, step by step, exactly how the curve was drawn and why it is the
correct, confound-free way to show whether a score separates real from
hallucinated objects.

---

## Why a plain ROC curve would be wrong here

A normal ROC sweeps **one global threshold** across all scores. That only makes
sense if the scores are comparable across all data points. **Our scores are
logits, and logits are not comparable across decoding steps** — a confident step
might peak near 25, an uncertain one near 12. If we pooled raw logits, the curve
would partly measure "real objects happen at higher-confidence steps than
hallucinations," which is *not* the question. So we build the curve **within
steps**, where the scale is shared.

---

## The scores being tested (per candidate token)

At each step the model scores the next token twice: **E** = expert logit (clean
image), **A** = amateur logit (degraded image). We test three rankers:
- **E** (plain model),
- **CD = 2E − A** (what contrastive decoding actually uses),
- **d = E − A** (the contrastive term on its own).

Positive labels = **real** objects, negative = **hallucinated** objects.

---

## Step-by-step construction of the curve

**Step 1 — collect object candidates per step.** For every decoding step of every
caption (500 images, greedy, LLaVA-1.5-7B), take the expert's **top-30** next-word
candidates. Keep the ones that are COCO objects, using the **word-initial filter**
(the token must start a word — SentencePiece "▁" marker — and be a COCO object
word), which excludes subword fragments like "ship" in "companionship". Record
each kept candidate's E, CD, d, and its label (real if its COCO category is in the
image's CHAIR ground-truth set, else hallucinated).

**Step 2 — keep only "contrastive" steps.** A within-step comparison needs both
classes present, so we keep only steps that contain **at least one real AND at
least one hallucinated** object candidate. (VCD: 2,535 such steps, 9,417 points;
SID: 2,432 steps, 9,315 points.) Steps with only one class carry no within-step
signal and are dropped.

**Step 3 — remove the per-step offset (centering).** Within each kept step,
subtract the step's mean of that score from every candidate:
`centered_score = score − (mean of that score over the step's object candidates)`.
This is the key move: it deletes the cross-step baseline difference (the confound)
while **preserving the order of candidates inside each step**, which is all a
within-step comparison depends on. After centering, a single global threshold is
finally meaningful, because every point is expressed *relative to its own step*.

**Step 4 — sweep the threshold to trace the curve.** Pool all centered scores.
Sort candidates from highest to lowest centered score. Walk down the sorted list;
at each threshold count:
- **TPR (y-axis)** = (real candidates above the threshold) ÷ (all real candidates)
- **FPR (x-axis)** = (hallucinated candidates above threshold) ÷ (all hallucinated)

Plotting TPR against FPR as the threshold moves from high to low traces the ROC
curve (starting at (0,0), ending at (1,1)).

**Step 5 — AUC.** The area under the curve = the probability that a randomly
chosen real candidate has a higher centered score than a randomly chosen
hallucinated one. 0.5 = chance; higher = better separation.

---

## Cross-check: the curve matches the pairwise number

We separately computed a **pairwise within-step concordance** (for every
real–hallucinated pair *at the same step*, how often the real one scores higher).
The two agree, which confirms the curve is faithful:

| | curve AUC (VCD) | pairwise (VCD) | curve AUC (SID) | pairwise (SID) |
|---|---|---|---|---|
| Expert E | 0.641 | 0.629 | 0.658 | 0.654 |
| CD 2E−A | 0.643 | 0.632 | 0.668 | 0.662 |
| Contrastive term d | 0.537 | 0.541 | 0.579 | 0.582 |

(The tiny gap is because centering equalises each step's *offset* but not its
*spread*; the pairwise number is offset- and spread-free. They land within ~0.01.)

---

## What the curve shows

- **Blue (Expert) and red (CD) lie almost exactly on top of each other.** Adding
  the contrastive subtraction moves the AUC by only +0.002 (VCD) / +0.010 (SID).
- **Gray (contrastive term d alone) is weak** — AUC 0.54 (VCD) / 0.58 (SID), only
  a little above the diagonal (chance).
- The only real separating power (~0.64–0.66) belongs to the **plain expert**, and
  CD does not meaningfully add to it.

**One-line conclusion:** *on a scale-free, within-step ROC, contrastive decoding's
score discriminates real from hallucinated objects no better than the plain expert
model — the contrastive term contributes essentially nothing.*

---

## Notes for a reviewer
- Greedy decoding, 500 COCO images, LLaVA-1.5-7B; reproducible.
- Labels are the standard CHAIR ground truth (COCO segmentation ∪ reference-caption
  objects, via the synonym list), with the word-initial filter removing subword
  false positives.
- Centering only removes an additive per-step constant, so it cannot manufacture
  or hide separation — it strictly preserves within-step ranking. The pairwise
  cross-check confirms no distortion was introduced.
