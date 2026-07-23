# Double-Verified Outputs

These are the **corrected** versions of the logit-level analyses. The earlier
versions used a naive per-token matcher that mislabeled subword fragments as
objects (e.g. "ship" inside "companion·ship", or "bear" split off from
"teddy bear"). Here every object is identified with the **caption-level CHAIR
mapping** — it works on the full generated word, collapses double-words
("teddy bear", "dining table"), and ignores subword fragments. So the
real-vs-hallucinated labels are trustworthy.

**Unit of analysis:** actually-emitted object words (a hallucination is a real
COCO object word the model wrote that is not in the image, per CHAIR). We use
emitted words because you can only know what word a token forms once it's
chosen — a candidate token that never won can't be labeled correctly.

---

## 1. Does the amateur "bonus" d = E − A single out hallucinations?

Files: `d_top10_{vcd,sid}_{hall,real}.png` and `d_top30_{vcd,sid}_{hall,real}.png`
— the top-10 and top-30 versions (object candidates in the expert's top-10 /
top-30 at each step, with the word-initial filter applied so subword fragments
like "ship" in "companionship" are excluded).

d is how much higher the clean-image "expert" scores a word than the
noised-image "amateur". VCD's final score is C = E + d, so **positive d means the
word gets pushed UP (amplified); negative d means pushed DOWN (suppressed).**
In the graphs, **red bars = positive d (amplified), blue bars = negative d
(suppressed)**; the dashed line is the mean.

Mean d by population:

| population | VCD real | VCD halluc | SID real | SID halluc |
|---|---|---|---|---|
| top-10 | +0.124 | +0.148 | +0.215 | +0.122 |
| top-30 | +0.034 | −0.010 | +0.149 | −0.007 |

**Plain English:** hallucinated objects are **not** systematically suppressed.
In the top-10 view VCD even amplifies hallucinations slightly more than real
objects (0.148 vs 0.124); in the broader top-30 view hallucinated d sits at
essentially **zero** (−0.01 / −0.01) while real objects lean mildly positive.
The sign flips between top-10 and top-30 and between VCD and SID — a real
suppression signal would be stable and clearly negative for hallucinations.
It is neither. (For the words the model *actually emits* — the `d_emitted_word/`
graphs — the effect is even stronger: emitted hallucinations get ~3× the
amplification of emitted real objects, mean d +0.79 vs +0.26.)

## 2. ROC — can the CD score tell real objects from hallucinated ones?

**Primary (confound-free): `roc_within_step_auc.png`** — full detail in `ROC_METHODS.md`.
AUC = probability the score ranks a real object above a hallucinated one
(0.5 = coin flip). Because raw logits are not comparable across steps, we compare
real vs. hallucinated candidates **within the same step** (same scale).

| score | VCD AUC | SID AUC |
|---|---|---|
| Expert logit E (plain model) | 0.629 | 0.654 |
| CD score 2E−A (after subtraction) | 0.632 | 0.662 |
| Contrastive term d alone | 0.541 | 0.582 |

**Plain English:** the plain expert already weakly separates real from
hallucinated (~0.63–0.65). Adding the subtraction changes this by only
+0.003 / +0.008 — **negligible**. The contrastive term alone is weak (0.54 / 0.58),
barely above chance. **CD does not meaningfully improve discrimination over the
plain model.**

> An earlier figure (`roc_emitted_objects.png`) pooled logits across steps for
> emitted words and *appeared* to show CD below the expert and d below chance
> (0.37). That was a cross-step-scale + selection-bias **confound**, not a real
> effect — the within-step numbers above are correct. Kept only for transparency
> (`ROC_METHODS.md` §4).

## 3. Where does the amateur (noised-image) branch put its probability?

`amateur_buckets_corrected.png`

| bucket | VCD | SID |
|---|---|---|
| denial ("no", "not") | 0.004 | 0.004 |
| generic ("the", "a") | 0.465 | 0.461 |
| present objects (really there) | 0.027 | 0.026 |
| absent objects (hallucination candidates) | **0.007** | **0.007** |
| other words | 0.499 | 0.503 |

**Plain English:** ~96% of the noised-image branch's probability is ordinary
non-object language. Only ~0.7% lands on absent (hallucinatable) objects —
*less* than on real objects (2.7%). So the degraded branch is not producing the
hallucinations VCD assumes it can subtract away. (This matches the earlier
result — it was already robust to the labeling bug, because objects are a tiny
slice either way.)

---

## Bottom line
With correct labels, the story is unchanged and slightly stronger:
1. The contrastive term amplifies emitted hallucinations ~3× more than real objects.
2. Subtraction *lowers* real-vs-hallucination discrimination (CD AUC < expert AUC); the term alone is below chance.
3. The degraded branch barely touches objects and does not hallucinate absent ones.

**Caveat:** ROC here is over *emitted* objects (all of which "won"), so read the
Expert-vs-CD *comparison* rather than the absolute AUC. The bucket analysis is
candidate-level with a word-initial filter; a residual multi-word case
("teddy bear" → "bear") remains but affects <1% of mass and both branches equally.
