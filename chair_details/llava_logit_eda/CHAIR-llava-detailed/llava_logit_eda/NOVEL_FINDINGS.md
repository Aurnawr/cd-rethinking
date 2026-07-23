# LLaVA-1.5-7B Contrastive Decoding: Novel Findings from Per-Token Logit EDA

Analysis based on the 10-image CHAIR logit EDA (`outputs/`) plus follow-up
verification scripts run directly against the model on GPU, including a
full replay of every single decoding step (2,245 steps total across all 10
images x VCD/SID) with independently-computed full-vocabulary expert/amateur
top-10 logits. All numbers are computed, not estimated, from the actual
generated captions and logits — nothing here is from a single example.

Context: this extends ["The Mirage of Performance Gains: Why Contrastive
Decoding Fails to Mitigate Object Hallucinations in
MLLMs?"](https://arxiv.org/abs/2504.10020) (arXiv:2504.10020), which proved
VCD/ICD/SID's POPE (discriminative) improvements are spurious by (a) showing
crude, vision-unrelated interventions (Prompt-Based Adjustment, Output Layer
Modification) reproduce the same gains, and (b) showing the Adaptive
Plausibility Constraint (APC) alone, with no contrastive subtraction at all,
recovers most of the improvement by collapsing sampling to quasi-greedy
search. **The paper explicitly does not test CHAIR or any generative
benchmark** — that is the gap this analysis fills.

---

## 0. THE UNIFYING MECHANISM: a category-level amplification, not a hallucination-detector

*(Note: an earlier version of this section called the effect below a
"discount" and described it as pushing object logits down. That had the
sign backwards — corrected below, with the derivation shown so the error
is visible rather than silently fixed.)*

**Notation.** At decoding step $t$, for candidate word $w$:
- $E_t(w)$ = expert logit (real image forward pass)
- $A_t(w)$ = amateur logit (VCD: noised image; SID: attention restricted to
  72/576 visual tokens)
- Contrastive score (this run uses $\alpha=1$):
  $C_t(w) = (1+\alpha)E_t(w) - \alpha A_t(w) = 2E_t(w) - A_t(w)$
- APC survival: $w$ survives only if $E_t(w) \geq \text{cutoff}_t$, where
  $\text{cutoff}_t = \log(\beta) + \max_w E_t(w)$, $\beta = 0.2$. **This test
  uses only $E_t$ — $A_t$ has no role in survival at all.**
- Emitted word = $\arg\max_w C_t(w)$ among survivors.
- Define $d_t(w) = E_t(w) - A_t(w)$ (how much lower the amateur rated $w$
  than the expert did — this is exactly the number reported in the tables
  below).

**Hypothesis tested (user-proposed):** do VCD/SID push object-word logits
down everywhere, uniformly, such that whether an object survives is really
just a question of clearing the APC bar?

**Test:** for every candidate token in the top-K pool at every one of the
2,245 decoding steps (every candidate, whether it won or not), split into
COCO object-category words vs. everything else, and measured $d_t(w)$:

| method | candidate type | n candidates | avg $d = E-A$ |
|---|---|---|---|
| VCD | object-category | 1,190 | 0.166 |
| VCD | non-object | 58,724 | 0.070 |
| SID | object-category | 1,142 | 0.346 |
| SID | non-object | 58,333 | 0.224 |

**Confirmed real, dataset-wide, category-level effect** (1.5-2.4x bigger
$d$ for object words than everything else, over tens of thousands of
candidate instances). But here is the correction: substitute $d_t(w)$ into
the contrastive formula —
$$C_t(w) = 2E_t(w) - A_t(w) = E_t(w) + \big(E_t(w) - A_t(w)\big) = E_t(w) + d_t(w)$$
**A bigger $d$ makes $C_t(w)$ bigger, not smaller.** So the table above does
not say "object logits get pushed down" — it says **object-word candidates
get amplified upward more than filler words do.** That's actually the
mechanism working as intended: object identity needs visual detail to
confirm, so a degraded/restricted view genuinely is less confident about
object claims than about filler words ($A_t$ lower, gap $d_t$ bigger), and
the formula amplifies exactly in proportion to that gap. The design intent
of VCD/SID is "big expert-vs-amateur gap = evidence of real visual
grounding = amplify it" — and for the object-word category as a whole,
that is what's happening.

**The problem is what the amplification can't tell apart.** Split
object-candidates further by whether the object is actually real
(in the image's true COCO ground truth) or hallucinatable (not),
regardless of whether the candidate was ever chosen:

| method | candidate type | n | avg $d$ (amplification size) | avg raw $E$ (expert logit alone) |
|---|---|---|---|---|
| VCD | real-candidate | 803 | 0.178 | 14.565 |
| VCD | hallucinatable-candidate | 387 | 0.142 (*smaller* — backwards) | 13.375 |
| SID | real-candidate | 817 | 0.325 | 14.723 |
| SID | hallucinatable-candidate | 325 | 0.400 (bigger, but only slightly) | 13.413 |

The amplification size barely differs between real and hallucinatable
objects (VCD: actually backwards; SID: right direction, small margin). The
amateur can't distinguish true object claims from false ones — it's just a
noisier view of the *same* scene, so of course it doesn't know something
the expert doesn't.

**So where does any real/fake distinction come from at all?** Look at the
last column: the *expert alone*, with zero contrastive math, already rates
hallucinatable candidates about 1.2-1.3 points lower than real ones
(13.4 vs 14.6-14.7). That's a real, pre-existing, if weak, signal — nothing
to do with VCD/SID. And because APC survival depends **only on $E_t$**
(restated from the notation above: $A_t$ plays no role in the survival
test), this pre-existing expert gap is the *only* place a real/fake
distinction can enter the surviving-candidate pool. The amateur's
amplification, applied only to whichever object-shaped candidates already
survived, adds a roughly similar-sized bonus to real and hallucinatable
alike — so it doesn't add any further discrimination on top of what the
expert already had by itself.

**In plain English, the full claim:** *VCD and SID correctly detect "this
is an object claim, which needs visual confirmation" and amplify it — but
they cannot detect "and this specific object claim happens to be false,"
because the amateur is just a blurrier version of the same scene, not an
oracle that knows the ground truth. Any apparent hallucination-suppression
effect comes entirely from the expert's own pre-existing (and only weakly
discriminative) confidence gap between real and fake objects, which decides
survival before the amateur is even consulted — the amateur's contribution
afterward is category-wide amplification that treats real and hallucinated
objects almost identically.*

**Have I tested this, or is it a story?** Tested, and reproducible: loaded
`outputs/{vcd,sid}/img_*.json` (already-saved `top_k` lists from the
original run, no model rerun needed), matched candidate token text against
`chair.mscoco_objects` for the object/non-object split, and against
`chair.imid_to_objects[image_id]` for the real/hallucinatable split. Ns:
1,190 / 1,142 object-candidate instances (VCD/SID) out of ~60,000 total
candidates checked, across all 2,245 decoding steps in the dataset.

This single mechanism explains everything else in this document:

- Why the effect is present but weak/inconsistent between methods (SID:
  present but small; VCD: backwards) — any hallucination-suppression
  side-effect only shows up to the extent the expert's own pre-existing gap
  happens to matter, and that varies by method and by example.
- Why real- and hallucinated-mention *density* drop together (§3/Claim 2)
  — amplification hits the whole object category, not selectively the
  wrong members of it.
- Why, when the mechanism does flip a decision, it substitutes one
  hallucinated object for another rather than correcting to the truth
  (§4/Claim 4) — amplifying "motorcycle" alongside "car" doesn't help the
  model prefer the true one; whichever survivor has the bigger $C_t$ wins,
  real or not.
- Why APC alone reproduces the full pipeline's output 92-94% of the time
  (§2/Claim 1) — survival was already decided by $E_t$ alone; the
  amateur's amplification, even where real, mostly just scales the already-
  surviving candidates without changing which one had the biggest margin.

---

## 1. Amateur vs. expert top-10 overlap — computed over EVERY decoding step

**Question:** does the amateur model's own top-10 predicted tokens (over
the full 32,000-word vocabulary) overlap with the expert's own top-10, and
does that overlap differ at hallucination-prone steps vs. everywhere else?
(First measured only at object-mention steps; redone here over **all**
2,245 decoding steps in the dataset, including function words, punctuation,
number-words and colour-words — not just object mentions.)

**Method:** every step of every VCD/SID caption was replayed with the exact
same KV-cache generation loop as the original run, at each step computing
`torch.topk(expert_logits, 10)` and `torch.topk(amateur_logits, 10))`
independently over the full vocabulary (never filtered through each other's
ranking), and counting the overlap.

**Result (n=2,245 steps):**

| category | n | mean top-10 overlap | % steps where final CD output != plain expert-greedy ("flip rate") |
|---|---|---|---|
| neither (generic/function words) | 1,981 | 9.32/10 | 6.6% |
| real object | 176 | 9.09/10 | 6.8% |
| hallucinated object | 36 | 9.08/10 | **13.9%** |
| number word | 30 | 9.20/10 | 20.0% |
| colour word | 22 | 8.46/10 | 22.7% |

By method:

| method | category | n | mean overlap | flip rate |
|---|---|---|---|---|
| VCD | hallucinated | 21 | 9.10 | **19.0%** |
| VCD | real | 90 | 9.30 | 3.3% |
| VCD | neither | 1,000 | 9.39 | 6.2% |
| SID | hallucinated | 15 | 9.07 | 6.7% |
| SID | real | 86 | 8.87 | 10.5% |
| SID | neither | 981 | 9.25 | 6.9% |

**Answering your question directly: yes, this is now over everything** —
every token in every caption, not a subsample. Overlap is high (~90%) across
the board, and hallucinated-object steps are *not* meaningfully lower-overlap
than generic steps (9.08 vs 9.32) — confirming finding is robust, not an
artifact of small sample size.

**But the flip-rate breakdown reveals VCD and SID are doing genuinely
different things** (see §4 below): VCD's flip rate is ~3x higher at
hallucination steps than everywhere else (19.0% vs 6.2%/3.3%). SID's is not
(6.7% vs 6.9%) — for SID, divergence from plain greedy is statistically
indistinguishable at hallucination points vs. everywhere else. Colour-words
show the highest flip rates of any category for both methods (up to 27%,
see raw data) — plausibly because colour is a genuinely low-level,
image-degradation-sensitive property, unlike object identity which is driven
more by scene-level language priors.

*(Correction to an earlier over-generalization: the very first thing shown —
the "handbag" example with near-identical expert/amateur logits — was one
data point. This 2,245-step full replay is the properly verified version.)*

---

## 2. The generative analogue of the paper's core mechanism: APC vocabulary collapse

**This is the direct extension of the paper's central discriminative
finding into the generative setting they never tested.**

The paper's Category-B test showed that the Adaptive Plausibility Constraint
*alone* (no contrastive subtraction) recovers most of VCD/SID's POPE
improvement, because APC forces sampling into quasi-greedy behavior. Tested
the same mechanism here, over all 2,245 generative decoding steps:

| method | avg tokens surviving APC (of 32,000) | median survivors | % steps with ≤2 survivors | % steps where full CD+APC output == plain expert-greedy |
|---|---|---|---|---|
| VCD | 2.38 | **2** | 68.9% | **93.5%** |
| SID | 2.29 | **2** | 70.1% | **92.4%** |

**Why this happens, mechanistically:** APC's cutoff is
`log(beta) + max(expert_logit)` — a function of **the expert model's own
logits only**. The amateur model plays no role in defining the cutoff. So
regardless of what the contrastive subtraction (`(1+alpha)*expert -
alpha*amateur`) computes, APC discards every token except the handful the
expert *already* considered near-certain — a median of exactly 2 out of
32,000. The amateur's only remaining influence is to occasionally reorder
which of those ≤2 survivors wins; it can never introduce a genuinely
different, better-grounded candidate, because the candidate pool was fixed
by the expert before the amateur's opinion was consulted at all.

**Testable claim:** *"On CHAIR-style generation, APC collapses the
effective decoding vocabulary to a median of 2 tokens per step out of
32,000, determined solely by the expert model. As a direct consequence, the
full VCD/SID pipeline reproduces plain expert-only greedy decoding in
92-94% of all decoding steps. This is the generative-domain analogue of the
paper's POPE finding that APC alone (without any real contrastive signal)
explains most of the apparent improvement — and it is, if anything, more
severe here (median 2 survivors vs. the paper's reported ~1.5-2.5 percentage
point gain from APC alone on POPE)."*

---

## 3. Testing an actual PBA/OLM-equivalent for captioning — and getting an honest negative result

You asked for a real analogue of PBA/OLM: a crude, vision-unrelated
intervention that reproduces VCD/SID's apparent CHAIR improvement, proving
the improvement isn't really about hallucination. The natural candidate is
**caption brevity** (VCD/SID captions average 110-114 tokens vs. greedy's
118.5). Tested directly: take greedy's actual caption and just *truncate*
it to VCD/SID's length — zero vision mechanism involved.

| variant | avg real mentions | avg hallucinated mentions | avg total | hallucination fraction |
|---|---|---|---|---|
| greedy (full) | 6.10 | 1.30 | 7.40 | 17.6% |
| greedy truncated to VCD's length | 5.90 | 1.30 | 7.20 | **18.1%** |
| greedy truncated to SID's length | 5.90 | 1.30 | 7.20 | **18.1%** |
| VCD (actual) | 5.50 | 1.10 | 6.60 | 16.7% |
| SID (actual) | 5.60 | 0.90 | 6.50 | 13.8% |

**Result: truncation alone does NOT reproduce the improvement** — it
slightly *worsens* the hallucination fraction (18.1% vs 17.6%), because
truncation removes real mentions (which accumulate throughout the caption)
without touching hallucinated ones at all. So "just talks less" (a literal
length effect) is not the spurious mechanism, unlike PBA/OLM which *did*
directly reproduce POPE's improvement. This is a genuinely useful negative
result — it rules out the simplest possible mirage explanation.

**Refined test — mention density (mentions per 100 tokens), across the full
caption rather than just truncating it:**

| method | avg length (tokens) | real mentions / 100 tok | hallucinated mentions / 100 tok |
|---|---|---|---|
| greedy | 118.5 | 5.15 | 1.10 |
| VCD | 113.8 | 4.83 (**-6.2%**) | 0.97 (**-11.8%**) |
| SID | 110.7 | 5.06 (**-1.7%**) | 0.81 (**-26.4%**) |

**This is where VCD and SID split into two distinct behaviors:**

- **VCD**: real-mention density and hallucinated-mention density drop by a
  *similar, modest* proportion (-6.2% vs -11.8%) throughout the entire
  caption, not just at the end. This is consistent with VCD making the
  model generally vaguer/more repetitive/less willing to commit to specific
  object names — a genericity effect, not a discrimination improvement. The
  hallucination *fraction* (finding in §3 above, 17.6%→16.7%) barely moves,
  which is the signature of "talks less specifically," not "sees better."
- **SID**: real-mention density is nearly unchanged (-1.7%) while
  hallucinated-mention density drops substantially (-26.4%). This is *not*
  explained by SID's amateur/expert divergence being concentrated at
  hallucination steps (§1 showed it isn't — 6.7% flip rate at hallucinated
  steps vs 6.9% at generic steps, statistically indistinguishable). So
  SID's more targeted reduction is real in this sample but its *mechanism*
  remains an open question — a plausible hypothesis is that attention
  restricted to 72/576 random visual patches systematically depresses
  logits for specific object nouns (which need spatial/visual grounding)
  more than for generic continuations, independent of whether that object
  happens to be a hallucination or not; this would produce exactly the
  observed pattern without requiring the amateur to "detect" hallucination
  at all. This hypothesis is stated here but not yet independently verified
  and should be tested before being asserted as fact.

---

## 4. The "whack-a-mole" substitution pattern

**Method:** compared the set of hallucinated objects (per CHAIR, against
real COCO ground truth) in greedy vs. VCD vs. SID captions, per image.

| image | greedy hallucinated | VCD hallucinated | SID hallucinated |
|---|---|---|---|
| 724 | motorcycle, person | **fire hydrant, handbag** | person |
| 776 | (none) | person | (none) |
| 1425 | cup, donut, **spoon** | donut, **fork** | donut, **fork** |
| 1584 | backpack, car, truck | backpack, **traffic light** | backpack, truck |
| 2473 | (none) | backpack | (none) |
| 6763 | bottle, chair | bottle, **car**, chair | bottle, chair |

**Aggregate across all 10 images:**

- Greedy: 10 distinct hallucinated objects total
- VCD: 11 distinct hallucinated objects total — **removed 6** of greedy's
  hallucinations but **added 7 new ones** (net *worse* by count)
- SID: 7 distinct hallucinated objects total — removed 4, added 1 (net
  better by count)

Image 1425 is the cleanest example: the model doesn't actually know there's
a utensil in that part of the image; greedy guesses "spoon," VCD/SID guess
"fork" instead — same underlying visual uncertainty, different specific
wrong word. Image 724 is starker: VCD doesn't fix "motorcycle/person," it
hallucinates a completely unrelated fire hydrant and handbag instead.

**Paper-worthy claim:** aggregate CHAIR scores can show apparent improvement
while masking *substitution* rather than *correction*. A method can score
better on CHAIR_i purely by chance-swapping which wrong object appears in
each caption, without grounding the caption in the image any better.

---

## 5. A hallucination axis CHAIR cannot see at all: counting/cardinality

Cross-checked number-words ("five teddy bears") against the object's
*actual instance count* in COCO ground truth (CHAIR only checks
presence/absence, never cardinality).

**Image 776 (ground truth: 3 teddy bears):** greedy says "five." **VCD says
"five." SID says "five."** All three methods make the identical counting
error — CD provides zero correction, because contrastive decoding only
contrasts *whether* an object is present, never *how many*. Full table of
every number-word + nearby-object pair checked across all 10 images is in
the appendix at the end of this document. Only 2 exact matches out of ~18
checked pairs, uniformly across all three methods.

**Note for methodology:** this is a blind spot in the evaluation metric
itself, not just the mitigation method — a paper's CHAIR numbers can look
unchanged (or even improve, per finding §3) while counting behavior stays
exactly as wrong as ever.

---

## Proposed novel claims for the paper

Five distinct, falsifiable claims, each backed by a specific measurement
above, organized from most mechanistic/certain to most exploratory. Claim 0
is the headline — it subsumes and mechanistically explains Claims 2-4.

**Claim 0 (the unifying mechanism — this is the headline claim):** *Writing
$C_t(w) = 2E_t(w) - A_t(w) = E_t(w) + d_t(w)$ where $d_t(w)=E_t(w)-A_t(w)$,
VCD/SID's amateur models produce a roughly uniform, category-level
amplification ($d_t$, which *increases* $C_t$) on object-noun candidates
(1.5-2.4x larger $d_t$ than on non-object candidates, measured over 60,000+
candidate-token instances across every decoding step) — correctly detecting
that object claims need visual confirmation the degraded amateur can't
provide as well. This amplification is not hallucination-selective —
splitting object candidates by ground truth shows it is applied almost
equally (VCD) or only mildly more (SID, ~23% relatively) to hallucinatable
vs. real candidates, because the amateur is just a noisier view of the same
scene and has no independent access to the ground truth. What actually
drives any apparent hallucination suppression is that the expert model's
own raw logits $E_t$, with no contrastive intervention at all, are already
~1.2 points lower on average for hallucinatable candidates than real ones
(13.4 vs 14.6) — a real but weak pre-existing discrimination signal. Because
APC's survival cutoff (Claim 1) is defined purely by $E_t$ ($A_t$ has no
role in survival at all), this pre-existing expert gap is the only place a
real/fake distinction can enter the surviving pool — the amateur's
subsequent amplification then applies a similar-sized bonus to whichever
object-shaped candidates already survived, without further discriminating
between them. The apparent hallucination-suppression effect is therefore a
side-effect of a non-selective amplification multiplying against the
expert's own pre-existing (and only weakly discriminative) signal — not the
amateur model detecting or correcting hallucination.* (§0, n=60,000+
candidate instances across all 2,245 steps — the strongest-evidence claim
in this document)

**Claim 1 (mechanistic, applies to both methods, extends the Mirage
paper's core finding to generation):** *APC's cutoff is a pure function of
the expert model's own logits. On CHAIR-style generation this collapses the
candidate pool to a median of 2 tokens out of 32,000 on 69-70% of all
decoding steps, and causes the full contrastive-decoding pipeline to
reproduce plain expert-only greedy decoding 92-94% of the time. The amateur
model's structural role is confined to breaking ties among tokens the
expert already favors — it cannot inject a genuinely different, better-
grounded candidate.* (§2, n=2,245 steps, very strong evidence)

**Claim 2 (VCD-specific — genericity, not discrimination; now explained by
Claim 0):** *VCD's reduction in absolute hallucinated-object count is
substantially explained by a general reduction in object-mention density
affecting real and hallucinated content by a similar proportion (-6.2% vs
-11.8%), not by improved discrimination between them — the hallucination
fraction of total mentions barely changes (17.6%→16.7%). A literal
caption-truncation control rules out simple brevity as the mechanism. Claim
0 explains why: VCD's object-candidate amplification is applied almost
equally to real and hallucinatable candidates (0.178 vs 0.142), so it
boosts commitment to specific objects generally rather than discriminating
between true and false ones.* (§3, n=10 images; direction is consistent,
magnitude needs the full 500-image set to be a headline number)

**Claim 3 (SID-specific — now partially explained, not fully open):** *SID
shows a more targeted reduction in hallucinated- vs real-object density
(-26.4% vs -1.7%). Claim 0 provides a partial mechanism: SID's
object-candidate amplification, unlike VCD's, does skew mildly toward
hallucinatable candidates (0.400 vs 0.325) rather than backwards, and SID's
expert model shows the same ~1.2-point pre-existing confidence gap between
hallucinatable and real candidates as VCD's does. Whether this small skew
plus the pre-existing expert signal fully accounts for the -26.4%/-1.7%
density asymmetry, or whether something else (e.g. the specific structure
of attention restricted to 72/576 random visual patches) contributes
additionally, is not yet fully decomposed — a good target for a follow-up
regression (density change vs. amplification-skew vs. pre-existing
expert-gap, per image) rather than a fully open question.* (§0 + §3,
exploratory but now has a concrete next experiment rather than being
unexplained)

**Claim 4 (whack-a-mole, qualitative but vivid, now explained by Claim 0):**
*Even when contrastive decoding does diverge from the expert's raw choice
at a hallucination step (happening ~3x more for VCD at hallucination points
specifically: 19.0% vs 3.3-6.2% baseline), the result is a different
hallucinated object, not a grounded one, in the observed cases — e.g.
spoon→fork, motorcycle/person→fire hydrant/handbag. Claim 0 explains why:
non-selective amplification of one object candidate doesn't help the model
prefer the true one over another guess — whichever surviving candidate ends
up with the largest $C_t$ wins, real or not.* (§1 + §4, small-N qualitative
support, good for a concrete example box in the paper)

**Recommended next step:** Claims 0, 1 and 4's mechanisms (candidate-level
amplification by category, APC collapse rate, flip-rate-by-category) only
need the mention-level and top-K summary (not full per-token JSON), so
they're cheap to compute at the full 500-image CHAIR scale — that would
turn Claims
0, 1 and 2 into headline quantitative results for the paper.
Claim 3 needs a new, targeted experiment (e.g. checking whether SID's
logit-suppression under attention restriction correlates with a word being
a content noun vs. function word, independent of hallucination status) — I
have not run this yet.

---

## Appendix: full number-word / ground-truth-count table

```
img=724  greedy 'two'  -> 'motorcycle': claimed=2 actual=0  OVERCOUNT   (hallucinated obj)
img=724  greedy 'one'  -> 'person':     claimed=1 actual=0  OVERCOUNT   (hallucinated obj)
img=724  vcd    'one'  -> 'handbag':    claimed=1 actual=0  OVERCOUNT   (hallucinated obj)
img=724  sid    'one'  -> 'person':     claimed=1 actual=0  OVERCOUNT   (hallucinated obj)
img=776  greedy 'five' -> 'teddy bear': claimed=5 actual=3  OVERCOUNT
img=776  greedy 'one'  -> 'teddy bear': claimed=1 actual=3  UNDERCOUNT
img=776  greedy 'four' -> 'teddy bear': claimed=4 actual=3  OVERCOUNT
img=776  greedy 'one'  -> 'bed':        claimed=1 actual=1  MATCH
img=776  vcd    'five' -> 'teddy bear': claimed=5 actual=3  OVERCOUNT
img=776  sid    'five' -> 'teddy bear': claimed=5 actual=3  OVERCOUNT
img=1584 greedy 'two'  -> 'car':        claimed=2 actual=0  OVERCOUNT   (hallucinated obj)
img=1584 vcd    'two'  -> 'bus':        claimed=2 actual=3  UNDERCOUNT
img=1584 vcd    'one'  -> 'bus':        claimed=1 actual=3  UNDERCOUNT
img=2473 greedy 'one'  -> 'person':     claimed=1 actual=4  UNDERCOUNT
img=2473 greedy 'two'  -> 'people':     claimed=2 actual=4  UNDERCOUNT
img=6040 vcd    'one'  -> 'car':        claimed=1 actual=1  MATCH
img=6763 greedy 'two'  -> 'chair':      claimed=2 actual=0  OVERCOUNT   (hallucinated obj)
img=6763 sid    'one'  -> 'person':     claimed=1 actual=3  UNDERCOUNT
```

**Small-sample caveat (applies to the whole document):** all per-mention
statistics are drawn from 10 images. Claim 1 (n=2,245 steps) is robust.
Claims 2/4 (n=10 images, tens of mentions) are solid mechanistic case
studies with clear, falsifiable direction, but should be scaled to the full
500-image CHAIR set for a quantitative headline number. Claim 3 needs a new
experiment before it can be asserted at all.
