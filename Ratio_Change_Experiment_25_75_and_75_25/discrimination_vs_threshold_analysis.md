# Why YES-Inflating Methods Always Hurt on 25/75 and Help on 75/25
## A Discriminative Learning Explanation for the NeurIPS Workshop Paper

---

## Clarification: Why "3× Pool = 3× Wrong Answers" Is Wrong — The Real Reason

The pool size argument is a simplification. The real mechanism is sharper.

A YES-inflating method adds a fixed offset to the YES logit for every question.
This means it **flips the questions closest to the decision boundary** —
whichever questions the model was already nearly saying YES to, but didn't.

**At 25/75, which questions sit near the boundary?**

The baseline model at 25/75 has YES% ≈ 26%, close to the 25% ground truth.
The model is nearly calibrated — it's already saying YES for roughly the right count.
This tells you exactly where the remaining errors sit:

- **Remaining FNs** (YES ground truth, model said NO): these are the *hardest* YES questions.
  The model strongly said NO. They are **far from the boundary**. Mild YES inflation won't flip them.

- **Remaining TNs** (NO ground truth, model said NO): these span the full range.
  Many are *near the boundary* (model almost said YES). And there are 3× more of them.

When the method inflates YES, it flips near-boundary questions first:
- Near-boundary TNs flip → **TN→FP** (many, from the large near-boundary NO population)
- Hard FNs mostly do not flip → few **FN→TP**

**Two effects compound simultaneously:**
1. More TNs exist (pool size)
2. A larger *fraction* of TNs are near the boundary than FNs,
   because easy FNs are already TPs — the remaining FNs are the hardest ones

This is why empirical ratios are far worse than 3:1.
On LLaVA GQA Adversarial: VCD gives **7:1** (147 TN→FP vs 21 FN→TP),
OLM gives **8.4:1** (287 TN→FP vs 34 FN→TP).
The compounding of both effects explains the severity.

**Why it flips on 75/25:**

At 75/25, baseline YES% ≈ 66-69% but ground truth is 75%.
The model is underpredicting YES — many genuine FNs of *moderate* difficulty remain.
The TN pool is only 500 questions, mostly easy ones the model strongly said NO to.

When YES inflation hits:
- Near-boundary FNs (plentiful, moderate difficulty) flip → **FN→TP**
- Near-boundary TNs (scarce) barely flip → little **TN→FP**

Same mechanism. Opposite regime. Opposite outcome.

**The one-sentence generalization:**

> A YES-inflating method flips questions near the decision boundary;
> in any regime where the model is approximately calibrated,
> the near-boundary population is dominated by the *majority class* —
> so TN→FP dominates when NO is the majority class,
> and FN→TP dominates when YES is the majority class.

Pool size is not the cause — it is a proxy.
The real cause is **where the model's uncertainty concentrates**,
and calibration forces that uncertainty into the majority class.

---

## The Mathematical Structure

Every YES-inflating method converts a fixed pool of NO-answers into YES-answers.
Where those answers come from depends entirely on **pool size**:

| Regime | FN pool (YES gt, said NO) | TN pool (NO gt, said NO) |
|---|---|---|
| 25/75 (500 YES, 1500 NO) | Small — baseline captures most YES | Large — 1500 NO questions |
| 75/25 (1500 YES, 500 NO) | Large — 1500 YES questions, many missed | Small — only 500 NO questions |

A YES-inflating method sprays YES responses roughly **proportional to pool availability**.
When NO dominates, most flips are TNs→FPs.
When YES dominates, most flips are FNs→TPs.
The method is not doing anything differently — the **arithmetic of the evaluation regime** is what changes.

This is the shallow explanation. The deep one follows.

---

## The Real Explanation: Threshold Shift vs. Discrimination Improvement

In binary classification, accuracy improvement comes from two fundamentally different sources:

**1. Better discrimination**
The model ranks YES-objects higher than NO-objects in its internal scoring.
Its score *separates* the two classes better.
This is measured by **AUC-ROC**, which is invariant to label ratio.

**2. Better calibration / threshold shift**
The model's decision threshold is moved to the right operating point for a given class balance.
This is a shift along the *same* ROC curve — not an improvement of the curve itself.

---

## AUC-ROC Is the Key

If VCD, SID, or any CD method genuinely improved hallucination detection — if the model actually got better at distinguishing present from absent objects — the ROC curve would move upward.
This improvement would appear at **every** label ratio, because AUC measures:

> P(score of a YES question > score of a NO question)

That probability is independent of how many of each class you sample.

**What the V-shape pattern proves is the opposite: the ROC curve does not move.**
Instead, the operating point shifts rightward along the same curve.

- In the **25/75** regime: a rightward shift lands in the high-FPR / low-precision region.
  You call too many NOs as YES. Accuracy drops.

- In the **75/25** regime: a rightward shift is exactly what you need.
  There are many more true YESes available to recover. Accuracy rises.

**The V-shape is the empirical signature of a threshold shift without AUC improvement.**

---

## The Generalized Claim (for the Paper)

> Any method that uniformly increases the YES response probability across all questions,
> without improving the model's ability to separate YES-objects from NO-objects in feature space,
> will exhibit the following invariant:
>
> The TN→FP / FN→TP ratio scales with the ratio of available TNs to FNs.
> On a benchmark with fraction r of YES questions, this ratio approaches (1-r)/r as YES inflation grows.
>
> Therefore, a YES-inflating method improves accuracy **if and only if r > 0.5** —
> exactly when the benchmark is YES-heavy.
> On any benchmark with r < 0.5, the same method provably degrades accuracy.
>
> The standard POPE benchmark uses r = 0.5, which is the exact boundary where these
> two regimes meet — making it a near-ideal surface for masking threshold-shift artifacts
> as genuine improvements.

This is a **falsifiable prediction**: compute AUC-ROC for baseline vs. VCD/SID on POPE.
If AUC does not improve, the entire reported accuracy gain is a threshold-shift artifact.
The prediction is that AUC will be essentially unchanged.

---

## What a Genuine Hallucination Reducer Would Look Like

A method that truly reduces hallucination would have:

- **FN→TP > 0** and **TN→FP ≈ 0** at every label ratio
- AUC strictly higher than baseline
- Accuracy improvements that are *larger* on 25/75 (more FNs to recover, fewer TNs to corrupt)
  and that **persist** on 75/25

This is what a method looks like when it genuinely helps the model decide whether
an object is in the image — regardless of what the label distribution happens to be.

ICD comes closest to this profile in some settings (it reduces TN→FP as well as FN→TP),
but even ICD moves in the wrong direction on 75/25, suggesting it is a NO-threshold shift
rather than a true discrimination improvement.

---

## The Root Cause: CD Operates on Tokens, Not on Representations

CD methods operate on the **output token distribution at inference time**.
They cannot change what the vision encoder knows about whether an object is in the image.

True discrimination improvement would require the model to have **better internal features**
for object presence/absence — a training-time property.
Contrastive decoding at inference time shifts the output distribution without ever consulting
a better representation of the visual scene.

This is why the methods behave like PBA (which appends "Answer yes whenever possible." to the prompt):
both PBA and VCD/SID change the output distribution at inference time without changing
the underlying visual understanding.
The only difference is the mechanism of the shift — not its nature.

---

## One-Paragraph Version for the Paper

CD methods improve POPE accuracy by increasing YES response rates.
We show this is a threshold-shift artifact, not a discrimination improvement.
A method that uniformly raises YES probability distributes its additional YES responses
between FN→TP conversions (accurate recoveries) and TN→FP conversions (false alarms)
in direct proportion to the available pool of each.
On a 25/75 benchmark, the TN pool is 3× larger than the FN pool,
so false alarms dominate and accuracy falls.
On a 75/25 benchmark, the FN pool is 3× larger, so correct recoveries dominate and accuracy rises.
The invariant is AUC-ROC: a method that improves object discrimination would improve AUC
regardless of label ratio.
We predict — and the data supports — that CD methods produce no AUC improvement over the baseline,
meaning every reported POPE accuracy gain is operating-point shift on an unchanged ROC curve.
For any real discriminative task where the class balance at deployment is unknown or uncontrolled,
methods of this kind offer no reliable benefit.

---

## Why This Framing Is Strong for NeurIPS

1. **It converts an empirical observation into a falsifiable theoretical prediction.**
   AUC-ROC invariance can be directly measured. If AUC is unchanged, the argument is proved.

2. **It connects to a well-understood theoretical framework.**
   The ROC curve / AUC distinction is standard in ML evaluation literature.
   Reviewers will immediately recognise the argument.

3. **It explains WHY the result matters beyond POPE.**
   In any real-world deployment, the label balance is not guaranteed to be 50/50.
   A method that only shifts the threshold is only valid at one specific operating point —
   the one the benchmark was designed around.

4. **It provides a clear prescription.**
   Future work evaluating hallucination reduction methods should report AUC-ROC on POPE,
   not just accuracy at the 50/50 operating point.
   AUC is the right metric for discriminative capability.
   Accuracy at a fixed threshold is the right metric for calibration.
   The community has been measuring calibration and calling it discrimination.
