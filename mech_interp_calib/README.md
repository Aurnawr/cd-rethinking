# Calibration experiments: oracle calibration and scalar bias

Two experiments from the discriminative half of the CD-rethinking write-up, implemented once and
run on both model families.
Neither needs a GPU or model weights: both read only the stored per-sample logit-lens margins
under `results/<model>/`.

```bash
bash mech_interp_calib/run_calibration.sh                 # both models, all three methods
MODELS=qwen2.5-vl-7b bash mech_interp_calib/run_calibration.sh
```

Runtime is about four minutes for six (model, method) pairs on CPU; the bootstrap CIs dominate.
Requires `numpy`, `scikit-learn` and `matplotlib` and nothing else.
Outputs land in `results/calibration/`.

## Notation

For each question the expert and amateur passes give a Yes-No margin per layer, and CD's
per-unit-beta shift is their difference.

```
m_l   = max_Yes z_l - max_No z_l          the decision margin read off layer l
D_l   = m_expert_l - m_amateur_l          what CD adds:  m_CD = m_expert + beta * D
H     = object absent, model said Yes     hallucinations
T     = object present, model said Yes    correct positives
```

`D > 0` pushes the decision *toward* Yes, which on a hallucination is the wrong direction.
Selectivity is the AUC with which `-D` ranks `H` above `T`: 0.5 means the shift cannot tell a
hallucinated Yes from a correct one.

## Experiment 3: oracle calibration

A selectivity of 0.5 has two readings, and the experiment separates them.
Either CD's shift genuinely carries no hallucination-specific information, or the metric is too
insensitive to register selectivity even when it is present.

**Part A, the ceiling.**
Fit a presence probe on the per-layer margin vector of `H u T`, out-of-fold, 5-fold stratified.
With `p+` the predicted presence probability, the oracle shift is `D_oracle = -(1 - p+)`, a
suppression proportional to predicted absence: selective by construction.
Pushing it through the identical `rank_auc(-D, H)` gives the ceiling the instrument can register
on this data, 0.885 for LLaVA and 0.901 for Qwen.
Fitting on margins rather than hidden states keeps the ceiling conservative, since presence is
more decodable from the residual stream (Probe A reaches 0.99) than from one number per layer.

**Part B, the graded oracle.**
Inject a scaled selective component into CD's own shift and sweep its size:

```
D_s = D_CD + s * u_oracle,     u_oracle = standardize(D_oracle)
```

`u_oracle` has unit standard deviation, so `s` is in log-odds.
`s*` is the smallest `s` whose selectivity clears the detection threshold, defined as the upper
end of the 95% stratified-bootstrap CI of CD's *own* readout selectivity.
That is the level an injected component has to reach before the metric separates it from what CD
alone already produces, which makes `s*` a detection limit rather than a significance test
against 0.5.
Reporting `s*` against CD's mean `|D|` turns "CD is not selective" into a quantitative claim: CD
moves the margin 9 to 24 times harder than the smallest selective component this test would have
caught.

Supporting figure `fig_selectivity_ci_*.png` adds two robustness checks: per-layer bootstrap CIs,
and a *residualized* selectivity curve in which `-D` is ranked after regressing out the expert
margin.
That second curve matters because `H` and `T` differ in expert margin by construction, so a shift
that is merely a function of the margin would score above chance while carrying no
hallucination-specific information at all.

## Scalar-bias experiment

On a binary decision the entire contrastive term is one number per sample.
If that number carried hallucination-specific information, replacing it with a single constant
shared by all 9,000 questions would have to cost accuracy.
The sweep asks exactly that:

```
decision(b) = [ m_expert_readout + b > 0 ]
```

and compares its POPE accuracy against the realized contrastive decision
`[ m_expert + beta * D > 0 ]` on the identical questions and readout.
Four reference points are reported: the expert at `b = 0`, CD realized, `b_equiv` (CD's own mean
shift flattened to a constant) and `b*` (the accuracy-maximizing constant).

The `b = 0` point is not an approximation of the model's greedy decision, it *is* that decision:
`sign(m_expert_readout)` matches the recorded greedy prediction on 100% of samples in five of the
six runs and 99.98% in the sixth.

One caveat belongs in any write-up of this: the sweep isolates the contrastive term.
A deployed decoder also applies the adaptive plausibility constraint, which is amateur-free and
independently known to account for most of the reported gain.
The point here is precisely that what is *left* after removing APC is a translation, and a
translation is fully described by one scalar.

## Files

| file | role |
|---|---|
| `calib_common.py` | npz loading, `H`/`T` populations, tie-aware AUC, stratified bootstrap, out-of-fold probe, POPE scoring |
| `oracle_calibration.py` | Experiment 3, both parts, plus the per-layer CI and residualized curves |
| `scalar_bias.py` | the bias sweep, pooled and per split, with accuracy / F1 / precision / recall / yes-rate |
| `compare_models.py` | cross-model tables and the three side-by-side figures |
| `run_calibration.sh` | driver over every (model, method) pair that has stored margins |

## Reproduction fidelity against the published LLaVA figures

The scalar-bias experiment reproduces the published LLaVA-1.5-7B numbers exactly: expert 85.49%,
VCD realized 85.04%, `b*` = +0.84 at 86.86%, `b_equiv` = +1.23.
(The published `b*` reads +0.85; the sweep grid here is 0.02 wide.)

Experiment 3 reproduces the ceiling and the scale: oracle 0.885 against a published 0.89, CD
selectivity 0.523 against 0.52, mean `|D|` 3.64 against 3.64, detection threshold 0.554 against
0.553.
It differs on `s*`: 0.15 here against a published 0.10, so the detection factor comes out 24x
rather than 36x.
The gap is a property of the probe's calibration near `s = 0`, where the sweep is steepest, and
the exact figure depends on choices the original script did not record.
Both numbers support the same claim by an order of magnitude, and the more conservative one is
reported here.

## Findings

Full tables in `results/calibration/comparison.md`, figures alongside them.

**The instrument works on both models.**
The oracle ceiling is 0.885 on LLaVA and 0.901 on Qwen, and `s*` never exceeds 0.45 log-odds
while CD's own mean `|D|` runs from 0.18 to 5.73.
Every null below is a measurement, not a blind spot.

**No method suppresses the false Yes on either model.**
The mean shift on `H` at the readout is positive in five of six (model, method) pairs: LLaVA VCD
+3.32, LLaVA SID +1.61, Qwen VCD +2.14, Qwen SID +3.05, Qwen ICD +0.03.
The single negative is LLaVA ICD at -0.17 log-odds, against a margin of median magnitude 3.2,
with `T` suppressed too, so the `H`-vs-`T` differential is +0.11.
CD reinforces the hallucination it is sold as removing, on both model families.

**Qwen's higher selectivity is the boost-correct-positives artifact, not a correction.**
Qwen VCD reaches 0.805 selectivity where LLaVA VCD sits at 0.523, which looks like a real
cross-model difference until the class means are read: Qwen VCD moves `H` by +2.14 and `T` by
+5.00 log-odds.
The ranking is right and the intervention is backwards.
This is the same diagnosis the write-up already made for corrected SID on LLaVA (+1.61 on `H`
against +3.11 on `T`), now reproduced on a different model family and a different amateur.

**Residualizing kills every above-chance selectivity.**
Regressing the expert margin out of `D` before ranking drops LLaVA ICD from 0.639 to 0.467,
LLaVA SID from 0.691 to 0.492, Qwen VCD from 0.805 to 0.558 and Qwen SID from 0.726 to 0.552.
What looked like selectivity is a function of the margin CD is added to, not of whether the
answer is a hallucination.

**One constant beats all three CD methods on both models.**
On LLaVA the best constant is `b*` = +0.84 for 86.86%, against a greedy 85.49% and realized CD of
85.04% (VCD), 85.04% (ICD) and 85.89% (SID).
On Qwen `b*` = +2.38 gives 90.02%, against a greedy 87.90% and realized CD of 88.30%, 87.16% and
89.10%.
The contrastive term is beaten by a single number in six of six cases, by 0.9 to 2.9 points.

**CD's mean shift, flattened to a constant, keeps most of what CD achieves.**
`b_equiv` reaches or exceeds the realized CD accuracy in four of six pairs: LLaVA VCD 86.48%
against 85.04%, LLaVA SID 86.43% against 85.89%, Qwen ICD 87.68% against 87.16%, Qwen SID 89.22%
against 89.10%.
The two exceptions are LLaVA ICD (84.93% against 85.04%, a 0.1 point gap) and Qwen VCD (87.34%
against 88.30%), where `b_equiv` is negative because the shift is negative on the No-answered
majority even though it is positive on `H u T`.

**The methods differ in size, not in kind.**
ICD's shift is an order of magnitude smaller than VCD's and SID's on both models (mean `|D|` 0.18
and 0.56 against 3.1 to 5.7), which is the structural cancellation the write-up predicts: ICD
keeps the clean image, so object presence is common-mode across its two branches and drops out of
the difference.
VCD and SID corrupt the image and produce large shifts, but those shifts land on `H` and `T`
alike.
