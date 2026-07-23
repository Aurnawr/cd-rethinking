# Proxy Design C: Oracle-Informed Bidirectional Noise (Upper-Bound Check)

**Sample**: same 70 images. See `PROXY_DESIGN_A_random_noise.md` for the
full caveat about real VCD/SID's noisy baseline on this subset.

**Important**: unlike Designs A, B, D, E, F_realonly/F_hallonly, **Design C
is NOT a deployable proxy** — it needs to know, at generation time, whether
a candidate word is actually true or false for the image (the COCO ground
truth). No real inference-time method could ever have this information. It
exists purely as an **upper/lower reference point**: "if the model somehow
had a perfect truth-oracle telling it which noise direction to apply, how
much could that alone move CHAIR?"

---

## What this design tests

If real VCD/SID's amateur bonus happened, by some accident of the
degraded/restricted forward pass, to correlate with ground truth (i.e.
systematically apply the "hallucination-flavored" noise distribution to
false words and the "real-flavored" noise distribution to true words), that
alone — with no other mechanism — could shift CHAIR scores. Design C tests
this directly: **the bonus is drawn from `Normal(real_mean, real_std)` if
the candidate word is actually correct for this image (per COCO ground
truth), or `Normal(hall_mean, hall_std)` if it is not** — using the
oracle-split statistics from `proxy_stats.json`:
- vcd: real=`Normal(0.1754, 0.8023)`, hall=`Normal(0.2334, 0.9215)`
- sid: real=`Normal(0.3273, 0.8552)`, hall=`Normal(0.2614, 0.9672)`

Note the real/hallucinated means are already very close to each other (the
whole reason the earlier flat-constant proxy failed) — so Design C is a
weak oracle at best; it mainly differs from Design A only in variance
per-arm, not in a strong "boost the truth, suppress the lie" signal.

---

## Results

| variant | CHAIR_s | CHAIR_i | avg objects/cap | avg length |
|---|---|---|---|---|
| greedy | 57.1% | 15.8% | 8.143 | 88.67 |
| VCD (real) | 51.4% | 15.6% | 8.057 | 92.11 |
| SID (real) | 54.3% | 14.1% | 8.086 | 88.17 |
| **C_vcd** | 58.6% | 15.2% | 8.557 | 88.67 |
| **C_sid** | 51.4% | 15.6% | 8.500 | 89.64 |

| variant | real objects/cap | hallucinated objects/cap | hallucination fraction | whack-a-mole net vs. greedy |
|---|---|---|---|---|
| greedy | 2.486 | 1.043 | 29.6% | — |
| VCD (real) | 2.586 | 1.000 | 27.9% | -3 |
| SID (real) | 2.629 | 0.857 | 24.6% | -13 |
| **C_vcd** | 2.600 | 1.029 | 28.3% | **-1** |
| **C_sid** | 2.629 | 1.029 | 28.1% | **-1** |

## Interpretation

Even with a **perfect oracle** splitting noise by ground-truth correctness,
Design C's effect is essentially neutral — CHAIR_i for both variants
(15.2%, 15.6%) sits close to greedy (15.8%) and close to real VCD/SID, and
whack-a-mole is mildly negative (-1/-1, i.e. a very slight net improvement)
for both. This is the **opposite** of what a "the real mechanism is secretly
truth-correlated" hypothesis would predict — if that were true, C should
show a strong, clean *improvement* (oracle correctly boosting true words
and suppressing false ones), clearly beating both Design A/B's truth-blind
noise and even real VCD/SID.

It doesn't. The reason is visible directly in the input statistics: the
real-vs-hallucinated mean gap (real=0.175 vs hall=0.233 for VCD; real=0.327
vs hall=0.261 for SID) is tiny relative to the shared standard deviation
(~0.8-0.97) — i.e., even a perfect oracle splitting by ground truth barely
changes the noise recipe, because real and hallucinated candidates get
*almost the same* real-world bonus distribution in the actual VCD/SID data.
This is direct confirmation of the core finding already established in
`CLAIM_VALIDATION_500.md`: **the amateur's bonus does not meaningfully
distinguish true from false objects** — it is close to indifferent to
correctness, which is precisely why VCD/SID's hallucination-increasing
effect cannot be a "smart" truth-tracking mechanism.

**Bottom line for Design C**: this rules out the hypothesis that some
hidden truth-correlation in the amateur model explains VCD/SID's behavior.
The oracle-informed version performs no better (and arguably slightly worse
than some of Designs A/B), confirming that whatever signal exists in the
real mechanism is not meaningfully about truth — it's dominated by
undirected variance, as Designs A and B already suggested.
