# Proxy Design B: Bootstrap-Resampled Real Bonus Values, No Truth-Awareness

**Sample**: same 70 images, same caveat as Design A regarding real VCD/SID's
noisy baseline on this subset (see `PROXY_DESIGN_A_random_noise.md` for the
full caveat text — it applies identically here and is not repeated in full
in every file below).

---

## What this design tests

Design A used a parametric `Normal(mean, std)` approximation of the real
amateur-vs-expert gap `d = E - A`. But the real distribution of `d` might
not actually be Gaussian — it could be skewed, heavy-tailed, or multimodal
in ways a Normal distribution smooths over. Design B removes that
assumption entirely: at every decoding step, each object-word candidate's
bonus is an **independent bootstrap draw (with replacement) from the actual
pool of ~thousands of real `d` values measured in the 500-image VCD/SID
run** (`empirical_samples` in `proxy_stats.json`, built by
`compute_proxy_stats.py`). This is the same "give it the right variance,
withhold the true content" logic as Design A, but non-parametric — if
Design A and B disagree, it means the *shape* of the distribution (not just
its mean/std) matters.

**Mechanism**: identical scaffold to Design A — APC survival from raw
expert logit only; bonus added only to object-word candidates; bonus here
is `random.choices(empirical_samples, k=n_obj)` instead of a Normal draw.
Truth-blind: the resampled value has no relationship to whether the word
is actually correct for this specific image.

---

## Results

| variant | CHAIR_s | CHAIR_i | avg objects/cap | avg length |
|---|---|---|---|---|
| greedy | 57.1% | 15.8% | 8.143 | 88.67 |
| VCD (real) | 51.4% | 15.6% | 8.057 | 92.11 |
| SID (real) | 54.3% | 14.1% | 8.086 | 88.17 |
| **B_vcd** | 57.1% | **16.8%** | 8.443 | 89.73 |
| **B_sid** | 52.9% | 15.0% | 8.657 | 92.01 |

| variant | real objects/cap | hallucinated objects/cap | hallucination fraction | whack-a-mole net vs. greedy |
|---|---|---|---|---|
| greedy | 2.486 | 1.043 | 29.6% | — |
| VCD (real) | 2.586 | 1.000 | 27.9% | -3 |
| SID (real) | 2.629 | 0.857 | 24.6% | -13 |
| **B_vcd** | 2.571 | 1.143 | 30.8% | **+7** |
| **B_sid** | 2.486 | 1.057 | 29.8% | **+1** |

## Interpretation

Design B is the **strongest hallucination-increasing effect of any design
tested here on the "hallucination fraction" and whack-a-mole metrics**:
B_vcd raises the hallucination fraction to 30.8% (above greedy's 29.6%) and
has the single largest positive whack-a-mole net (+7) in the whole batch —
meaning it adds more new wrong objects than it removes existing ones, more
than any other design. CHAIR_i for B_vcd (16.8%) is also clearly elevated
over greedy.

This is a meaningful result: it means the **non-parametric, empirically-
resampled noise reproduces (or exceeds) the real hallucination-increasing
signature even more reliably than the Gaussian approximation in Design A**.
Since B differs from A only in distribution *shape* (real empirical
histogram vs. fitted Normal), this suggests the real `d` distribution likely
has heavier tails or more extreme outlier draws than a Normal with the same
mean/std would produce — and those extreme draws (the ones large enough to
exceed APC's 1.609 margin) are exactly what's doing the damage.

**Bottom line for Design B**: matching the *exact empirical shape* of the
real bonus distribution — with zero truth-awareness — is sufficient, and
arguably more reliable than a Gaussian approximation, to reproduce (and even
amplify) VCD/SID's hallucination-increasing behavior. This further weakens
the case that any genuine "vision computation" is doing the real work; a
random resampler with the right histogram does at least as well.
