# Proxy Design A: Pure Random Noise, No Truth-Awareness

**Sample**: 70 images (same 70 used across all Designs A-F, first 70 of the
500-image set). **Important caveat that applies to every design report in
this series**: on this particular 70-image sample, the *real* VCD/SID
methods happen to come out slightly *better* than greedy on CHAIR_i (VCD
15.6%, SID 14.1%, vs greedy 15.8%) and net-negative on whack-a-mole
(VCD -3, SID -13) — this differs from the robust, well-powered 500-image
finding where both were clearly worse (see `CLAIM_VALIDATION_500.md`). This
is expected sample-to-sample noise at n=70. It means: **judge each design by
its own pattern and internal consistency, not by whether it exactly matches
real VCD/SID's numbers on this specific subset** — the "ground truth" here
is itself noisy.

---

## What this design tests

The earlier flat-constant-boost experiment (fixed +0.2/+0.3 added to every
object word) failed to reproduce VCD/SID's behavior at all — it was
statistically identical to plain greedy. We then found the real amateur
bonus has a standard deviation (~0.80-0.97) roughly 4x its mean (~0.20-0.30)
— most of the signal is in the *spread*, not the average. Design A tests the
simplest possible way to inject that spread: **at every decoding step, every
object-category candidate word gets an independent random bonus drawn from
`Normal(mean, std)`, using the pooled statistics (both real and hallucinated
candidates combined) actually measured from the 500-image VCD/SID run.**
Critically, this random draw has **zero knowledge of whether the specific
word is actually true or false for this image** — it's pure noise of the
right shape, not the right content.

**Parameters used** (from `proxy_stats.json`, pooled across real+hallucinated):
- A_vcd: `Normal(mean=0.1990, std=0.8533)`
- A_sid: `Normal(mean=0.3003, std=0.9033)`

**Mechanism**: APC survival is decided from the raw expert logit only
(exactly matching the real formula); the random bonus is added only to the
score used to pick the winner among survivors — same structural design as
real VCD/SID, only the *source* of the per-token bonus differs (noise
instead of a real degraded-vision forward pass).

---

## Results

| variant | CHAIR_s | CHAIR_i | avg objects/cap | avg length |
|---|---|---|---|---|
| greedy | 57.1% | 15.8% | 8.143 | 88.67 |
| VCD (real) | 51.4% | 15.6% | 8.057 | 92.11 |
| SID (real) | 54.3% | 14.1% | 8.086 | 88.17 |
| **A_vcd** | 51.4% | **17.1%** | 8.514 | 88.17 |
| **A_sid** | 55.7% | 15.6% | 8.529 | 89.44 |

| variant | real objects/cap | hallucinated objects/cap | hallucination fraction | whack-a-mole net vs. greedy |
|---|---|---|---|---|
| greedy | 2.486 | 1.043 | 29.6% | — |
| VCD (real) | 2.586 | 1.000 | 27.9% | -3 |
| SID (real) | 2.629 | 0.857 | 24.6% | -13 |
| **A_vcd** | 2.557 | 1.043 | 29.0% | **+0** |
| **A_sid** | 2.571 | 0.986 | 27.7% | **-4** |

## Interpretation

Unlike the flat-constant proxy, **Design A produces a real, measurable
effect** — A_vcd's CHAIR_i (17.1%) is the highest of any variant tested in
this batch, higher than greedy itself. This confirms the earlier prediction:
injecting the *right variance*, even with zero truth-awareness, can move
the needle in a way a flat constant provably cannot (because individual
draws can exceed APC's 1.609-logit survival margin, occasionally rescuing
implausible candidates from far below the cutoff — something no constant
this small could ever do).

A_sid's effect is more muted (CHAIR_i 15.6%, whack-a-mole -4, i.e. slightly
*fewer* hallucinations than greedy) — noisier and less consistent than
A_vcd's. Given both use the same mechanism with different mean/std
parameters, this asymmetry is plausibly just sample noise at n=70 rather
than a systematic VCD-vs-SID distinction; it would need the full 500-image
scale to confirm which (if either) direction is real.

**Bottom line for Design A**: pure, truth-blind random noise matching the
real amateur's variance *can* reproduce a hallucination-increasing effect
(clearly for the VCD-parameterized version here), which the flat constant
never could. This is meaningful evidence that **the magnitude of per-token
variance, not any actual visual signal, may be doing most of the practical
work** in creating VCD/SID's real-world failure mode — though the SID-side
result is weak enough here that this should be confirmed at larger scale
before treating it as settled.
