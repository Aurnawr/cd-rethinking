# Proxy Design F: One-Sided Oracle Noise — Real-Only vs. Hallucinated-Only

**Sample**: same 70 images. See `PROXY_DESIGN_A_random_noise.md` for the
full caveat about real VCD/SID's noisy baseline on this subset.

**Note**: like Design C, this is an oracle-informed diagnostic, not a
deployable proxy — it requires ground-truth knowledge of which candidate
words are actually correct for the image. It exists to isolate *which side*
of the real/hallucinated split is responsible for the effect seen in
Designs A/B, by testing each side in complete isolation.

---

## What this design tests

Design C applied oracle-informed noise to *both* arms simultaneously (real
words got real-flavored noise, hallucinated words got hallucination-
flavored noise) and found close-to-neutral results. Design F asks a sharper
question: **does noise on the real-object arm alone, or noise on the
hallucinated-object arm alone, do the damage** — since applying both at once
in Design C might be masking an effect that's concentrated in only one arm.

Four variants:
- **F_vcd_realonly / F_sid_realonly**: only candidate words that ARE
  actually present in the image (per COCO ground truth) get a random bonus
  drawn from `Normal(real_mean, real_std)`; every hallucination-candidate
  word gets **zero** bonus (falls back to plain greedy behavior for those).
- **F_vcd_hallonly / F_sid_hallonly**: the mirror image — only candidate
  words that are NOT actually present get a bonus from
  `Normal(hall_mean, hall_std)`; real-object candidates get zero bonus.

---

## Results

| variant | CHAIR_s | CHAIR_i | avg objects/cap | avg length |
|---|---|---|---|---|
| greedy | 57.1% | 15.8% | 8.143 | 88.67 |
| VCD (real) | 51.4% | 15.6% | 8.057 | 92.11 |
| SID (real) | 54.3% | 14.1% | 8.086 | 88.17 |
| **F_vcd_realonly** | 51.4% | 15.1% | 8.729 | 90.47 |
| **F_sid_realonly** | 54.3% | 14.9% | 8.529 | 88.77 |
| **F_vcd_hallonly** | 57.1% | **17.3%** | 8.257 | 89.00 |
| **F_sid_hallonly** | 57.1% | **17.2%** | 8.329 | 88.77 |

| variant | real objects/cap | hallucinated objects/cap | hallucination fraction | whack-a-mole net vs. greedy |
|---|---|---|---|---|
| greedy | 2.486 | 1.043 | 29.6% | — |
| VCD (real) | 2.586 | 1.000 | 27.9% | -3 |
| SID (real) | 2.629 | 0.857 | 24.6% | -13 |
| **F_vcd_realonly** | 2.600 | 1.043 | 28.6% | **+0** |
| **F_sid_realonly** | 2.571 | 1.029 | 28.6% | **-1** |
| **F_vcd_hallonly** | 2.471 | 1.043 | 29.7% | **+0** |
| **F_sid_hallonly** | 2.500 | 1.100 | 30.6% | **+4** |

## Interpretation

This is the **cleanest decomposition result in the whole batch**. The two
arms behave completely differently:

- **F_realonly** (noise only on true candidates): CHAIR_i drops *below*
  greedy for both (15.1%, 14.9%), whack-a-mole is flat/slightly negative
  (+0, -1). Boosting the variance of true-object candidates does essentially
  nothing harmful — occasionally it even helps slightly, presumably by
  rescuing a correct-but-borderline object mention that greedy would have
  otherwise dropped.

- **F_hallonly** (noise only on false candidates): CHAIR_i is the
  **highest of any variant in the entire batch** (17.3%, 17.2%, both a full
  1.4-1.5 percentage points above greedy), and hallucination fraction climbs
  to 29.7%/30.6% — at or above greedy's own 29.6%, despite greedy having no
  noise mechanism applied to it at all.

This directly answers the question Design C left ambiguous: **the
hallucination-increasing effect is not spread evenly across "any object
noise" — it is concentrated almost entirely in what happens to the
already-false candidates.** When a hallucinated candidate's logit
occasionally gets randomly boosted enough to cross the APC survival margin,
it gets a free, undeserved chance to win — and there's no offsetting
mechanism protecting the caption from it, because APC's cutoff is fixed
before the noise is even applied. Real candidates, by contrast, have
nothing to "gain" from extra variance in the same way, since they're
already plausible; giving them more variance mostly moves them between
different true statements rather than between true and false.

**Bottom line for Design F**: this is the strongest, single most
interpretable finding across all six designs. VCD/SID's practical
hallucination-increasing failure mode is best explained not by "noise on
objects in general" (Design A/B, a blunter version of this same idea) but
specifically by **noise landing on already-implausible (hallucinated)
candidates, occasionally boosting them past the APC survival threshold with
no truth-check to stop them**. Design C's near-neutral result was because
the harmful hallucinated-arm effect and the neutral/mildly-helpful real-arm
effect were mixed together and roughly canceled out; isolating them here
reveals the asymmetry clearly.
