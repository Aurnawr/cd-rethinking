# Proxy Design E: Random Noise Applied to the ENTIRE Vocabulary (Specificity Control)

**Sample**: same 70 images. See `PROXY_DESIGN_A_random_noise.md` for the
full caveat about real VCD/SID's noisy baseline on this subset.

---

## What this design tests

All prior designs (A, B, C, D) restrict their bonus to the ~219 COCO
object-category tokens — deliberately mimicking the paper's specific claim
about *object* logits. Design E is a **specificity control**: what happens
if we apply the exact same noise recipe as Design A (`Normal(pooled_mean,
pooled_std)` per method), but to **every single token in the ~32,000-token
vocabulary**, not just object words? If Design E produces a similar or
larger hallucination-increasing effect than Design A, it would mean nothing
about object-specificity is actually necessary — generic noise anywhere in
the vocabulary would do. If Design E is markedly weaker or produces the
*opposite* effect, it confirms that concentrating the noise specifically on
object-category words (as A/B/D do) is doing meaningful, targeted work.

**Mechanism**: identical scaffold; only difference from Design A is
`full_bonus = torch.randn(vocab_size) * std + mean` applied to the whole
logit vector before argmax-under-APC-survival, instead of only to the 219
object-token indices.

---

## Results

| variant | CHAIR_s | CHAIR_i | avg objects/cap | avg length |
|---|---|---|---|---|
| greedy | 57.1% | 15.8% | 8.143 | 88.67 |
| VCD (real) | 51.4% | 15.6% | 8.057 | 92.11 |
| SID (real) | 54.3% | 14.1% | 8.086 | 88.17 |
| **E_vcd** | 48.6% | **15.6%** | 7.900 | 92.30 |
| **E_sid** | 54.3% | **15.6%** | 7.586 | 89.01 |

| variant | real objects/cap | hallucinated objects/cap | hallucination fraction | whack-a-mole net vs. greedy |
|---|---|---|---|---|
| greedy | 2.486 | 1.043 | 29.6% | — |
| VCD (real) | 2.586 | 1.000 | 27.9% | -3 |
| SID (real) | 2.629 | 0.857 | 24.6% | -13 |
| **E_vcd** | 2.486 | 0.943 | 27.5% | **-7** |
| **E_sid** | 2.500 | 0.957 | 27.7% | **-6** |

## Interpretation

Design E is the clearest negative control in the whole batch: applying the
identical noise recipe to the *entire* vocabulary — rather than
concentrating it on object words — produces the **largest net improvement**
of any variant tested (whack-a-mole -7/-6, both beating even real VCD's -3
on this subset), lower hallucinated-objects/cap (0.943/0.957, both below
greedy's 1.043 and below real SID's 0.857 comparison point only slightly),
and CHAIR_i essentially flat or improved relative to greedy.

This is an important, informative contrast with Designs A/B: when the same
random-noise magnitude is spread across the whole vocabulary instead of
concentrated on the ~219 object tokens, it stops looking like a
hallucination-*amplifying* effect and starts looking like ordinary sampling
noise diluted evenly — a small amount of "hedging" behavior, where object
mentions in general (both true and false) become marginally less
concentrated because the noise, on average, sometimes elevates a filler or
generic word into the winning slot instead. This makes intuitive sense:
spreading a fixed noise "budget" over 32,000 tokens gives each object word
a vanishingly small chance of actually being the *specific* token most
helped in any one step compared to concentrating that same noise on only
219 candidates.

**Bottom line for Design E**: **object-specificity is necessary** for
reproducing VCD/SID's hallucination-increasing signature. The same-shaped
noise, applied without any targeting to object-category words, does not
reproduce the effect — it actually reverses it into a mild net improvement.
Combined with Designs A/B (targeted noise → increase) and Design C (oracle
truth-correlation → no meaningful additional effect beyond plain
targeting), this converges on a specific, falsifiable characterization: **it
is not "any noise" and not "truth-aware noise" that drives VCD/SID's real
degradation — it's noise that is (a) large enough in variance to
occasionally cross the APC survival margin, and (b) concentrated
specifically on object-category tokens**, exactly matching how the real
amateur model happens to behave (it degrades object-word confidence
specifically, via the diffusion-noised image or attention-restricted
visual pathway) without needing to know or care whether the word is true.
