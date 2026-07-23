# Proxy Design D: Fixed Per-Word Constant (No Randomness At All)

**Sample**: same 70 images. See `PROXY_DESIGN_A_random_noise.md` for the
full caveat about real VCD/SID's noisy baseline on this subset.

---

## What this design tests

Designs A-C all inject randomness at generation time. Design D asks a
different question: **is even that randomness necessary, or is it enough
that different WORDS get different (but fixed, deterministic) treatment?**
Every object word gets its own precomputed constant bonus — e.g. "sandwich"
always gets its own fixed number, "dog" gets a different fixed number —
computed once as that word's own average `d` value across the entire
500-image run (falling back to the pooled mean for words with fewer than 20
observed instances, to avoid overfitting to small-n words). No randomness,
no image-dependence, no truth-awareness: the same word gets the exact same
bonus on every single image, every single time it comes up as a candidate.

This isolates "word identity effects" from "per-instance variance effects."
If Design D matches Designs A/B/real VCD/SID, it would mean VCD/SID's real
effect is really just "some words are systematically boosted more than
others" (a vocabulary-level property) rather than genuine per-instance
noise. If it looks more like the flat-constant proxy (no effect), it
confirms per-instance variance — not word identity — is what matters.

**Parameters**: `per_word_bonus` dict from `proxy_stats.json`
(`compute_proxy_stats.py`, `word_d` grouped means, ≥20-instance threshold).

---

## Results

| variant | CHAIR_s | CHAIR_i | avg objects/cap | avg length |
|---|---|---|---|---|
| greedy | 57.1% | 15.8% | 8.143 | 88.67 |
| VCD (real) | 51.4% | 15.6% | 8.057 | 92.11 |
| SID (real) | 54.3% | 14.1% | 8.086 | 88.17 |
| **D_vcd** | 55.7% | **16.0%** | 8.314 | 89.21 |
| **D_sid** | 57.1% | **16.4%** | 8.471 | 89.10 |

| variant | real objects/cap | hallucinated objects/cap | hallucination fraction | whack-a-mole net vs. greedy |
|---|---|---|---|---|
| greedy | 2.486 | 1.043 | 29.6% | — |
| VCD (real) | 2.586 | 1.000 | 27.9% | -3 |
| SID (real) | 2.629 | 0.857 | 24.6% | -13 |
| **D_vcd** | 2.514 | 1.086 | 30.2% | **+3** |
| **D_sid** | 2.514 | 1.100 | 30.4% | **+4** |

## Interpretation

Design D produces a small but consistent hallucination-increasing effect —
CHAIR_i for both variants (16.0%, 16.4%) sits slightly above greedy (15.8%)
and both real VCD/SID numbers on this subset, and whack-a-mole is
positive for both (+3, +4), meaning it adds more new false objects than it
removes. This is a real, measurable — if modest — effect, clearly different
from the flat-constant proxy's null result.

This is informative: it means a fixed, **word-identity-only** signal (no
per-instance randomness, no per-image adaptivity) is enough to produce
*some* hallucination increase, because some individual words (e.g. ones
that happen to have an unusually large average `d` in the 500-image data,
possibly due to how often they co-occur with visually ambiguous scenes) get
a permanently elevated bonus that occasionally exceeds the APC survival
margin regardless of context. However, the effect (+3/+4 net whack-a-mole)
is clearly smaller than Design B's empirical resampling (+7/+1) or Design
A's random-noise VCD variant (CHAIR_i 17.1%) — suggesting **word identity
alone explains only part of the story; per-instance random variance (as in
A/B) contributes an additional, independent effect on top of it.**

**Bottom line for Design D**: word-level fixed biases do contribute to
VCD/SID's hallucination-increasing signature, but they are not the dominant
driver — per-instance randomness (Designs A/B) produces a larger and more
consistent effect. The true mechanism is likely a combination of both: some
words are intrinsically more "amplifiable" than others, but the bulk of the
effect still comes from step-to-step noise, not fixed word identity.
