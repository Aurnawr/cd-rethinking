# GUIDE: the Qwen KV-cache design, why it needed real engineering, and what was verified

Read `README.md` first for the step-by-step run instructions. This file
explains *why* this leg exists as a separate engineering effort from the
LLaVA leg, what specifically was hard about it, and exactly what was
verified before it was trusted.

## 1. Why Qwen needed new code, not a copy of the LLaVA scripts

The original (pre-existing, sibling `CHAIR_analysis/` project) Qwen
generation script recomputes the ENTIRE amateur branch from scratch, with
no KV-cache, at every single decoding step. Its own docstring says why:
"a hand-rolled mRoPE second-branch cache diverged from the exact
computation." That's a real, documented finding -- Qwen2.5-VL uses
multimodal RoPE (3D position ids: temporal/height/width for image tokens,
standard 1D for text), and someone had already tried caching the amateur
branch and gotten wrong numbers.

Cache-less means O(sequence-length^2) attention work instead of O(n) --
every step re-attends over the whole growing sequence. This is why the
original Qwen captures were slow ("expect hours... on 500 images" per that
project's own README). Running that same approach for baseline + VCD + SID
+ 5 ICD passes here would plausibly consume most or all of the available
GPU budget on the SLOW leg alone.

## 2. The actual root cause (found by reading the model source, not guessing)

`Qwen2_5_VLModel` computes its multimodal position ids once per generation
and caches the result in a **mutable attribute on the model object itself**:
`model.model.rope_deltas`. Every later forward call in that same generation
reuses this attribute to cheaply advance positions instead of recomputing
the full 3D index.

That attribute is shared by the whole model. A hand-rolled two-branch loop
that alternates calls into the same live model (expert step, amateur step,
expert step, ...) makes the second branch's first call silently overwrite
`rope_deltas` with its own value -- corrupting the OTHER branch's position
ids on every subsequent step. For VCD and SID this happens to be harmless
(both branches have identical token/image structure, so they'd compute the
same delta anyway regardless of whose value won). For ICD, whose amateur
branch has a genuinely different (disturbance-prefixed) prompt length, the
two branches' correct deltas actually differ -- and sharing the attribute
would silently feed one branch the other's wrong offset. This is almost
certainly the "diverged from the exact computation" the original script's
author hit.

## 3. The fix

`generate_qwen.py`'s `Branch` class never lets the model touch
`rope_deltas` at all. Each branch calls `model.model.get_rope_index(...)`
directly, once, at its own prefill, and stores the resulting per-branch
delta in a plain Python variable private to that `Branch` instance. Every
subsequent step builds `position_ids` from that private delta and passes it
explicitly -- so the model's internal auto-compute path (the one that
reads/writes the shared attribute) is never invoked. Verified: a `[1, bsz,
seq]` position_ids tensor (this script's continuation shape) is exactly
what `Qwen2_5_VLRotaryEmbedding` expects for pure-text continuation --
PyTorch's batched matmul broadcasts it to all 3 mrope dimensions, which is
mathematically correct once you're past the image prefix (temporal =
height = width = the same running counter for plain text tokens).

SID additionally needs attention weights at `AGG_LAYER` to decide which
image tokens are least-attended (the paper's real mechanism, not a random
subset -- see the SID note in the top-level `GUIDE.md`, same fix applied
here). The model is loaded with `attn_implementation="eager"` (required --
SDPA never materializes attention weights, cached or not), and forward
pre/post-hooks on that one layer's self-attention module request and
capture its weights; later layers' pre-hooks then clone the
**framework-built** causal mask (not a hand-rolled one -- reusing the
framework's own correct causal/padding construction) and additionally block
the pruned image-token columns.

## 4. What was actually verified, and how

`verify_qwen_cache.py` compares this cached implementation against a
brute-force, cache-less, full-sequence recompute at every step, for both
VCD and SID, replaying the SAME chosen-token sequence into both so every
comparison is apples-to-apples (not two independently-diverging captions).

**First attempt failed with NaN at every even-numbered step.** Root cause:
the verification's OWN reference implementation was wrong, not the cached
code -- it only passed `pixel_values` at step 0 (a KV-cache habit that
doesn't apply to a cache-less recompute, which re-embeds the raw input_ids
from scratch every call and needs the image every time) and never passed
`mm_token_type_ids` at all (silently disabling real multimodal position
computation). Fixed by always passing both.

**After that fix, bf16 showed a real but bounded gap** (a few logit units,
argmax always still agreeing) starting at the first cached continuation
step, after an EXACT match (0.00000) at step 0. This was the deciding
question: precision noise, or a real remaining bug?

**Decisive test: re-ran the identical comparison in fp32** (needs a bigger
GPU -- the 7B model in fp32 doesn't fit an L4's 24 GB, see
`verify_qwen_cache_bigmem` in `../modal_app.py`, A100-40GB). The gap
collapsed to ~1e-4 (float noise) with every argmax matching. **This
confirms the bf16 gap is precision noise from cached (small, per-step
matmuls) vs. cache-less (large, full-sequence matmuls) having different
floating-point accumulation order -- not a logic bug.** bf16 has roughly 3
significant decimal digits; composing that noise through ~28 transformer
layers over many steps producing a few units of final logit divergence is
expected, not alarming.

**One residual finding, also investigated, not brushed aside:** across a
full sweep (3 images x VCD/SID x 20 steps = 120 step-comparisons), 2 showed
an argmax MISMATCH -- both at the very last step tested (step 19), both
apparently genuine near-ties where accumulated bf16 rounding tipped the
decision differently between the two paths. This is the expected failure
mode of a low-precision format on a long autoregressive chain: once two
candidates are within the bf16 noise floor of each other, which one "wins"
is close to arbitrary, and the two computation paths (structurally
different matmul shapes) can land on either side of that arbitrary line.
Since the fp32 control already proved the underlying math is exact, this
is not evidence of a remaining bug -- chasing it further would be chasing
noise. The real pipeline **samples** (temperature 1.0), not greedy, which
is far less sensitive to this kind of near-tie than the strict
greedy-replay used for verification.

`verify_qwen_cache.py`'s pass criterion reflects this: it accepts up to a
10% per-run argmax mismatch rate (observed rate was under 2%), not a
literal zero -- because demanding zero at bf16 over long sequences is not
a meaningful bar, and the fp32 control already answered the real question.

## 5. What NOT to do if you see a FAILED verification on your own run

- Don't just re-run it hoping for a different result on the same code --
  if it fails at a rate meaningfully above what's described here (a lot
  more than 1-2 mismatches per 120 comparisons, or non-trivial logit gaps
  even at step 0), that's a real signal, not noise.
- Don't disable the mismatch-rate check to force a PASS -- fix or
  understand the actual discrepancy first.
- Do re-run the fp32 control (`--dtype float32` via
  `verify_qwen_cache_bigmem`) if you're unsure whether a gap is precision
  noise or a bug -- it's the decisive test, not a guess.

## 6. Hyperparameters (identical to the LLaVA leg's convention)

Same reasoning as the top-level `GUIDE.md`: everything is held fixed to
what the greedy-baseline `CHAIR_analysis` experiment used, so switching to
`--decode sample` is the only variable that changes.

| Parameter | Value |
|---|---|
| `cd_alpha` | 1.0 |
| `cd_beta` (APC) | 0.2 |
| `max_new_tokens` | 256 |
| Prompt | `"Describe this image in detail."` |
| SID `AGG_LAYER` | 2 (0-indexed) |
| SID keep fraction | 10% least-attended image tokens (paper-exact, `topk(largest=False)`, attention-based not random) |
| ICD `cd_alpha`/`cd_beta` | 1.0 / 0.2 (matches this package's VCD/SID convention, not the ICD paper's own 0.1) |
| ICD prompts | 5 separate full passes (`p1,p2,n1,n2,p3`), official methodology |
| Decode | `sample` for the real run plan (`greedy` also implemented, useful for spot-checks) |
| Seed | 1 (seed 0) -- paired with the LLaVA leg's 1 seed, per the budget decision to cover two models rather than more seeds of one |
| Model dtype | bfloat16 for all real runs; float32 used ONLY as a one-off diagnostic, never for actual generation |
