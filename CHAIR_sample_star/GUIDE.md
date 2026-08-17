# GUIDE: what Sample* is, why beta=0.1, and what's reused vs. new

Read `README.md` first for the step-by-step instructions. This file
explains the methodology and the decisions behind it.

## 1. What Sample* actually is

Contrastive decoding's score is `C = (1+alpha)*E - alpha*A`, gated by the
Adaptive Plausibility Constraint (APC): candidates with `E < log(beta) +
max(E)` are excluded before the final token is chosen. That gate is
computed from the raw EXPERT logits only -- it never looks at the amateur
branch.

**Sample\*** asks: what if you keep the APC gate but throw away the
amateur branch and the contrastive subtraction entirely -- just sample
from the plain, gated expert distribution? If Sample* alone recovers most
of what VCD/SID appear to achieve, that's evidence the "contrastive"
mechanism itself (the amateur branch, the actual visual/attention
degradation) isn't doing the real work -- the plausibility mask is. This
is exactly the ablation Yin et al.'s "Mirage of Performance Gains" runs in
their Table 6, alongside plain (unmasked) Sample, VCD(sample), and
SID(sample).

Concretely, per decoding step:
```
cutoff = log(beta) + max(E)
scored = E.clone()
scored[E < cutoff] = -inf
chosen = multinomial(softmax(scored))       # or argmax for --decode greedy
```
No `A`, no `alpha`, no amateur forward pass at all -- Sample* is a
single-branch method, computationally cheap (same cost as the sibling
package's unmasked baseline).

## 2. Why cd_beta = 0.1 here, not 0.2

The sibling `CHAIR_sampling+ICD` package used `cd_beta=0.2` for
Sample*'s counterpart (the unmasked "Direct Sampling" baseline didn't use
beta at all, but VCD/SID/ICD there used 0.2) -- chosen specifically to
match the ORIGINAL greedy-baseline `CHAIR_analysis` experiment's own
convention, so that package's whole point (isolating the effect of
greedy-vs-sampling) held every other variable fixed against already-reported
numbers.

This package's comparison is different: Sample* vs. VCD vs. SID vs. ICD,
all under sampling, asking how much the amateur branch adds ON TOP of the
plausibility mask. For THAT comparison to be meaningful, beta must be the
SAME across all four conditions, and the natural choice is VCD's own paper
default (0.1), which is also what the Mirage paper itself uses in Table 6.
Using 0.2 here would still be internally consistent, but wouldn't match
the reference table this package is designed to replicate/extend -- so
0.1 was chosen deliberately, not by oversight. See `generate_llava.py`'s
module docstring for where this is enforced in code.

**Do not silently change this back to 0.2** -- that would defeat the
purpose of building a separate package instead of just adding a fourth
condition to the sibling one.

## 3. What's reused unmodified vs. what's new

**Reused, unmodified in spirit (only cd_beta/mode-name changed):**
- VCD's noise-diffusion amateur branch, SID's paper-correct attention-based
  least-attended-10% token selection (not random), ICD's 5-separate-pass
  disturbance-prompt methodology -- all identical mechanism to the sibling
  package, just at beta=0.1 instead of 0.2.
- Qwen's KV-cached `Branch` class and its whole design rationale (the
  `model.model.rope_deltas` shared-mutable-state bug, and the fix of
  computing each branch's multimodal position delta independently) --
  this is **entirely beta-independent**, a pure decoding-mechanics
  correctness question, so the sibling package's fp32-verified fix applies
  here unchanged. `verify_qwen_cache.py` is re-run here (see `qwen/README.md`)
  as a sanity check on this deployment, not because the math needed
  re-deriving.
- The whole analysis toolkit (`agreement_analysis.py`, `contrastive_analysis.py`,
  `plot_d_histograms.py`, `plot_d_per_step.py`, `verify_outputs.py`) --
  schema-identical captures mean these run unmodified, just pointed at
  `sample_star` instead of `baseline` as the single-branch condition name.

**New in this package:**
- `--mode sample_star` in both `generate_llava.py` and `qwen/generate_qwen.py`:
  applies the APC mask to the single expert branch (the sibling package's
  `--mode baseline` had no mask at all). Captures an `expert_top_survives_apc`
  field per step (didn't exist in the sibling's baseline capture, since
  there was no mask to report survival against).
- `modal_app.py`: new app name (`cd-rethink-sample-star`), but points at
  the SAME Modal Volume as the sibling package (`cd-rethink-sampling-icd-vol`)
  to reuse already-downloaded weights, with all outputs namespaced under
  `outputs/sample_star/` so the two packages' results can never collide on
  the shared volume.

## 4. Capture file schema

Same shape as the sibling package, with one addition for the single-branch
condition:

```jsonc
{
  "image_id": 139,
  "method": "sample_star",       // "sample_star" | "vcd" | "sid" | "icd" (see prompt_key)
  "decode": "sample",
  "caption": "...",
  "steps": [
    {
      "step": 0,
      "chosen_id": 1234,
      "apc_cutoff": -8.17,                        // ALWAYS present now, even for sample_star
      "expert_top_ids": [...], "expert_top_logits": [...],
      "expert_top_survives_apc": [true, false, ...],  // sample_star only -- whether each
                                                        // top-10 candidate's raw E passed the mask
      "amateur_top_ids": [...], "amateur_top_logits": [...],   // vcd/sid/icd only
      "cd_top_ids": [...], "cd_top_pre_apc_logits": [...],     // vcd/sid/icd only
      "cd_top_survives_apc": [...]                              // vcd/sid/icd only
    }
  ]
}
```

## 5. Compute budget

Same per-run cost as the sibling package (Sample* is single-branch, same
cost class as its unmasked baseline; VCD/SID/ICD unchanged): ~15 GPU-hr for
the LLaVA leg (1 seed, full 5-prompt ICD), similarly for Qwen. Total for
both models, 1 seed each: ~30 GPU-hr, well inside the budget already
established for the sibling package.

## 6. What NOT to do

- Don't run the full 500-image jobs without explicit go-ahead -- this
  package is prepared and smoke-tested only until told otherwise.
- Don't change cd_beta back to 0.2 (see Section 2).
- Don't skip `verify_qwen_cache_run.py` before Qwen generation, even
  though the cache math is already proven correct elsewhere -- re-verify
  on this deployment, don't assume.
- Don't re-download the ~30GB of model weights if the sibling package's
  volume already has them -- `modal_app.py`'s `setup_assets()` is
  idempotent and will just print "already present" and skip.
