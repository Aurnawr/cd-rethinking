# GUIDE: methodology, dataset, schema, and rationale

Read `README.md` first for the step-by-step run instructions. This file
explains *why* the package is built the way it is, so you can judge any
result it produces (or debug a problem) without having to guess.

## 1. What question this answers

The original CHAIR study asked: under **greedy decoding**, does VCD/SID's
contrastive step actually change what the model says, or does it behave
like the amateur branch is barely doing anything (a "mirage")? That
experiment is `CHAIR_analysis/` (sibling directory) and already has
reported CHAIR-S/CHAIR-I numbers, agreement rates, and d = E-A
distributions for VCD and SID under greedy decoding, plus a Gaussian-noise
proxy comparison.

This package asks the SAME question under **direct sampling** instead of
greedy, and adds **ICD** (Instruction Contrastive Decoding) as a third
contrastive method, since it wasn't part of the original greedy-baseline
study. Sampling matters because greedy decoding is deterministic — a single
run — while sampling introduces its own randomness that interacts with
whatever contrastive decoding is doing. If CD's effect is real (not a
mirage), it should still show up as a d < 0 shift for hallucinated
candidates and a lower CHAIR score under sampling too, not just under
greedy.

## 2. Dataset — exact image identity

`image_ids_500.json` at the repo root is the **exact same 500 MSCOCO
val2017 image ids, in the exact same order**, as used in `CHAIR_analysis/`.
This is not a resample — it's a byte-identical copy of that file. Using the
same images means any CHAIR-S/I or d-distribution comparison between this
package's results and the original greedy-baseline results is a true
apples-to-apples comparison (only the decoding rule differs), not
confounded by a different image sample.

Do not regenerate or reshuffle this file. If you need to verify it's the
right one: it should contain exactly 500 integers, each an MSCOCO val2017
image id (e.g. `139`, `285`, `632`, ... decodes to filenames like
`000000000139.jpg` via `f"{id:012d}.jpg"`).

## 3. Hyperparameters — and why they're what they are

| Parameter | Value | Why |
|---|---|---|
| `cd_alpha` | 1.0 | Matches the original `CHAIR_analysis` greedy-baseline experiment and the VCD/SID papers' own default. |
| `cd_beta` (APC) | **0.2** | Matches `CHAIR_analysis`'s own convention (see its `README.md` / `common/generate_llava.py`), NOT the VCD/SID papers' more common 0.1, and NOT the 0.1 used in this project's separate `jaccard_experiment/` llava-bench work. This is deliberate: the point of this package is to isolate the effect of switching greedy→sampling, so every other knob is held at exactly what the already-reported greedy numbers used. If you want paper-default beta=0.1 numbers later, that's a separate, clearly-labeled run — don't silently change this constant, it breaks comparability with the existing results. |
| VCD `noise_step` | 500 | Same reasoning — matches `CHAIR_analysis`'s existing VCD captures, not the VCD paper's 900 (used elsewhere in this project for llava-bench). |
| `max_new_tokens` | 256 | Matches `CHAIR_analysis`. |
| Prompt | `"Describe this image in detail."` | Matches `CHAIR_analysis`'s CHAIR captioning prompt (not the llava-bench per-image questions used elsewhere in this project). |
| SID `ATTENTION_RANK` | 58 (10% of 576 image tokens) | Matches arXiv:2408.02032 exactly — the least-ATTENDED 10% of image tokens are kept for the amateur branch, computed via real attention weights (`topk(..., largest=False)`), NOT randomly. This was independently paper-verified earlier in this project's history; do not change it back to a random subset. |
| SID `AGG_LAYER` | 2 (0-indexed) | The paper's layer i=3 (1-indexed). |
| SID KV-cache | Used normally | The per-layer KV cache accumulates exactly as in any standard forward pass; the attention-based pruning decision is recomputed fresh at every decoding step from that step's query against the (cached) key sequence — this is already efficient and does not require disabling the cache. (Qwen is different — see §6.) |
| ICD `cd_alpha`/`cd_beta` | 1.0 / **0.2** | Deliberately matches this package's VCD/SID convention above, NOT the ICD paper's own 0.1 default (which was used correctly, separately, for this project's llava-bench ICD run). Internal consistency within this package takes priority over matching the ICD paper's exact number, since the goal here is a fair 3-way VCD/SID/ICD comparison under one shared recipe. |
| ICD disturbance prompts | 5 separate full passes (`p1,p2,n1,n2,p3`) | Verified against the official ICD repo (`github.com/p1k0pan/ICD`) earlier in this project: the official method runs each of the 5 prompts as its own complete pass over the dataset, never randomly mixed within one run. `generate_llava_icd.py --prompt-key` enforces this by construction. |
| Decode rule | **sample** (multinomial, temperature 1.0, from the post-APC-masked score) | This is the one deliberate change this whole package makes relative to `CHAIR_analysis`. `--decode greedy` is still implemented in both generation scripts (useful for a quick sanity check against the old numbers on a handful of images) but the real run plan (`spawn_jobs.py --full`) uses `sample` everywhere. |
| Seeds | 1 (seed 0) | Budget is deliberately spent covering a second MODEL (Qwen, added later) at 1 seed rather than extra seeds of LLaVA alone — a conscious choice given the fixed GPU-hour budget, not an oversight. |

## 4. The baseline is new — read this carefully

The original experiment's baseline was **plain greedy decoding** — a single
deterministic run, no seed needed (`captions_greedy.jsonl` in the original
package). That baseline cannot be reused here, because this package's whole
point is comparing methods **under sampling**, and a plain-sampling
baseline is itself stochastic — it needs a seed and can vary run to run,
exactly like VCD/SID/ICD now do.

`generate_llava.py --mode baseline --decode sample` is that new baseline:
single branch (expert only), no contrastive term, no APC mask — pure
temperature-1.0 sampling from the raw expert distribution, matching the
"Direct Sampling" convention already used elsewhere in this project
(`jaccard_experiment/inference/llava_bench_infer_sample.py`). Unlike the
old greedy baseline, it DOES capture the expert's own top-10 logits at
every step (the old greedy mode didn't bother, since there was nothing to
compare them against) — this package logs everything, on the theory that
storage is cheap and re-running is not.

## 5. Capture file schema

Every capture file (`outputs/captures/llava_<method>_seed<N>.jsonl`) is one
JSON object per line, one line per image:

```jsonc
{
  "image_id": 139,
  "method": "vcd",              // "baseline" | "vcd" | "sid" | "icd" (see prompt_key for which ICD pass)
  "decode": "sample",           // "greedy" | "sample"
  "prompt_key": "n2",           // ICD files only
  "caption": "A kitchen with a wooden table and...",
  "steps": [
    {
      "step": 0,
      "chosen_id": 1234,
      "apc_cutoff": -12.34,                       // capture/ICD only, not baseline
      "expert_top_ids": [1234, 5678, ...],         // top-10, always present
      "expert_top_logits": [3.21, 2.87, ...],
      "amateur_top_ids": [...],                    // capture/ICD only
      "amateur_top_logits": [...],
      "cd_top_ids": [...],                          // capture/ICD only
      "cd_top_pre_apc_logits": [...],               // = (1+alpha)*E - alpha*A, BEFORE the APC mask
      "cd_top_survives_apc": [true, false, ...]     // whether each cd_top_id's raw E passed the APC cutoff
    },
    ...
  ]
}
```

`chosen_id` is what was actually emitted at that step — under `--decode
sample`, this is a *sampled* token, so unlike the greedy captures it will
NOT always equal `expert_top_ids[0]` or `cd_top_ids[0]` even when the
branches agree closely; that's expected, not a bug. `verify_outputs.py`
checks the schema and the `cd_top_pre_apc_logits` arithmetic but does not
(cannot) check that sampling "looks right" beyond that — the only real
check for sampling correctness is re-running the same seed twice and
diffing (should be byte-identical; see below).

## 6. Compute budget

Measured throughput (from this project's own prior LLaVA runs, same
256-token / two-branch capture pattern): roughly **2 GPU-hr per 500-image,
1-seed, two-branch capture** (VCD, SID, or one ICD prompt pass), and
roughly **1 GPU-hr per 500-image, 1-seed, single-branch run** (baseline).

LLaVA leg, 1 seed, full 5-prompt ICD:

| Job | GPU-hr |
|---|---|
| baseline | ~1 |
| VCD | ~2 |
| SID | ~2 |
| ICD × 5 prompts | ~10 |
| **Total** | **~15** |

Out of a 35-40 hour budget, this leaves roughly 20-25 GPU-hr for the Qwen
leg described below.

## 7. The Qwen leg (do not start until told to)

A second phase adds `generate_qwen.py` / `generate_qwen_icd.py` (Qwen2.5-VL-7B-Instruct),
1 seed, mirroring the LLaVA plan above. **One important open issue to
resolve first, not assume away:** the original `CHAIR_analysis/README.md`
states that Qwen's SID amateur branch was deliberately run WITHOUT a
KV-cache, because a cached second branch didn't numerically match the
uncached computation under Qwen's multimodal rotary position embeddings —
i.e. caching silently produced wrong logits for Qwen specifically (this is
not an issue for LLaVA, which uses standard RoPE). Before wiring up
KV-cached SID for Qwen in this package, that numerical-mismatch needs to be
either fixed at its root or confirmed no longer applicable — enabling the
cache without resolving it risks a Qwen SID run that completes cleanly but
produces silently incorrect logits. This will be investigated and reported
before any Qwen jobs are spawned.

## 8. Reproducibility

Every image's generation is reseeded from `iseed = (seed * 1_000_003 +
image_id) % (2**31 - 1)` right before that image starts — this reset
covers PyTorch's global RNG, which both VCD's diffusion-noise draw and the
`--decode sample` multinomial draws pull from. A run is therefore fully
reproducible on restart: killing and resuming a job produces byte-identical
output for images not yet processed, given the same seed. (VCD/SID capture
under `--decode sample` is more stochastic than the old greedy captures —
by design — but it is still deterministic given the seed, not
run-to-run-random.)

## 9. What NOT to change without re-reading this file

- `image_ids_500.json` — must stay byte-identical to `CHAIR_analysis`'s copy.
- `cd_alpha`, `cd_beta`, `noise_step`, `max_new_tokens`, the prompt string —
  changing any of these breaks comparability with the already-reported
  greedy-baseline numbers, which is the entire point of holding them fixed.
- SID's `ATTENTION_RANK=58` / `AGG_LAYER=2` — paper-verified, do not revert
  to a random subset or a different percentage without a documented reason.
- ICD's 5-separate-passes structure — the official methodology; a
  `random.choice()`-per-question version was an earlier, since-fixed bug in
  this project, don't reintroduce it.
