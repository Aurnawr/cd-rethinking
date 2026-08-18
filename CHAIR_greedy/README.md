# CHAIR_greedy: consolidated greedy-decode outputs (LLaVA + Qwen)

This folder gathers, in one place with consistent naming, the existing
**greedy-decode** CHAIR generation outputs for both models -- baseline
(no contrastive decoding), VCD, and SID -- pulled from where they were
originally produced elsewhere in this repo. Nothing here was regenerated;
these are copies (Qwen files decompressed) of already-existing data,
organized for easy browsing.

## Layout

```
CHAIR_greedy/
  README.md                          this file
  outputs/
    llava/
      llava_baseline_greedy.jsonl      500 captions, no CD, no logits
      llava_vcd_greedy_seed0.jsonl      \
      llava_vcd_greedy_seed1.jsonl       > top-10 logits/step, 3 real seeds
      llava_vcd_greedy_seed2.jsonl      /
      llava_sid_greedy_seed0.jsonl      \
      llava_sid_greedy_seed1.jsonl       > top-10 logits/step, see note below
      llava_sid_greedy_seed2.jsonl      /
    qwen/
      qwen_baseline_greedy.jsonl        500 captions, no CD, no logits
      qwen_vcd_greedy_seed0.jsonl        top-10 logits/step, single seed
      qwen_sid_greedy_seed0.jsonl        top-10 logits/step, single seed -- see CAVEAT below
```

Every file is 500 lines (one JSON record per image, 500 MSCOCO val2017
images, same fixed image set used everywhere in this repo).

## Where each file actually came from

| File | Source |
|---|---|
| `llava_baseline_greedy.jsonl` | `chair_full_handoff/data/llava/captions_greedy.jsonl` |
| `llava_vcd_greedy_seed{0,1,2}.jsonl` | `CHAIR_analysis/outputs/vcd_capture/seed{0,1,2}.jsonl` |
| `llava_sid_greedy_seed{0,1,2}.jsonl` | `CHAIR_analysis/outputs/sid_capture/seed{0,1,2}.jsonl` |
| `qwen_baseline_greedy.jsonl` | `chair_full_handoff/data/qwen/captions_greedy.jsonl` |
| `qwen_vcd_greedy_seed0.jsonl` | `chair_full_handoff/data/captures/qwen_vcd.jsonl.gz` (decompressed) |
| `qwen_sid_greedy_seed0.jsonl` | `chair_full_handoff/data/captures/qwen_sid.jsonl.gz` (decompressed) |

## Schema

Baseline files: `{"image_id": ..., "caption": "..."}` -- plain caption,
no per-step data, since there is nothing to log without a contrastive branch.

LLaVA VCD/SID files (`CHAIR_analysis` lineage): `{"image_id", "caption",
"steps": [{"step", "chosen_id", "apc_cutoff", "expert_top_ids",
"expert_top_logits", "amateur_top_ids", "amateur_top_logits", "cd_top_ids",
"cd_top_pre_apc_logits", "cd_top_survives_apc"}, ...]}` -- **top-10** ids/logits
per branch per step.

Qwen VCD/SID files (`chair_full_handoff` lineage): same idea, also
**top-10** ids/logits per branch per step now -- the source data
(`chair_full_handoff/data/captures/qwen_{vcd,sid}.jsonl.gz`) actually
records top-30 per step, but it's truncated to the first 10 (already
sorted descending by logit, so this is a true top-10, not a resample) when
copied in here, both to match the LLaVA files' width and to keep the file
size sane for a plain git push (untruncated, these two files were
~199MB/~198MB -- comfortably over GitHub's 100MB hard limit; truncated
they're ~83MB each). The full top-30 version still exists at the original
path if anyone needs the wider window.

## Two things worth knowing before using this data

**1. SID's three LLaVA "seeds" are byte-identical, on purpose.** SID's
image-token selection is attention-based (least-attended 10% of image
tokens, `topk(largest=False)`), not random -- it's fully deterministic
given the same model and input, so there is no actual seed-to-seed
variation to observe. This was checked directly: all three
`llava_sid_greedy_seed*.jsonl` files hash identically (md5
`dafe3465be6a095f7dfedb5c78408f79`). They're included as three
copies anyway (rather than silently deduplicated to one file) so the
per-seed naming stays consistent with VCD's, and so this determinism is
visible/checkable rather than hidden. VCD's three seeds *do* differ
(diffusion noise draw is genuinely random per seed), so those three files
are real independent runs.

**2. Qwen's SID data here has an unconfirmed correctness caveat.**
`CHAIR_analysis`'s LLaVA SID capture is the paper-exact, attention-based
implementation (fixed from an earlier version that used a random token
subset instead of attention scores -- see `CHAIR_analysis`'s own history).
`qwen_sid_greedy_seed0.jsonl` here comes from `chair_full_handoff`, an
**older** capture that predates that fix being ported/verified on the Qwen
side. It is not confirmed whether this specific file's SID masking used
the correct attention-based selection or the older random-subset logic --
treat it as provisional until it's re-run through `CHAIR_analysis`'s
current Qwen pipeline (`contrastive_adjustment/qwen-2.5-7b/run.sh`) the
same way the LLaVA leg already was. `qwen_vcd_greedy_seed0.jsonl` doesn't
have this issue -- VCD's mechanism (diffusion noise on the image) didn't
change between capture vintages.

## Not included here

- ICD greedy data: no 500-image CHAIR-format ICD greedy capture exists yet
  for either model (only a 60-question llava-bench ICD run exists, a
  different benchmark/task, in `CHAIR_analysis/icd_llava_bench/` -- not
  copied here since it isn't CHAIR data).
- The pre-fix `chair_full_handoff/data/captures/llava_vcd.jsonl.gz` /
  `llava_sid.jsonl.gz` were intentionally left out in favor of
  `CHAIR_analysis`'s corrected, seeded versions above.
