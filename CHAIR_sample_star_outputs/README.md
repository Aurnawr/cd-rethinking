# CHAIR_sample_star_outputs: full generation outputs (LLaVA + Qwen, Sample* leg)

Full per-step generation outputs from the `CHAIR_sample_star` package (the
Sample*/beta=0.1 experiment, see `CHAIR_sample_star/README.md` and `GUIDE.md`
in the repo for the methodology). All 500-image runs, both models, pulled
directly from the run on the Lightning AI Studio. Each file includes the full
per-step top-10 expert/amateur logits needed to compute the contrastive
adjustment `d = E - A`, plus the final caption.

## Layout

```
CHAIR_sample_star_outputs/
  README.md
  outputs/
    llava/
      llava_sample_star_seed0.jsonl   500 records, single-branch (no amateur), APC-only
      llava_vcd_seed0.jsonl           500 records, expert+amateur (VCD)
      llava_sid_seed0.jsonl           500 records, expert+amateur (SID)
      llava_icd_p1_seed0.jsonl        500 records, expert+amateur (ICD, disturbance prompt p1)
      llava_icd_n2_seed0.jsonl        500 records, expert+amateur (ICD, disturbance prompt n2)
    qwen/
      qwen_sample_star_seed0.jsonl    500 records, single-branch (no amateur), APC-only
      qwen_vcd_seed0.jsonl            500 records, expert+amateur (VCD)
      qwen_sid_seed0.jsonl            500 records, expert+amateur (SID)
      qwen_icd_p1_seed0.jsonl         500 records, expert+amateur (ICD, disturbance prompt p1)
      qwen_icd_n2_seed0.jsonl         500 records, expert+amateur (ICD, disturbance prompt n2)
```

All 10 files: 500 records each, one JSON object per line (one per MSCOCO
image, same fixed 500-image set used across the whole repo). Verified: every
file has exactly 500 lines, every decoding step has exactly 10 entries in
every top-list, and `apc_cutoff` matches `log(0.1) + max(expert_top_logits)`
on every single step across all 5,000 records.

**Note on ICD:** the official methodology uses 5 disturbance prompts
(`p1, p2, n1, n2, p3`). This run only covers 2 of them (`p1`, `n2`) — a
deliberate scope reduction to cut total GPU time, not a data-quality issue.

## Schema

`sample_star` files (no amateur branch):
```json
{"image_id": ..., "method": "sample_star", "decode": "sample", "caption": "...",
 "steps": [{"step": 0, "chosen_id": ..., "apc_cutoff": ...,
            "expert_top_ids": [...10...], "expert_top_logits": [...10...],
            "expert_top_survives_apc": [...10 bools...]}, ...]}
```

`vcd` / `sid` / `icd_*` files (expert + amateur branches):
```json
{"image_id": ..., "caption": "...",
 "steps": [{"step": 0, "chosen_id": ...,
            "apc_cutoff": ...,
            "expert_top_ids": [...10...], "expert_top_logits": [...10...],
            "amateur_top_ids": [...10...], "amateur_top_logits": [...10...],
            "cd_top_ids": [...10...], "cd_top_pre_apc_logits": [...10...],
            "cd_top_survives_apc": [...10 bools...]}, ...]}
```

To compute `d` per candidate at a given step: match a token id between
`expert_top_ids` and `amateur_top_ids` (both lists are independently top-10
by their own branch, so a given id may appear in one list and not the other
-- only compute `d` for ids present in both), then
`d = expert_top_logits[i] - amateur_top_logits[j]` at the matching positions.
`cd_top_pre_apc_logits` is the already-computed `(1+alpha)*E - alpha*A` if you
don't want to recompute it yourself (`alpha=1`, `beta=0.1` for this whole run
-- see `CHAIR_sample_star/GUIDE.md` for why 0.1 rather than the sibling
package's 0.2).

## Hyperparameters (same for every file here)

| Parameter | Value |
|---|---|
| `cd_alpha` | 1.0 |
| `cd_beta` (APC) | 0.1 |
| `max_new_tokens` | 256 |
| Prompt | "Describe this image in detail." |
| Decode | `sample` |
| Seed | 0 |
| Images | 500 (same fixed MSCOCO val2017 set as the rest of the repo) |

## Provenance

Generated directly on a Lightning AI Studio (NVIDIA L4), not Modal -- run
end-to-end via `run_qwen_full.sh` / `run_llava_full.sh` driver scripts calling
`CHAIR_sample_star`'s `common/generate_llava.py`, `common/generate_llava_icd.py`,
`qwen/generate_qwen.py`, `qwen/generate_qwen_icd.py` directly. Qwen's KV-cache
implementation was verified against a brute-force cache-less recompute before
this run (`qwen/verify_qwen_cache.py`, PASS on this deployment). Every file
passed structural verification (`common/verify_outputs.py`) and a content
sanity check (schema, `apc_cutoff` formula, caption coherence) before being
copied here.
