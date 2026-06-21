# Implementation Plan: Qwen2.5-VL MME Integration

## Overview

Add Qwen2.5-VL-7B-Instruct support to the MME pipeline at full parity with LLaVA across five
method families. Each method ships as a separate Qwen-specific script built on the HF
`Qwen2_5_VLForConditionalGeneration` + `AutoProcessor` contract with chat-template messages,
reusing the pure helpers in `inference/mme_infer_common.py` verbatim. Build order is
base → pba → olm → apc → cd (base establishes the shared skeleton each later script copies),
then the three shell drivers. LLaVA scripts, LLaVA drivers, `mme_infer_common.py`, and
`eval/mme_eval.py` are left untouched. Work is lean and precise; no smoke tests.

## Tasks

- [ ] 1. Implement base Qwen script (shared skeleton)
  - [ ] 1.1 Create `inference/mme_infer_base_qwen.py`
    - Establish the shared skeleton reused by all later scripts: `load_qwen` (loads
      `Qwen2_5_VLForConditionalGeneration` + `AutoProcessor`, derives `model_name` as the
      checkpoint basename), `build_inputs` (chat-template message with image + question, no
      `IMAGE_TOKEN_INDEX`/`conv_templates`/`tokenizer_image_token`), `decode_trimmed`,
      argparse block (LLaVA args minus `--conv-mode`), and the `eval_model` loop.
    - Import `split_list`, `get_chunk`, `derive_do_sample`, `build_answer_record` verbatim from
      `mme_infer_common`; append `INSTRUCTION_SUFFIX` to each question.
    - `run_method` calls stock `model.generate` (greedy at `--temperature 0`, sampling at
      `--temperature 1`); write records via `build_answer_record` with `model_id` = model name.
    - _Requirements: 1.1, 1.4, 2.1, 2.2, 2.3, 2.4, 4.1, 4.3, 4.4, 5.1, 5.2, 5.3, 6.3, 6.4_

- [ ] 2. Implement pba Qwen script
  - [ ] 2.1 Create `inference/mme_infer_pba_qwen.py`
    - Copy the base skeleton; replace `INSTRUCTION_SUFFIX` with `PBA_SUFFIX`
      (`" Answer the question using a single word or phrase. Answer yes whenever possible."`).
    - Stock `model.generate` (no logit surgery); driver runs it greedy only.
    - _Requirements: 1.1, 1.4, 2.4, 4.1, 4.3, 4.4, 5.1, 5.2, 5.3, 6.3, 6.4_

- [ ] 3. Implement olm Qwen script
  - [ ] 3.1 Create `inference/mme_infer_olm_qwen.py`
    - Add `--use-olm` flag; resolve yes/no token ids from the Qwen tokenizer at load time
      (`yes_no_ids`) instead of hardcoded Llama ids.
    - Manual greedy loop using the model's own `prepare_inputs_for_generation` /
      `_update_model_kwargs_for_generation`; apply the verbatim yes/no decision logic on
      greedy-step probabilities. Driver runs greedy (`--temperature 0`) only.
    - _Requirements: 1.1, 1.4, 2.1, 2.2, 2.3, 2.4, 4.1, 4.3, 4.4, 5.1, 5.2, 5.3, 6.3, 6.4_

- [ ] 4. Implement apc Qwen script
  - [ ] 4.1 Create `inference/mme_infer_apc_qwen.py`
    - Add `--use-apc` flag; manual sample loop (single stream). Apply the adaptive-plausibility
      cutoff mask directly to sampling logits, then temperature/top_p warper + multinomial
      (drop the redundant second forward, mathematically equivalent to LLaVA APC sample path).
    - Driver runs sampling (`--temperature 1`) only.
    - _Requirements: 1.1, 1.4, 2.1, 2.2, 2.3, 2.4, 4.1, 4.3, 4.4, 5.1, 5.2, 5.3, 6.3, 6.4_

- [ ] 5. Implement cd Qwen script (vcd/icd/sid)
  - [ ] 5.1 Create `inference/mme_infer_cd_qwen.py`
    - Add `--use-vcd`/`--use-icd`/`--use-sid` (store_true) and `--noise-step` (default 900);
      call `validate_cd_flags(args)` first to reject conflicting flags.
    - Implement self-contained `contrastive_generate` two-stream loop using the model's own
      `prepare_inputs_for_generation` / `_update_model_kwargs_for_generation`; verbatim CD
      logit combination (`cd_alpha=1.0`, `cd_beta=0.2`, cutoff mask). Both greedy and sampling
      paths.
    - Per-method contrastive stream: vcd = diffusion-noised pixels (`add_diffusion_noise`
      verbatim from `cd_utils/vcd_utils.py`); icd = negative-system-prompt message
      (`get_random_icd_prompt` verbatim from `cd_utils/icd_utils.py`); sid = text-only
      (image-free) prompt.
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 4.1, 4.3, 4.4, 5.1, 5.2, 5.3, 6.3, 6.4_

- [ ] 6. Checkpoint - verify all five scripts import and parse args
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Implement shell drivers
  - [ ] 7.1 Create `scripts/mme_infer_base_qwen.sh`
    - Mirror `scripts/mme_infer_base.sh`, drop `--conv-mode`, default `MODEL_PATH` to the Qwen
      checkpoint (overridable via env), run base script greedy + sample writing
      `outputs/mme/baseline/qwen25-7b-mme-{greedy,sample}.jsonl`.
    - _Requirements: 4.2, 6.1, 6.2, 7.1, 7.4_

  - [ ] 7.2 Create `scripts/mme_infer_cd_qwen.sh`
    - Mirror `scripts/mme_infer_cd.sh`; loop vcd/icd/sid passing matching flag, greedy + sample,
      writing `outputs/mme/{vcd,icd,sid}/qwen25-7b-mme-{greedy,sample}.jsonl`.
    - _Requirements: 4.2, 6.1, 6.2, 7.2, 7.4_

  - [ ] 7.3 Create `scripts/mme_infer_spurious_qwen.sh`
    - Mirror `scripts/mme_infer_spurious.sh`: pba greedy →
      `outputs/mme/pba/qwen25-7b-mme-greedy.jsonl`; olm `--use-olm` greedy →
      `outputs/mme/olm/qwen25-7b-mme-greedy.jsonl`; apc `--use-apc` sampling →
      `outputs/mme/apc/qwen25-7b-mme-sample.jsonl`.
    - _Requirements: 4.2, 6.1, 6.2, 7.3, 7.4_

- [ ] 8. Verify reused-helper property tests and example checks
  - [ ]* 8.1 Verify/add property test for contrastive-flag conflict rejection
    - In `inference/test_mme_infer_common.py`; add only if not already present.
    - **Property 1: Contrastive-flag conflict rejection**
    - **Validates: Requirements 1.3**

  - [ ]* 8.2 Verify/add property test for question chunking
    - **Property 2: Question chunking is a lossless partition**
    - **Validates: Requirements 5.1**

  - [ ]* 8.3 Verify/add property test for decoding mode selection
    - **Property 3: Decoding mode follows temperature**
    - **Validates: Requirements 5.2, 6.3**

  - [ ]* 8.4 Verify/add property test for answer-record schema
    - **Property 4: Answer-record schema and field provenance**
    - **Validates: Requirements 4.3, 4.4, 5.3**

  - [ ]* 8.5 Add light example checks for per-script wiring
    - Suffix selection (Instruction vs PBA), filename prefix mapping
      (`greedy.jsonl`/`sample.jsonl`), and `model_id` basename derivation.
    - _Requirements: 2.4, 4.2, 4.4_

- [ ] 9. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional (test sub-tasks) and can be skipped for a faster MVP.
- Property tests exercise only the reused `mme_infer_common` helpers; per the fast/no-smoke-test
  scope, no heavy model or integration tests are added. Add property tests only if not already
  present in `inference/test_mme_infer_common.py`.
- Each script references the specific requirements it satisfies for traceability.
- LLaVA scripts/drivers, `mme_infer_common.py`, and `eval/mme_eval.py` are never modified
  (Requirements 3.1, 3.2, 4.5, 5.4 satisfied by omission).

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1", "3.1", "4.1", "5.1"] },
    { "id": 2, "tasks": ["7.1", "7.2", "7.3", "8.1"] },
    { "id": 3, "tasks": ["8.2"] },
    { "id": 4, "tasks": ["8.3"] },
    { "id": 5, "tasks": ["8.4"] },
    { "id": 6, "tasks": ["8.5"] }
  ]
}
```
