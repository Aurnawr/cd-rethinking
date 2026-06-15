# Implementation Plan: MME Results Table — Data Preparation

## Overview

This plan builds the `eval/mme_prepare.py` converter incrementally and test-first,
mirroring the repo's existing pattern (pure helpers in a module + thin argparse
entry point + a sibling `test_*.py` with pytest + Hypothesis, as in
`eval/mme_eval.py` / `eval/test_mme_eval.py`).

Pure, filesystem-free helpers (`parse_annotation`, `make_question_id`,
`build_image_field`, `build_records_for_image`, `validate_reference`,
`derive_questions`) are implemented and property-tested first. The I/O-bound
functions (`detect_layout`, `enumerate_pairs`, `convert_subtask`,
`reconcile_images`, `main`) come next, with example/edge-case tests for their
error branches. An end-to-end integration smoke test (run the converter against
the committed raw tree via `--subtasks`, then feed the reference back through
`eval/mme_eval.py`) and a final verification task close the loop.

The 10 Correctness Properties from the design are each turned into their own
explicit Hypothesis property-based test sub-task, placed close to the code they
exercise so errors surface early.

The inference modules (`inference/mme_infer_*.py`) and the evaluator
(`eval/mme_eval.py`, `eval/mme_constants.py`) are **fixed** — no task modifies
them. The converter only *reuses* `eval/mme_constants.py` (`SUBTASKS`,
`PERCEPTION`, `COGNITION`).

## Tasks

- [x] 1. Scaffold the converter module and its test file
  - [x] 1.1 Create the converter module skeleton and test scaffold
    - Create `eval/mme_prepare.py` with the module docstring, standard-library
      imports (`os`, `json`, `argparse`, `collections`, `shutil`), and the reuse
      import `from mme_constants import SUBTASKS, PERCEPTION, COGNITION`.
    - Define the core constants: `SPLIT_IMAGE_DIR = "images"`,
      `SPLIT_TEXT_DIR = "questions_answers_YN"`,
      `IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")`, `VALID_LABELS = {"yes", "no"}`.
    - Create `eval/test_mme_prepare.py` with pytest + Hypothesis imports, mirroring
      the structure/conventions of `eval/test_mme_eval.py`, plus a shared
      Hypothesis strategy module-section that builds synthetic raw trees (random
      subtask from `SUBTASKS`, stems, question strings, Yes/No labels, FLAT or
      SPLIT layout, `.jpg`/`.png` extension) to be reused by the property tests.
    - _Requirements: 5.1_

- [x] 2. Implement annotation parsing (pure)
  - [x] 2.1 Implement `parse_annotation(raw_text)`
    - Split into non-empty lines (≥1 non-whitespace char); require exactly two.
    - For each line: require a TAB, set question = text left of first TAB
      (stripped), set label = text right of first TAB (stripped, lower-cased);
      accept only `yes`/`no` (case-insensitive) → emit exactly `yes`/`no`.
    - Raise `ValueError` naming the file/line/value for: wrong non-empty line
      count, missing TAB, empty question, or non-Yes/No label.
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9_

  - [x]* 2.2 Write Hypothesis property test for label validity
    - **Property 3: Label validity** — every parsed label ∈ {"yes", "no"}.
    - Use a strategy generating mixed-casing `Yes`/`No` answers.
    - **Validates: Requirements 1.3, 1.4, 8.3**

  - [x]* 2.3 Write example/edge-case tests for `parse_annotation` error branches
    - Cover: != 2 non-empty lines, whitespace-only lines ignored, missing TAB,
      empty question, label not Yes/No; assert each raises `ValueError`.
    - Include a known FLAT-style body and a SPLIT-style body yielding expected pairs.
    - _Requirements: 1.1, 1.6, 1.7, 1.8, 1.9_

- [x] 3. Implement id, image-field, and per-image record builders (pure)
  - [x] 3.1 Implement `make_question_id` and `build_image_field`
    - `make_question_id(subtask, stem, line_index)` → `f"{subtask}/{stem}_{line_index}"`.
    - `build_image_field(subtask, image_filename)` → `f"{subtask}/{image_filename}"`
      (real on-disk extension preserved by the caller).
    - _Requirements: 2.4, 2.5, 4.3_

  - [x] 3.2 Implement `build_records_for_image(subtask, image_filename, raw_text)`
    - Derive `stem` via `os.path.splitext`; call `parse_annotation`; emit exactly
      two records with keys `{question_id, image, text, label, category}` where
      `question_id` ends in `_0`/`_1`, `image == build_image_field(...)`,
      `category == subtask`, and `text`/`label` come from the parsed pairs.
    - _Requirements: 2.1, 2.2, 2.3, 2.6, 2.7_

  - [x]* 3.3 Write Hypothesis property test for schema completeness
    - **Property 2: Schema completeness** — every record has exactly
      `{question_id, image, text, label, category}` and no extra keys.
    - **Validates: Requirements 2.2, 7.4**

  - [x]* 3.4 Write Hypothesis property test for image-field shape
    - **Property 5: Image-field shape** — every record's `image` equals
      `f"{category}/{filename}"` with the real on-disk extension.
    - **Validates: Requirements 1.2, 2.4, 4.3, 6.1**

  - [x]* 3.5 Write Hypothesis property test for two-records-per-image
    - **Property 1: Two lines per image** — `build_records_for_image` always
      returns exactly two records sharing the same `image`.
    - **Validates: Requirements 1.1, 2.1, 8.1**

- [x] 4. Implement self-validation and questions projection (pure)
  - [x] 4.1 Implement `validate_reference(records)`
    - Raise `ValueError` (naming offenders) if any `question_id` is duplicated,
      any `image` does not have exactly two records, any `label ∉ {yes,no}`, or
      any `category ∉ SUBTASKS`; complete all checks before any write happens.
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9_

  - [x] 4.2 Implement `derive_questions(reference)`
    - Order-preserving projection to exactly `{question_id, image, text}` per record.
    - _Requirements: 7.4, 7.5_

  - [x]* 4.3 Write Hypothesis property test for question_id uniqueness
    - **Property 6: question_id uniqueness** — `len({r.question_id}) == len(R)`
      for any valid generated reference.
    - **Validates: Requirements 2.5, 8.2**

  - [x]* 4.4 Write Hypothesis property test for positional alignment (round-trip)
    - **Property 7: Positional alignment** — `len(Q) == len(R)` and
      `Q[i].question_id == R[i].question_id`; assert `eval/mme_eval.py`'s
      `validate_alignment(R, Q)` never raises.
    - **Validates: Requirements 7.5, 10.1**

  - [x]* 4.5 Write Hypothesis property test for downstream acceptance
    - **Property 10: Downstream acceptance** — `group_by_image(R)` and
      `validate_alignment(R, Q)` from `eval/mme_eval.py` never raise on a valid
      generated reference.
    - **Validates: Requirements 7.1, 10.2, 10.3**

- [x] 5. Checkpoint - pure helpers complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement layout detection and deterministic pairing (I/O)
  - [x] 6.1 Implement `detect_layout(subtask_dir)`
    - Return `"split"` when both `images/` and `questions_answers_YN/`
      sub-folders exist; otherwise `"flat"`.
    - _Requirements: 3.1, 3.2_

  - [x] 6.2 Implement `enumerate_pairs(subtask_dir, layout)`
    - FLAT: read images/`.txt` as siblings in the subtask dir; SPLIT: images from
      `images/`, annotations from `questions_answers_YN/`.
    - Treat `.jpg`/`.jpeg`/`.png` (case-insensitive) as images and `.txt`
      (case-insensitive) as annotations; pair by exact case-sensitive stem;
      return pairs sorted by stem in ascending Unicode order.
    - Raise `ValueError` for: image with no matching `.txt`, `.txt` with no image,
      a stem mapping to >1 image (`.jpg`+`.png`), or a folder with zero pairs.
    - _Requirements: 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8_

  - [x]* 6.3 Write example/edge-case tests for `detect_layout` and `enumerate_pairs`
    - `detect_layout` → `"flat"` for `OCR`/`code_reasoning`, `"split"` for
      `artwork`/`celebrity` (use committed raw tree + tmp fixtures).
    - Each `enumerate_pairs` error branch (unpaired image, unpaired txt,
      duplicate-extension stem, empty folder) raises `ValueError`; happy path
      returns stem-sorted pairs.
    - _Requirements: 3.1, 3.2, 4.4, 4.5, 4.6, 4.7, 4.8_

- [x] 7. Implement subtask conversion and scheduling (I/O)
  - [x] 7.1 Implement `convert_subtask(raw_root, subtask)`
    - Join `raw_root`/`subtask`, detect layout, enumerate pairs, read each `.txt`,
      and accumulate `build_records_for_image` output in stem order.
    - _Requirements: 2.6, 3.5, 4.4_

  - [x] 7.2 Implement subtask scheduling helper used by `main`
    - When no `--subtasks` subset is given, process exactly `SUBTASKS` in order;
      with a subset, process only those names in `SUBTASKS` order; raise
      `ValueError` for an unrecognized name (process nothing); ignore stray
      raw-tree folders; raise `ValueError` for a scheduled subtask whose folder
      is missing (produce no output).
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x]* 7.3 Write Hypothesis property test for layout invariance
    - **Property 8: Layout invariance** — a FLAT tree and a SPLIT tree with the
      same logical content (same stems/questions/labels) produce identical `R`
      (same records, same order). Build both trees from one generated dataset.
    - **Validates: Requirements 3.5, 4.1, 4.2, 7.2**

  - [x]* 7.4 Write example/edge-case tests for subtask scheduling
    - Unrecognized `--subtasks` name raises; missing scheduled folder raises;
      stray folder ignored; subset processed in `SUBTASKS` order.
    - _Requirements: 5.2, 5.3, 5.4, 5.5_

- [x] 8. Implement image reconciliation (I/O)
  - [x] 8.1 Implement `reconcile_images(raw_root, subtask, layout, out_images_root, copy=False)`
    - Create `data/mme/images/{subtask}/` if absent; materialize each image at
      `images/{subtask}/{filename}` preserving the real extension; default to a
      relative symlink, physical byte-for-byte copy under `copy=True`.
    - Overwrite existing entries so a re-run yields the same set of paths and
      identical contents/targets (no leftovers/duplicates).
    - Abort without materializing further entries and emit an error naming the
      missing `{subtask}/{filename}` if a source image is absent; on symlink
      mode where symlinks are unsupported, abort and emit an actionable error
      directing the operator to `--copy`.
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [x]* 8.2 Write example/edge-case tests for `reconcile_images`
    - Symlink mode creates relative links resolving to the raw source; `--copy`
      creates byte-identical copies; re-run is idempotent (same paths, same
      targets/contents); missing source aborts with a naming error.
    - _Requirements: 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 9. Implement output writers and the argparse entry point (I/O)
  - [x] 9.1 Implement JSONL writers and `main()` orchestration
    - Add a `write_jsonl` helper writing one JSON object per line, each line
      (including the last) terminated by a single `\n`, creating parent dirs as
      needed; add a per-subtask writer for `reference/{subtask}.jsonl`.
    - Implement `parse_args()` (`--raw-root`, `--out-root`, `--copy`,
      `--per-subtask`, `--subtasks`) and `main()`: schedule subtasks, accumulate
      reference in `SUBTASKS`-then-stem order, reconcile images, call
      `validate_reference` BEFORE any write, then write `mme_reference.jsonl`,
      optional per-subtask files, and `mme_questions.jsonl` (via
      `derive_questions`); write nothing if validation fails.
    - _Requirements: 7.1, 7.2, 7.3, 7.6, 7.7, 7.8_

  - [x]* 9.2 Write Hypothesis property test for idempotency
    - **Property 9: Idempotency** — running `main` twice on the same generated
      raw tree with identical options yields byte-identical
      `mme_reference.jsonl` and `mme_questions.jsonl`, and a stable `images/` view.
    - **Validates: Requirements 6.4, 9.1, 9.2, 9.3**

  - [x]* 9.3 Write example tests for writer behavior and validation gating
    - Final-line `\n` present; combined order is `SUBTASKS`-then-stem; questions
      file is positional projection; a validation failure leaves no files written.
    - _Requirements: 7.1, 7.2, 7.5, 7.7, 7.8, 9.4, 9.5_

- [x] 10. Add the thin wrapper script
  - [x] 10.1 Create `scripts/mme_prepare.sh`
    - Mirror `scripts/mme_eval.sh` conventions (`#!/usr/bin/env bash`,
      `set -euo pipefail`, env-var overridable defaults). Invoke
      `python ./eval/mme_prepare.py --raw-root ./data/mme/MME_Benchmark_release_version/MME_Benchmark --out-root ./data/mme --per-subtask`.
    - _Requirements: 11.1_

- [x] 11. Integration / end-to-end smoke test
  - [x]* 11.1 Write integration smoke test feeding the converter output to the evaluator
    - Run `main` against the committed raw tree restricted via `--subtasks`
      (e.g. `OCR` or `existence`), then feed the produced `mme_reference.jsonl`
      back through `eval/mme_eval.py` as a perfect-answer result file and assert
      the table renders for all processed subtasks with no alignment/grouping error.
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [x] 12. Update README_MME.md
  - [x] 12.1 Rewrite Section 4, update Sections 8 and 9
    - Section 4 ("Prepare the data"): rewrite around `python eval/mme_prepare.py`
      (and `bash scripts/mme_prepare.sh`) as the prep step; list all four outputs
      (`mme_reference.jsonl`, `mme_questions.jsonl`, `reference/{subtask}.jsonl`,
      `images/{subtask}/`); describe FLAT vs SPLIT auto-detection and the
      observable property that selects each; describe image reconciliation and
      when to pass `--copy`; remove the manual `python -c "..."` placeholder.
    - Section 8 ("Tests"): add `python -m pytest eval/test_mme_prepare.py -q`.
    - Section 9 ("End-to-end summary"): show the converter as step 1 and remove
      the prior "(Once) make sure data/mme/ holds the full MME data" placeholder.
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7_

- [x] 13. Final verification
  - Run `python -m pytest eval/test_mme_prepare.py -q` and ensure the full suite
    (including all Hypothesis property tests) passes.
  - Run an end-to-end smoke: `python ./eval/mme_prepare.py --raw-root ./data/mme/MME_Benchmark_release_version/MME_Benchmark --out-root ./data/mme --subtasks OCR --per-subtask`,
    then `python ./eval/mme_eval.py --ref-files ./data/mme/mme_reference.jsonl --res-files ./data/mme/mme_reference.jsonl`
    and confirm the table renders with no alignment or grouping error.
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional (tests) and can be skipped for a faster MVP,
  but the property-based tests are central to this feature — they encode the
  design's 10 Correctness Properties and should not be skipped in normal flow.
- Each task references specific requirement sub-clauses for traceability.
- Pure helpers are built and property-tested before I/O-bound functions
  (`enumerate_pairs`, `reconcile_images`, `main`) and before the integration test.
- The inference and eval modules are fixed; only `eval/mme_constants.py` is reused.
- All Hypothesis property tests use the `hypothesis` library, mirroring
  `eval/test_mme_eval.py`.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "10.1"] },
    { "id": 1, "tasks": ["2.1", "3.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "3.2"] },
    { "id": 3, "tasks": ["3.3", "3.4", "3.5", "4.1", "4.2"] },
    { "id": 4, "tasks": ["4.3", "4.4", "4.5", "6.1", "6.2"] },
    { "id": 5, "tasks": ["6.3", "7.1", "7.2", "8.1"] },
    { "id": 6, "tasks": ["7.3", "7.4", "8.2", "9.1"] },
    { "id": 7, "tasks": ["9.2", "9.3", "11.1", "12.1"] }
  ]
}
```
