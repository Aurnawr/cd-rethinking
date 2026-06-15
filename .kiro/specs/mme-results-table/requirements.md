# Requirements Document

## Introduction

This feature adds a one-time data-preparation converter, `eval/mme_prepare.py`, that
turns the committed raw MME release tree
(`data/mme/MME_Benchmark_release_version/MME_Benchmark/{subtask}/`) into the exact
JSONL + image layout the already-built MME inference and evaluation pipeline consumes.
The converter parses the raw two-line `question<TAB>answer` `.txt` annotations into
reference records, reconciles the two on-disk image layouts (FLAT and SPLIT) into a
single uniform image view, derives the combined inference-input file, and validates its
own output so the downstream evaluator's invariants hold by construction. A
`README_MME.md` update documents how to run the converter end-to-end to reproduce the
paper's MME results table.

The inference modules (`inference/mme_infer_*.py`) and the evaluator
(`eval/mme_eval.py`, `eval/mme_constants.py`) are treated as **fixed downstream
consumers**. These requirements do not redesign them; instead they require that the
converter's output be accepted by them unchanged.

## Glossary

- **Converter**: The new `eval/mme_prepare.py` program (pure helper functions plus a
  thin argparse entry point) that transforms the raw MME tree into the pipeline's input
  files.
- **Raw_Tree**: The committed MME release directory tree rooted at
  `data/mme/MME_Benchmark_release_version/MME_Benchmark/`, containing one folder per
  subtask.
- **Subtask**: One of the fourteen fixed MME subtask names defined in
  `eval/mme_constants.py` (`SUBTASKS = PERCEPTION + COGNITION`).
- **FLAT_Layout**: A subtask folder where `{stem}.jpg|.png` image files and `{stem}.txt`
  annotation files are siblings directly inside `{subtask}/`.
- **SPLIT_Layout**: A subtask folder where images live in `{subtask}/images/` and
  annotations live in `{subtask}/questions_answers_YN/`.
- **Annotation_File**: A raw `{stem}.txt` file containing exactly two non-empty lines,
  each of the form `question<TAB>answer` with answer in {`Yes`, `No`} (case-insensitive).
- **Non-empty line**: A line containing at least one non-whitespace character; lines that
  are empty or contain only whitespace are ignored.
- **Stem**: An image or annotation filename with its extension removed.
- **Reference_Record**: One line of `mme_reference.jsonl`, a JSON object with exactly the
  keys `{question_id, image, text, label, category}`.
- **Question_Record**: One line of `mme_questions.jsonl`, a JSON object with exactly the
  keys `{question_id, image, text}`.
- **Reference_File**: The combined output file `data/mme/mme_reference.jsonl`.
- **Questions_File**: The combined output file `data/mme/mme_questions.jsonl`.
- **Per_Subtask_Reference**: An optional output file `data/mme/reference/{subtask}.jsonl`
  containing the Reference_Records for a single subtask.
- **Image_View**: The uniform reconciled image tree at
  `data/mme/images/{subtask}/{filename}`.
- **Question_Id**: The deterministic identifier `f"{subtask}/{stem}_{line_index}"` with
  `line_index` in {0, 1}.
- **Evaluator**: The fixed downstream `eval/mme_eval.py`, including its `group_by_image`
  and `validate_alignment` functions.
- **README**: The `README_MME.md` documentation file.

## Requirements

### Requirement 1: Parse raw annotation files into question/label pairs

**User Story:** As a benchmark operator, I want each raw two-line annotation file parsed
into question and label pairs, so that the raw MME text becomes structured records the
evaluator can consume.

#### Acceptance Criteria

1. WHEN an Annotation_File body is parsed, THE Converter SHALL treat only lines with at
   least one non-whitespace character as non-empty lines and SHALL ignore lines that are
   empty or contain only whitespace.
2. WHEN an Annotation_File body contains exactly two non-empty `question<TAB>answer`
   lines, THE Converter SHALL produce exactly two `(question, label)` pairs in
   top-to-bottom line order.
3. WHEN a `question<TAB>answer` line is parsed, THE Converter SHALL set the question to
   the text preceding the first TAB with surrounding whitespace removed.
4. WHEN a `question<TAB>answer` line is parsed, THE Converter SHALL set the label to the
   text following the first TAB with surrounding whitespace removed and converted to
   lower case.
5. WHEN a parsed answer is `Yes` or `No` in any letter casing, THE Converter SHALL accept
   it and produce a label of exactly `yes` or `no` respectively.
6. IF an Annotation_File body does not contain exactly two non-empty lines, THEN THE
   Converter SHALL raise an error naming the file and the non-empty line count found, and
   SHALL produce no pairs for that file.
7. IF a non-empty line contains no TAB separator, THEN THE Converter SHALL raise an error
   naming the file and the offending line, and SHALL produce no pairs for that file.
8. IF a parsed question is empty after whitespace removal, THEN THE Converter SHALL raise
   an error naming the file and the offending line, and SHALL produce no pairs for that
   file.
9. IF a parsed answer is not `Yes` or `No` (case-insensitive), THEN THE Converter SHALL
   raise an error naming the file and the offending value, and SHALL produce no pairs for
   that file.

### Requirement 2: Build reference records for each image

**User Story:** As a benchmark operator, I want two complete reference records built per
image, so that the combined reference matches the evaluator's exact schema.

#### Acceptance Criteria

1. WHEN reference records are built for one image, THE Converter SHALL produce exactly two
   Reference_Records for that image.
2. THE Converter SHALL set each Reference_Record to contain exactly the five keys
   `question_id`, `image`, `text`, `label`, and `category`, and no additional keys.
3. WHEN reference records are built for one image, THE Converter SHALL set the `category`
   of each Reference_Record to the name of the Subtask folder the image came from.
4. WHEN reference records are built for one image, THE Converter SHALL set the `image`
   field of each Reference_Record to `f"{subtask}/{filename}"`, where `subtask` is the
   Subtask folder name and `filename` is the image's on-disk filename including its real
   extension.
5. WHEN reference records are built for one image, THE Converter SHALL set the
   `question_id` of the first Reference_Record to `f"{subtask}/{stem}_0"` and the
   `question_id` of the second Reference_Record to `f"{subtask}/{stem}_1"`, where `stem`
   is the image filename with its extension removed.
6. WHEN reference records are built for one image, THE Converter SHALL set the `text` of
   the first and second Reference_Records to the first and second question prompts
   associated with that image, respectively.
7. WHEN reference records are built for one image, THE Converter SHALL set the `label` of
   the first and second Reference_Records to the ground-truth answer of the first and
   second question respectively, where each `label` value is exactly `yes` or `no`.

### Requirement 3: Detect and handle FLAT and SPLIT layouts

**User Story:** As a benchmark operator, I want the converter to auto-detect each
subtask's on-disk layout, so that both raw layouts are converted without manual
configuration.

#### Acceptance Criteria

1. WHEN a subtask folder contains both a sub-folder named `images` and a sub-folder named
   `questions_answers_YN`, THE Converter SHALL classify that subtask as SPLIT_Layout.
2. IF a subtask folder is not classified as SPLIT_Layout, THEN THE Converter SHALL
   classify that subtask as FLAT_Layout.
3. WHILE processing a FLAT_Layout subtask, THE Converter SHALL read image files
   (`.jpg`, `.jpeg`, or `.png`) and annotation files (`.txt`) as siblings directly inside
   the subtask folder.
4. WHILE processing a SPLIT_Layout subtask, THE Converter SHALL read image files from the
   `images` sub-folder and annotation files from the `questions_answers_YN` sub-folder.
5. WHEN a FLAT_Layout subtask and a SPLIT_Layout subtask carry the same logical content
   (the same set of stems, the same question strings, and the same Yes/No labels), THE
   Converter SHALL produce Reference_Records with identical field values in identical
   order for both.

### Requirement 4: Pair images with annotations deterministically

**User Story:** As a benchmark operator, I want images paired with their annotation files
in a stable order, so that the produced output is deterministic and complete.

#### Acceptance Criteria

1. WHEN enumerating a subtask, THE Converter SHALL pair each image file with the
   annotation file located in the same subtask folder whose stem is an exact,
   case-sensitive match.
2. WHERE a file's extension is `.jpg`, `.jpeg`, or `.png` (matched case-insensitively),
   THE Converter SHALL treat that file as an image of the subtask.
3. WHERE a file's extension is `.txt` (matched case-insensitively), THE Converter SHALL
   treat that file as an annotation file of the subtask.
4. THE Converter SHALL return image/annotation pairs ordered by stem in ascending
   lexicographic (Unicode code point) order.
5. IF an image file has no annotation file sharing its stem, THEN THE Converter SHALL
   raise an error listing the unpaired stems and SHALL NOT produce output.
6. IF an annotation file has no image file sharing its stem, THEN THE Converter SHALL
   raise an error listing the unpaired stems and SHALL NOT produce output.
7. IF a single stem maps to more than one image file (for example both `.jpg` and
   `.png`), THEN THE Converter SHALL raise an error naming the ambiguous stem and SHALL
   NOT produce output.
8. IF a subtask folder contains zero image/annotation pairs, THEN THE Converter SHALL
   raise an error naming the empty subtask folder and SHALL NOT produce output.

### Requirement 5: Iterate over the fixed subtask vocabulary

**User Story:** As a benchmark operator, I want only the fourteen known subtasks
processed, so that stray folders in the raw tree are ignored and missing subtasks are
reported.

#### Acceptance Criteria

1. WHEN the operator does not supply an explicit subtask subset, THE Converter SHALL
   process exactly the fourteen subtask names defined by `SUBTASKS` in
   `eval/mme_constants.py` and no others.
2. WHERE the operator supplies an explicit subtask subset, THE Converter SHALL process
   only the subtasks in that subset, in the order the names appear in `SUBTASKS`.
3. IF the operator-supplied subset contains a name that is not one of the fourteen
   `SUBTASKS` names, THEN THE Converter SHALL raise an error naming the unrecognized
   subtask and SHALL process no subtasks.
4. WHEN the Raw_Tree contains a folder whose name does not exactly match (case-sensitive)
   one of the subtasks scheduled for processing, THE Converter SHALL exclude that folder
   from processing and SHALL continue processing the remaining scheduled subtasks.
5. IF a subtask scheduled for processing has no folder whose name exactly matches it in
   the Raw_Tree, THEN THE Converter SHALL raise an error naming the missing subtask
   folder and SHALL produce no output for any subtask.

### Requirement 6: Reconcile images into a uniform image view

**User Story:** As a benchmark operator, I want all images materialized at
`data/mme/images/{subtask}/{filename}`, so that the `image` field resolves under
`--image-folder ./data/mme/images` regardless of the raw layout.

#### Acceptance Criteria

1. WHEN reconciling images for a subtask, THE Converter SHALL create the directory
   `data/mme/images/{subtask}/` if it does not already exist.
2. WHEN reconciling an image for a subtask, THE Converter SHALL materialize it at
   `data/mme/images/{subtask}/{filename}`, where the extension of `{filename}` equals the
   actual on-disk extension of the corresponding source file in the Raw_Tree.
3. WHERE the operator does not pass the `--copy` option, THE Converter SHALL materialize
   each reconciled image as a relative symlink whose target resolves to the corresponding
   source file in the Raw_Tree.
4. WHERE the operator passes the `--copy` option, THE Converter SHALL materialize each
   reconciled image as a byte-for-byte physical copy of the corresponding source file in
   the Raw_Tree.
5. WHEN reconciliation runs against an Image_View previously materialized from the same
   Raw_Tree, THE Converter SHALL overwrite each existing entry so that the resulting
   Image_View contains exactly the same set of paths and identical file contents (for
   copies) or symlink targets (for symlinks) as a fresh reconciliation, with no leftover
   or duplicate entries.
6. IF the source file referenced by a record's `image` field (`{subtask}/{filename}`) is
   absent from the Raw_Tree, THEN THE Converter SHALL abort reconciliation without
   materializing any further entries and SHALL emit an error identifying the missing
   `{subtask}/{filename}`.
7. IF the filesystem does not support symlinks while symlink mode is selected, THEN THE
   Converter SHALL abort reconciliation without materializing any entries and SHALL emit
   an actionable error that identifies the unsupported-symlink condition and directs the
   operator to re-run with the `--copy` option.

### Requirement 7: Emit reference, per-subtask, and questions outputs

**User Story:** As a benchmark operator, I want the converter to write the combined
reference, optional per-subtask references, and the derived questions file, so that both
the evaluator and the inference scripts have their required inputs.

#### Acceptance Criteria

1. THE Converter SHALL write all Reference_Records to `data/mme/mme_reference.jsonl` with
   exactly one JSON object per line, each line terminated by a single `\n` newline
   character including the final line.
2. THE Converter SHALL order the combined Reference_File first by Subtask following the
   `SUBTASKS` order defined in `eval/mme_constants.py`, and within each Subtask in
   ascending stem order.
3. WHERE the operator requests per-subtask output, THE Converter SHALL write each
   processed subtask's Reference_Records to `data/mme/reference/{subtask}.jsonl` with
   exactly one JSON object per line in ascending stem order.
4. THE Converter SHALL derive `data/mme/mme_questions.jsonl` from the combined
   Reference_File by projecting each Reference_Record to exactly the keys `question_id`,
   `image`, and `text`, copying each value unchanged from the source Reference_Record.
5. WHEN deriving the Questions_File, THE Converter SHALL emit exactly one Question_Record
   per Reference_Record in the same line order, so that the Question_Record at each line
   position has the same `question_id` as the Reference_Record at that position and the
   Questions_File and Reference_File contain an equal number of lines.
6. WHEN self-validation of the combined reference succeeds, THE Converter SHALL write the
   Reference_File, the Questions_File, and any requested Per_Subtask_Reference files.
7. IF self-validation of the combined reference fails, THEN THE Converter SHALL write none
   of the Reference_File, Questions_File, or Per_Subtask_Reference files.
8. WHEN writing an output file whose parent directory does not exist, THE Converter SHALL
   create the parent directory before writing the file.

### Requirement 8: Self-validate output before writing

**User Story:** As a benchmark operator, I want the converter to validate its output
against the evaluator's invariants before writing, so that a failure never leaves a
half-written reference the evaluator would reject.

#### Acceptance Criteria

1. WHEN self-validation runs, THE Converter SHALL confirm that every distinct `image`
   value is associated with exactly two Reference_Records.
2. WHEN self-validation runs, THE Converter SHALL confirm that every `question_id` value
   occurs exactly once across the combined reference.
3. WHEN self-validation runs, THE Converter SHALL confirm that every `label` value is
   exactly one of the two strings `yes` or `no`.
4. WHEN self-validation runs, THE Converter SHALL confirm that every `category` value is
   exactly one of the fourteen `SUBTASKS` names.
5. THE Converter SHALL complete all self-validation checks before creating, opening, or
   writing any output file.
6. IF self-validation finds any `image` not associated with exactly two Reference_Records,
   THEN THE Converter SHALL raise an error naming the affected images and their record
   counts, and SHALL leave no output file written.
7. IF self-validation finds a `question_id` that occurs more than once, THEN THE Converter
   SHALL raise an error naming the duplicated `question_id`, and SHALL leave no output
   file written.
8. IF self-validation finds a `label` value that is not `yes` or `no`, THEN THE Converter
   SHALL raise an error naming the offending value, and SHALL leave no output file
   written.
9. IF self-validation finds a `category` value that is not one of the fourteen `SUBTASKS`,
   THEN THE Converter SHALL raise an error naming the offending value, and SHALL leave no
   output file written.

### Requirement 9: Produce deterministic and idempotent output

**User Story:** As a benchmark operator, I want re-running the converter to reproduce the
same results, so that the prepared data is stable and reproducible.

#### Acceptance Criteria

1. WHEN the Converter is run twice on the same Raw_Tree with identical command-line
   options, THE Converter SHALL produce a byte-identical `mme_reference.jsonl` on both
   runs.
2. WHEN the Converter is run twice on the same Raw_Tree with identical command-line
   options, THE Converter SHALL produce a byte-identical `mme_questions.jsonl` on both
   runs.
3. WHERE per-subtask output is requested, WHEN the Converter is run twice on the same
   Raw_Tree with identical command-line options, THE Converter SHALL produce
   byte-identical `reference/{subtask}.jsonl` files on both runs.
4. WHEN the Converter is run twice on the same Raw_Tree with identical command-line
   options, THE Converter SHALL produce an Image_View containing the same set of relative
   paths, each resolving to the same Raw_Tree source file and materialized in the same
   mode (symlink or copy), on both runs.
5. IF the Converter is re-run against output already materialized from the same Raw_Tree
   with identical command-line options, THEN THE Converter SHALL complete without error
   and leave the output files byte-identical and the Image_View equivalent to the prior
   run.

### Requirement 10: Guarantee downstream acceptance

**User Story:** As a benchmark operator, I want the converter's output accepted by the
fixed evaluator without modification, so that the prepared data reproduces the MME results
table.

#### Acceptance Criteria

1. WHEN the Evaluator's `validate_alignment` is applied to the combined Reference_File and
   the derived Questions_File, THE Converter SHALL have produced a Reference_File and
   Questions_File containing an equal number of records.
2. WHEN the Evaluator's `validate_alignment` is applied to the combined Reference_File and
   the derived Questions_File, THE Converter SHALL have produced output in which the
   `question_id` at each line position is identical between the Reference_File and the
   Questions_File.
3. WHEN the Evaluator's `group_by_image` is applied to the combined Reference_File, THE
   Converter SHALL have produced output in which every distinct `image` value groups to
   exactly two Reference_Records.
4. WHEN the combined Reference_File is fed to `eval/mme_eval.py` as a perfect-answer result
   file (every predicted answer equal to its record's `label`), THE Converter SHALL have
   produced output for which the results table renders for all processed `SUBTASKS`
   without raising an alignment error or a grouping error.

### Requirement 11: Update README_MME.md to document the converter

**User Story:** As a benchmark operator, I want the README to document the converter
end-to-end, so that I can reproduce the full MME results table from the raw tree.

#### Acceptance Criteria

1. THE README Section 4 ("Prepare the data") SHALL contain the command invocation
   `python eval/mme_prepare.py` identified as the data-preparation step that runs against
   the committed Raw_Tree.
2. THE README Section 4 SHALL list all four Converter output artifacts by name:
   `mme_reference.jsonl`, `mme_questions.jsonl`, the per-subtask `reference/{subtask}.jsonl`
   files, and the `images/{subtask}/` view.
3. THE README Section 4 SHALL describe the FLAT versus SPLIT auto-detection behavior,
   stating the observable property of the Raw_Tree that determines which layout the
   Converter selects.
4. THE README Section 4 SHALL describe the image reconciliation behavior and the `--copy`
   option, including the specific condition under which an operator should pass `--copy`.
5. THE README Section 8 ("Tests") SHALL contain the command
   `python -m pytest eval/test_mme_prepare.py -q`.
6. THE README Section 9 ("End-to-end summary") SHALL show the `eval/mme_prepare.py`
   invocation as the data-preparation step.
7. THE README Section 9 ("End-to-end summary") SHALL NOT contain the prior manual
   data-preparation placeholder.
