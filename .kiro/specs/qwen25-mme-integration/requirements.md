# Requirements Document

## Introduction

This feature adds Qwen2.5-VL-7B-Instruct support to the existing MME inference pipeline at full parity with the current LLaVA implementation across all five decoding/mitigation method families: base, contrastive decoding (vcd/icd/sid), apc, olm, and pba. Because the existing scripts hook LLaVA model internals (custom `model.generate` keyword arguments, `IMAGE_TOKEN_INDEX`, `conv_templates`, `tokenizer_image_token`), each method requires a method-specific Qwen reimplementation built on the Hugging Face `Qwen2_5_VLForConditionalGeneration` + `AutoProcessor` API using chat-template messages.

Each method is delivered as a separate Qwen-specific inference script that mirrors its LLaVA counterpart (e.g. `inference/mme_infer_base_qwen.py`). The existing LLaVA scripts remain untouched as reference. Qwen output is kept separable from LLaVA output by writing to the existing per-method output directories with a distinct filename prefix (`qwen25-7b-mme-*.jsonl` vs `llava-7b-mme-*.jsonl`), preserving the identical answer-record schema so that `eval/mme_eval.py` works unchanged. Shared pure helpers in `inference/mme_infer_common.py` are reused verbatim. New shell drivers mirror the existing LLaVA shell drivers.

## Glossary

- **Qwen_Inference_Script**: Any new Qwen-specific MME inference script under `inference/` named `mme_infer_{method}_qwen.py`, where method is one of base, cd, apc, olm, pba.
- **Qwen_Model**: A Hugging Face `Qwen2_5_VLForConditionalGeneration` model instance loaded together with its `AutoProcessor`.
- **HF_API**: The Hugging Face Transformers interface comprising `Qwen2_5_VLForConditionalGeneration`, `AutoProcessor`, and chat-template message construction.
- **Chat_Message**: A chat-template message list passed to the Qwen `AutoProcessor`, containing the image reference and the question text, instead of LLaVA's `IMAGE_TOKEN_INDEX`/`conv_templates`/`tokenizer_image_token` machinery.
- **Answer_Record**: A single JSONL output record with exactly the keys `question_id`, `prompt`, `text`, `answer_id`, `model_id`, `metadata`.
- **Common_Helpers**: The shared pure helper functions in `inference/mme_infer_common.py`: `split_list`, `get_chunk`, `derive_do_sample`, `build_answer_record`, `validate_cd_flags`.
- **Method_Family**: One of the five supported families: base, cd (covering vcd/icd/sid), apc, olm, pba.
- **CD_Method**: A contrastive-decoding method, one of vcd, icd, or sid.
- **Output_Directory**: An existing per-method directory under `outputs/mme/`: one of `baseline`, `vcd`, `icd`, `sid`, `apc`, `olm`, `pba`.
- **Filename_Prefix**: The leading portion of a Qwen output filename, fixed as `qwen25-7b-mme-`.
- **Greedy_Decoding**: Generation with temperature 0 (`do_sample = False`).
- **Sample_Decoding**: Generation with temperature 1 (`do_sample = True`).
- **Default_Checkpoint**: The local filesystem path `/teamspace/studios/this_studio/models/Qwen2.5-VL-7B-Instruct`.
- **MODEL_PATH**: The environment variable that overrides the model checkpoint path used by a Qwen shell driver.
- **Qwen_Shell_Driver**: A new shell script under `scripts/` that runs one or more Qwen_Inference_Scripts, mirroring an existing LLaVA shell driver.
- **Instruction_Suffix**: The fixed string `Answer the question using a single word or phrase.` appended to the question text for eval consistency.
- **PBA_Suffix**: The fixed string `Answer the question using a single word or phrase. Answer yes whenever possible.` used by the pba method in place of the standard Instruction_Suffix.

## Requirements

### Requirement 1: Full method-family parity

**User Story:** As a researcher, I want a Qwen reimplementation of every MME method family, so that I can run the full benchmark suite on Qwen2.5-VL with the same coverage as LLaVA.

#### Acceptance Criteria

1. THE Qwen_Inference_Script set SHALL provide one script for each Method_Family: base, cd, apc, olm, and pba.
2. THE Qwen_Inference_Script for the cd Method_Family SHALL support each CD_Method (vcd, icd, sid) selectable through command-line flags `--use-vcd`, `--use-icd`, and `--use-sid`.
3. WHERE the cd Method_Family script receives more than one CD_Method flag, THE Qwen_Inference_Script SHALL reject the invocation by calling `validate_cd_flags`.
4. THE Qwen_Inference_Script set SHALL preserve each LLaVA method's greedy/sampling decoding behavior, including the olm greedy-only path, the apc sampling-only path, and the pba greedy path.

### Requirement 2: Hugging Face Qwen API with chat templates

**User Story:** As a researcher, I want the Qwen scripts to use the Hugging Face Qwen API with chat templates, so that the implementation matches the Qwen2.5-VL model contract rather than LLaVA internals.

#### Acceptance Criteria

1. THE Qwen_Inference_Script SHALL load the model using `Qwen2_5_VLForConditionalGeneration` together with `AutoProcessor` from the HF_API.
2. THE Qwen_Inference_Script SHALL construct each model input as a Chat_Message using the processor chat template containing the image reference and the question text.
3. THE Qwen_Inference_Script SHALL NOT use `IMAGE_TOKEN_INDEX`, `conv_templates`, or `tokenizer_image_token`.
4. THE Qwen_Inference_Script for the pba Method_Family SHALL append the PBA_Suffix to the question text, and THE Qwen_Inference_Script for every other Method_Family SHALL append the Instruction_Suffix to the question text.

### Requirement 3: Preserve LLaVA reference scripts

**User Story:** As a researcher, I want the existing LLaVA scripts left untouched, so that they remain available as a reference implementation.

#### Acceptance Criteria

1. THE feature SHALL leave every existing LLaVA inference script under `inference/` unchanged.
2. THE feature SHALL leave every existing LLaVA shell driver under `scripts/` unchanged.

### Requirement 4: Separable output and unchanged evaluator

**User Story:** As a researcher, I want Qwen output written alongside LLaVA output but separable by filename, so that both result sets coexist and the evaluator runs without modification.

#### Acceptance Criteria

1. THE Qwen_Inference_Script SHALL write its output into the existing per-method Output_Directory used by the corresponding LLaVA method.
2. THE Qwen_Inference_Script SHALL write greedy output to a file named with the Filename_Prefix followed by `greedy.jsonl` and sampling output to a file named with the Filename_Prefix followed by `sample.jsonl`.
3. THE Qwen_Inference_Script SHALL write each Answer_Record with exactly the keys `question_id`, `prompt`, `text`, `answer_id`, `model_id`, and `metadata`, built using `build_answer_record`.
4. THE Qwen_Inference_Script SHALL set the `model_id` field of each Answer_Record to the Qwen model name.
5. THE feature SHALL leave `eval/mme_eval.py` unchanged.

### Requirement 5: Reuse shared pure helpers

**User Story:** As a researcher, I want the Qwen scripts to reuse the shared pure helpers, so that chunking, sampling, and record-building behavior stays identical to the LLaVA pipeline.

#### Acceptance Criteria

1. THE Qwen_Inference_Script SHALL import and use `split_list` and `get_chunk` from Common_Helpers for question chunking.
2. THE Qwen_Inference_Script SHALL determine the `do_sample` flag using `derive_do_sample` from Common_Helpers.
3. THE Qwen_Inference_Script SHALL build each Answer_Record using `build_answer_record` from Common_Helpers.
4. THE feature SHALL leave `inference/mme_infer_common.py` unchanged.

### Requirement 6: Configurable model checkpoint and decoding

**User Story:** As a researcher, I want a configurable model checkpoint, so that I can point the Qwen scripts at a local model by default and override it when needed.

#### Acceptance Criteria

1. WHERE no override is provided, THE Qwen_Shell_Driver SHALL use the Default_Checkpoint as the model path.
2. WHERE the MODEL_PATH environment variable is set, THE Qwen_Shell_Driver SHALL use its value as the model path.
3. THE Qwen_Inference_Script SHALL run Greedy_Decoding with temperature 0 and Sample_Decoding with temperature 1.
4. THE Qwen_Inference_Script SHALL omit the `--conv-mode` command-line argument used by the LLaVA scripts.

### Requirement 7: Qwen shell drivers

**User Story:** As a researcher, I want shell drivers for the Qwen scripts, so that I can launch the full Qwen benchmark with the same workflow as the LLaVA drivers.

#### Acceptance Criteria

1. THE feature SHALL provide a Qwen_Shell_Driver mirroring `scripts/mme_infer_base.sh` that runs the base Qwen_Inference_Script for both Greedy_Decoding and Sample_Decoding.
2. THE feature SHALL provide a Qwen_Shell_Driver mirroring `scripts/mme_infer_cd.sh` that runs the cd Qwen_Inference_Script for vcd, icd, and sid, each for both Greedy_Decoding and Sample_Decoding.
3. THE feature SHALL provide a Qwen_Shell_Driver mirroring `scripts/mme_infer_spurious.sh` that runs the pba, olm, and apc Qwen_Inference_Scripts following each method's decoding path.
4. THE Qwen_Shell_Driver SHALL write each method's output into the corresponding Output_Directory using the Filename_Prefix.
