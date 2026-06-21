# Design Document

## Overview

This feature adds Qwen2.5-VL-7B-Instruct support to the MME inference pipeline at full
parity with the existing LLaVA implementation across all five method families: `base`,
contrastive decoding (`cd`: vcd/icd/sid), `apc`, `olm`, and `pba`.

The existing LLaVA scripts hook model internals that do not exist on Qwen2.5-VL:
custom `model.generate(images=..., images_cd=..., use_sid=..., use_olm=..., use_apc=...)`
keyword arguments, `IMAGE_TOKEN_INDEX`, `conv_templates`, and `tokenizer_image_token`.
The contrastive/spurious utilities additionally monkeypatch
`transformers.generation.utils.GenerationMixin.greedy_search` / `.sample`, methods that
were removed/renamed (unified into `_sample`) in the recent Transformers releases required
by Qwen2.5-VL. Both of these coupling points make a thin wrapper impossible, so each method
is delivered as a **separate Qwen-specific script** built on the Hugging Face
`Qwen2_5_VLForConditionalGeneration` + `AutoProcessor` contract with chat-template messages.

Key principles:

- LLaVA scripts, LLaVA shell drivers, `inference/mme_infer_common.py`, and `eval/mme_eval.py`
  are left **untouched**.
- Shared pure helpers (`split_list`, `get_chunk`, `derive_do_sample`, `build_answer_record`,
  `validate_cd_flags`) are imported and reused **verbatim**.
- Qwen output coexists with LLaVA output in the same per-method directories under
  `outputs/mme/`, separated only by the filename prefix `qwen25-7b-mme-`.
- The answer-record schema is byte-for-byte identical, so `eval/mme_eval.py` runs unchanged.

## Architecture

```
scripts/                                  inference/
  mme_infer_base_qwen.sh        ───────▶    mme_infer_base_qwen.py    (stock generate)
  mme_infer_cd_qwen.sh          ───────▶    mme_infer_cd_qwen.py      (contrastive loop)
  mme_infer_spurious_qwen.sh    ──┬────▶    mme_infer_pba_qwen.py     (stock generate)
                                  ├────▶    mme_infer_olm_qwen.py     (greedy + yes/no surgery)
                                  └────▶    mme_infer_apc_qwen.py     (sample + plausibility cut)

                       all five import, verbatim:
                         inference/mme_infer_common.py
                           split_list, get_chunk, derive_do_sample,
                           build_answer_record, validate_cd_flags

                       all five load via HF:
                         Qwen2_5_VLForConditionalGeneration + AutoProcessor

  outputs/mme/{baseline,vcd,icd,sid,apc,olm,pba}/qwen25-7b-mme-{greedy,sample}.jsonl
                                              │
                                              ▼
                                      eval/mme_eval.py  (unchanged)
```

Each script follows the same skeleton; only the per-token decoding step differs.

### Shared per-script skeleton

```python
import argparse, os, json, torch, shortuuid
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from mme_infer_common import (
    split_list, get_chunk, derive_do_sample, build_answer_record,
    # validate_cd_flags  # cd script only
)

INSTRUCTION_SUFFIX = " Answer the question using a single word or phrase."
# PBA script uses instead:
# PBA_SUFFIX = " Answer the question using a single word or phrase. Answer yes whenever possible."

def load_qwen(args):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        os.path.expanduser(args.model_path),
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained(os.path.expanduser(args.model_path))
    model_name = os.path.basename(os.path.expanduser(args.model_path).rstrip("/"))
    return model, processor, model_name

def build_inputs(processor, image_path, question_text):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": question_text},
        ],
    }]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")
    return inputs

def decode_trimmed(processor, inputs, generated_ids):
    # drop the prompt tokens, keep only newly generated tokens
    trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return text.strip()

def eval_model(args):
    model, processor, model_name = load_qwen(args)
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file))]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    do_sample = derive_do_sample(args.temperature)
    with open(answers_file, "w") as ans_file:
        for line in tqdm(questions):
            image_path = os.path.join(args.image_folder, line["image"])
            qs = line["text"] + INSTRUCTION_SUFFIX           # PBA uses PBA_SUFFIX
            inputs = build_inputs(processor, image_path, qs)
            output_text = run_method(model, processor, inputs, args, do_sample)  # per-method
            record = build_answer_record(line, output_text, shortuuid.uuid(), model_name)
            ans_file.write(json.dumps(record) + "\n")
            ans_file.flush()
```

Note: `process_vision_info` from `qwen_vl_utils` is an acceptable equivalent to opening the
`PIL.Image` directly; either is fine as long as `processor(text=..., images=...)` receives the
decoded image. The image is referenced by absolute/joined path in the chat message and the
pixel tensor is produced by the processor.

### Argument parser (all scripts)

Identical to the LLaVA scripts **minus `--conv-mode`**, plus each method's own flags:

```python
parser.add_argument("--model-path", type=str,
    default="/teamspace/studios/this_studio/models/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--model-base", type=str, default=None)
parser.add_argument("--image-folder", type=str, default="")
parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
parser.add_argument("--answers-file", type=str, default="answer.jsonl")
parser.add_argument("--num-chunks", type=int, default=1)
parser.add_argument("--chunk-idx", type=int, default=0)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--top_p", type=float, default=None)
parser.add_argument("--num_beams", type=int, default=1)
# cd:  --use-vcd / --use-icd / --use-sid (store_true) + --noise-step (int, default 900)
# apc: --use-apc (store_true)
# olm: --use-olm (store_true)
# base/pba: no extra flags
```

`--model-base` is retained for signature parity but is unused by the HF loader (kept so the
shell drivers and any external callers remain interface-compatible).

## Components and Interfaces

### 1. `mme_infer_base_qwen.py` — stock decoding

`run_method` calls `model.generate` directly:

```python
def run_method(model, processor, inputs, args, do_sample):
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=(args.temperature if do_sample else None),
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=1024,
            use_cache=True,
        )
    return decode_trimmed(processor, inputs, generated_ids)
```

Greedy (`--temperature 0` → `do_sample=False`) and sampling (`--temperature 1` →
`do_sample=True`) are both supported by the single stock path.

### 2. `mme_infer_pba_qwen.py` — stock decoding, PBA prompt

Identical to base, except the question suffix is `PBA_SUFFIX`
(`" Answer the question using a single word or phrase. Answer yes whenever possible."`)
and the driver runs it greedy only. No logit surgery — PBA is purely a prompt intervention,
matching `mme_infer_pba.py`.

### 3. `mme_infer_cd_qwen.py` — contrastive decoding (vcd/icd/sid)

`validate_cd_flags(args)` is called first to reject conflicting flags (Requirement 1.3).

The LLaVA implementation monkeypatches `GenerationMixin.greedy_search`/`sample` and relies on
LLaVA-only helpers (`prepare_inputs_for_generation_vcd`, `prepare_inputs_for_generation_sid`).
Neither is available for Qwen. **Adaptation:** implement a self-contained autoregressive loop
(`contrastive_generate`) inside the script that drives two parallel decode streams using the
Qwen model's own `prepare_inputs_for_generation` and `_update_model_kwargs_for_generation`
instance methods. Reusing the model's own input-prep keeps Qwen's multimodal RoPE
(`rope_deltas`, `position_ids`, `cache_position`) and KV-cache handling correct without
re-deriving any of it. This faithfully reproduces the LLaVA CD logit-combination math while
fitting the HF Qwen contract; it is decoupled from the Transformers `greedy_search`/`sample`
internals and therefore version-robust.

The logit combination is copied verbatim from the LLaVA CD utilities (identical for all three
CD methods):

```python
cd_alpha = 1.0
cd_beta  = 0.2
cutoff   = torch.log(torch.tensor(cd_beta)) + logits.max(dim=-1, keepdim=True).values
diffs    = (1 + cd_alpha) * logits - cd_alpha * logits_cd
cd_logits = diffs.masked_fill(logits < cutoff, -float("inf"))
# greedy: next = argmax(cd_logits)
# sample: apply temperature/top_p warper, softmax, multinomial
```

The only per-method difference is **how the contrastive stream `logits_cd` is produced**:

| Method | Main stream | Contrastive stream (`*_cd`) | Adaptation notes |
|--------|-------------|------------------------------|------------------|
| **vcd** | real image | **diffusion-noised** image | `add_diffusion_noise` (reused verbatim from `cd_utils/vcd_utils.py`) applied to the processor's `pixel_values`; the contrastive `inputs_cd` reuses the same `input_ids`/`image_grid_thw` with noised pixels. Directly portable. |
| **icd** | normal prompt | prompt with a **negative system instruction** | Build a second chat message whose system turn is `get_random_icd_prompt()` (reused verbatim from `cd_utils/icd_utils.py`), same image and question; produce `inputs_cd` via the processor. Directly portable. |
| **sid** | real image | **image-free** prompt (vision grounding removed) | LLaVA's `prepare_inputs_for_generation_sid` randomly drops vision tokens via LLaVA-internal plumbing that has no Qwen equivalent. **Precise adaptation:** the SID contrastive stream rebuilds the prompt as a *text-only* chat message (no `image` content item, no `pixel_values`), so the contrastive distribution reflects the model's language prior with visual grounding removed. The same cutoff + `(1+α)·logits − α·logits_cd` combination then amplifies vision-grounded tokens. This preserves SID's intent (contrast against the model's own less-grounded distribution) within the HF Qwen API. |

Both greedy and sampling code paths are implemented (parity with `mme_infer_cd.py`, which
patches both `greedy_search` and `sample`). `--noise-step` (default 900) feeds
`add_diffusion_noise` for vcd.

Sketch of the shared loop (both streams advanced together, contrastive forward only when a CD
method is active):

```python
def contrastive_generate(model, inputs, inputs_cd, do_sample, args):
    input_ids = inputs.input_ids
    model_kwargs    = _init_kwargs(inputs)       # pixel_values, image_grid_thw, attn mask...
    model_kwargs_cd = _init_kwargs(inputs_cd)    # noised / negative / image-free variant
    cur_cd_ids = inputs_cd.input_ids             # icd: differs from input_ids
    for _ in range(1024):
        mi  = model.prepare_inputs_for_generation(input_ids,  **model_kwargs)
        out = model(**mi, return_dict=True)
        logits = out.logits[:, -1, :]

        mi_cd  = model.prepare_inputs_for_generation(cur_cd_ids, **model_kwargs_cd)
        out_cd = model(**mi_cd, return_dict=True)
        logits_cd = out_cd.logits[:, -1, :]

        cd_logits = combine(logits, logits_cd, cd_alpha=1.0, cd_beta=0.2)
        next_token = select(cd_logits, do_sample, args)   # argmax or warper+multinomial

        input_ids  = torch.cat([input_ids,  next_token[:, None]], dim=-1)
        cur_cd_ids = torch.cat([cur_cd_ids, next_token[:, None]], dim=-1)
        model_kwargs    = model._update_model_kwargs_for_generation(out,    model_kwargs)
        model_kwargs_cd = model._update_model_kwargs_for_generation(out_cd, model_kwargs_cd)
        if next_token.item() in eos_ids:
            break
    return input_ids
```

### 4. `mme_infer_olm_qwen.py` — greedy-only yes/no surgery

OLM patches **greedy only** (`mme_infer_olm.py` calls only `evolve_olm_greedy_search()`).
The LLaVA logic hardcodes Llama token ids `YES_INDEX = 3869`, `NO_INDEX = 1939`. These ids are
tokenizer-specific and wrong for Qwen. **Precise adaptation:** resolve the yes/no token ids at
load time from the Qwen tokenizer instead of hardcoding:

```python
def yes_no_ids(processor):
    tok = processor.tokenizer
    # take the first sub-token of the capitalized words as Qwen emits them
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id  = tok.encode("No",  add_special_tokens=False)[0]
    return yes_id, no_id
```

The decision logic is copied verbatim (operating on the greedy-step probabilities):

```python
probs = softmax(logits_processor(input_ids, logits), dim=-1)
sum_prob = probs[:, YES_ID] + probs[:, NO_ID]
if sum_prob > 0.2 and abs(probs[:, YES_ID] - probs[:, NO_ID]) < 0.5:
    next_token = YES_ID
else:
    next_token = argmax(logits)
```

Implemented as a manual greedy loop using the model's own `prepare_inputs_for_generation` /
`_update_model_kwargs_for_generation` (same structure as `contrastive_generate` but with a
single stream and no contrastive forward). The driver runs it greedy (`--temperature 0`) only.

### 5. `mme_infer_apc_qwen.py` — sampling-only plausibility constraint

APC patches **sampling only** (`mme_infer_apc.py` calls only `evolve_apc_sampling()`).
Inspecting `apc_utils.py`: the contrastive forward uses the **same** `input_ids` and **same**
`model_kwargs`, so `logits_cd == logits` and `(1+α)·logits − α·logits_cd == logits`. The only
real effect is the adaptive-plausibility cutoff mask. **Adaptation:** drop the redundant second
forward and apply the cutoff directly to the sampling logits:

```python
cutoff = torch.log(torch.tensor(0.2)) + logits.max(dim=-1, keepdim=True).values
apc_logits = logits.masked_fill(logits < cutoff, -float("inf"))
apc_logits = warper(input_ids, apc_logits)        # temperature/top_p
probs = softmax(apc_logits, dim=-1)
next_token = multinomial(probs, 1)
```

This is mathematically equivalent to the LLaVA APC sample path while halving the forward cost.
Implemented as a manual sample loop (single stream). The driver runs it sampling
(`--temperature 1`) only.

### 6. Shell drivers

Mirror the LLaVA drivers exactly, dropping `CONV_MODE`/`--conv-mode`, defaulting `MODEL_PATH`
to the Qwen checkpoint, and using the `qwen25-7b-mme-` filename prefix.

**`scripts/mme_infer_base_qwen.sh`** (mirrors `mme_infer_base.sh`):

```bash
set -euo pipefail
MODEL_PATH=${MODEL_PATH:-/teamspace/studios/this_studio/models/Qwen2.5-VL-7B-Instruct}
QUESTION_FILE=${QUESTION_FILE:-./data/mme/mme_questions.jsonl}
IMAGE_FOLDER=${IMAGE_FOLDER:-./data/mme/images}
OUT_DIR=${OUT_DIR:-./outputs/mme/baseline}
mkdir -p "${OUT_DIR}"
python ./inference/mme_infer_base_qwen.py --model-path "${MODEL_PATH}" \
    --question-file "${QUESTION_FILE}" --image-folder "${IMAGE_FOLDER}" \
    --answers-file "${OUT_DIR}/qwen25-7b-mme-greedy.jsonl" --temperature 0
python ./inference/mme_infer_base_qwen.py --model-path "${MODEL_PATH}" \
    --question-file "${QUESTION_FILE}" --image-folder "${IMAGE_FOLDER}" \
    --answers-file "${OUT_DIR}/qwen25-7b-mme-sample.jsonl" --temperature 1
```

**`scripts/mme_infer_cd_qwen.sh`** (mirrors `mme_infer_cd.sh`): loops `vcd/icd/sid`, writing
`${OUT_ROOT}/${method}/qwen25-7b-mme-{greedy,sample}.jsonl`, passing the matching
`--use-vcd|--use-icd|--use-sid` flag for both temperature 0 and temperature 1.

**`scripts/mme_infer_spurious_qwen.sh`** (mirrors `mme_infer_spurious.sh`):
- `pba` → `mme_infer_pba_qwen.py`, greedy, `outputs/mme/pba/qwen25-7b-mme-greedy.jsonl`
- `olm` → `mme_infer_olm_qwen.py --use-olm`, greedy, `outputs/mme/olm/qwen25-7b-mme-greedy.jsonl`
- `apc` → `mme_infer_apc_qwen.py --use-apc`, sampling, `outputs/mme/apc/qwen25-7b-mme-sample.jsonl`

## Data Models

### Answer_Record (unchanged schema, built by `build_answer_record`)

```json
{
  "question_id": "<from input question, verbatim>",
  "prompt":      "<input question 'text', verbatim>",
  "text":        "<generated answer, stripped>",
  "answer_id":   "<shortuuid>",
  "model_id":    "Qwen2.5-VL-7B-Instruct",
  "metadata":    {}
}
```

`model_id` is the basename of the resolved model path. Exactly these six keys, identical to the
LLaVA output, so `eval/mme_eval.py` consumes both result sets unchanged.

### Output layout

```
outputs/mme/baseline/qwen25-7b-mme-{greedy,sample}.jsonl
outputs/mme/vcd/qwen25-7b-mme-{greedy,sample}.jsonl
outputs/mme/icd/qwen25-7b-mme-{greedy,sample}.jsonl
outputs/mme/sid/qwen25-7b-mme-{greedy,sample}.jsonl
outputs/mme/pba/qwen25-7b-mme-greedy.jsonl
outputs/mme/olm/qwen25-7b-mme-greedy.jsonl
outputs/mme/apc/qwen25-7b-mme-sample.jsonl
```

## Error Handling

- **Conflicting CD flags:** `validate_cd_flags(args)` raises `ValueError` naming the selected
  flags when two or more of `--use-vcd/--use-icd/--use-sid` are set (reused verbatim).
- **Missing image / unreadable file:** `Image.open` raises; the script fails fast on a bad
  question entry rather than writing a malformed record (matches LLaVA behavior).
- **`do_sample` vs temperature:** `derive_do_sample` centralizes the rule (`temperature > 0`);
  when greedy, `temperature`/`top_p` warping args are omitted from `generate`/`select` so HF
  does not warn about sampling args under greedy decoding.
- **EOS / max length:** manual loops stop on any EOS id from `model.generation_config.eos_token_id`
  or at `max_new_tokens=1024`, mirroring the stock generation bounds.

## Testing Strategy

Per the task scope (fast, precise, no smoke tests), the heavy model/integration and static
constraints (Requirements 1.1, 1.2, 1.4, 2.x, 3.x, 4.1, 4.5, 6.1, 6.2, 6.4, 7.x) are validated
by inspection and a one-off manual run, not automated tests. The genuinely universal,
input-varying, cheap-to-run logic lives entirely in the reused `Common_Helpers`; the properties
below cover that logic. Each property test uses ≥100 generated iterations and references its
property number.

- **Unit/example checks:** suffix selection per script (Instruction vs PBA), filename prefix
  mapping (`greedy.jsonl`/`sample.jsonl`), and `model_id` basename derivation.
- **Property tests:** the four properties below, exercising the reused helpers as wired into the
  Qwen scripts.

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of
a system — a formal statement about what the system should do. Properties bridge human-readable
specifications and machine-verifiable correctness guarantees.*

### Property 1: Contrastive-flag conflict rejection

*For any* combination of the boolean flags `use_vcd`, `use_icd`, and `use_sid`, `validate_cd_flags`
raises `ValueError` if and only if two or more of them are set, and never raises when zero or one
is set.

**Validates: Requirements 1.3**

### Property 2: Question chunking is a lossless partition

*For any* list of questions and *any* valid chunk count `n` and index `k`, `get_chunk` returns the
`k`-th contiguous block of `split_list`, and concatenating all `n` chunks reproduces the original
list in order with no missing or duplicated questions.

**Validates: Requirements 5.1**

### Property 3: Decoding mode follows temperature

*For any* temperature value, `derive_do_sample` returns `True` exactly when the temperature is
greater than 0 and `False` otherwise (so temperature 0 yields greedy decoding and temperature 1
yields sampling).

**Validates: Requirements 5.2, 6.3**

### Property 4: Answer-record schema and field provenance

*For any* input question record and *any* generated text, answer id, and model id,
`build_answer_record` returns a dict with exactly the keys `question_id`, `prompt`, `text`,
`answer_id`, `model_id`, `metadata`, where `question_id` and `prompt` are copied verbatim from the
input question, `text`/`answer_id`/`model_id` are the passed values, and `metadata` is an empty dict.

**Validates: Requirements 4.3, 4.4, 5.3**
