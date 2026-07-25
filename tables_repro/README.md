# POPE table reproduction

Code to reproduce the original POPE object-hallucination tables for two
vision-language models:

- `llava1.5-7b/` : LLaVA-1.5-7B (Vicuna-7B + CLIP ViT-L/14@336)
- `qwen2.5-7B/`  : Qwen2.5-VL-7B-Instruct

Each model is evaluated on the three POPE datasets (**GQA**, **COCO**, **AOKVQA**)
under the three negative-sampling splits (**random**, **popular**, **adversarial**),
for six decoding methods:

| Method     | Type                     | What it does                                            |
|------------|--------------------------|---------------------------------------------------------|
| `baseline` | greedy                   | plain greedy decoding, no intervention                  |
| `vcd`      | contrastive decoding     | contrast against a diffusion-noised image               |
| `icd`      | contrastive decoding     | contrast against a negative system prompt               |
| `sid`      | contrastive decoding     | contrast against a self-introspective masked branch     |
| `pba`      | naive yes-inflation      | appends "Answer yes whenever possible." to the prompt   |
| `olm`      | naive yes-inflation      | pushes the yes/no token logits at decode time           |

`pba` and `olm` are included as naive controls: they inflate the yes-rate
without any visual reasoning, so they bound how much of a method's change is
attributable to a yes/no threshold shift rather than to grounding.

## Layout

Both model folders share the same layout:

```
inference/   model + decoding code, writes one answer file per run
  cd_utils/         VCD, ICD, SID logit machinery
  spurious_utils/   OLM logit machinery
eval/        scores answer files against the POPE ground truth
scripts/     shell scripts that run the full sweep and the scoring
data/        POPE annotation files (gqa / coco / aokvqa, 3 splits each)
```

`llava1.5-7b/` additionally contains `llava_patch/`, see the LLaVA setup note below.

The `outputs/` folder is not shipped; it is created by the run scripts.

## Requirements

A GPU (an NVIDIA L4 with 24 GB is enough), the model weights, and the POPE
images. Decoding is greedy (`temperature 0`), so results are deterministic
except for VCD, whose contrastive branch draws a fresh Gaussian noise image
per question (the noise *level* is fixed at `--noise-step 900`).

### 1. Point the scripts at the weights and images

The scripts read three environment variables and never contain absolute paths:

```bash
export MODEL_PATH=/path/to/weights          # LLaVA-1.5-7B  or  Qwen2.5-VL-7B-Instruct
export GQA_IMAGE_DIR=/path/to/gqa/images     # GQA images (e.g. 2405722.jpg)
export COCO_IMAGE_DIR=/path/to/coco/val2014  # COCO val2014 images; used by COCO and AOKVQA
```

POPE annotations are already in `data/`. The images are not redistributed here:
GQA questions use GQA images, while both COCO and AOKVQA questions use the COCO
`val2014` images (their `image` fields are `COCO_val2014_*.jpg`).

### 2. Install the model code

**Qwen** has no LLaVA dependency:

```bash
pip install "transformers>=4.49" accelerate qwen-vl-utils torch
```

**LLaVA** needs the LLaVA package, plus a small patch that makes its
`generate` path work on modern `transformers` (the stock code mis-detects the
pre-allocated `DynamicCache` on step 0 and skips image processing) and exposes
the VCD/SID `prepare_inputs` variants:

```bash
git clone https://github.com/haotian-liu/LLaVA
# overlay the two patched model files onto the clone:
cp llava1.5-7b/llava_patch/language_model/llava_llama.py \
   llava1.5-7b/llava_patch/language_model/custom_modeling_llama.py \
   LLaVA/llava/model/language_model/
pip install -e LLaVA
```

### 3. Run

From inside a model folder:

```bash
cd llava1.5-7b        # or  cd qwen2.5-7B
bash scripts/run_all.sh
```

`run_all.sh` runs, in order:

- `scripts/run_baseline.sh`  writes `outputs/pope/baseline/...`
- `scripts/run_cd.sh`        writes `outputs/pope/{vcd,icd,sid}/...`
- `scripts/run_spurious.sh`  writes `outputs/pope/{pba,olm}/...`
- `scripts/run_eval.sh`      scores every method and prints the metrics

You can also run any single stage on its own.

## Scoring

`eval/pope_eval_base.py` takes a ground-truth file and an answer file and prints
accuracy, precision, recall, F1, and the predicted yes-proportion:

```bash
python eval/pope_eval_base.py \
  --ref-files ./data/gqa/gqa_pope_random.json \
  --res-files ./outputs/pope/vcd/llava-7b-gqa-random-greedy.jsonl
```

`eval/pope_eval_transfer.py` compares two answer files against the ground truth
and reports the per-question transition counts (TP->FN, FN->TP, TN->FP, ...),
which is how the yes-inflation of a method relative to the baseline is measured:

```bash
python eval/pope_eval_transfer.py \
  --ref-files    ./data/gqa/gqa_pope_random.json \
  --res-rg-files ./outputs/pope/baseline/llava-7b-gqa-random-greedy.jsonl \
  --res-cd-files ./outputs/pope/olm/llava-7b-gqa-random-greedy.jsonl
```

## Notes on reproducibility

- Decoding is greedy; the only stochastic element is VCD's noised contrastive
  image, drawn fresh per question at a fixed noise level.
- Answer files are JSONL, one line per question, keyed by `question_id`; the
  scorers align on `question_id` and refuse to run on mismatched files.
- The scorers treat any answer containing "yes" as positive and any answer
  containing "no" as negative, matching the original POPE protocol.
