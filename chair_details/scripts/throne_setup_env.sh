#!/usr/bin/env bash
# Install the extra dependencies needed for the THRONE evaluation pipeline
# without disturbing the repo's pinned torch/transformers versions.
#
# Run once on the evaluation machine before throne_generate.sh / throne_eval.sh:
#
#   bash scripts/throne_setup_env.sh
#
# What this does
# --------------
# - pycocotools: required by throne_aqa_evaluation.py and throne_generate_cd.py
# - sentencepiece: required by Flan-T5 tokenizer (google/flan-t5-*)
# - protobuf: sentencepiece dependency
# - requests: used by image loading helpers
#
# What this deliberately does NOT touch
# --------------------------------------
# - torch, torchvision, transformers, Pillow — all pinned in pyproject.toml;
#   reinstalling THRONE's requirements.txt would downgrade transformers from
#   4.31.0 to 4.37.2 and break this repo's custom LLaVA patching.
set -euo pipefail

pip install --quiet \
    pycocotools \
    sentencepiece \
    protobuf \
    requests

echo "THRONE env setup complete."
echo "Verify pycocotools is importable:"
python -c "from pycocotools.coco import COCO; print('  OK:', COCO)"
echo "Verify sentencepiece is importable:"
python -c "import sentencepiece; print('  OK:', sentencepiece.__version__)"
