#!/usr/bin/env bash
# Full local asset download: model weights + the 500 COCO images + COCO
# annotations. Use this when running directly on a GPU machine (Lightning
# AI Studio, or any other machine with a CUDA GPU) instead of through
# modal_app.py -- the generation scripts (common/generate_llava.py,
# common/generate_llava_icd.py, qwen/generate_qwen.py,
# qwen/generate_qwen_icd.py) have no Modal dependency at all and run
# directly once these assets exist locally. Idempotent -- skips anything
# already present, safe to re-run/resume.
#
# Usage:
#   bash download_assets.sh            # both models
#   bash download_assets.sh llava      # LLaVA only
#   bash download_assets.sh qwen       # Qwen only
set -euo pipefail
cd "$(dirname "$0")"

WHAT="${1:-both}"

echo "== model weights =="
mkdir -p models
# Uses huggingface_hub's Python API (snapshot_download) directly rather than
# shelling out to a CLI -- huggingface_hub renamed its CLI from
# `huggingface-cli` to `hf` in newer releases and the old name now just
# prints a deprecation notice and does nothing (silently downloads
# nothing), so pinning to either CLI name is fragile across versions. The
# Python API has been stable and is what modal_app.py already uses.
if [ "$WHAT" = "both" ] || [ "$WHAT" = "llava" ]; then
  if [ ! -f models/llava-v1.5-7b/config.json ]; then
    python -c "from huggingface_hub import snapshot_download; snapshot_download('liuhaotian/llava-v1.5-7b', local_dir='models/llava-v1.5-7b')"
  fi
fi
if [ "$WHAT" = "both" ] || [ "$WHAT" = "qwen" ]; then
  if [ ! -f models/Qwen2.5-VL-7B-Instruct/config.json ]; then
    python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='models/Qwen2.5-VL-7B-Instruct')"
  fi
fi

echo "== COCO val2017 annotations =="
mkdir -p data/coco/annotations
if [ ! -f data/coco/annotations/instances_val2017.json ]; then
  tmp=$(mktemp -d)
  wget -q --show-progress -O "$tmp/a.zip" http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  python - "$tmp/a.zip" <<'PY'
import sys, zipfile, shutil, os
z = zipfile.ZipFile(sys.argv[1])
for name in ("annotations/instances_val2017.json", "annotations/captions_val2017.json"):
    with z.open(name) as s, open(os.path.join("data/coco/annotations", os.path.basename(name)), "wb") as d:
        shutil.copyfileobj(s, d)
PY
  rm -rf "$tmp"
fi

echo "== the 500 COCO val2017 images =="
mkdir -p data/coco/val2017
python - <<'PY'
import json, os, urllib.request
ids = json.load(open("image_ids_500.json")); base = "http://images.cocodataset.org/val2017/"
got = 0
for i in ids:
    dst = f"data/coco/val2017/{i:012d}.jpg"
    if os.path.exists(dst):
        continue
    urllib.request.urlretrieve(base + f"{i:012d}.jpg", dst); got += 1
    if got % 100 == 0:
        print(f"  {got} images...", flush=True)
print(f"images ready ({len(ids)} total)")
PY
echo "assets ready."
