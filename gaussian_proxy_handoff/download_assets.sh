#!/usr/bin/env bash
# =====================================================================
# download_assets.sh -- fetch the model weights and the 500 COCO images that
# are NOT stored in git (too large / redundant). Run this once after cloning,
# before run_all.sh. Everything lands in the exact paths the scripts expect.
#
#   models/llava-v1.5-7b/                (~13 GB, liuhaotian/llava-v1.5-7b)
#   models/Qwen2.5-VL-7B-Instruct/       (~16 GB, Qwen/Qwen2.5-VL-7B-Instruct)
#   data/coco/annotations/               instances_val2017.json, captions_val2017.json
#   data/coco/val2017/                   the 500 images listed in image_ids_500.json
#
# Idempotent: existing files are skipped.
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"

echo "########## 1/3  model weights ##########"
mkdir -p models
if [ ! -f models/llava-v1.5-7b/config.json ]; then
  echo ">>> downloading liuhaotian/llava-v1.5-7b"
  huggingface-cli download liuhaotian/llava-v1.5-7b --local-dir models/llava-v1.5-7b
else
  echo "llava-v1.5-7b already present, skipping"
fi
if [ ! -f models/Qwen2.5-VL-7B-Instruct/config.json ]; then
  echo ">>> downloading Qwen/Qwen2.5-VL-7B-Instruct"
  huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/Qwen2.5-VL-7B-Instruct
else
  echo "Qwen2.5-VL-7B-Instruct already present, skipping"
fi

echo "########## 2/3  COCO annotations ##########"
mkdir -p data/coco/annotations
if [ ! -f data/coco/annotations/instances_val2017.json ]; then
  echo ">>> downloading COCO 2017 annotations (~241 MB)"
  tmp=$(mktemp -d)
  wget -q --show-progress -O "$tmp/ann.zip" http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  python - "$tmp/ann.zip" <<'PY'
import sys, zipfile, shutil, os
z = zipfile.ZipFile(sys.argv[1])
for name in ("annotations/instances_val2017.json", "annotations/captions_val2017.json"):
    with z.open(name) as src, open(os.path.join("data/coco/annotations", os.path.basename(name)), "wb") as dst:
        shutil.copyfileobj(src, dst)
print("extracted instances_val2017.json + captions_val2017.json")
PY
  rm -rf "$tmp"
else
  echo "annotations already present, skipping"
fi

echo "########## 3/3  the 500 COCO val2017 images ##########"
mkdir -p data/coco/val2017
python - <<'PY'
import json, os, urllib.request
ids = json.load(open("image_ids_500.json"))
base = "http://images.cocodataset.org/val2017/"
got = skip = 0
for i in ids:
    fn = f"{i:012d}.jpg"
    dst = os.path.join("data/coco/val2017", fn)
    if os.path.exists(dst):
        skip += 1; continue
    urllib.request.urlretrieve(base + fn, dst)
    got += 1
    if got % 50 == 0:
        print(f"  downloaded {got} images...", flush=True)
print(f"images: downloaded {got}, already had {skip}, total {len(ids)}")
PY

echo "ALL ASSETS READY. Now run:  bash run_all.sh --smoke"
