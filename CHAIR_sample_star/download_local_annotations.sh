#!/usr/bin/env bash
# The analysis scripts (agreement_analysis.py, contrastive_analysis.py,
# plot_d_histograms.py) run locally and only need the small COCO annotation
# JSONs for ground truth -- NOT the model weights or the 500 images, which
# stay on the Modal volume where generation happens. Idempotent.
set -euo pipefail
cd "$(dirname "$0")"

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
echo "annotations ready in data/coco/annotations/"
