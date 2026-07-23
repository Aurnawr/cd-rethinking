#!/usr/bin/env bash
# Download COCO val2017 images and annotations for THRONE evaluation.
#
# Usage:
#   bash scripts/throne_fetch_coco.sh             # downloads to data/coco/
#   DATA_ROOT=/mydata bash scripts/throne_fetch_coco.sh
#
# Produces:
#   $DATA_ROOT/annotations/instances_val2017.json   (~121 MB)
#   $DATA_ROOT/val2017/                             (~1 GB, 5000 images)
#
# Both paths are the defaults used by throne_generate.sh / throne_eval.sh.
# If you already have a COCO download, set DATA_ROOT to point at it and skip
# this script — the layout just needs to match the above.
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-./data/coco}
ANNO_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL="http://images.cocodataset.org/zips/val2017.zip"

mkdir -p "${DATA_ROOT}"

# --- annotations ---
if [ ! -f "${DATA_ROOT}/annotations/instances_val2017.json" ]; then
    echo "Downloading COCO annotations (~241 MB) …"
    curl -L "${ANNO_URL}" -o "${DATA_ROOT}/annotations_trainval2017.zip"
    unzip -q "${DATA_ROOT}/annotations_trainval2017.zip" -d "${DATA_ROOT}"
    rm  "${DATA_ROOT}/annotations_trainval2017.zip"
    echo "Annotations extracted to ${DATA_ROOT}/annotations/"
else
    echo "Annotations already present — skipping."
fi

# --- val2017 images ---
if [ ! -d "${DATA_ROOT}/val2017" ] || [ -z "$(ls -A "${DATA_ROOT}/val2017" 2>/dev/null)" ]; then
    echo "Downloading COCO val2017 images (~778 MB) …"
    curl -L "${IMG_URL}" -o "${DATA_ROOT}/val2017.zip"
    unzip -q "${DATA_ROOT}/val2017.zip" -d "${DATA_ROOT}"
    rm  "${DATA_ROOT}/val2017.zip"
    echo "Images extracted to ${DATA_ROOT}/val2017/"
else
    echo "val2017 images already present — skipping."
fi

echo ""
echo "COCO data ready at: ${DATA_ROOT}"
echo "  annotations: ${DATA_ROOT}/annotations/instances_val2017.json"
echo "  images:      ${DATA_ROOT}/val2017/"
