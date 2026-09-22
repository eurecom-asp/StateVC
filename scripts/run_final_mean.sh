#!/usr/bin/env bash
set -euo pipefail

SOURCE=${1:?source wav required}
REFERENCE=${2:?reference wav required}
OUTPUT=${3:?output wav required}
DEVICE=${DEVICE:-cuda}

python convert.py \
  --source "$SOURCE" \
  --reference "$REFERENCE" \
  --output "$OUTPUT" \
  --device "$DEVICE" \
  --operator mean \
  --routing soft \
  --cluster-dim 24 \
  --k-max 4 \
  --n-min 20 \
  --m-min 20 \
  --smooth-window 5 \
  --strength 1.0
