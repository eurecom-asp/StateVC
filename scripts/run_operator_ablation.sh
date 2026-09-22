#!/usr/bin/env bash
set -euo pipefail

SOURCE=${1:?source wav required}
REFERENCE=${2:?reference wav required}
OUTDIR=${3:?output directory required}
DEVICE=${DEVICE:-cuda}
mkdir -p "$OUTDIR"

for OP in mean diagonal block; do
  python convert.py \
    --source "$SOURCE" \
    --reference "$REFERENCE" \
    --output "$OUTDIR/${OP}.wav" \
    --device "$DEVICE" \
    --operator "$OP" \
    --routing soft \
    --cluster-dim 24 \
    --k-max 4 \
    --n-min 20 \
    --m-min 20 \
    --smooth-window 5 \
    --strength 1.0 \
    --block-size 2 \
    --eps 1e-4
done
