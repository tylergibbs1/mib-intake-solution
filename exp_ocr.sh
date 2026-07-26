#!/usr/bin/env bash
# OCR experiment harness: re-extract the fixed 250-case dev subsample with
# current extraction code, then score officially against its labels.
set -uo pipefail
cd "$(dirname "$0")"
PY=/Users/tylergibbs/Projects/8090chalfable/.venv/bin/python
BASE=/Users/tylergibbs/Projects/8090chalfable
export TMPDIR=/private/tmp/claude-501/-Users-tylergibbs-Projects-8090chalfable/036cdba6-723b-4782-9587-8c2d02523d8e/scratchpad
export MIB_TRAIN_DIR=$BASE/cache/ocr_exp_pdfs
export MIB_CACHE=$BASE/cache/ocr_exp_cache

rm -rf "$MIB_CACHE"
$PY dev_run.py extract > run.log 2>&1 || { echo "OCR250=CRASH"; exit 1; }
$PY dev_run.py predict >> run.log 2>&1 || { echo "OCR250=CRASH"; exit 1; }

OUT=$(python3 $BASE/mib-doc-challenge/scripts/evaluate.py \
  --truth $BASE/cache/labels_ocr250.csv \
  --submission $BASE/cache/predictions.jsonl 2>>run.log)
TOTAL=$(echo "$OUT" | awk '/Deterministic score/ {print $3}')
EXTR=$(echo "$OUT" | awk '/Field extraction/ {print $3}')
echo "OCR250=${TOTAL:-CRASH} EXTR=${EXTR:-?}"
