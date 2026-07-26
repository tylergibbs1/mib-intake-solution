#!/usr/bin/env bash
# Experiment harness: rules-only dev score (selection) + honest ensemble
# holdout score (report). Prints METRIC lines; all detail goes to run.log.
set -uo pipefail
cd "$(dirname "$0")"
PY=/Users/tylergibbs/Projects/8090chalfable/.venv/bin/python
export MIB_CACHE=/Users/tylergibbs/Projects/8090chalfable/cache/evidence2

$PY dev_run.py predict > run.log 2>&1 || { echo "DEV=CRASH"; exit 1; }
DEV=$($PY dev_run.py score dev 2>>run.log | awk '/Deterministic score/ {print $3}')
DEVFA=$($PY dev_run.py score dev 2>>run.log | awk '/Catastrophic/ {print $4}')

$PY - >> run.log 2>&1 <<'EOF'
import json
mine = {}
for line in open('/Users/tylergibbs/Projects/8090chalfable/cache/predictions.jsonl'):
    r = json.loads(line)
    mine[r['case_id']] = r
theirs = {}
for line in open('/Users/tylergibbs/Projects/8090chalfable/cache/other_holdout.jsonl'):
    r = json.loads(line)
    theirs[r['case_id']] = r
DELEGATE = {'clean_strong', 'clean_weak', 'damaged_packet', 'sponsor_blank'}
out = []
for cid, t in theirs.items():
    m = dict(mine[cid])
    if m.get('_path') in DELEGATE:
        m['adjudication'] = t['adjudication']
        m['confidence'] = t['confidence']
    m.pop('_path', None)
    out.append(m)
with open('/Users/tylergibbs/Projects/8090chalfable/cache/ensemble_holdout.jsonl', 'w') as f:
    for r in out:
        f.write(json.dumps(r) + '\n')
EOF

HOLDRAW=$(python3 /Users/tylergibbs/Projects/8090chalfable/mib-doc-challenge/scripts/evaluate.py \
  --truth /Users/tylergibbs/Projects/8090chalfable/cache/labels_holdout.csv \
  --submission /Users/tylergibbs/Projects/8090chalfable/cache/ensemble_holdout.jsonl 2>>run.log)
HOLD=$(echo "$HOLDRAW" | awk '/Deterministic score/ {print $3}')
HOLDFA=$(echo "$HOLDRAW" | awk '/Catastrophic/ {print $4}')

echo "DEV=${DEV:-CRASH} DEVFA=${DEVFA:-?} HOLD=${HOLD:-CRASH} HOLDFA=${HOLDFA:-?}"
