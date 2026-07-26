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
import csv, json, sys
import joblib
import numpy as np
sys.path.insert(0, '/Users/tylergibbs/Projects/8090chalfable/solution')
import solution
BASE = '/Users/tylergibbs/Projects/8090chalfable'
mine = {}
for line in open(f'{BASE}/cache/predictions.jsonl'):
    r = json.loads(line)
    mine[r['case_id']] = r
truth = sorted(r['case_id'] for r in csv.DictReader(open(f'{BASE}/mib-doc-challenge/data/train_labels.csv')))
hold = [cid for i, cid in enumerate(truth) if i % 10 >= 7]
bundle = joblib.load(f'{BASE}/solution/models/graybox_dev.joblib')
model, classes = bundle['model'], list(bundle['classes'])
out = []
for cid in hold:
    m = dict(mine[cid])
    if m.get('_path') in solution.DELEGATED_BUCKETS:
        e = json.loads(open(f'{BASE}/cache/evidence2/{cid}.json').read())
        for p in e['pages']:
            solution.enrich_page(p)
        probs = model.predict_proba(np.array([solution.graybox_vector(e)]))[0]
        adj, conf = solution.graybox_decide(probs, classes)
        m['adjudication'] = adj
        m['confidence'] = conf
    m.pop('_path', None)
    out.append(m)
with open(f'{BASE}/cache/ensemble_holdout.jsonl', 'w') as f:
    for r in out:
        f.write(json.dumps(r) + '\n')
EOF

HOLDRAW=$(python3 /Users/tylergibbs/Projects/8090chalfable/mib-doc-challenge/scripts/evaluate.py \
  --truth /Users/tylergibbs/Projects/8090chalfable/cache/labels_holdout.csv \
  --submission /Users/tylergibbs/Projects/8090chalfable/cache/ensemble_holdout.jsonl 2>>run.log)
HOLD=$(echo "$HOLDRAW" | awk '/Deterministic score/ {print $3}')
HOLDFA=$(echo "$HOLDRAW" | awk '/Catastrophic/ {print $4}')

echo "DEV=${DEV:-CRASH} DEVFA=${DEVFA:-?} HOLD=${HOLD:-CRASH} HOLDFA=${HOLDFA:-?}"
