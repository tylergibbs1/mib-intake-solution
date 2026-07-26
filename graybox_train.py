#!/usr/bin/env python3
"""Train the gray-zone adjudication model from cached evidence.

Default trains on the dev split only (honest holdout evaluation). Pass --all
to fit the shipped artifact on all 1000 training cases.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, "/Users/tylergibbs/Projects/8090chalfable/solution")
import solution

BASE = Path("/Users/tylergibbs/Projects/8090chalfable")
CACHE = BASE / "cache/evidence2"
LABELS = BASE / "mib-doc-challenge/data/train_labels.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="train on all cases (shipped artifact)")
    parser.add_argument("--out", default=str(BASE / "solution/models/graybox.joblib"))
    args = parser.parse_args()

    truth = {r["case_id"]: r for r in csv.DictReader(open(LABELS))}
    ids = sorted(truth)
    use = ids if args.all else [cid for i, cid in enumerate(ids) if i % 10 < 7]

    X, y = [], []
    for cid in use:
        e = json.loads((CACHE / f"{cid}.json").read_text())
        for p in e["pages"]:
            solution.enrich_page(p)
        X.append(solution.graybox_vector(e))
        y.append(truth[cid]["adjudication"])
    X, y = np.array(X), np.array(y)
    print("train matrix", X.shape, "(all)" if args.all else "(dev only)")

    model = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3,
                                 class_weight="balanced", random_state=7,
                                 n_jobs=4)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=1)
    classes_order = np.array(sorted(set(y)))
    print("OOF accuracy:", (classes_order[oof.argmax(1)] == y).mean())

    model.fit(X, y)
    joblib.dump({"model": model, "classes": model.classes_}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
