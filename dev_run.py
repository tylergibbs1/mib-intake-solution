#!/usr/bin/env python3
"""Dev harness: run extraction over train with caching, then evaluate.

Usage:
  dev_run.py extract [N]    # extract evidence for first N cases (default all)
  dev_run.py predict        # resolve+adjudicate from cache, write predictions
  dev_run.py analyze        # residual analysis against train labels
"""

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import solution

import os

ROOT = Path("/Users/tylergibbs/Projects/8090chalfable/mib-doc-challenge")
TRAIN = Path(os.environ.get("MIB_TRAIN_DIR", str(ROOT / "data/train")))
LABELS = ROOT / "data/train_labels.csv"
import os

CACHE = Path(os.environ.get(
    "MIB_CACHE", "/Users/tylergibbs/Projects/8090chalfable/cache/evidence"))
PRED = Path("/Users/tylergibbs/Projects/8090chalfable/cache/predictions.jsonl")


def _extract_one(pdf):
    cid = pdf.stem
    out = CACHE / f"{cid}.json"
    if out.exists():
        return cid
    evidence = solution.extract_case(str(pdf))
    out.write_text(json.dumps(evidence))
    return cid


def cmd_extract(n=None):
    CACHE.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(TRAIN.glob("*.pdf"))
    if n:
        pdfs = pdfs[:n]
    todo = [p for p in pdfs if not (CACHE / f"{p.stem}.json").exists()]
    print(f"{len(pdfs)} cases, {len(todo)} to extract")
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, cid in enumerate(pool.map(_extract_one, pdfs, chunksize=4)):
            if i % 100 == 0:
                print(i, cid, flush=True)


def load_evidence(cid):
    return json.loads((CACHE / f"{cid}.json").read_text())


def cmd_predict():
    evidences = []
    for f in sorted(CACHE.glob("*.json")):
        evidence = json.loads(f.read_text())
        for page in evidence["pages"]:
            solution.enrich_page(page)
        evidences.append(evidence)

    dates = []
    for e in evidences:
        d, score = solution.resolve_fields(e).get("arrival_date", (None, 0))[:2]
        if d and score >= 0.9:
            dates.append(d)
    cutoff = solution.batch_stale_cutoff(dates)
    print("stale cutoff:", cutoff)

    records = []
    for e in evidences:
        record, path = solution.build_record(e, cutoff)
        record["_path"] = path
        records.append(record)
    with open(PRED, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} predictions to {PRED}")


def holdout_split(truth):
    """Deterministic 70/30 split for rule tuning vs validation."""
    ids = sorted(truth)
    dev = {cid for i, cid in enumerate(ids) if i % 10 < 7}
    return dev, set(ids) - dev


def cmd_analyze(split=None):
    truth = {r["case_id"]: r for r in csv.DictReader(open(LABELS))}
    if split in ("dev", "holdout"):
        dev, hold = holdout_split(truth)
        keep = dev if split == "dev" else hold
        truth = {k: v for k, v in truth.items() if k in keep}
        print(f"[{split} split: {len(truth)} cases]")
    preds = {}
    for line in open(PRED):
        r = json.loads(line)
        preds[r["case_id"]] = r

    from collections import Counter
    adj_conf = Counter()
    field_miss = Counter()
    path_stats = {}
    for cid, t in truth.items():
        p = preds.get(cid)
        if not p:
            adj_conf[(t["adjudication"], "MISSING")] += 1
            continue
        adj_conf[(t["adjudication"], p["adjudication"])] += 1
        path = p.get("_path", "?")
        ps = path_stats.setdefault(path, [0, 0])
        ps[1] += 1
        if p["adjudication"] == t["adjudication"]:
            ps[0] += 1
        for f in ["applicant_name", "species_code", "home_world", "visa_class",
                  "sponsor_id", "arrival_date", "declared_purpose",
                  "risk_flags", "fee_status"]:
            tv, pv = t[f], p[f]
            if f == "risk_flags":
                tv = "|".join(sorted(x for x in tv.split("|") if x and x != "none"))
                pv = "|".join(sorted(x for x in pv.split("|") if x and x != "none"))
            if solution.norm(str(tv)).lower() != solution.norm(str(pv)).lower():
                field_miss[f] += 1

    n = len(truth)
    correct = sum(v for (t_adj, p_adj), v in adj_conf.items() if t_adj == p_adj)
    print(f"adjudication accuracy: {correct}/{n} = {correct/n:.3f}")
    print("confusion:")
    for (t_adj, p_adj), v in sorted(adj_conf.items()):
        marker = " <-- FALSE APPROVAL" if (t_adj == "DENIED" and p_adj == "APPROVED") else ""
        print(f"  {t_adj:13s} -> {p_adj:13s} {v}{marker}")
    print("field misses (out of", n, "):")
    for f, v in field_miss.most_common():
        print(f"  {f:18s} {v}")
    print("decision-path accuracy:")
    for path, (c, tot) in sorted(path_stats.items(), key=lambda kv: -kv[1][1]):
        print(f"  {path:20s} {c}/{tot} = {c/tot:.3f}")


def cmd_xtab():
    from collections import Counter, defaultdict
    truth = {r["case_id"]: r for r in csv.DictReader(open(LABELS))}
    world_embargo = Counter()
    regstatus_flags = defaultdict(Counter)
    note_agree = Counter()
    stamp_flags = defaultdict(Counter)
    biocnf_flags = defaultdict(Counter)
    for f in sorted(CACHE.glob("*.json")):
        e = json.loads(f.read_text())
        t = truth.get(e["case_id"])
        if not t:
            continue
        tflags = set(x for x in t["risk_flags"].split("|") if x and x != "none")
        merged = solution.resolve_fields(e)
        hw = merged.get("home_world", (None,))[0]
        if hw:
            world_embargo[(hw, "planetary_embargo" in tflags or "embargo" in t["adjudication"].lower())] += 1
        rs = merged.get("registry_status", (None,))[0]
        if rs:
            regstatus_flags[rs][t["adjudication"] + "/" + (t["risk_flags"] or "none")] += 1
        fnd = merged.get("finding", (None,))[0]
        if fnd:
            note_agree[(fnd, t["adjudication"])] += 1
        stamps = set()
        for p in e["pages"]:
            stamps.update(s.upper() for s in p.get("stamps", []))
        for s in stamps:
            stamp_flags[s][t["adjudication"] + ("|RESC" if "rescinded_denial" in tflags else "")] += 1
    print("== home_world vs embargo truth")
    for (hw, emb), n in sorted(world_embargo.items()):
        print(f"  {hw:18s} emb={emb} {n}")
    print("== registry_status vs adjudication/flags")
    for rs, c in regstatus_flags.items():
        print(f"  {rs}: {dict(c.most_common(6))}")
    print("== note finding vs truth adjudication")
    for (fnd, adj), n in sorted(note_agree.items()):
        print(f"  note={fnd:13s} truth={adj:13s} {n}")
    print("== stamps vs adjudication")
    for s, c in sorted(stamp_flags.items()):
        print(f"  {s}: {dict(c.most_common(8))}")


def cmd_score(split="dev"):
    """Run the official evaluator on a split of train."""
    import subprocess
    truth = {r["case_id"]: r for r in csv.DictReader(open(LABELS))}
    if split == "all":
        keep = set(truth)
    else:
        dev, hold = holdout_split(truth)
        keep = dev if split == "dev" else hold
    tdir = Path("/Users/tylergibbs/Projects/8090chalfable/cache")
    tcsv = tdir / f"labels_{split}.csv"
    with open(tcsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(truth.values())).keys()))
        w.writeheader()
        for cid in sorted(keep):
            w.writerow(truth[cid])
    tpred = tdir / f"pred_{split}.jsonl"
    with open(tpred, "w") as f:
        for line in open(PRED):
            r = json.loads(line)
            if r["case_id"] in keep:
                r.pop("_path", None)
                f.write(json.dumps(r) + "\n")
    subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"),
                    "--truth", str(tcsv), "--submission", str(tpred)])


def cmd_calibrate():
    """Per-decision-path confidence = smoothed dev-split accuracy."""
    from collections import defaultdict
    truth = {r["case_id"]: r for r in csv.DictReader(open(LABELS))}
    dev, _ = holdout_split(truth)
    stats = defaultdict(lambda: [0, 0])
    for line in open(PRED):
        r = json.loads(line)
        cid = r["case_id"]
        if cid not in dev or cid not in truth:
            continue
        path = r.get("_path", "?")
        stats[path][1] += 1
        if r["adjudication"] == truth[cid]["adjudication"]:
            stats[path][0] += 1
    print("CONFIDENCE_BY_PATH = {")
    for path, (c, n) in sorted(stats.items(), key=lambda kv: -kv[1][1]):
        conf = (c + 1) / (n + 2)
        print(f'    "{path}": {conf:.2f},  # {c}/{n}')
    print("}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "extract"
    if cmd == "extract":
        cmd_extract(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif cmd == "predict":
        cmd_predict()
    elif cmd == "analyze":
        cmd_analyze(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "xtab":
        cmd_xtab()
    elif cmd == "calibrate":
        cmd_calibrate()
    elif cmd == "score":
        cmd_score(sys.argv[2] if len(sys.argv) > 2 else "dev")
