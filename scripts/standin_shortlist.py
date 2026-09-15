#!/usr/bin/env python3
"""
Stand-in Shortlists (TF-IDF label centroids) — development only
================================================================
Writes top-50 candidate lists in the handoff-4 / 4b format (WORK_PLAN.md §6) so the Gemma code
can be built and timed before SciBERT's real lists exist. The generator is the word-counting
stand-in the plan's recall ceiling (81.6% tail coverage at 50) was measured on: TF-IDF over
title + abstract, one centroid per topic from the training papers, cosine score per topic.

Fitted on the 15,822-paper training carve only (data/split.json). Nothing produced here is a
result: the validation file is where the prompt and parser are developed, the train file is a
dress-rehearsal input for the 5b trainer. The test split is written only with --include-test,
and only so the pipeline can be smoke-tested end to end; it is never scored.

Outputs (results/shortlists/):
    standin_tfidf_top50_val.jsonl    {"paper_id", "candidates": [50 names, score order], "scores"}
    standin_tfidf_top50_train.jsonl  same + "gold_in_list" (gold ∩ candidates, ordered by score)
"""

import os
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import scipy.sparse as sp
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from scibert_dataset import clean_astronomy_text

K = 50


def paper_text(row):
    return clean_astronomy_text(row["title"]) + " " + clean_astronomy_text(row["abstract"])


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "shortlists"))
    ap.add_argument("--include-test", action="store_true", help="also write the test split (pipeline smoke test only)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = json.load(open(DATA_DIR / "split.json", encoding="utf-8"))
    id_to_name = json.load(open(DATA_DIR / "id_to_name.json", encoding="utf-8"))
    label_order = json.load(open(DATA_DIR / "label_order.json", encoding="utf-8"))
    names = [id_to_name[str(i)] for i in label_order]
    col = {uat_id: j for j, uat_id in enumerate(label_order)}

    ds = load_dataset(split["source"])
    rows = {r["bibcode"]: r for part in ("train", "val") for r in ds[part]}
    train_ids, val_ids, test_ids = split["train"], split["validation"], split["test"]
    print(f"labels {len(names)}; train {len(train_ids)}, validation {len(val_ids)}, test {len(test_ids)}", flush=True)

    vec = TfidfVectorizer(max_features=50000, min_df=2, sublinear_tf=True, stop_words="english")
    X_train = vec.fit_transform(paper_text(rows[b]) for b in train_ids)

    # one centroid per topic: mean TF-IDF vector of the training papers carrying it
    r_idx, c_idx = [], []
    for i, b in enumerate(train_ids):
        for uat_id in rows[b]["verified_uat_ids"]:
            if uat_id in col:
                r_idx.append(col[uat_id]); c_idx.append(i)
    M = sp.csr_matrix((np.ones(len(r_idx)), (r_idx, c_idx)), shape=(len(names), len(train_ids)))
    C = normalize(M @ X_train)                       # topic x vocab

    def shortlists(bibcodes, with_target):
        X = normalize(vec.transform(paper_text(rows[b]) for b in bibcodes))
        S = np.asarray((X @ C.T).todense())          # paper x topic cosine
        top = np.argsort(-S, axis=1)[:, :K]
        recs, covered, total = [], 0, 0
        for i, b in enumerate(bibcodes):
            cands = [names[j] for j in top[i]]
            rec = {"paper_id": b, "candidates": cands, "scores": [round(float(S[i, j]), 5) for j in top[i]]}
            gold = {id_to_name[str(u)] for u in rows[b]["verified_uat_ids"] if str(u) in id_to_name}
            total += len(gold); covered += len(gold & set(cands))
            if with_target:
                rec["gold_in_list"] = [c for c in cands if c in gold]   # already in score order
            recs.append(rec)
        return recs, covered / total

    val_recs, val_cov = shortlists(val_ids, with_target=False)
    write_jsonl(out_dir / "standin_tfidf_top50_val.jsonl", val_recs)
    train_recs, train_cov = shortlists(train_ids, with_target=True)
    write_jsonl(out_dir / "standin_tfidf_top50_train.jsonl", train_recs)
    print(f"gold coverage@{K}: validation {val_cov:.3f} (out-of-sample), train {train_cov:.3f} (in-sample); "
          f"train lists with empty gold_in_list: {sum(1 for r in train_recs if not r['gold_in_list'])}")
    if args.include_test:
        test_recs, test_cov = shortlists(test_ids, with_target=False)
        write_jsonl(out_dir / "standin_tfidf_top50_test.jsonl", test_recs)
        print(f"test coverage@{K}: {test_cov:.3f}  (smoke-test file only; never scored)")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
