#!/usr/bin/env python3
"""
SciBERT Shortlists from the committed probability tables — handoff 4 (validation + test)
========================================================================================
Itai's exporter (scripts/export_scibert_full_handoffs.py) writes the top-50 JSONL shortlists to
artifacts/, which git ignores, so they never reach the repo. The probability tables it also
writes — results/scibert_full/scibert_full_{val,test}.parquet — ARE tracked, and the shortlist is
nothing but the top-50 of each row. This script rebuilds the validation and test lists from those
tables in the handoff-4 format the Gemma scripts read (WORK_PLAN.md §6): the 50 candidates are the
same ones the exporter's torch.topk(k=50) picks, in score order, up to float ties.

The training list (handoff 4b) is NOT derivable here: it needs SciBERT's checkpoint run over the
15,822 training papers. Itai exports it to results/scibert_full/scibert_full_top50_train.jsonl
(tracked since e037005) under his exporter's schema — key "candidate_names" and ["NONE"] for an
empty target — so when that file exists this script also writes a copy in the schema the Gemma
scripts read ("candidates", [] for empty), never touching his bytes.

Provenance of a list = this script + the git blob of the file it was read from (printed).

Outputs (results/shortlists/, git-ignored, ~1 s to rebuild):
    scibert_full_top50_val.jsonl    {"paper_id", "candidates": [50 names, SciBERT score order], "scores"}
    scibert_full_top50_test.jsonl   same
    scibert_full_top50_train.jsonl  same + "gold_in_list" (gold ∩ candidates in score order, [] if none)
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "scibert_full"

import numpy as np
import pandas as pd

K = 50


def git_blob(path):
    """Hash of the committed bytes (empty if the file is not tracked or git is unavailable)."""
    try:
        return subprocess.check_output(["git", "rev-parse", f"HEAD:{path.relative_to(PROJECT_ROOT).as_posix()}"],
                                       cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return ""


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "shortlists"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = json.load(open(DATA_DIR / "split.json", encoding="utf-8"))
    id_to_name = json.load(open(DATA_DIR / "id_to_name.json", encoding="utf-8"))
    label_order = json.load(open(DATA_DIR / "label_order.json", encoding="utf-8"))
    names = [id_to_name[str(i)] for i in label_order]
    assert len(set(names)) == len(names), "label names must be distinct for a name-keyed shortlist"

    for part, split_key in (("val", "validation"), ("test", "test")):
        parquet = RESULTS_DIR / f"scibert_full_{part}.parquet"
        df = pd.read_parquet(parquet)
        ids = df["paper_id"].tolist()
        assert ids == split[split_key], f"{parquet.name}: paper_id order differs from split.json[{split_key}]"
        cols = [c for c in df.columns if c != "paper_id"]
        assert cols == [str(i) for i in label_order], f"{parquet.name}: columns differ from label_order.json"
        P = df[cols].to_numpy(dtype=np.float64)
        assert not np.isnan(P).any(), f"{parquet.name}: NaN probabilities"

        top = np.argsort(-P, axis=1)[:, :K]                # descending score, exporter's order up to ties
        recs = [{"paper_id": pid,
                 "candidates": [names[j] for j in top[i]],
                 "scores": [round(float(P[i, j]), 5) for j in top[i]]}
                for i, pid in enumerate(ids)]
        assert all(len(set(r["candidates"])) == K for r in recs)

        out = out_dir / f"scibert_full_top50_{part}.jsonl"
        write_jsonl(out, recs)
        print(f"{out.name}: {len(recs)} lists from {parquet.relative_to(PROJECT_ROOT).as_posix()} "
              f"(blob {git_blob(parquet)[:12] or 'untracked'}); max prob {P.max():.3f}, "
              f"median row-max {np.median(P.max(axis=1)):.3f}", flush=True)

    # handoff 4b: Itai's exported training list, re-keyed for gemma_train.py
    src = RESULTS_DIR / "scibert_full_top50_train.jsonl"
    if not src.exists():
        print(f"{src.relative_to(PROJECT_ROOT).as_posix()} not present; training list not written", flush=True)
        return
    with open(src, encoding="utf-8") as f:
        his = [json.loads(line) for line in f if line.strip()]
    assert [r["paper_id"] for r in his] == split["train"], "train list: paper_id order differs from split.json[train]"
    recs, n_empty = [], 0
    for r in his:
        cands = r.get("candidates", r.get("candidate_names"))
        assert len(cands) == K and len(set(cands)) == K, f"{r['paper_id']}: not 50 distinct candidates"
        gold_in_list = [] if r["gold_in_list"] == ["NONE"] else r["gold_in_list"]
        assert all(g in cands for g in gold_in_list), f"{r['paper_id']}: gold_in_list not a subset of candidates"
        n_empty += not gold_in_list
        recs.append({"paper_id": r["paper_id"], "candidates": cands, "scores": r["scores"], "gold_in_list": gold_in_list})
    out = out_dir / "scibert_full_top50_train.jsonl"
    write_jsonl(out, recs)
    print(f"{out.name}: {len(recs)} lists re-keyed from {src.relative_to(PROJECT_ROOT).as_posix()} "
          f"(blob {git_blob(src)[:12] or 'untracked'}); empty gold_in_list {n_empty}", flush=True)


if __name__ == "__main__":
    main()
