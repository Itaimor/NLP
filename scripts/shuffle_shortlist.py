#!/usr/bin/env python3
"""
Shuffled copy of a handoff-4 shortlist — the Stage 5 order-robustness pass
==========================================================================
WORK_PLAN Stage 5: both arms see SciBERT's order at validation and test, and 5b is trained on
shuffled lists so it stops learning "the answer is near the top". The robustness check is one
extra inference pass — "score 5b on shuffled test lists as well; the difference should be within
a point or two" (ListT5 Table 5: a shuffle-trained re-ranker loses 0.4–1.7 nDCG when the test
order is randomised, an untouched one 8.8).

This script writes that shuffled file. Each paper's 50 candidates are permuted with a
deterministic per-(seed, paper_id) RNG — the same convention as gemma_train.py's per-epoch
training shuffle, keyed with the tag "test-shuffle" so it never coincides with a training
epoch's permutation. "scores" (when present) is permuted identically so the file stays
self-consistent; every other key is copied through untouched. Nothing about generation changes:
gemma_select.py reads the shuffled file exactly like the original.

Usage:
    python scripts/shuffle_shortlist.py results/shortlists/scibert_full_top50_test.jsonl
    -> results/shortlists/scibert_full_top50_test_shuffled.jsonl   (seed 42)
"""

import sys
import json
import random
import hashlib
import argparse
from pathlib import Path


def shuffled(record, seed):
    rng = random.Random(f"{seed}|test-shuffle|{record['paper_id']}")
    perm = list(range(len(record["candidates"])))
    rng.shuffle(perm)
    out = dict(record)
    out["candidates"] = [record["candidates"][i] for i in perm]
    if "scores" in record:
        out["scores"] = [record["scores"][i] for i in perm]
    return out, perm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shortlist", help="handoff-4 jsonl (paper_id + candidates [+ scores])")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="default: <shortlist>_shuffled.jsonl next to the input")
    args = ap.parse_args()

    src = Path(args.shortlist)
    dst = Path(args.out) if args.out else src.with_name(src.stem + "_shuffled" + src.suffix)
    raw = src.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]

    moved = 0
    with open(dst, "w", encoding="utf-8") as f:
        for r in records:
            out, perm = shuffled(r, args.seed)
            moved += sum(1 for i, p in enumerate(perm) if i != p)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    n_cand = sum(len(r["candidates"]) for r in records)
    print(f"shuffle_shortlist: {src} (sha256 {hashlib.sha256(raw).hexdigest()[:12]}, {len(records)} papers) "
          f"-> {dst} | seed {args.seed} | {moved}/{n_cand} candidates changed position")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
