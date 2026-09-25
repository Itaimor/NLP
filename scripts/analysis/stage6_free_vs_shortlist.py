#!/usr/bin/env python3
"""
Stage 6 against Stage 5, on the same 200 papers
===============================================
The Stage 6 pass (`scripts/gemma_free_generation.py`) asks Gemma to name UAT topics with no
candidate list in the prompt. Stage 5 asks the same model, on the same papers, to pick from
SciBERT's top 50. This script puts the two conditions side by side for both arms, which is the
comparison the shortlist-justification paragraph rests on (`SETTLED_QUESTIONS.md:341`).

The point of doing it here rather than reading the two summary files is that **the same
measurement has to be applied to both conditions.** Stage 5's own `off_list` field is relative
to the 50 candidates shown, so a line naming a real UAT topic that simply was not in the top 50
counts as off-list there. That is the right field for Stage 5's purpose and the wrong one here.
So both conditions are re-parsed from their stored `raw_output` against the full 1,864-name
vocabulary, using the Stage 6 functions unchanged, and every rate below means the same thing in
both columns.

Rates are unique-line-weighted throughout, because the fine-tuned checkpoint loops on a minority
of papers and line-weighted rates are distorted wherever it does.

Reads only tracked files, so a fresh clone reproduces it.
"""

import os
import sys
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from gemma_common import parse_picks
from gemma_free_generation import (
    emitted_lines, build_vocab_indexes, classify_off_list, load_vocab_and_bands, load_papers,
)

# condition -> arm -> picks file. The 200-paper sample is whatever the free files contain.
FILES = {
    "free": {
        "untouched": ROOT / "results/gemma_free/free-5a_validation.jsonl",
        "fine-tuned": ROOT / "results/gemma_free/free-5b_validation.jsonl",
    },
    "shortlist": {
        "untouched": ROOT / "results/gemma_select/5a_validation.jsonl",
        "fine-tuned": ROOT / "results/gemma_select/5b-epoch-2_validation.jsonl",
    },
}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def measure(records, paper_ids, vocab, indexes, name_band, papers):
    """Re-parse `raw_output` against the full vocabulary and score the papers in `paper_ids`."""
    by_id = {r["paper_id"]: r for r in records}
    missing = [p for p in paper_ids if p not in by_id]
    assert not missing, f"{len(missing)} papers absent from this arm, e.g. {missing[:3]}"
    vocab_lower = {v.lower(): v for v in vocab}

    n_lines = n_uniq = n_uniq_oov = n_dup = n_loop = 0
    kinds = Counter()
    bands = Counter()
    tp = fp = fn = gold_total = 0
    for pid in paper_ids:
        text = by_id[pid]["raw_output"]
        raw = emitted_lines(text)
        uniq = {l.lower() for l in raw}
        n_lines += len(raw)
        n_uniq += len(uniq)
        n_dup += len(raw) - len(uniq)
        n_loop += 1 if len(raw) > len(uniq) else 0
        # Same rule in both conditions: exact case-folded match against all 1,864 names.
        picks, off = parse_picks(text, vocab)
        seen = set()
        for line in off:
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            kind, _ = classify_off_list(line, *indexes)
            kinds[kind] += 1
            n_uniq_oov += 1
        gold = set(papers[pid]["gold"])
        pick_set = set(picks)
        tp += len(gold & pick_set); fp += len(pick_set - gold); fn += len(gold - pick_set)
        gold_total += len(gold)
        for p in pick_set:
            bands[name_band.get(p, "unbanded")] += 1

    near = sum(v for k, v in kinds.items() if k.startswith("near_miss"))
    p_ = tp / (tp + fp) if tp + fp else 0.0
    r_ = tp / (tp + fn) if tp + fn else 0.0
    band_tot = sum(bands.values())
    return {
        "papers": len(paper_ids),
        "mean_cardinality_unique": n_uniq / len(paper_ids),
        "mean_cardinality_emitted": n_lines / len(paper_ids),
        "out_of_vocabulary_rate_unique_lines": n_uniq_oov / n_uniq if n_uniq else 0.0,
        "near_miss_rate_of_oov": near / n_uniq_oov if n_uniq_oov else None,
        "hallucination_rate_of_oov": kinds.get("hallucination", 0) / n_uniq_oov if n_uniq_oov else None,
        "oov_breakdown_unique": dict(kinds),
        "duplicate_rate_lines": n_dup / n_lines if n_lines else 0.0,
        "papers_with_duplicates": n_loop / len(paper_ids),
        "micro_precision": p_, "micro_recall": r_,
        "micro_f1": 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0,
        "mean_gold": gold_total / len(paper_ids),
        "in_vocab_band_profile": {k: v / band_tot for k, v in bands.items()} if band_tot else {},
        "in_vocab_band_counts": dict(bands),
    }


def main():
    vocab, name_band = load_vocab_and_bands()
    indexes = build_vocab_indexes(vocab)
    papers = load_papers("validation")

    free_5a = read_jsonl(FILES["free"]["untouched"])
    paper_ids = [r["paper_id"] for r in free_5a]
    print(f"Stage 6 sample: {len(paper_ids)} validation papers; vocabulary {len(vocab)} names")
    print("Both conditions re-parsed from raw_output against the full vocabulary.\n")

    out = {"papers": len(paper_ids), "vocabulary_size": len(vocab), "conditions": {}}
    for cond in ("free", "shortlist"):
        out["conditions"][cond] = {}
        for arm, path in FILES[cond].items():
            out["conditions"][cond][arm] = measure(
                read_jsonl(path), paper_ids, vocab, indexes, name_band, papers)

    hdr = f"{'':14}" + "".join(f"{c + '/' + a:>22}" for c in ("free", "shortlist") for a in ("untouched", "fine-tuned"))
    print(hdr)
    rows = [
        ("unique/paper", "mean_cardinality_unique", "{:.2f}"),
        ("emitted/paper", "mean_cardinality_emitted", "{:.2f}"),
        ("OOV rate", "out_of_vocabulary_rate_unique_lines", "{:.4f}"),
        ("  halluc of OOV", "hallucination_rate_of_oov", "{:.3f}"),
        ("  near-miss", "near_miss_rate_of_oov", "{:.3f}"),
        ("loop rate", "papers_with_duplicates", "{:.3f}"),
        ("micro P", "micro_precision", "{:.4f}"),
        ("micro R", "micro_recall", "{:.4f}"),
        ("micro F1", "micro_f1", "{:.4f}"),
    ]
    for label, key, fmt in rows:
        line = f"{label:14}"
        for cond in ("free", "shortlist"):
            for arm in ("untouched", "fine-tuned"):
                v = out["conditions"][cond][arm][key]
                line += f"{(fmt.format(v) if v is not None else '-'):>22}"
        print(line)

    print("\nin-vocabulary picks by band (share of picks):")
    for band in ("head", "torso", "tail", "unbanded"):
        line = f"  {band:12}"
        for cond in ("free", "shortlist"):
            for arm in ("untouched", "fine-tuned"):
                prof = out["conditions"][cond][arm]["in_vocab_band_profile"]
                line += f"{prof.get(band, 0.0):>22.4f}"
        print(line)

    print(f"\ngold topics per paper: {out['conditions']['free']['untouched']['mean_gold']:.2f}")
    dest = Path(__file__).with_suffix(".json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {dest.name}")


if __name__ == "__main__":
    main()
