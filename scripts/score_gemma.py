#!/usr/bin/env python3
"""
Gemma Output -> evaluate.py Adapter (bridges Stage 5's picks files to Shai's scorer)
=====================================================================================
evaluate.py scores a [papers x 1,864 labels] probability matrix. Gemma's prompt-and-select
arms (5a/5b) emit something different: an ordered list of picked candidate NAMES per paper,
drawn from a 50-candidate shortlist, with no per-label score (WORK_PLAN Stage 5: "The Gemma
arms have no tau at all, because prompt-and-select emits a list of labels rather than a score
per topic"). Nothing has bridged the two before this script -- no Gemma picks file has ever
gone through evaluate.py, on either branch.

What this does, step by step:
  1. Reads a Stage-5/6 picks file (WORK_PLAN.md section 6 Gemma format): one JSON line per
     paper with paper_id, candidates_shown, picks (ordered, on-list), off_list, raw_output.
     This is gemma_select.py's output shape -- schemas/example_gemma_picks.jsonl is a 3-row
     example of it.
  2. Places each paper's picks and gold topics into the full 1,864-column vocabulary
     (data/label_order.json order), using data/id_to_name.json to translate topic names to
     UAT ids to column indices. Candidate names in every shortlist producer (SciBERT,
     standin_shortlist.py) come from this same id_to_name.json, so the mapping is exact --
     an unmapped name means a real vocabulary mismatch and this script raises rather than
     silently dropping it.
  3. Runs the placed 0/1 matrices through evaluate.evaluate_test() at a constant pass-through
     threshold (0.5) for every band. Gemma's predictions are already binary decisions, not
     scores, so this is not a threshold search -- 0.5 just reproduces the input matrix exactly
     (predictions = probs >= threshold, and every entry is 0.0 or 1.0) -- but it reuses the
     exact same per-band precision/recall/micro-F1/macro-F1 (both denominators) and per-label
     TP/FP/FN accounting that every other arm gets, instead of a second implementation that
     could quietly drift from it.
  4. REPLACES evaluate_test's built-in P@k/R@k. That function ranks the full 1,864-column
     score vector and, for any paper with fewer than k real picks, pads the top-k with
     arbitrary zero-score labels (tie-broken by column index) -- which could spuriously score
     a hit on a label Gemma never picked. Stage 5 states the correct rule for a set-emitter
     explicitly: "P@k counts hits among its first min(m, k) picks and still divides by k ...
     R@k divides by the number of correct topics, and nothing is padded in below its last
     pick." gemma_ranking_metrics() below implements exactly that, from the ordered picks
     list itself, not from the reconstructed matrix.
  5. Adds the Gemma-only diagnostics the results table needs and evaluate.py has no concept
     of: mean picks per paper, parse rate, empty-output rate, off-list rate (lines and
     papers), all computed straight from the picks file.
  6. Saves results with evaluate.save_test_results(), the same test_results_<arm>.json /
     test_predictions_<arm>.npz shape every other arm uses, so the results table is read from
     one file format regardless of which arm produced it.

NOT implemented here, because the data does not exist yet: the cardinality-matched SciBERT
row and the recall@50 ceiling both need SciBERT's full 1,864-column probability table
(handoff 5), which is not on disk on any branch as of this writing. Score those once that
table exists -- this script only produces the Gemma-arm side of the comparison.

Usage:
    python scripts/score_gemma.py --picks results/gemma_select/smoke4b_validation.jsonl \
        --split validation --arm smoke4b_validation
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from gemma_select import read_jsonl, load_papers
import evaluate as evaluate_module

RANK_KS = (1, 3, 5)
# Gemma has no per-label score to threshold (Stage 5: "no tau at all"); predictions are
# already binary, so this constant is a pass-through, not a threshold search.
PASS_THROUGH_TAU = {"head": 0.5, "torso": 0.5, "tail": 0.5}


def load_vocab():
    """name -> column index (data/label_order.json order), column index -> band
    (data/band_map_train.json, the primary training-basis map), and the label list
    evaluate.save_test_results wants for its per-label JSON keys."""
    label_order = json.load(open(DATA_DIR / "label_order.json", encoding="utf-8"))
    id_to_name = json.load(open(DATA_DIR / "id_to_name.json", encoding="utf-8"))
    bands = json.load(open(DATA_DIR / "band_map_train.json", encoding="utf-8"))["bands"]

    name_to_col, col_band = {}, {}
    for idx, uat_id in enumerate(label_order):
        name_to_col[id_to_name[str(uat_id)]] = idx
        col_band[idx] = bands[str(uat_id)]
    return name_to_col, col_band, label_order


def build_matrices(records, papers, name_to_col, num_labels):
    """[papers x num_labels] binary prediction and gold matrices, in name_to_col's column
    order. Raises on any topic name absent from the shared vocabulary -- that is a real
    mismatch between the shortlist/gold source and data/id_to_name.json, not something to
    paper over with a silent skip."""
    n = len(records)
    predictions = np.zeros((n, num_labels), dtype=np.float64)
    true_labels = np.zeros((n, num_labels), dtype=np.float64)
    paper_ids = []

    for i, r in enumerate(records):
        pid = r["paper_id"]
        paper_ids.append(pid)

        for name in papers[pid]["gold"]:
            col = name_to_col.get(name)
            if col is None:
                raise ValueError(f"gold topic {name!r} for {pid} is not in label_order/id_to_name")
            true_labels[i, col] = 1.0

        for name in r["picks"]:
            col = name_to_col.get(name)
            if col is None:
                raise ValueError(
                    f"pick {name!r} for {pid} is not in label_order/id_to_name "
                    "-- parse_picks should have routed this to off_list, not picks"
                )
            predictions[i, col] = 1.0

    return predictions, true_labels, paper_ids


def gemma_ranking_metrics(records, papers):
    """Stage 5's P@k/R@k rule for a set-emitter: hits counted only among the picks
    actually made (picks[:k], never padded), P@k still divides by k, R@k divides by the
    number of correct topics."""
    hits_at = {k: [] for k in RANK_KS}
    recall_at = {k: [] for k in RANK_KS}

    for r in records:
        gold = set(papers[r["paper_id"]]["gold"])
        n_gold = len(gold)
        correct = [1 if pick in gold else 0 for pick in r["picks"]]

        for k in RANK_KS:
            hits = sum(correct[:k])
            hits_at[k].append(hits / k)
            recall_at[k].append(hits / n_gold if n_gold else 0.0)

    metrics = {}
    for k in RANK_KS:
        metrics[f"precision_at_{k}"] = float(np.mean(hits_at[k]))
        metrics[f"recall_at_{k}"] = float(np.mean(recall_at[k]))
    return metrics


def gemma_diagnostics(records):
    """Set-emitter columns the results table needs that evaluate.py has no concept of."""
    n = len(records)
    lines = sum(len(r["picks"]) + len(r["off_list"]) for r in records)
    off = sum(len(r["off_list"]) for r in records)
    return {
        "papers": n,
        "mean_picks": sum(len(r["picks"]) for r in records) / n,
        "parse_rate": sum(1 for r in records if r["picks"]) / n,
        "empty_output_rate": sum(1 for r in records if not r["picks"] and not r["off_list"]) / n,
        "off_list_rate_lines": off / lines if lines else 0.0,
        "papers_with_off_list": sum(1 for r in records if r["off_list"]) / n,
    }


def score(picks_path, split, papers=None):
    """Runs the full adapter and returns (results dict, paper_ids, label_order) without
    touching disk -- the pieces main() needs to both print and save."""
    records = read_jsonl(picks_path)
    if papers is None:
        papers = load_papers(split)

    missing = [r["paper_id"] for r in records if r["paper_id"] not in papers]
    assert not missing, f"{len(missing)} picks paper_ids are not in the {split} split, e.g. {missing[:3]}"

    name_to_col, col_band, label_order = load_vocab()
    predictions, true_labels, paper_ids = build_matrices(records, papers, name_to_col, len(label_order))

    results = evaluate_module.evaluate_test(
        test_probs=predictions,
        test_labels=true_labels,
        band_map=col_band,
        chosen_taus=PASS_THROUGH_TAU,
    )
    results["ranking_metrics"] = gemma_ranking_metrics(records, papers)
    results["gemma_diagnostics"] = gemma_diagnostics(records)
    return results, paper_ids, label_order


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--picks", required=True, help="a Stage-5 Gemma picks jsonl (gemma_select.py's output)")
    ap.add_argument("--split", required=True, choices=["validation", "test"])
    ap.add_argument("--arm", required=True, help="name used in the output files, e.g. 5a_test, smoke4b_validation")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "gemma_select" / "scored"))
    args = ap.parse_args()

    results, paper_ids, label_order = score(args.picks, args.split)

    per_band = results["per_band"]
    rank = results["ranking_metrics"]
    diag = results["gemma_diagnostics"]
    print(f"score_gemma: {args.picks} | {args.split} | {diag['papers']} papers", flush=True)
    for band in ("head", "torso", "tail"):
        b = per_band[band]
        print(f"  {band.upper():5s} P {b['precision']:.3f} R {b['recall']:.3f} micro-F1 {b['micro_f1']:.3f} "
              f"macro-F1(present) {b['macro_f1_present_labels']}", flush=True)
    print(f"  delta head-tail micro-F1: {results['delta_head_tail']:.3f}", flush=True)
    print(f"  P@1/3/5 {rank['precision_at_1']:.3f}/{rank['precision_at_3']:.3f}/{rank['precision_at_5']:.3f} "
          f"R@1/3/5 {rank['recall_at_1']:.3f}/{rank['recall_at_3']:.3f}/{rank['recall_at_5']:.3f}", flush=True)
    print(f"  mean picks {diag['mean_picks']:.2f} | parse {diag['parse_rate']:.2f} | "
          f"empty {diag['empty_output_rate']:.2f} | off-list lines {diag['off_list_rate_lines']:.3f}", flush=True)

    evaluate_module.save_test_results(
        test_results=results,
        arm_name=args.arm,
        output_directory=args.out_dir,
        paper_ids=paper_ids,
        label_order=label_order,
    )
    # save_test_results only keeps keys it knows about in the JSON path via dict-copy of
    # test_results minus "predictions" -- ranking_metrics/gemma_diagnostics are plain dict
    # values so they round-trip through it already; nothing further to write here.
    print(f"wrote {Path(args.out_dir) / f'test_results_{args.arm}.json'} and "
          f"test_predictions_{args.arm}.npz", flush=True)


if __name__ == "__main__":
    main()
