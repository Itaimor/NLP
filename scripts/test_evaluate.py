#!/usr/bin/env python3
# Acceptance tests for the scoring harness in evaluate.py.
#
# These are the checks WORK_PLAN.md section 8 names for Stage 2, written so that
# somebody who did not build the harness can run them from the repository:
#
#   - a perfect prediction scores 1.0, per band and overall
#   - an empty prediction scores 0.0, and so does an inverted one
#   - the band sizes come out of whichever band map the script is handed,
#     12 / 383 / 1,469 under the primary map and 17 / 429 / 1,418 under the
#     corpus map, with neither hard-coded in evaluate.py
#   - choose_tau returns the same thresholds for the same inputs
#
# No model, no GPU and no pytest. Run it with:
#
#   python scripts/test_evaluate.py
#
# Exit code 0 means every check passed.

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate import (
    BANDS,
    TAU_CANDIDATES,
    choose_tau,
    evaluate_test,
    evaluate_thresholds,
)

DATA_DIR = PROJECT_ROOT / "data"

# The counts WORK_PLAN.md section 8 requires, per band map. The harness must
# read whichever map it is given and never hard-code either triple.
EXPECTED_BAND_SIZES = {
    "band_map_train.json": {"head": 12, "torso": 383, "tail": 1469},
    "band_map_corpus.json": {"head": 17, "torso": 429, "tail": 1418},
}


## Section: Helpers ##
def load_column_band_map(band_map_filename):
    """
    Builds a column-index-to-band mapping from a saved band map.

    Kept local rather than imported from main.py so that these tests need only
    numpy, and do not pull in torch, transformers or the dataset loader.

    Input:
    --- band_map_filename: filename inside data/, such as "band_map_train.json".
    Output:
    --- dictionary mapping each label-column index to "head", "torso" or "tail".
    """

    with (DATA_DIR / "label_order.json").open(encoding="utf-8") as file:
        label_order = json.load(file)

    with (DATA_DIR / band_map_filename).open(encoding="utf-8") as file:
        bands = json.load(file)["bands"]

    return {
        column_index: bands[str(label_id)]
        for column_index, label_id in enumerate(label_order)
    }


def build_small_case(num_papers=12, labels_per_band=4, seed=42):
    """
    Builds a small synthetic case with every band populated.

    Every band gets at least one positive in every split, because choose_tau
    searches only the labels that occur in validation and would otherwise be
    handed an empty column selection.

    Inputs:
    --- num_papers: number of rows.
    --- labels_per_band: columns per band, at least two (evaluate_test refuses
        a band with fewer than two columns).
    --- seed: seed for the gold pattern.
    Output:
    --- tuple of (gold label matrix, column band map).
    """

    if labels_per_band < 2:
        raise ValueError("evaluate_test requires at least two columns per band.")

    num_labels = labels_per_band * len(BANDS)

    band_map = {
        column_index: BANDS[column_index // labels_per_band]
        for column_index in range(num_labels)
    }

    rng = np.random.default_rng(seed)
    gold = (rng.random((num_papers, num_labels)) < 0.35).astype(np.int8)

    # Guarantees at least one positive in every column, so that every band has
    # scoreable labels and no band selection comes back empty.
    for column_index in range(num_labels):
        gold[column_index % num_papers, column_index] = 1

    return gold, band_map


def check(condition, description):
    """Raises AssertionError with a readable message when a check fails."""

    if not condition:
        raise AssertionError(description)


## Section: The tests ##
def test_band_sizes_come_from_the_given_map():
    """
    Both saved band maps produce their documented band sizes.

    This is the check that the harness reads whichever map it is handed. It also
    guards the two triples the paper quotes.
    """

    for filename, expected in EXPECTED_BAND_SIZES.items():
        band_map = load_column_band_map(filename)

        counts = {
            band: sum(1 for value in band_map.values() if value == band)
            for band in BANDS
        }

        check(
            counts == expected,
            f"{filename} band sizes are {counts}, expected {expected}",
        )

        check(
            len(band_map) == sum(expected.values()),
            f"{filename} must map every one of {sum(expected.values())} columns",
        )

    # The two maps must disagree, otherwise one of them is not being read.
    check(
        EXPECTED_BAND_SIZES["band_map_train.json"]
        != EXPECTED_BAND_SIZES["band_map_corpus.json"],
        "the primary and corpus maps must differ",
    )


def test_perfect_prediction_scores_one():
    """A prediction identical to the gold answers scores 1.0 everywhere."""

    gold, band_map = build_small_case()
    taus = {band: 0.5 for band in BANDS}

    results = evaluate_test(
        test_probs=gold.astype(float),
        test_labels=gold,
        band_map=band_map,
        chosen_taus=taus,
    )

    for band in BANDS:
        for metric in ("precision", "recall", "micro_f1"):
            value = results["per_band"][band][metric]
            check(
                value == 1.0,
                f"perfect prediction gave {band} {metric} = {value}, expected 1.0",
            )

    for metric in ("precision", "recall", "micro_f1", "macro_f1_all_labels"):
        value = results["overall"][metric]
        check(
            value == 1.0,
            f"perfect prediction gave overall {metric} = {value}, expected 1.0",
        )

    check(
        results["delta_head_tail"] == 0.0,
        f"perfect prediction gave delta {results['delta_head_tail']}, expected 0.0",
    )

    check(
        results["delta_head_tail_basis"] == "micro_f1",
        "delta must name micro_f1 as its basis",
    )


def test_empty_prediction_scores_zero():
    """A prediction of nothing scores 0.0 everywhere, without dividing by zero."""

    gold, band_map = build_small_case()
    taus = {band: 0.5 for band in BANDS}

    results = evaluate_test(
        test_probs=np.zeros_like(gold, dtype=float),
        test_labels=gold,
        band_map=band_map,
        chosen_taus=taus,
    )

    for band in BANDS:
        for metric in ("precision", "recall", "micro_f1"):
            value = results["per_band"][band][metric]
            check(
                value == 0.0,
                f"empty prediction gave {band} {metric} = {value}, expected 0.0",
            )

    check(
        results["overall"]["micro_f1"] == 0.0,
        "empty prediction must give overall micro_f1 0.0",
    )

    check(
        results["mean_predictions_per_paper"] == 0.0,
        "empty prediction must predict nothing per paper",
    )

    counts = results["per_label_counts"]

    check(
        sum(counts["tp"]) == 0 and sum(counts["fp"]) == 0,
        "empty prediction must record no true and no false positives",
    )

    check(
        sum(counts["fn"]) == int(gold.sum()),
        "every gold label must be recorded as a false negative",
    )


def test_inverted_prediction_scores_zero():
    """Predicting the complement of the gold answers scores 0.0."""

    gold, band_map = build_small_case()
    taus = {band: 0.5 for band in BANDS}

    results = evaluate_test(
        test_probs=1.0 - gold.astype(float),
        test_labels=gold,
        band_map=band_map,
        chosen_taus=taus,
    )

    for band in BANDS:
        value = results["per_band"][band]["micro_f1"]
        check(
            value == 0.0,
            f"inverted prediction gave {band} micro_f1 = {value}, expected 0.0",
        )


def test_choose_tau_is_deterministic():
    """choose_tau returns identical thresholds for identical inputs."""

    gold, band_map = build_small_case()

    rng = np.random.default_rng(7)
    probs = rng.random(gold.shape)

    first_taus, first_sweep = choose_tau(probs, gold, band_map)
    second_taus, second_sweep = choose_tau(probs, gold, band_map)

    check(
        first_taus == second_taus,
        f"choose_tau is not deterministic: {first_taus} then {second_taus}",
    )

    check(
        first_sweep == second_sweep,
        "the recorded tau sweep is not deterministic",
    )

    for band, tau in first_taus.items():
        check(
            tau in TAU_CANDIDATES,
            f"{band} tau {tau} is not a member of the fixed grid",
        )


def test_tau_grid_spans_both_extremes():
    """
    The grid still reaches below the old floor and above the old ceiling.

    The 22 September correction is disclosed in the paper, so a later edit that
    silently narrows the grid would invalidate every threshold-tuned row.
    """

    check(
        min(TAU_CANDIDATES) <= 0.0005,
        f"grid floor is {min(TAU_CANDIDATES)}, expected 0.0005 or lower",
    )

    check(
        max(TAU_CANDIDATES) >= 0.98,
        f"grid ceiling is {max(TAU_CANDIDATES)}, expected 0.98 or higher",
    )

    check(
        list(TAU_CANDIDATES) == sorted(set(TAU_CANDIDATES)),
        "the grid must be sorted and free of duplicates",
    )


def test_plateau_tie_break_keeps_the_lowest_tau():
    """On a flat region the lower, more permissive threshold wins."""

    # Every candidate above the scores predicts nothing and every candidate at
    # or below them predicts everything, so the winning region is a plateau.
    probs = np.full((6, 2), 0.9)
    labels = np.ones((6, 2), dtype=np.int8)

    result = evaluate_thresholds(probs, labels)

    candidates_at_or_below = [tau for tau in TAU_CANDIDATES if tau <= 0.9]

    check(
        result["best_tau"] == min(candidates_at_or_below),
        f"tie-break gave {result['best_tau']}, expected {min(candidates_at_or_below)}",
    )

    check(
        result["best_micro_f1"] == 1.0,
        "a fully correct prediction must score 1.0 in the sweep",
    )


## Section: Runner ##
TESTS = (
    test_band_sizes_come_from_the_given_map,
    test_perfect_prediction_scores_one,
    test_empty_prediction_scores_zero,
    test_inverted_prediction_scores_zero,
    test_choose_tau_is_deterministic,
    test_tau_grid_spans_both_extremes,
    test_plateau_tie_break_keeps_the_lowest_tau,
)


def main():
    """Runs every check and reports which passed. Returns a process exit code."""

    failures = []

    for test in TESTS:
        try:
            test()
        except AssertionError as error:
            failures.append((test.__name__, str(error)))
            print(f"FAIL  {test.__name__}\n      {error}")
        except Exception as error:  # noqa: BLE001 - a crash is a failure too
            failures.append((test.__name__, repr(error)))
            print(f"ERROR {test.__name__}\n      {error!r}")
        else:
            print(f"ok    {test.__name__}")

    print()

    if failures:
        print(f"{len(failures)} of {len(TESTS)} checks failed.")
        return 1

    print(f"All {len(TESTS)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
