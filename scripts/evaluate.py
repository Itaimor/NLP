# This file implements the evaluation of the pipeline

import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import hashlib
import json
import subprocess
from pathlib import Path
from datetime import datetime


# Global Variable:
SHIFT = "   "
BANDS = ("head", "torso", "tail")
# The grid must span both extremes: an arm whose best tau lands on the first or
# last candidate has not had a threshold selected, it has had the search truncated.
# SciBERT-full pinned at the old 0.54 ceiling (validation Micro-F1 still rising there)
# and TF-IDF+LR pinned at the old 0.005 floor. This grid is a strict SUPERSET of the
# previous one, so any arm whose optimum was already interior keeps its exact tau.
TAU_CANDIDATES = tuple(sorted(set(
      [round(i * 0.0005, 4) for i in range(1, 10)]  # 0.0005 to 0.0045 - below the old floor
    + [round(i * 0.005, 3) for i in range(1, 21)]  # 0.005 to 0.10 (include)
    + [round(0.10 + i * 0.02, 3) for i in range(1, 45)] # 0.12 to 0.98 (include)
)))

# A top-k boundary that falls inside a group of equal scores is decided by column
# order, not by the model. Above this share of papers, P@k/R@k stop being a property
# of the arm and evaluate_test refuses to report them. Probability-emitting encoders
# sit at ~0; set-emitting arms (0/1 matrices, or rank scores over a shortlist with
# zeros elsewhere) sit near 1.
RANKING_TIE_TOLERANCE = 0.01

## Section: Validating inputs ##
def validate_evaluation_inputs(
        probabilities,
        true_labels,
        band_map=None,
        isEvaluateThreshold=False,
):
    """
    Validates NumPy score and answer matrices and their band mapping.
    Raises ValueError for invalid inputs.

    Inputs:
    --- probabilities: 2D NumPy array of prediction probabilities.
    --- true_labels: 2D NumPy array of binary correct answers, with the same
        shape as probabilities.
    --- band_map: optional dictionary mapping each label-column index to
        "head", "torso" or "tail".
    --- isEvaluateThreshold: if True, skips the band_map validation because
        evaluate_thresholds receives data from one band only.
    Output: None
    """

    # Checks that scores and answers have matching matrix dimensions.
    if probabilities.ndim != 2 or probabilities.shape != true_labels.shape:
        raise ValueError("Scores and labels must have the same 2D shape.")

    # Checks that the matrices contain papers and label columns.
    if probabilities.shape[0] == 0 or probabilities.shape[1] == 0:
        raise ValueError("Evaluation inputs must not be empty.")

    # Checks that scores are finite and within [0, 1].
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0) | (probabilities > 1)
    ).any():
        raise ValueError("Scores must be finite and between 0 and 1.")

    # Checks that correct answers contain only 0 and 1.
    if not np.isin(true_labels, [0, 1]).all():
        raise ValueError("True labels must contain only 0 and 1.")

    # These checks are irrelevant when evaluating one band's tau thresholds.
    if not isEvaluateThreshold:

        # Checks that every column index has a band assignment.
        num_labels = probabilities.shape[1]
        if set(band_map) != set(range(num_labels)):
            raise ValueError("band_map must map every label column index.")

        # Checks that all assigned band names are valid.
        if any(band not in BANDS for band in band_map.values()):
            raise ValueError("Band assignments must be head, torso or tail.")

    return None


## Section: Loading ##
def file_checksum(path):
    """
    Calculates the SHA-256 fingerprint of a file, reading it in chunks.

    Input:
    --- path: path to an existing file.
    Output:
    --- the hexadecimal digest as a string.
    """

    checksum = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            checksum.update(chunk)

    return checksum.hexdigest()


def run_provenance():
    """
    Collects the stamp that ties a saved artefact to the code that produced it.

    The timestamp is read here rather than at import, so that two arms scored in
    one session do not both carry the moment the module was loaded.

    The Git lookup is deliberately forgiving. A scoring run that produced correct
    numbers must not be lost because git is absent from PATH or the tree was
    copied without its history, so a failure records None instead of raising.

    Output:
    --- dictionary containing run_datetime, commit and has_tracked_changes.
        commit and has_tracked_changes are None when Git could not be read.
    """

    stamp = {
        "run_datetime": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "commit": None,
        "has_tracked_changes": None,
    }

    project_directory = Path(__file__).resolve().parent

    try:
        # Reads the Git commit of the project containing evaluate.py.
        stamp["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_directory,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        # Checks whether tracked files contain changes outside that commit.
        tracked_changes = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=project_directory,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        stamp["has_tracked_changes"] = bool(tracked_changes)

    except (OSError, subprocess.SubprocessError):
        pass

    return stamp


def save_tau_selection(
    chosen_taus,
    tau_sweep,
    arm_name,
    objective,
    validation_filename,
    output_directory,
):
    """
    Saves selected thresholds and their selection metadata as
     tau_<arm_name>.json inside output_directory.

    Inputs:
    --- chosen_taus: dictionary containing head, torso and tail thresholds.
    --- tau_sweep: dictionary containing the Micro-F1 results for every tested
        threshold, separately for head, torso and tail.
    --- arm_name: experiment name, such as "majority" or "scibert_full".
    --- objective: description of the threshold-selection rule.
    --- validation_filename: validation score filename inside output_directory.
    --- output_directory: directory containing the validation score file.
    Output: None

    """

    # Builds the paths for the existing score file and the new JSON file.
    output_dir = Path(output_directory)
    validation_path = output_dir / validation_filename
    output_path = output_dir / f"tau_{arm_name}.json"

    # Records, per band, whether the chosen threshold sits on an end of the grid.
    # Derived from the sweep rather than passed in, so that choose_tau keeps its
    # two-value return and none of its call sites change. A band reading "floor"
    # or "ceiling" has either a degenerate model or a truncated search, and must
    # not be reported until somebody has decided which.
    on_grid_edge = {}

    for band in BANDS:
        band_sweep = tau_sweep.get(band) or []
        chosen = float(chosen_taus[band])
        edge = None

        if band_sweep:
            if chosen == float(band_sweep[0]["tau"]):
                edge = "floor"
            elif chosen == float(band_sweep[-1]["tau"]):
                edge = "ceiling"

        on_grid_edge[band] = edge

    # Collects the selected thresholds and their selection metadata.
    record = {
        **run_provenance(),
        "taus": {
            band: float(chosen_taus[band])
            for band in BANDS
        },
        "objective": objective,
        "on_grid_edge": on_grid_edge,
        "validation_file": validation_filename,
        "validation_checksum": file_checksum(validation_path),
        "tau_sweep": tau_sweep,
    }

    # Saves the record using the filename required by the work plan.
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=4)

    return None


def save_test_results(
    test_results,
    arm_name,
    output_directory,
    paper_ids,
    label_order,
    test_filename=None,
):
    """
    Saves test metrics as JSON and predictions as a compressed NumPy file.

    Per-label TP, FP and FN counts are saved together with their label IDs.
    Labels whose count is zero are omitted.

    The record carries the same provenance stamp as tau_<arm_name>.json, so that
    a results-table cell can be traced to the commit that produced it.

    ALIGNMENT, the assumption this function cannot check for you: rows of
    test_results["predictions"] are matched to paper_ids POSITIONALLY, and the
    caller is responsible for having built the score matrix and the gold matrix
    from the same sequence of papers. The checks below catch a length mismatch
    and duplicate identifiers, which is every misalignment that leaves a visible
    trace, but a caller that silently reorders one matrix and not the other
    produces numbers this function will accept.

    Inputs:
    --- test_results: dictionary returned by evaluate_test, including predictions.
    --- arm_name: experiment name, such as "majority" or "scibert_full".
    --- output_directory: directory for saving the results.
    --- paper_ids: paper identifiers in prediction-row order.
    --- label_order: UAT IDs in prediction-column order.
    --- test_filename: optional test score filename inside output_directory. When
        given, its SHA-256 is recorded, pinning which table produced the metrics.

    Output: None.
    """

    # Separates the prediction matrix from the remaining test results.
    predictions = np.asarray(test_results["predictions"])

    metrics = {
        **run_provenance(),
        **{
            key: value
            for key, value in test_results.items()
            if key != "predictions"
        },
    }

    # Checks that identifiers match the prediction matrix.
    if predictions.shape != (len(paper_ids), len(label_order)):
        raise ValueError(
            "Prediction shape must match paper IDs and label order."
        )

    # Duplicate identifiers mean the rows cannot be attributed to papers, which
    # is the misalignment worth refusing rather than recording.
    if len(set(map(str, paper_ids))) != len(paper_ids):
        raise ValueError(
            "paper_ids must be unique, otherwise prediction rows cannot be "
            "attributed to papers."
        )

    if len(set(map(str, label_order))) != len(label_order):
        raise ValueError("label_order must not contain duplicate label IDs.")

    metrics["row_alignment"] = "positional, paper_ids given in prediction-row order"

    if not np.isin(predictions, [0, 1]).all():
        raise ValueError("Predictions must contain only 0 and 1.")

    # Converts per-label count arrays into readable label/count entries.
    readable_counts = {}

    for count_name, counts in metrics["per_label_counts"].items():
        if len(counts) != len(label_order):
            raise ValueError(
                f"'{count_name}' count length must match label_order."
            )

        readable_counts[count_name] = []

        for label_id, count in zip(label_order, counts):
            if count > 0:
                readable_counts[count_name].append({
                    "label_id": str(label_id),
                    "count": int(count),
                })

    metrics["per_label_counts"] = readable_counts

    # Creates the output directory and output paths.
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"test_results_{arm_name}.json"
    predictions_path = output_dir / f"test_predictions_{arm_name}.npz"

    # Saves readable metrics as JSON.
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            indent=4,
            allow_nan=False,
        )

    # Saves predictions with their row and column identifiers.
    np.savez_compressed(
        predictions_path,
        predictions=predictions.astype(bool),
        paper_ids=np.asarray(paper_ids, dtype=str),
        label_order=np.asarray(label_order, dtype=str),
    )

    return None


## Section: Evaluations ##

def evaluate_thresholds(
    probabilities,
    true_labels,
    tau_candidates=TAU_CANDIDATES,
):
    """
    Searches tau for a given band's validation scores.
    Maximises Micro-F1 over labels occurring in validation.

    Tie-break: candidates are scanned in ascending order and a new best must be
    strictly greater, so on a plateau the LOWEST tau wins - the more permissive
    threshold, which predicts more labels. This is deliberate (on a flat region a
    lower threshold buys recall in the tail at no measured micro-F1 cost) and it is
    recorded here because it is not recoverable from the saved tau alone.

    This function can be called independently of choose_tau, but it is not advised.

    Assumptions:
    --- probabilities and true_labels have the same shape: [num_papers, num_band_labels].
    Inputs:
    --- probabilities: scores of shape [num_papers, num_band_labels]. (numpy or torch)
    --- true_labels: binary answers with the same shape and ordering. (numpy or torch)
    --- tau_candidates: optional thresholds to search, tuple of floats.
    Output:
    --- dict containing best_tau, best_micro_f1 and tau_sweep.
    """

    # Sanity checks:
    # Checks that at least one threshold was provided.
    if len(tau_candidates) == 0:
        raise ValueError("tau_candidates must not be empty.")

    # Converts input PyTorch tensors to NumPy arrays:
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.detach().cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.detach().cpu().numpy()

    probabilities = np.asarray(probabilities)
    true_labels = np.asarray(true_labels)

    # Validates the score and answer matrices.
    validate_evaluation_inputs(
        probabilities,
        true_labels,
        isEvaluateThreshold=True,
    )

    # Initialization:
    best_tau = float(tau_candidates[0])
    best_micro_f1 = -1.0
    tau_sweep = []

    # Searches for the tau that maximizes Micro-F1.
    for tau in tau_candidates:
        predictions = probabilities >= tau

        if true_labels.shape[1] == 1:
            micro_f1 = float(f1_score(
                true_labels[:, 0],
                predictions[:, 0],
                average="binary",
                zero_division=0,
            ))
        else:
            micro_f1 = float(f1_score(
                true_labels,
                predictions,
                average="micro",
                zero_division=0,
            ))

        tau_sweep.append({"tau": float(tau), "micro_f1": micro_f1})

        # Strictly greater, so an equal score never displaces the lower tau
        # already held. See the tie-break note in the docstring.
        if micro_f1 > best_micro_f1:
            best_tau = float(tau)
            best_micro_f1 = micro_f1

    # Enforces the rule stated at the top of this file. A winner sitting on the
    # first or last candidate means the search was truncated, not that a
    # threshold was selected, so the condition is recorded rather than left to a
    # reader to notice. TF-IDF+LR's head and torso legitimately pin at the floor
    # because the model is degenerate there, which is why this flags and does not
    # raise.
    on_grid_edge = None

    if best_tau == float(tau_candidates[0]):
        on_grid_edge = "floor"
    elif best_tau == float(tau_candidates[-1]):
        on_grid_edge = "ceiling"

    if on_grid_edge is not None:
        print(
            f"{SHIFT}WARNING: best_tau {best_tau} sits on the grid "
            f"{on_grid_edge}. Either the model is degenerate at that threshold "
            "or the grid is truncated. Check before reporting this arm."
        )

    return {
        "best_tau": best_tau,
        "best_micro_f1": best_micro_f1,
        "tau_sweep": tau_sweep,
        "tie_break": "lowest tau wins on equal Micro-F1",
        "on_grid_edge": on_grid_edge,
    }


def choose_tau(val_probs, val_labels, band_map):
    """
    Chooses one tau per band using validation Micro-F1.

    DENOMINATOR WARNING: the search below runs only over the labels that occur at
    least once in validation, because a label with no positive instance cannot
    inform a threshold. evaluate_test then scores EVERY label of the band,
    present or not. The two therefore sit on different denominators. Under the
    primary band map the tail is selected over 1,062 validation-present columns
    and scored over all 1,469, so best_micro_f1 in tau_<arm>.json and micro_f1 in
    test_results_<arm>.json ARE NOT COMPARABLE and must never be tabled beside
    each other. This is deliberate and is not to be "fixed" by aligning them:
    changing either denominator re-selects every threshold and moves every
    number already reported in the paper.

    Inputs:
    --- val_probs: score matrix [num_papers, num_labels], NumPy or PyTorch.
    --- val_labels: binary answers with the same shape and ordering.
    --- band_map: dictionary mapping column index to head, torso or tail.
    Outputs:
    --- chosen_taus: dictionary containing the chosen tau for each band.
    --- tau_sweep: dictionary containing the evaluation results over the tau candidates
    """

    # Converts inputs to NumPy before processing the three bands.
    if isinstance(val_probs, torch.Tensor):
        val_probs = val_probs.detach().cpu().numpy()
    if isinstance(val_labels, torch.Tensor):
        val_labels = val_labels.detach().cpu().numpy()

    val_probs = np.asarray(val_probs)
    val_labels = np.asarray(val_labels)

    # Validates the converted matrices and band mapping.
    validate_evaluation_inputs(val_probs, val_labels, band_map)

    # Builds the band-name array in label-column order.
    column_bands = np.array([
        band_map[idx] for idx in range(val_probs.shape[1])
    ])

    # Marks labels occurring at least once in validation.
    labels_present_in_validation = (val_labels == 1).any(axis=0)

    chosen_taus = {}
    tau_sweep = {}
    for band in BANDS:
        # Selects this band's labels occurring in validation, keeping all papers.
        selected_columns = (
            (column_bands == band) & labels_present_in_validation
        )

        # Searches this band's threshold using the default grid.
        result = evaluate_thresholds(
            probabilities=val_probs[:, selected_columns],
            true_labels=val_labels[:, selected_columns],
        )
        chosen_taus[band] = result["best_tau"]
        tau_sweep[band] = result["tau_sweep"]

        print(
            f"{SHIFT}{band.upper()}: ",
            f"best_tau={result['best_tau']}, best_micro_f1={result['best_micro_f1']}"
        )

    return chosen_taus, tau_sweep


def ranking_tie_fractions(probabilities, k_values=(1, 3, 5)):
    """
    Measures how often a top-k boundary falls inside a group of equal scores.

    When the k-th and the (k+1)-th score are equal, which labels land in the top k
    is settled by column order rather than by the model, so P@k and R@k measure
    label_order.json instead of the arm. A probability-emitting encoder essentially
    never ties; a set-emitting arm ties on almost every paper.

    Inputs:
    --- probabilities: score matrix [num_papers, num_labels], NumPy.
    --- k_values: the k values that will be reported.
    Output:
    --- dictionary mapping each k to the share of papers whose top-k boundary
        sits inside a tie.
    """

    num_labels = probabilities.shape[1]
    ordered = np.sort(probabilities, axis=1)[:, ::-1]  # Descending.

    fractions = {}
    for k in k_values:
        if k >= num_labels:
            continue
        fractions[int(k)] = float(np.mean(ordered[:, k - 1] == ordered[:, k]))

    return fractions


def evaluate_test(
    test_probs,
    test_labels,
    band_map,
    chosen_taus,
    compute_ranking="auto",
):
    """
    Evaluates test scores using thresholds selected on validation.

    Inputs:
    --- test_probs: score matrix [num_papers, num_labels], NumPy or PyTorch.
    --- test_labels: binary answers with the same shape and ordering.
    --- band_map: dictionary mapping column index to head, torso or tail.
    --- chosen_taus: dictionary containing one threshold per band.
    --- compute_ranking: "auto" (default) reports P@k/R@k only when the ranking
        is not decided by ties, and returns None with a note when it is; True
        forces them, False skips them. Set-emitting arms must not use the
        forced setting - they need a ranking counted from their own ordered
        picks, as score_gemma.gemma_ranking_metrics does.
    Output:
    --- dictionary containing per-band metrics, overall metrics, ranking metrics,
        head-tail delta, prediction counts and per-label TP/FP/FN.

    DENOMINATOR WARNING: every label of a band is scored here, including labels
    with no test instance, while choose_tau selected the threshold over the
    validation-present labels only. See the matching note on choose_tau. The two
    micro-F1 figures sit on different denominators and must not be tabled side by
    side. macro_f1_all_labels and macro_f1_present_labels are both reported for
    the same reason, because the published work this is compared against does not
    state which convention it uses.
    """

    # Converts inputs to NumPy before calculating test metrics.
    if isinstance(test_probs, torch.Tensor):
        test_probs = test_probs.detach().cpu().numpy()
    if isinstance(test_labels, torch.Tensor):
        test_labels = test_labels.detach().cpu().numpy()

    test_probs = np.asarray(test_probs)
    test_labels = np.asarray(test_labels)

    # Validates the converted matrices and band mapping.
    validate_evaluation_inputs(test_probs, test_labels, band_map)

    # Gets matrix dimensions and band names in label-column order.
    num_papers, num_labels = test_probs.shape
    column_bands = np.array([
        band_map[idx] for idx in range(num_labels)
    ])

    # Validates the three thresholds selected on validation.
    if set(chosen_taus) != set(BANDS):
        raise ValueError("chosen_taus must contain head, torso and tail.")

    if any(not 0 < float(chosen_taus[band]) <= 1 for band in BANDS):
        raise ValueError("Thresholds must be in (0, 1].")

    # Applies the selected threshold of each label's band.
    thresholds = np.array([
        float(chosen_taus[band]) for band in column_bands
    ])  # Each column gets its band threshold

    true_labels = test_labels.astype(bool)
    predictions = test_probs >= thresholds

    # Counts true positives, false positives and false negatives per label.
    tp = (predictions & true_labels).sum(axis=0)
    fp = (predictions & ~true_labels).sum(axis=0)
    fn = (~predictions & true_labels).sum(axis=0)

    labels_present_in_test = true_labels.any(axis=0)
    per_band = {}
    for band in BANDS:
        # Keeps every label of this band, including labels absent from test.
        band_columns = column_bands == band
        if band_columns.sum() < 2:
            raise ValueError(f"Band '{band}' requires at least two label columns.")

        # Calculates required metrics
        band_labels = true_labels[:, band_columns]
        band_predictions = predictions[:, band_columns]

        precision = precision_score(
            band_labels,
            band_predictions,
            average="micro",
            zero_division=0,
        )

        recall = recall_score(
            band_labels,
            band_predictions,
            average="micro",
            zero_division=0,
        )

        micro_f1 = f1_score(
            band_labels,
            band_predictions,
            average="micro",
            zero_division=0,
        )

        # Calculates F1 separately for each label of this band.
        label_f1_scores = f1_score(
            band_labels,
            band_predictions,
            average=None,
            zero_division=0,
        )

        # Identifies this band's labels with at least one test occurrence.
        present_labels = labels_present_in_test[band_columns]

        per_band[band] = {
            "tau": float(chosen_taus[band]),
            "num_labels": int(band_columns.sum()),
            "num_labels_present_in_test": int(present_labels.sum()),
            "precision": float(precision),
            "recall": float(recall),
            "micro_f1": float(micro_f1),
            "macro_f1_all_labels": float(label_f1_scores.mean()),
            "macro_f1_present_labels": (
                float(label_f1_scores[present_labels].mean())
                if present_labels.any()
                else None
            ),
        }

    # Calculates the same metrics pooled over every label, ignoring bands.
    # Alkan et al.'s Table 5 is an overall figure, so a comparison against it
    # needs this row rather than a band row.
    overall_label_f1 = f1_score(
        true_labels,
        predictions,
        average=None,
        zero_division=0,
    )

    overall = {
        "num_labels": int(num_labels),
        "num_labels_present_in_test": int(labels_present_in_test.sum()),
        "precision": float(precision_score(
            true_labels, predictions, average="micro", zero_division=0,
        )),
        "recall": float(recall_score(
            true_labels, predictions, average="micro", zero_division=0,
        )),
        "micro_f1": float(f1_score(
            true_labels, predictions, average="micro", zero_division=0,
        )),
        "macro_f1_all_labels": float(overall_label_f1.mean()),
        "macro_f1_present_labels": (
            float(overall_label_f1[labels_present_in_test].mean())
            if labels_present_in_test.any()
            else None
        ),
    }

    # Done with calculates metrics.
    # Decides whether a ranking can be read off these scores at all.
    tie_fractions = ranking_tie_fractions(test_probs)
    ranking_is_ambiguous = any(
        fraction > RANKING_TIE_TOLERANCE
        for fraction in tie_fractions.values()
    )

    if compute_ranking == "auto":
        report_ranking = not ranking_is_ambiguous
    else:
        report_ranking = bool(compute_ranking)

    ranking_note = None
    if not report_ranking:
        ranking_note = (
            "P@k/R@k not reported: the top-k boundary falls inside a group of "
            f"equal scores for {tie_fractions} of papers (tolerance "
            f"{RANKING_TIE_TOLERANCE}), so the top-k set would be decided by "
            "column order, not by the arm. A set-emitting arm must count its "
            "ranking from its own ordered picks instead."
        )

    ranking_metrics = None

    # Ranks labels by score, resolving ties by the fixed column order.
    if report_ranking:
        top_indices = np.argsort(
            -test_probs.astype(np.float64),  # Minus turns the order to be from high to low.
            axis=1,
            kind="stable",
        )[:, :min(5, num_labels)] # takes at most 5 top labels (less if the list is shorter)

        ranked_correct = np.take_along_axis(true_labels, top_indices, axis=1)
        correct_counts = true_labels.sum(axis=1)
        ranking_metrics = {}

        for k in (1, 3, 5):
            if k > num_labels:
                continue

            # Counts correct labels among each paper's top-k scores.
            hits = ranked_correct[:, :k].sum(axis=1)

            # Calculates per-paper recall, using zero for papers with no gold labels.
            recall_per_paper = np.divide(
                hits,
                correct_counts,
                out=np.zeros(num_papers, dtype=float),
                where=correct_counts > 0,
            )
            ranking_metrics[f"precision_at_{k}"] = float((hits / k).mean())      # P@
            ranking_metrics[f"recall_at_{k}"] = float(recall_per_paper.mean())   # R@


    results = {
        "per_band": per_band,
        "overall": overall,
        "ranking_metrics": ranking_metrics,
        "ranking_note": ranking_note,
        "ranking_tie_fractions": tie_fractions,
        "delta_head_tail": per_band["head"]["micro_f1"] - per_band["tail"]["micro_f1"],
        # Named because the published deltas this is compared against do not state
        # their own basis; a micro-F1 delta must not be tabled beside a macro one.
        "delta_head_tail_basis": "micro_f1",
        "mean_predictions_per_paper": float(predictions.sum(axis=1).mean()),
        "per_label_counts": {
            "tp": tp.tolist(),
            "fp": fp.tolist(),
            "fn": fn.tolist(),
        },
        "predictions": predictions,
    }

    return results


def paired_bootstrap(
    predictions_a,
    predictions_b,
    true_labels,
    band_map,
    num_bootstraps=2000,
    seed=42,
):
    """
    Estimates 95% percentile intervals for model A minus model B
    in per-band Micro-F1, using paired resampling of test papers.

    Inputs:
    --- predictions_a, predictions_b: binary prediction matrices.
    --- true_labels: binary answers with the same shape and ordering.
    --- band_map: dictionary mapping column index to head, torso or tail.
    --- num_bootstraps: number of resampled test datasets.
    --- seed: random seed for reproducibility.

    Output:
    --- dictionary containing difference, ci_low and ci_high per band.
    """

    # Converts PyTorch inputs to NumPy.
    arrays = []
    for array in (predictions_a, predictions_b, true_labels):
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
        arrays.append(np.asarray(array))

    predictions_a, predictions_b, true_labels = arrays

    # Validates binary predictions and answers.
    validate_evaluation_inputs(predictions_a, true_labels, band_map)
    validate_evaluation_inputs(predictions_b, true_labels, band_map)

    if not np.isin(predictions_a, [0, 1]).all():
        raise ValueError("Model A predictions must contain only 0 and 1.")

    if not np.isin(predictions_b, [0, 1]).all():
        raise ValueError("Model B predictions must contain only 0 and 1.")

    if not isinstance(num_bootstraps, int) or num_bootstraps < 1:
        raise ValueError("num_bootstraps must be a positive integer.")

    predictions_a = predictions_a.astype(bool)
    predictions_b = predictions_b.astype(bool)
    true_labels = true_labels.astype(bool)

    num_papers, num_labels = true_labels.shape
    column_bands = np.array([
        band_map[idx] for idx in range(num_labels)
    ])

    # Stores each paper's TP, FP and FN for both models and every band.
    counts_a = []
    counts_b = []

    for band in BANDS:
        columns = column_bands == band

        if not columns.any():
            raise ValueError(f"Band '{band}' has no label columns.")

        answers = true_labels[:, columns]

        for predictions, counts in (
            (predictions_a, counts_a),
            (predictions_b, counts_b),
        ):
            band_predictions = predictions[:, columns]

            counts.append(np.column_stack([
                (band_predictions & answers).sum(axis=1),
                (band_predictions & ~answers).sum(axis=1),
                (~band_predictions & answers).sum(axis=1),
            ]))

    # Shapes: [num_papers, num_bands, 3], with 3 representing TP/FP/FN.
    counts_a = np.stack(counts_a, axis=1)
    counts_b = np.stack(counts_b, axis=1)

    def micro_f1_from_counts(counts):
        # Calculates pooled Micro-F1 separately for each band.
        tp = counts[:, 0]
        fp = counts[:, 1]
        fn = counts[:, 2]

        denominator = 2 * tp + fp + fn

        return np.divide(
            2 * tp,
            denominator,
            out=np.zeros(len(BANDS), dtype=float),
            where=denominator > 0,
        )

    # Calculates the difference on the original test dataset.
    original_difference = (
        micro_f1_from_counts(counts_a.sum(axis=0))
        - micro_f1_from_counts(counts_b.sum(axis=0))
    )

    rng = np.random.default_rng(seed)
    sampled_differences = np.zeros((num_bootstraps, len(BANDS)))

    for iteration in range(num_bootstraps):
        # Uses the same sampled papers for both models, with replacement.
        indices = rng.integers(0, num_papers, size=num_papers)

        sampled_differences[iteration] = (
            micro_f1_from_counts(counts_a[indices].sum(axis=0))
            - micro_f1_from_counts(counts_b[indices].sum(axis=0))
        )

    # Takes the 2.5th and 97.5th percentiles of the sampled differences.
    lower, upper = np.percentile(
        sampled_differences, [2.5, 97.5], axis=0
    )

    return {
        band: {
            "difference": float(original_difference[idx]),
            "ci_low": float(lower[idx]),
            "ci_high": float(upper[idx]),
            "confidence_level": 0.95,
            "num_bootstraps": num_bootstraps,
            "seed": seed,
        }
        for idx, band in enumerate(BANDS)
    }



