# This file implements the evaluation of the pipeline

import torch
import numpy as np
from sklearn.metrics import f1_score


# Global Variable:
TAU_CANDIDATES = tuple(sorted(set(
      [round(i * 0.005, 3) for i in range(1, 21)]  # 0.005 to 0.10 (include)
    + [round(0.10 + i * 0.02, 3) for i in range(1, 23)] # 0.12 to 0.54 (include)
)))

BANDS = ("head", "torso", "tail")


def evaluate_thresholds(
        probabilities,
        true_labels,
        tau_candidates=TAU_CANDIDATES,
):
    """
    Searches tau for a given band's validation scores.
    Maximises Micro-F1 over labels occurring in validation.
    This function can be called independently of choose_tau, but it's not advised.

    This function was originally implemented by Itai in SciBERT_experiment.py file.
    Shai copied it here and modified it (Macro to Micro F1, for example).

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
    # Converts input PyTorch tensors to NumPy arrays:
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.detach().cpu().numpy()

    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.detach().cpu().numpy()

    probabilities = np.asarray(probabilities)
    true_labels = np.asarray(true_labels)

    # Validates structure:
    if probabilities.ndim != 2 or probabilities.shape != true_labels.shape:
        raise ValueError("Scores and labels must have the same 2D shape.")

    if len(tau_candidates) == 0:
        raise ValueError("tau_candidates must not be empty.")

    if probabilities.shape[0] == 0 or probabilities.shape[1] < 2:
        raise ValueError("Expected papers and at least two label columns.")

    # Initialization:
    best_tau = float(tau_candidates[0])
    best_micro_f1 = -1.0
    tau_sweep = []

    # Searches after the tau that maximizes Micro-F1 score
    for tau in tau_candidates:
        predictions = probabilities >= tau

        micro_f1 = float(f1_score(
            true_labels,
            predictions,
            average="micro",
            zero_division=0,
        ))

        tau_sweep.append({"tau": float(tau), "micro_f1": micro_f1})

        if micro_f1 > best_micro_f1:
            best_tau = float(tau)
            best_micro_f1 = micro_f1

    return {
        "best_tau": best_tau,
        "best_micro_f1": best_micro_f1,
        "tau_sweep": tau_sweep,
    }



def choose_tau(val_probs, val_labels, band_map):
    """
    Chooses one tau per band using validation Micro-F1.

    Inputs:
    --- val_probs: score matrix [num_papers, num_labels], NumPy or PyTorch.
    --- val_labels: binary answers with the same shape and ordering.
    --- band_map: dictionary mapping column index to head, torso or tail.
    Output:
    --- dictionary containing the chosen tau for each band.
    """

    # Converts inputs to NumPy once before processing the three bands.
    if isinstance(val_probs, torch.Tensor):
        val_probs = val_probs.detach().cpu().numpy()
    if isinstance(val_labels, torch.Tensor):
        val_labels = val_labels.detach().cpu().numpy()

    val_probs = np.asarray(val_probs)
    val_labels = np.asarray(val_labels)

    # Checks that scores and answers have matching dimensions.
    if val_probs.ndim != 2 or val_probs.shape != val_labels.shape:
        raise ValueError("Scores and labels must have the same 2D shape.")

    # Checks that every column has a valid band assignment.
    num_labels = val_probs.shape[1]
    if set(band_map) != set(range(num_labels)):
        raise ValueError("band_map must map every label column index.")

    column_bands = np.array([band_map[idx] for idx in range(num_labels)])
    if not np.isin(column_bands, list(BANDS)).all():
        raise ValueError("Band assignments must be head, torso or tail.")

    # Checks that scores and correct answers contain valid values.
    if not np.isfinite(val_probs).all() or (
        (val_probs < 0) | (val_probs > 1)
    ).any():
        raise ValueError("Scores must be finite and between 0 and 1.")

    if not np.isin(val_labels, [0, 1]).all():
        raise ValueError("True labels must contain only 0 and 1.")


    # Marks labels that have at least one occurrence in validation set's labels.
    # Namely have at least one paper of validation set.
    labels_present_in_validation = (val_labels == 1).any(axis=0)

    chosen_taus = {}
    for band in BANDS:
        # Selects the labels of the given band from all the validation set's labels:
        selected_columns = (column_bands == band) & labels_present_in_validation

        # Searches this band's threshold using the default threshold grid.
        result = evaluate_thresholds(
            probabilities=val_probs[:, selected_columns],
            true_labels=val_labels[:, selected_columns],
        )
        chosen_taus[band] = result["best_tau"]

    return chosen_taus