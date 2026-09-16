# This file implements the following baselines:
# majority-frequency and TF-IDF + logistic regression baselines.

import numpy as np
import pandas as pd
from pathlib import Path

from evaluate import choose_tau, save_tau_selection, evaluate_test, save_test_results


## Shared function for both baselines ##
def build_score_dataframe_for_evaluation(df, scores, label_order):
    """
    Builds a score pandas dataFrame, while wrapping the scores with relevant columns and IDs.
    This function is shared and fits for both of the baselines.

    All the entries contains values in [0,1] (as the results of majority_vector),
    hence the output fits the requirement of evaluation.py according to the work plan.

    Inputs:
    --- df: DataFrame containing bibcode, in the same row order as scores.
    --- scores: matrix of shape [len(df), len(label_order)],
        with columns ordered according to label_order.
    --- label_order: fixed ordered list of UAT label IDs.
    Output:
    --- DataFrame with paper_id followed by one score column per label.
    """

    # Sanity check:
    if scores.shape != (len(df), len(label_order)):
        raise ValueError("Scores shape must match the number of papers and labels.")

    result = pd.DataFrame(scores, columns=[str(label) for label in label_order])  # Adds columns names
    result.insert(0, "paper_id", df["bibcode"].tolist())  # Adds "paper_id" column
    return result


def build_true_labels_dataframe_for_evaluation(df, label_order):
    """
    Builds a binary correct-answer pandas dataFrame for evaluation.
    This function is shared and fits for both of the baselines.

    Since labelID != index, and since all the columns in the future matrix
    are labels with "labelID <number>", we first map labelID to index, and
    then use this mapping for creating the matrix.

    Inputs:
    --- df: papers containing bibcode and verified_uat_ids.
    --- label_order: fixed ordered list of UAT label IDs.
    Output:
    --- DataFrame with paper_id followed by one 0/1 column per label.
        Row order follows df; column order follows label_order.
    """

    # Converts label IDs to strings and maps each ID to its column index:
    label_ids = [str(label) for label in label_order]
    topic_id_to_idx = {label: idx for idx, label in enumerate(label_ids)}

    # Creates an all-zero matrix with one row per paper and one column per label:
    true_labels = np.zeros((len(df), len(label_ids)), dtype=np.uint8)

    # Marks each paper's correct labels as 1, leaving all other labels as 0:
    for row_idx, paper_labels in enumerate(df["verified_uat_ids"]):
        for label in paper_labels:
            true_labels[row_idx, topic_id_to_idx[str(label)]] = 1

    # Builds the answer table in label_order and adds the corresponding paper IDs.
    # Adss column names:
    result = pd.DataFrame(true_labels, columns=label_ids)
    result.insert(0, "paper_id", df["bibcode"].tolist())
    return result


## Majority baseline ##
def build_appearances_counter_vector(train_df, label_order):
    """
    Calculates the fraction of training papers containing each label.
    Output positions follow label_order:
    result[i] is the fraction of training papers containing label_order[i].

    This method is used to produce the majority baseline scores.

    Assumption:
    --- train_df contains a verified_uat_ids column containing label collections.
    Inputs:
    --- train_df: pandas dataFrame.
    --- label_order: ordered list of unique UAT IDs.
    Output:
    --- result: float32 NumPy vector of length len(label_order), with values in [0, 1].
    """

    # Initializes the counter:
    counter = np.zeros(len(label_order), dtype=np.float32)

    # Builds inner map dictionary of {topic_ID : index in label_order and in counts}
    topic_id_to_idx = {
        str(topic_id): index
        for index, topic_id in enumerate(label_order)
    }

    # Counts appearances over the training set:
    for paper_labels in train_df["verified_uat_ids"]:
        unique_labels = {str(label) for label in paper_labels}
        for label in unique_labels:
            label_order_index = topic_id_to_idx[label]
            counter[label_order_index] += 1

    result = counter / len(train_df)
    return result


def run_majority_baseline(
        train_df,
        validation_df,
        test_df,
        label_order,
        train_band_map,
        save_directory,
):
    """
    Runs the complete majority-frequency baseline experiment.

    The baseline assigns every paper the same vector of label frequencies,
    calculated from the training set. It selects a separate threshold for each
    frequency band using validation data, evaluates the resulting predictions
    on the test set, and saves the scores, thresholds, metrics and predictions.

    Inputs:
    --- train_df: training DataFrame used to calculate label frequencies.
    --- validation_df: validation DataFrame used to select the thresholds.
    --- test_df: test DataFrame used only for final evaluation.
    --- label_order: UAT IDs defining the label-column order.
    --- train_band_map: dictionary mapping each label-column index to
        "head", "torso" or "tail".
    --- save_directory: directory in which the baseline artifacts are saved.
    Output: None
    """

    # Sanity check:
    if len(train_df) == 0:
        raise ValueError("Baselines: train_df is empty.")

    # Creates the directory:
    print("Builds directory ...", end="")
    output_directory = Path(save_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    print(" Done.")

    # Calculate label frequencies from the training set:
    print("Calculates majority vector ...", end="")
    majority_vector = build_appearances_counter_vector(train_df, label_order)
    print(" Done.")

    # Calculates the evaluation tables for evaluation.py's function:
    matrix_for_evaluation_dict = {}
    for split_name, temp_dataset in [
        ("val", validation_df),
        ("test", test_df),
    ]:

        # Repeats the same vector for every paper in the given dataset:
        df_scores = np.tile(majority_vector, (len(temp_dataset), 1))

        # Build the evaluation table:
        df_output = build_score_dataframe_for_evaluation(
            temp_dataset,
            df_scores,
            label_order,
        )

        # Saves output:
        # Note: here the table is saved with the column "paper_id".
        print( f"Saving majority_{split_name}.parquet in dir ...", end="")
        name = f"majority_{split_name}.parquet"
        temp_saves_path = output_directory / name
        df_output.to_parquet(temp_saves_path, index=False)
        matrix_for_evaluation_dict[split_name] = df_output
        print(" Done.")


    # Runs validation check with taus
    # Builds the validation correct-answer table:
    validation_real_labels_answers = build_true_labels_dataframe_for_evaluation(
        validation_df,
        label_order,
    )

    # Extracts numeric matrices, skipping the first column which is the paper_id column:
    val_probs = matrix_for_evaluation_dict["val"].iloc[:, 1:].to_numpy()
    val_labels = validation_real_labels_answers.iloc[:, 1:].to_numpy()

    # Calls choose_tau from evaluate.py and gets dict of {band:float}:
    print("Calling choose_tau() from evaluation.py. ")
    chosen_taus, tau_sweep = choose_tau(
        val_probs=val_probs,
        val_labels=val_labels,
        band_map=train_band_map,
    )

    # Saves information as instructed:
    objective = "Maximise each band's validation Micro-F1 while applying TAUs candidates."
    print("Saving selected validation TAUs ...", end="")
    save_tau_selection(  # Function in evaluation.py
        chosen_taus=chosen_taus,
        tau_sweep=tau_sweep,
        arm_name="majority",
        objective=objective,
        validation_filename="majority_val.parquet",  # Which validation file already exists for documentation
        output_directory=output_directory,
    )
    print(" Done.")

    # Runs test with already chosen taus values
    # Builds the test correct-answer table.
    test_real_labels_answers = build_true_labels_dataframe_for_evaluation(
        test_df,
        label_order,
    )

    # Extracts numeric matrices, skipping the paper_id column.
    test_probs = matrix_for_evaluation_dict["test"].iloc[:, 1:].to_numpy()
    test_real_labels = test_real_labels_answers.iloc[:, 1:].to_numpy()

    # Evaluates test using the thresholds selected on validation.
    print("Calling evaluate_test() from evaluate.py.")
    test_results = evaluate_test(
        test_probs=test_probs,
        test_labels=test_real_labels,
        band_map=train_band_map,   # Band map according to the training set
        chosen_taus=chosen_taus,
    )

    print("Saving test results ...", end="")
    save_test_results(
        test_results=test_results,
        arm_name="majority",
        output_directory=output_directory,
        paper_ids=test_df["bibcode"].tolist(),
        label_order=label_order,
    )
    print(" Done.")

    return None



## TF-IDF + Logistic Regression baseline ##






