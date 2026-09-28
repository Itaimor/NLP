# This file implements the following baselines:
# majority-frequency and TF-IDF + logistic regression baselines.

import time
import json
import resource
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from evaluate import (
    choose_tau,
    save_tau_selection,
    evaluate_test,
    save_test_results,
    run_provenance,
)
from scibert_dataset import clean_astronomy_text


def get_peak_memory_mb():
    """Returns peak resident set size (RSS) in MB for the current process."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return raw / (1024 * 1024)
    return raw / 1024



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


def rescore_encoder_from_disk(
    arm_name,
    results_dir,
    val_df,
    test_df,
    label_order,
    band_map,
    val_parquet_path=None,
    test_parquet_path=None,
    objective="Maximise each band's validation Micro-F1 over the fixed TAU candidates.",
):
    """
    Re-scores an encoder arm directly from its saved §6 Parquet tables on disk.

    Reads validation and test Parquet files, verifies row alignment against
    bibcodes, executes choose_tau on validation probabilities, saves tau_<arm_name>.json
    with validation checksum, evaluates on test probabilities using the selected thresholds,
    and saves test_results_<arm_name>.json and test_predictions_<arm_name>.npz with
    test checksum.

    Satisfies §8's requirement: "regenerated from the files by one command, by someone
    who did not train the model", running on CPU in seconds.

    Inputs:
    --- arm_name: experiment identifier, e.g. "majority", "tfidf_lr", "scibert_full", "scibert_lora".
    --- results_dir: directory path containing or intended for the arm's artifacts.
    --- val_df: validation DataFrame containing 'bibcode' and 'verified_uat_ids'.
    --- test_df: test DataFrame containing 'bibcode' and 'verified_uat_ids'.
    --- label_order: fixed ordered list of unique UAT label IDs (strings or ints).
    --- band_map: dictionary mapping column index to band ("head", "torso", "tail").
    --- val_parquet_path: optional path to validation Parquet file. If None, resolves from results_dir.
    --- test_parquet_path: optional path to test Parquet file. If None, resolves from results_dir.
    --- objective: description string for threshold optimization goal.

    Output:
    --- test_results: dictionary containing test evaluation metrics returned by evaluate_test.
    """
    results_dir = Path(results_dir)

    # Resolves validation Parquet path:
    if val_parquet_path is not None:
        val_path = Path(val_parquet_path)
        if not val_path.is_file() and (results_dir / val_path).is_file():
            val_path = results_dir / val_path
    else:
        candidates = [
            results_dir / f"{arm_name}_val.parquet",
            results_dir / f"{arm_name.replace('_baseline', '')}_val.parquet",
        ]
        if arm_name in ("tfidf_lr", "tfidf_logistic_regression"):
            candidates.insert(0, results_dir / "tfidf_lr_val.parquet")
        val_path = next((p for p in candidates if p.is_file()), None)
        if val_path is None:
            raise FileNotFoundError(
                f"Validation Parquet file not found in {results_dir}. Tried: {[str(c.name) for c in candidates]}"
            )

    # Resolves test Parquet path:
    if test_parquet_path is not None:
        test_path = Path(test_parquet_path)
        if not test_path.is_file() and (results_dir / test_path).is_file():
            test_path = results_dir / test_path
    else:
        candidates = [
            results_dir / f"{arm_name}_test.parquet",
            results_dir / f"{arm_name.replace('_baseline', '')}_test.parquet",
        ]
        if arm_name in ("tfidf_lr", "tfidf_logistic_regression"):
            candidates.insert(0, results_dir / "tfidf_lr_test.parquet")
        test_path = next((p for p in candidates if p.is_file()), None)
        if test_path is None:
            raise FileNotFoundError(
                f"Test Parquet file not found in {results_dir}. Tried: {[str(c.name) for c in candidates]}"
            )

    # 1. Reads validation Parquet from disk:
    print(f"Reading validation Parquet from {val_path.name} ... ", end="")
    val_score_df = pd.read_parquet(val_path)
    print("Done.")

    # Validates shape and row alignment:
    if len(val_score_df) != len(val_df):
        raise ValueError(
            f"Row count mismatch in {val_path.name}: "
            f"expected {len(val_df)} rows, found {len(val_score_df)}"
        )
    if "paper_id" in val_score_df.columns:
        if val_score_df["paper_id"].tolist() != val_df["bibcode"].tolist():
            raise ValueError(f"Paper ID alignment mismatch between {val_path.name} and validation_df.")
        val_probs = val_score_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    else:
        val_probs = val_score_df.to_numpy(dtype=np.float32)

    if val_probs.shape[1] != len(label_order):
        raise ValueError(
            f"Column count mismatch in {val_path.name}: "
            f"expected {len(label_order)} labels, found {val_probs.shape[1]}"
        )

    # 2. Builds validation ground truth:
    val_answers_df = build_true_labels_dataframe_for_evaluation(val_df, label_order)
    val_labels = val_answers_df.iloc[:, 1:].to_numpy(dtype=np.uint8)

    # 3. Selects thresholds per band via choose_tau:
    print(f"Selecting validation TAUs for {arm_name} via choose_tau() ...")
    chosen_taus, tau_sweep = choose_tau(
        val_probs=val_probs,
        val_labels=val_labels,
        band_map=band_map,
    )

    # 4. Saves tau selection metadata and validation file checksum:
    print(f"Saving selected validation TAUs to tau_{arm_name}.json ... ", end="")
    save_tau_selection(
        chosen_taus=chosen_taus,
        tau_sweep=tau_sweep,
        arm_name=arm_name,
        objective=objective,
        validation_filename=val_path.name,
        output_directory=results_dir,
    )
    print("Done.")

    # 5. Reads test Parquet from disk:
    print(f"Reading test Parquet from {test_path.name} ... ", end="")
    test_score_df = pd.read_parquet(test_path)
    print("Done.")

    # Validates shape and row alignment:
    if len(test_score_df) != len(test_df):
        raise ValueError(
            f"Row count mismatch in {test_path.name}: "
            f"expected {len(test_df)} rows, found {len(test_score_df)}"
        )
    if "paper_id" in test_score_df.columns:
        paper_ids = test_score_df["paper_id"].tolist()
        if paper_ids != test_df["bibcode"].tolist():
            raise ValueError(f"Paper ID alignment mismatch between {test_path.name} and test_df.")
        test_probs = test_score_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    else:
        paper_ids = test_df["bibcode"].tolist()
        test_probs = test_score_df.to_numpy(dtype=np.float32)

    if test_probs.shape[1] != len(label_order):
        raise ValueError(
            f"Column count mismatch in {test_path.name}: "
            f"expected {len(label_order)} labels, found {test_probs.shape[1]}"
        )

    # 6. Builds test ground truth:
    test_answers_df = build_true_labels_dataframe_for_evaluation(test_df, label_order)
    test_labels = test_answers_df.iloc[:, 1:].to_numpy(dtype=np.uint8)

    # 7. Evaluates test set:
    print(f"Calling evaluate_test() for {arm_name} ...")
    test_results = evaluate_test(
        test_probs=test_probs,
        test_labels=test_labels,
        band_map=band_map,
        chosen_taus=chosen_taus,
    )

    # 8. Saves test results and predictions (with test filename for checksum):
    print(f"Saving test results for {arm_name} ... ", end="")
    save_test_results(
        test_results=test_results,
        arm_name=arm_name,
        output_directory=results_dir,
        paper_ids=paper_ids,
        label_order=label_order,
        test_filename=test_path.name,
    )
    print("Done.")

    return test_results


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

    t0 = time.perf_counter()

    # Creates the directory:
    print("Builds directory ...", end="")
    output_directory = Path(save_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    print(" Done.")

    # Calculate label frequencies from the training set:
    print("Calculates majority vector ...", end="")
    t_fit_start = time.perf_counter()
    majority_vector = build_appearances_counter_vector(train_df, label_order)
    fit_duration = time.perf_counter() - t_fit_start
    print(" Done.")

    val_parquet_path = output_directory / "majority_val.parquet"
    test_parquet_path = output_directory / "majority_test.parquet"

    # Builds and saves standard §6 score tables:
    val_scores = np.tile(majority_vector, (len(validation_df), 1))
    df_val_output = build_score_dataframe_for_evaluation(
        validation_df,
        val_scores,
        label_order,
    )
    print(f"Saving {val_parquet_path.name} in dir ...", end="")
    df_val_output.to_parquet(val_parquet_path, index=False)
    print(" Done.")

    test_scores = np.tile(majority_vector, (len(test_df), 1))
    df_test_output = build_score_dataframe_for_evaluation(
        test_df,
        test_scores,
        label_order,
    )
    print(f"Saving {test_parquet_path.name} in dir ...", end="")
    df_test_output.to_parquet(test_parquet_path, index=False)
    print(" Done.")

    # Evaluates from the saved tables on disk:
    test_results = rescore_encoder_from_disk(
        arm_name="majority",
        results_dir=output_directory,
        val_df=validation_df,
        test_df=test_df,
        label_order=label_order,
        band_map=train_band_map,
        val_parquet_path=val_parquet_path,
        test_parquet_path=test_parquet_path,
        objective="Maximise each band's validation Micro-F1 while applying TAUs candidates.",
    )

    total_duration = time.perf_counter() - t0
    peak_mb = get_peak_memory_mb()

    # Emits run-log and cost telemetry required by §8 Stage 3:
    cost_data = {
        "arm": "majority_baseline",
        "base_architecture": "Majority Frequency Prior",
        "seed": 42,
        "hyperparameters": {
            "method": "train_label_frequencies",
            "num_labels": len(label_order),
            "selection_metric": "validation_micro_f1_per_band",
        },
        "training_cost": {
            "device": f"{platform.processor() or 'Apple Silicon'} (CPU)",
            "training_wall_clock_seconds": round(fit_duration, 4),
            "total_wall_clock_seconds": round(total_duration, 4),
            "peak_memory": f"~{peak_mb:.1f} MB",
            "trainable_parameters": 0,
            "total_parameters": len(label_order),
        },
        "model_checkpoint": {
            "test_micro_f1_head": round(test_results["per_band"]["head"]["micro_f1"], 4),
            "test_micro_f1_torso": round(test_results["per_band"]["torso"]["micro_f1"], 4),
            "test_micro_f1_tail": round(test_results["per_band"]["tail"]["micro_f1"], 4),
            "test_micro_f1_overall": round(test_results.get("overall", {}).get("micro_f1", 0.0), 4),
            "mean_predictions_per_paper": round(test_results.get("mean_predictions_per_paper", 0.0), 2),
        },
        "provenance": {
            **run_provenance(),
        },
    }
    cost_file = output_directory / "cost.json"
    print(f"Saving cost telemetry to {cost_file.name} ... ", end="")
    with cost_file.open("w", encoding="utf-8") as f:
        json.dump(cost_data, f, indent=2)
    print("Done.")

    return test_results



## TF-IDF + Logistic Regression baseline ##


def build_paper_texts(df):
    """
    Cleans and combines each paper's title and abstract.
    Uses the same logic as Itai's original implementation in
    scibert_dataset.SciXDataset.__getitem__().

    Input:
    --- df: DataFrame containing title and abstract columns.
    Output:
    --- list of cleaned texts in the original row order.
    """

    required_columns = {"title", "abstract"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"DataFrame is missing required columns: {sorted(missing_columns)}"
        )

    texts = []

    for title, abstract in zip(df["title"], df["abstract"]):
        clean_title = clean_astronomy_text(title)
        clean_abstract = clean_astronomy_text(abstract)

        if clean_title and clean_abstract:
            full_text = f"{clean_title}\n{clean_abstract}"
        elif clean_title:
            full_text = clean_title
        else:
            full_text = clean_abstract

        texts.append(full_text)

    return texts



def run_tfidf_logistic_regression_baseline(
        train_df,
        validation_df,
        test_df,
        label_order,
        train_band_map,
        save_directory,
):
    """
    Runs the complete TF-IDF + Logistic Regression baseline.
    TF-IDF and Logistic Regression are fitted using the training set only.
    The trained model produces one probability per paper and label for
    validation and test.

    Inputs:
    --- train_df: training DataFrame used to fit TF-IDF and Logistic Regression.
    --- validation_df: validation DataFrame used to select thresholds.
    --- test_df: test DataFrame used only for final evaluation.
    --- label_order: UAT IDs defining the label-column order.
    --- train_band_map: dictionary mapping each label-column index to
        "head", "torso" or "tail".
    --- save_directory: directory in which baseline artifacts are saved.
    Output: None.
    """

    t0 = time.perf_counter()

    # Creates the output directory:
    print("Builds directory ...", end="")
    output_directory = Path(save_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    print(" Done.")

    # Builds title + abstract texts while preserving DataFrame row order:
    print("Builds paper texts ...", end="")
    train_texts = build_paper_texts(train_df)
    validation_texts = build_paper_texts(validation_df)
    test_texts = build_paper_texts(test_df)
    print(" Done.")

    # Fits TF-IDF on train only and transforms validation and test:
    # The parameters were chosen due to train's statistics analysis.
    print("Fits TF-IDF vectorizer ...", end="")
    t_vec_start = time.perf_counter()
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents='unicode',
        ngram_range=(1, 2),  # Uses both unigrams and adjacent bigrams
        min_df=3,  # Keeps feature only if it appears at least three papers
        max_df=0.95,
        sublinear_tf=True,
        # Common practice, TF(word) = 1+log(#word),
        # hence the 10th appearance in *the same paper* is not 10 times more important
        max_features=None,
        dtype=np.float32,
    )
    train_features = vectorizer.fit_transform(train_texts)  # Trains and applies
    validation_features = vectorizer.transform(validation_texts)  # Applies
    test_features = vectorizer.transform(test_texts)  # Applies
    vec_duration = time.perf_counter() - t_vec_start
    print(" Done.")

    # Builds train correct-answer matrix in label_order.
    train_answers_df = build_true_labels_dataframe_for_evaluation(
        train_df,
        label_order,
    )
    train_labels = train_answers_df.iloc[:, 1:].to_numpy() # removes "id" column

    # Sanity checks before training LR:
    # Every label must contain at least one positive training example.
    labels_without_train_examples = np.flatnonzero(
        train_labels.sum(axis=0) == 0
    )
    if len(labels_without_train_examples) > 0:
        raise ValueError(
            f"{len(labels_without_train_examples)} labels have no "
            "positive training examples."
        )
    # Checks dimensions:
    if train_features.shape[0] != train_labels.shape[0]:
        raise ValueError(
            "train_features and train_labels must contain the same "
            "number of papers."
        )

    if train_labels.shape[1] != len(label_order):
        raise ValueError(
            "The number of train-label columns must match label_order."
        )

    # Trains one binary Logistic Regression classifier per label.
    # OneVsRestClassifier splits the multi-label task to 1864 binary tasks.
    #   train_features: [15,822 papers, TF-IDF features].
    #   train_labels: [15,822 papers, 1,864 labels].
    #   OneVsRest trains one binary LR per label using train_labels[:, i].
    print("Trains multi-label logistic regression classifier ...", end="")
    t_fit_start = time.perf_counter()
    classifier = OneVsRestClassifier(
        LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=1000,
            random_state=42,
        ),
        n_jobs=1,
    )
    classifier.fit(train_features, train_labels)
    fit_duration = time.perf_counter() - t_fit_start
    print(" Done.")

    # Produces one probability per paper and label.
    print("Calculates validation and test probabilities ...", end="")
    t_pred_start = time.perf_counter()
    validation_scores = np.asarray(
        classifier.predict_proba(validation_features),
        dtype=np.float32,
    )
    test_scores = np.asarray(
        classifier.predict_proba(test_features),
        dtype=np.float32,
    )
    pred_duration = time.perf_counter() - t_pred_start
    print(" Done.")

    # Wraps scores with paper IDs and label-ID column names.
    validation_score_df = build_score_dataframe_for_evaluation(
        validation_df,
        validation_scores,
        label_order,
    )
    test_score_df = build_score_dataframe_for_evaluation(
        test_df,
        test_scores,
        label_order,
    )

    # Saves the standard score tables required by the work plan.
    val_parquet_path = output_directory / "tfidf_lr_val.parquet"
    test_parquet_path = output_directory / "tfidf_lr_test.parquet"

    print(f"Saving {val_parquet_path.name} ...", end="")
    validation_score_df.to_parquet(
        val_parquet_path,
        index=False,
    )
    print(" Done.")

    print(f"Saving {test_parquet_path.name} ...", end="")
    test_score_df.to_parquet(
        test_parquet_path,
        index=False,
    )
    print(" Done.")

    # Evaluates from the saved tables on disk:
    test_results = rescore_encoder_from_disk(
        arm_name="tfidf_lr",
        results_dir=output_directory,
        val_df=validation_df,
        test_df=test_df,
        label_order=label_order,
        band_map=train_band_map,
        val_parquet_path=val_parquet_path,
        test_parquet_path=test_parquet_path,
        objective=(
            "Maximise each band's validation Micro-F1 "
            "over the fixed TAU candidates."
        ),
    )

    total_duration = time.perf_counter() - t0
    peak_mb = get_peak_memory_mb()
    num_features = train_features.shape[1]
    num_labels = len(label_order)
    trainable_params = num_features * num_labels + num_labels
    seconds_per_classifier = round(fit_duration / num_labels, 4)

    # Emits run-log and cost telemetry required by §8 Stage 3:
    cost_data = {
        "arm": "tfidf_logistic_regression",
        "base_architecture": "TfidfVectorizer + OneVsRestClassifier(LogisticRegression)",
        "seed": 42,
        "hyperparameters": {
            "vectorizer": {
                "max_features": None,
                "ngram_range": [1, 2],
                "sublinear_tf": True,
                "min_df": 3,
                "max_df": 0.95,
                "strip_accents": "unicode",
            },
            "classifier": {
                "C": 1.0,
                "solver": "liblinear",
                "max_iter": 1000,
                "n_jobs": 1,
                "random_state": 42,
            },
            "num_labels": num_labels,
            "vocab_features": num_features,
            "selection_metric": "validation_micro_f1_per_band",
        },
        "training_cost": {
            "device": f"{platform.processor() or 'Apple Silicon'} (CPU)",
            "num_classifiers": num_labels,
            "feature_extraction_seconds": round(vec_duration, 2),
            "fitting_wall_clock_seconds": round(fit_duration, 2),
            "prediction_wall_clock_seconds": round(pred_duration, 2),
            "total_wall_clock_seconds": round(total_duration, 2),
            "seconds_per_classifier": seconds_per_classifier,
            "seconds_per_step": seconds_per_classifier,
            "peak_memory": f"~{peak_mb:.1f} MB",
            "trainable_parameters": trainable_params,
            "total_parameters": trainable_params,
        },
        "model_checkpoint": {
            "test_micro_f1_head": round(test_results["per_band"]["head"]["micro_f1"], 4),
            "test_micro_f1_torso": round(test_results["per_band"]["torso"]["micro_f1"], 4),
            "test_micro_f1_tail": round(test_results["per_band"]["tail"]["micro_f1"], 4),
            "test_micro_f1_overall": round(test_results.get("overall", {}).get("micro_f1", 0.0), 4),
            "mean_predictions_per_paper": round(test_results.get("mean_predictions_per_paper", 0.0), 2),
        },
        "provenance": {
            **run_provenance(),
        },
    }
    cost_file = output_directory / "cost.json"
    print(f"Saving cost telemetry to {cost_file.name} ... ", end="")
    with cost_file.open("w", encoding="utf-8") as f:
        json.dump(cost_data, f, indent=2)
    print("Done.")

    return test_results