
import os
import sys
import json
import random
import argparse

from pathlib import Path

import numpy as np
import pandas as pd
import torch

# main.py is located inside the scripts directory:
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = PROJECT_ROOT / "results"

# Hugging Face cache directory:
os.environ.setdefault(
    "HF_HOME",
    str(PROJECT_ROOT / ".hf_cache"),
)

# Allows imports starting from the project root and scripts directory:
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Data-loading and splitting imports:
from datasets import load_dataset, concatenate_datasets
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# Project imports:
from data.build_split import load_split
from EDA_Analysis import EDA_Analysis
from baselines import (
    run_majority_baseline,
    run_tfidf_logistic_regression_baseline,
    rescore_encoder_from_disk,
)
from score_gemma import score as score_gemma_picks
import evaluate as evaluate_module

# Global variables:
SEED = 42


def set_seed(seed):
    """ Fixes given random seed value. """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return None


def data_loading():
    """
    Loads the original dataset splits and creates a combined dataframe as well.

    Assumption:
    --- Data already exists in hf_cache (user already ran "download_data.py").
    Input: None
    Output:
    --- d : dictionary of pandas dataframes
    """

    print("\nLoading data from hf_cache ... ", end="")
    dataset = load_dataset("adsabs/SciX_UAT_keywords")
    print(" Done!")

    # Extracts datasets:
    train_df = dataset["train"].to_pandas()
    original_validation_df = dataset["val"].to_pandas()

    # Merges:
    full_dataset = concatenate_datasets([dataset["train"], dataset["val"]])
    full_df = full_dataset.to_pandas()

    return {
        "train": train_df,
        "validation": [],  # later it will be overridden, it's for inner order.
        "test": original_validation_df,
        "full": full_df
    }


def split_train_validation(train_df, seed, validation_size=0.15):
    """
    Splits the original training data into train and validation,
    approximately preserving each label's frequency.

    Inputs:
    --- train_df : pandas dataframes
    --- seed: int
    --- validation_size : float
    Outputs:
    --- new_train_df, validation_df : pandas dataframes
    """

    print("Splits the original train set to train and validation ... ", end="")
    # Temporary binary label matrix, used only for stratification:
    mlb = MultiLabelBinarizer()
    labels_matrix = mlb.fit_transform(train_df["verified_uat_ids"])

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_size,
        random_state=seed
    )

    train_indices, validation_indices = next(
        splitter.split(
            np.zeros((len(train_df), 1)),
            labels_matrix
        )
    )

    new_train_df = train_df.iloc[train_indices].reset_index(drop=True)
    validation_df = train_df.iloc[validation_indices].reset_index(drop=True)

    print(" Done!")
    return new_train_df, validation_df


def load_dataset_from_split_json_file():
    """
    Loads the fixed train, validation and test datasets according
    to the paper IDs stored in data/split.json.

    The rows of each DataFrame follow the order saved in split.json.

    Input: None
    Output:
    --- datasets_dict: dictionary containing train, validation,
        test and full DataFrames.
    """

    print("\nLoading fixed datasets from split.json ... ")

    loaded_split = load_split(
        data_dir=PROJECT_ROOT / "data",
        with_dataframes=True,
    )

    train_df = loaded_split["train_df"]
    validation_df = loaded_split["validation_df"]
    test_df = loaded_split["test_df"]

    # Reconstructs the complete dataset for EDA.
    full_df = pd.concat(
        [train_df, validation_df, test_df],
        ignore_index=True,
    )

    expected_sizes = loaded_split["split"]["sizes"]
    actual_sizes = {
        "train": len(train_df),
        "validation": len(validation_df),
        "test": len(test_df),
    }

    if actual_sizes != expected_sizes:
        raise ValueError(
            "Loaded dataset sizes do not match split.json. "
            f"Expected {expected_sizes}, received {actual_sizes}."
        )

    print(
        f"Train: {len(train_df)}, "
        f"Validation: {len(validation_df)}, "
        f"Test: {len(test_df)}"
    )

    return {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
        "full": full_df,
    }


def load_json_from_data(filename):
    """
    Loads a JSON file from the project's data directory.

    Input:
    --- filename: name of the JSON file inside the data directory.
    Output:
    --- The Python object stored in the JSON file.
    """

    file_path = PROJECT_ROOT / "data" / filename

    if not file_path.is_file():
        raise FileNotFoundError(
            f"JSON file was not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        loaded_data = json.load(file)

    return loaded_data


def load_label_order():
    """
    Loads and validates the fixed label-column order.

    Input: None
    Output:
    --- label_order: ordered list of unique label IDs represented as strings.
    """

    label_order = load_json_from_data("label_order.json")

    if not isinstance(label_order, list) or not label_order:
        raise ValueError(
            "label_order.json must contain a non-empty list."
        )

    label_order = [str(label) for label in label_order]

    if len(label_order) != len(set(label_order)):
        raise ValueError(
            "label_order.json contains duplicate label IDs."
        )

    return label_order


def load_train_band_map(label_order):
    """
        Loads the training-based label-to-band mapping and converts it
        into a column-index-to-band mapping according to label_order.

        Input:
        --- label_order: ordered list of label IDs defining the score columns.
        Output:
        --- column_band_map: dictionary mapping each column index
            to "head", "torso" or "tail".
        """

    saved_data = load_json_from_data("band_map_train.json")

    if not isinstance(saved_data, dict):
        raise ValueError(
            "band_map_train.json must contain a JSON object."
        )

    label_band_map = saved_data.get("bands")

    if not isinstance(label_band_map, dict) or not label_band_map:
        raise ValueError(
            "band_map_train.json must contain a non-empty 'bands' dictionary."
        )

    # JSON label IDs are strings.
    label_band_map = {
        str(label_id): band
        for label_id, band in label_band_map.items()
    }

    valid_bands = {"head", "torso", "tail"}

    if any(band not in valid_bands for band in label_band_map.values()):
        raise ValueError(
            "Every value in 'bands' must be 'head', 'torso' or 'tail'."
        )

    # Ensures that every score column has a band assignment.
    missing_labels = [
        str(label_id)
        for label_id in label_order
        if str(label_id) not in label_band_map
    ]

    if missing_labels:
        raise ValueError(
            "Some labels from label_order are missing from "
            f"band_map_train.json: {missing_labels[:10]}"
        )

    # Converts Label-ID mapping into column-index mapping.
    column_band_map = {
        column_index: label_band_map[str(label_id)]
        for column_index, label_id in enumerate(label_order)
    }

    present_bands = set(column_band_map.values())

    if present_bands != valid_bands:
        raise ValueError(
            "The selected label_order must contain labels from all three bands. "
            f"Found: {sorted(present_bands)}"
        )

    return column_band_map


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run or rescore NLP project models and baselines pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--arm",
        choices=["all", "majority", "tfidf", "scibert_full", "scibert_lora", "gemma", "rescore_all"],
        default="all",
        help="Model arm to run or rescore.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for deterministic execution.",
    )
    parser.add_argument(
        "--eda",
        action="store_true",
        help="Run exploratory data analysis (EDA).",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Re-score baselines from committed §6 Parquet tables instead of retraining.",
    )
    return parser.parse_args()


def print_table_row(name, results):
    if not results or "per_band" not in results:
        print(f"{name:30s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'N/A':>12s}")
        return
    head_f1 = results["per_band"]["head"]["micro_f1"]
    torso_f1 = results["per_band"]["torso"]["micro_f1"]
    tail_f1 = results["per_band"]["tail"]["micro_f1"]
    overall_f1 = results.get("overall", {}).get("micro_f1", 0.0)
    preds_per_paper = results.get("mean_predictions_per_paper")
    if preds_per_paper is None:
        preds_per_paper = results.get("gemma_diagnostics", {}).get("mean_picks", 0.0)
    print(f"{name:30s} {head_f1:8.4f} {torso_f1:8.4f} {tail_f1:8.4f} {overall_f1:8.4f} {preds_per_paper:12.2f}")


def main(args=None, chosen_seed=None):
    """
    Runs the complete project pipeline or re-scores models directly from disk.
    Satisfies §8's requirement: 'regenerated from the files by one command,
    by someone who did not train the model', running on CPU in seconds.
    """
    if isinstance(args, int):
        chosen_seed = args
        args = None

    if args is None:
        args = parse_args()

    if chosen_seed is not None:
        args.seed = chosen_seed

    set_seed(args.seed)

    # Loads datasets according to split.json file:
    datasets_dict = load_dataset_from_split_json_file()

    # Runs EDA analysis if requested:
    if args.eda:
        print("\nRunning Exploratory Data Analysis (EDA):")
        eda_save_dir = RESULT_PATH / "eda_plots"
        EDA_Analysis(datasets_dict, save_dir=eda_save_dir)

    # Loads label_order.json and band_map file from data directory:
    label_order = load_label_order()
    train_band_map = load_train_band_map(label_order)

    table_rows = {}

    # 1. Majority baseline:
    if args.arm in ("all", "majority", "rescore_all"):
        majority_path = RESULT_PATH / "majority_baseline"
        if args.rescore or args.arm == "rescore_all":
            print("\nRe-scoring Majority Baseline from §6 tables:")
            table_rows["Majority prior"] = rescore_encoder_from_disk(
                arm_name="majority",
                results_dir=majority_path,
                val_df=datasets_dict["validation"],
                test_df=datasets_dict["test"],
                label_order=label_order,
                band_map=train_band_map,
            )
        else:
            print("\nRunning Majority Baseline:")
            table_rows["Majority prior"] = run_majority_baseline(
                train_df=datasets_dict["train"],
                validation_df=datasets_dict["validation"],
                test_df=datasets_dict["test"],
                label_order=label_order,
                train_band_map=train_band_map,
                save_directory=majority_path,
            )
        print("Majority Baseline is completed.")

    # 2. TF-IDF + Logistic Regression:
    if args.arm in ("all", "tfidf", "rescore_all"):
        tfidf_path = RESULT_PATH / "tfidf_logistic_regression"
        if args.rescore or args.arm == "rescore_all":
            print("\nRe-scoring TF-IDF + Logistic Regression from §6 tables:")
            table_rows["TF-IDF + LR"] = rescore_encoder_from_disk(
                arm_name="tfidf_lr",
                results_dir=tfidf_path,
                val_df=datasets_dict["validation"],
                test_df=datasets_dict["test"],
                label_order=label_order,
                band_map=train_band_map,
            )
        else:
            print("\nRunning TF-IDF + Logistic Regression Baseline:")
            table_rows["TF-IDF + LR"] = run_tfidf_logistic_regression_baseline(
                train_df=datasets_dict["train"],
                validation_df=datasets_dict["validation"],
                test_df=datasets_dict["test"],
                label_order=label_order,
                train_band_map=train_band_map,
                save_directory=tfidf_path,
            )
        print("TF-IDF + Logistic Regression Baseline is completed.")

    # 3. SciBERT - full (re-scoring from committed §6 tables):
    if args.arm in ("all", "scibert_full", "rescore_all"):
        print("\nRe-scoring SciBERT-full from committed §6 tables:")
        scibert_full_path = RESULT_PATH / "scibert_full"
        table_rows["SciBERT-alone (tuned)"] = rescore_encoder_from_disk(
            arm_name="scibert_full",
            results_dir=scibert_full_path,
            val_df=datasets_dict["validation"],
            test_df=datasets_dict["test"],
            label_order=label_order,
            band_map=train_band_map,
        )
        print("SciBERT-full re-scoring is completed.")

    # 4. SciBERT - LoRA (re-scoring from committed §6 tables):
    if args.arm in ("all", "scibert_lora", "rescore_all"):
        print("\nRe-scoring SciBERT-LoRA:")
        scibert_lora_path = RESULT_PATH / "scibert_lora"
        lora_val_parquet = scibert_lora_path / "scibert_lora_val.parquet"
        if lora_val_parquet.is_file():
            table_rows["SciBERT-LoRA (tuned)"] = rescore_encoder_from_disk(
                arm_name="scibert_lora",
                results_dir=scibert_lora_path,
                val_df=datasets_dict["validation"],
                test_df=datasets_dict["test"],
                label_order=label_order,
                band_map=train_band_map,
            )
        else:
            # Fallback to existing test_results JSON if parquet tables haven't been exported yet
            lora_results_json = scibert_lora_path / "run2" / "lora" / "test_results_scibert_lora.json"
            if lora_results_json.is_file():
                with open(lora_results_json, "r", encoding="utf-8") as f:
                    table_rows["SciBERT-LoRA (tuned)"] = json.load(f)
                print("  (Loaded existing results from run2/lora/test_results_scibert_lora.json; export Parquet in Package 3)")
            else:
                print(f"  Warning: SciBERT-LoRA artifacts not found in {scibert_lora_path}")
        print("SciBERT-LoRA processing is completed.")

    # 5. Gemma (re-scoring from committed picks files):
    if args.arm in ("all", "gemma", "rescore_all"):
        print("\nRe-scoring Gemma arms from committed picks files:")
        gemma_dir = RESULT_PATH / "gemma_select"
        scored_dir = gemma_dir / "scored"
        scored_dir.mkdir(parents=True, exist_ok=True)

        # 5a: Untuned Gemma 2B
        picks_5a = gemma_dir / "5a_test.jsonl"
        if picks_5a.is_file():
            res_5a, pids_5a, ord_5a = score_gemma_picks(str(picks_5a), "test")
            evaluate_module.save_test_results(
                test_results=res_5a,
                arm_name="5a_test",
                output_directory=scored_dir,
                paper_ids=pids_5a,
                label_order=ord_5a,
            )
            table_rows["Gemma, untuned"] = res_5a
        elif (scored_dir / "test_results_5a_test.json").is_file():
            with open(scored_dir / "test_results_5a_test.json", "r", encoding="utf-8") as f:
                table_rows["Gemma, untuned"] = json.load(f)

        # 5b: Fine-tuned Gemma (QLoRA)
        picks_5b = gemma_dir / "5b_test.jsonl"
        if picks_5b.is_file():
            res_5b, pids_5b, ord_5b = score_gemma_picks(str(picks_5b), "test")
            evaluate_module.save_test_results(
                test_results=res_5b,
                arm_name="5b_test",
                output_directory=scored_dir,
                paper_ids=pids_5b,
                label_order=ord_5b,
            )
            table_rows["Gemma, fine-tuned (QLoRA)"] = res_5b
        elif (scored_dir / "test_results_5b_test.json").is_file():
            with open(scored_dir / "test_results_5b_test.json", "r", encoding="utf-8") as f:
                table_rows["Gemma, fine-tuned (QLoRA)"] = json.load(f)

        print("Gemma re-scoring is completed.")

    # Prints summary Table 1:
    if table_rows:
        print("\n" + "=" * 80)
        print("TABLE 1: TEST-SET RESULTS SUMMARY")
        print("=" * 80)
        print(f"{'System':30s} {'Head':>8s} {'Torso':>8s} {'Tail':>8s} {'Overall':>8s} {'Preds/Paper':>12s}")
        print("-" * 80)
        for name, res in table_rows.items():
            print_table_row(name, res)
        print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    main()
