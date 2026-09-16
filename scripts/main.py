
import os
import sys
import json
import random

from pathlib import Path

import numpy as np
import pandas as pd
import torch


# main.py is located inside the scripts directory:
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = PROJECT_ROOT / "result"

# Hugging Face cache directory:
os.environ.setdefault(
    "HF_HOME",
    str(PROJECT_ROOT / ".hf_cache"),
)

# Allows imports starting from the project root:
sys.path.insert(0, str(PROJECT_ROOT))

# Data-loading and splitting imports:
from datasets import load_dataset, concatenate_datasets
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# Project imports:
from data.build_split import load_split
from EDA_Analysis import EDA_Analysis
from baselines import run_majority_baseline

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


def main(chosen_seed):
    """  Runs the complete project pipeline.  """

    set_seed(chosen_seed)

    # Loads datasets according to split.json file:
    datasets_dict = load_dataset_from_split_json_file()

    # Runs EDA analysis:
    # EDA_Analysis(datasets_dict)

    # For running baselines and models:
    # Loads label_order.json and band_map file from data directory
    label_order = load_label_order()
    train_band_map = load_train_band_map(label_order)

    # Baselines:
    # Runs majority baseline:
    print("\nRuns Majority Baseline:")
    print("----------------------")
    majority_path = RESULT_PATH / "majority_baseline"
    run_majority_baseline(
        train_df=datasets_dict["train"],
        validation_df=datasets_dict["validation"],
        test_df=datasets_dict["test"],
        label_order=label_order,
        train_band_map=train_band_map,
        save_directory=majority_path,
    )

    print("\nMajority Baseline is completed.")


    # Runs TF-IDF + Logistic Regression:
    # TODO

    # Runs SciBert - full
    # TODO

    # Runs SciBERT - LORA
    # TODO

    # Runs Gemma
    # TODO

    # Runs another experiment if needed
    # TODO


    return 0  # Success




if __name__ == "__main__":
    main(chosen_seed=SEED)
