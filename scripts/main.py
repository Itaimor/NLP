# This file is the main project file, running the full experiment.

# For loading dataset from hf_cache:
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))

# General libraries:
import random
import numpy as np
import torch
import time

from datasets import load_dataset, concatenate_datasets
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# Project's imports:
from EDA_Analysis import EDA_Analysis


# Global variables:
SAVE_PATH = ""
SEED = 42
TAU_THRESHOLDS = []




def set_seed(seed):
    """
    Fixes given random seed value.

    Input:
    --- seed: int
    Output: None
    """

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
    print("Done!")

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

    print("Done!")
    return new_train_df, validation_df




def main(chosen_seed,tau_thresholds):
    """  Runs the complete project pipeline.  """

    set_seed(chosen_seed)

    # Loads and splits the training set to train and validation:
    datasets_dict = data_loading()
    train_df, validation_df = split_train_validation(
        train_df=datasets_dict["train"],
        seed=chosen_seed
    )
    datasets_dict["train"] = train_df  # Override
    datasets_dict["validation"] = validation_df  # Overrides

    # Runs EDA analysis:
    EDA_Analysis(datasets_dict)


    # Runs SciBert
    # TODO

    # Runs Gemma
    # TODO


    return 0  # Success




if __name__ == "__main__":
    main(chosen_seed=SEED,tau_thresholds=TAU_THRESHOLDS)