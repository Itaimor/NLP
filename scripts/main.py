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
from EDA_Analysis import EDA_Analysis

# Global variables:
SAVE_PATH = ""




## General: fixes seed and data loading ##

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

    print("Loading data from hf_cache ... ", end="")
    dataset = load_dataset("adsabs/SciX_UAT_keywords")
    print("Done!")

    # Extracts datasets:
    train_df = dataset["train"].to_pandas()
    validation_df = dataset["val"].to_pandas()

    # Merges:
    full_dataset = concatenate_datasets([dataset["train"], dataset["val"]])
    full_df = full_dataset.to_pandas()

    return {
        "train": train_df,
        "test": validation_df,
        "full": full_df
    }



def main(chosen_seed):
    """
    Runs the complete project pipeline.

    Assumption:
    --- User has already run "download_data.py".
    Inputs: None
    Output: None
    """

    set_seed(chosen_seed)

    # Loads datasets and runs EDA analysis
    datasets_dict = data_loading()
    EDA_Analysis(datasets_dict)








    return 0 # Success







if __name__ == "__main__":
    main(chosen_seed=42)