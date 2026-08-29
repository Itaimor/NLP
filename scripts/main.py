# By Shai Habi, 29/08/2026
# This file is the main project file, running the full experiment.

# General libraries:
import random
import numpy as np
import torch
import time

# Imports from other files:
from EDA_DataSplit import main_EDA_DataSplit


# Global variables:
SAVE_PATH = ""


## Subsection: General ##
def set_seed(seed):
    """
    Fixes random seeds for reproducibility.

    Input:
    --- seed: int
    Output: None
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


## Subsection: Preprocessing ##
def create_input_text_to_models(dataframe):
    """
    Creates the same textual input for both models: SciBert and Gemma.
    The exact format can be changed later, but both models must receive the same information.

    Input:
    --- dataframe : pandas.DataFrame
    Output:
    --- dataframe : pandas.DataFrame
    """

    dataframe = dataframe.copy()
    titles = dataframe["title"].fillna("").astype(str)
    abstracts = dataframe["abstract"].fillna("").astype(str)
    dataframe["input_text"] = ("Title: " + titles + "\nAbstract: " + abstracts)

    return dataframe



## Section: Main ##
def main(chosen_k, chosen_seed):
    """
    Runs the current project pipeline given our chosen k value.

    Assumption:
    --- User has already run "download_data.py".
    Inputs:
    --- chosen_k, chosen_seed : int
    """

    # Fixes the seed throughout the experiment:
    set_seed(chosen_seed)

    # Runs main_EDA_DataSplit:
    # This function loads the original dataset from HF, runs full EDA, removes and split according to k.
    datasets = main_EDA_DataSplit(k_list=[chosen_k], chosen_seed=chosen_seed)
    print(" ")

    # Preprocesses before sending to SciBert and Gemma:



    return 0 # Success







if __name__ == "__main__":
    main(chosen_k=14, chosen_seed=42)