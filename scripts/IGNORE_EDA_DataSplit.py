# By: Shai Habi;  August 24th, 2026

# This python file mainly implements the Exploratory Data Analysis (EDA) and the Data Splitting part.
# Since the original split contains only "train" and "validation", and since we remove any label that appears less than k* times,
# we cannot use the original split.
# Overall, this file contains the following sections: EDA and Data Splitting


# For loading dataset from hf_cache:
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))

# General imports:
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.ticker import MultipleLocator
from datasets import load_dataset, concatenate_datasets

# For splitting the dataset according to k:
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


########################################################################################################################
# Section: Data Loading
# This section loads the original dataset, and combines its original splits ("train" and "validation") to one dataframe.
########################################################################################################################

def data_loading_combining():
    """
    Loads the dataset and combines the original splits to one dataframe.

    Assumption:
    --- Data already exists in hf_cache. (user already ran "download_data.py")
    Input: None
    Output:
    --- d : dictionary of pandas dataframes
    """

    print("Loading data from hf_cache ... ", end="")
    dataset = load_dataset("adsabs/SciX_UAT_keywords")
    print("Done!")
    # print(dataset)

    # Merges:
    train_df = dataset["train"].to_pandas()
    validation_df = dataset["val"].to_pandas()
    full_dataset = concatenate_datasets([dataset["train"], dataset["val"]])
    full_df = full_dataset.to_pandas()
    print(f"Total papers: {len(full_df):,}")

    return {
        "train": train_df,
        "validation": validation_df,
        "full": full_df
    }


########################################################################################################################
# Section:  Statistical Analysis of the Dataset
# This section counts how many unique labels each dataframe has, and returns it for future plotting.
########################################################################################################################

def get_label_stats(name, dataframe, toPrint=False, labels_col="verified_uat_labels"):
    """
    Calculates label-frequency statistics for a dataframe using UAT IDs.

    Inputs:
    --- name : str
    --- dataframe : pandas.DataFrame
    --- labels_col : str
    --- toPrint : boolean
    Output:
    --- pandas.DataFrame
    """

    label_counts = Counter(
        label
        for labels in dataframe[labels_col]
        for label in labels
    )

    label_stats = pd.DataFrame(
        label_counts.items(),
        columns=["label", "count"]
    )

    label_stats = (
        label_stats
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    if toPrint:
        print(
            f"{name} dataset has:\n"
            f"{len(dataframe):,} papers.\n"
            f"{len(label_stats):,} unique labels."
        )

    return label_stats


########################################################################################################################
# Section:  Plot Distributions
# Plots the distributions of given dataframes.
########################################################################################################################

def plot_label_frequency_distribution(
        label_stats,
        y_tick_interval=50,
        figsize=(16, 7),
        title="UAT Label Frequency Distribution",
        ax=None
):
    """
    Plot the frequency distribution of labels, sorted from most frequent to least frequent.
    Each label is represented by a bar whose height is proportional to its frequency.
    The function sorts the input internally for robustness, so label_stats does not need to be sorted beforehand.

    Inputs:
    --- label_stats : pandas.DataFrame
    --- y_tick_interval : int
    --- figsize : tuple
    --- title : str
    --- ax : matplotlib.axes.Axes, optional
        If provided, the plot is drawn onto this axis (e.g. one panel of a combined
        figure) instead of creating and showing its own standalone figure.
    Output: None
    """

    # Validates requirements:
    if label_stats.empty:
        raise ValueError("label_stats cannot be empty.")

    required_columns = {"label", "count"}
    if not required_columns.issubset(label_stats.columns):
        raise ValueError("label_stats must contain 'label' and 'count' columns.")

    # Sort labels from most frequent to least frequent
    stats = (
        label_stats
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    n_labels = len(stats)
    tick_positions = [
        0,
        n_labels // 4,
        n_labels // 2,
        3 * n_labels // 4,
        n_labels - 1
    ]

    percentages = [0, 25, 50, 75, 100]

    # If no axis was supplied, this plot is being made standalone: create its
    # own figure and remember to show() it at the end. If an axis WAS supplied
    # (e.g. from plot_label_analysis), just draw onto it and let the caller
    # handle figure-level layout/showing.
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)

    ax.bar(range(n_labels), stats["count"])
    ax.set_xlabel("Labels (sorted by frequency)")
    ax.set_ylabel("Number of papers")
    ax.set_title(title)

    tick_labels = [
        f"{pct}%\n"
        f"#{pos + 1}\n"
        f"{stats.iloc[pos]['label']}\n"
        f"Y = {stats.iloc[pos]['count']}"
        for pos, pct in zip(tick_positions, percentages)
    ]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        tick_labels,
        rotation=30,
        ha="right"
    )
    ax.yaxis.set_major_locator(MultipleLocator(y_tick_interval))

    for pos in tick_positions:
        y = stats.iloc[pos]["count"]

        ax.annotate(
            f"{y}",
            xy=(pos, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold"
        )

    ax.grid(axis="y", alpha=0.3)

    if standalone:
        plt.tight_layout()
        plt.show()

    return None


def plot_label_frequency_bins(
        label_stats,
        bins=None,
        bin_labels=None,
        figsize=(14, 6),
        title="Distribution of UAT Labels by Frequency Range",
        ax=None
):
    """
    Plot the number of labels that fall into different frequency ranges.
    Each bar represents a range of label frequencies.
    The height of the bar indicates how many labels occur within that range.

    Inputs:
    --- label_stats : pandas.DataFrame
    --- bins : list
    --- bin_labels : list
    --- figsize : tuple
    --- title : str
    --- ax : matplotlib.axes.Axes, optional
        If provided, the plot is drawn onto this axis instead of creating and
        showing its own standalone figure.
    Output: None
    """

    # Validate input
    if label_stats.empty:
        raise ValueError("label_stats cannot be empty.")

    if "count" not in label_stats.columns:
        raise ValueError("label_stats must contain a 'count' column.")

    # Default frequency bins
    if bins is None:
        bins = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, float("inf")]

    if bin_labels is None:
        bin_labels = [
            "1", "2", "3-5", "6-10", "11-20", "21-50", "51-100",
            "101-200", "201-500", "501-1000", "1000+"
        ]

    if len(bin_labels) != len(bins) - 1:
        raise ValueError("bin_labels must contain exactly len(bins) - 1 elements.")

    # Assign each label to a frequency bin
    frequency_bins = pd.cut(
        label_stats["count"],
        bins=bins,
        labels=bin_labels,
        right=True,
        include_lowest=True
    )

    # Count labels in each bin
    binned_distribution = (
        frequency_bins
        .value_counts()
        .sort_index()
    )

    # Plot
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        binned_distribution.index.astype(str),
        binned_distribution.values
    )

    ax.set_xlabel("Number of papers containing the label")
    ax.set_ylabel("Number of labels")
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(axis="y", alpha=0.3)

    # Display the number of labels above each bar
    for bar, value in zip(bars, binned_distribution.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom"
        )

    if standalone:
        plt.tight_layout()
        plt.show()

    return None


def plot_label_analysis(
        label_stats,
        dataset_name="Dataset",
        y_tick_interval=50,
        combined=True,
        figsize=(20, 7)
):
    """
    Plot both label-frequency visualizations for a dataset.

    This function is a wrapper around:
    --- plot_label_frequency_distribution
    --- plot_label_frequency_bins

    Inputs:
    --- label_stats : pandas.DataFrame
    --- dataset_name : str
    --- y_tick_interval : int
    --- combined : bool
        If True (default), both plots are drawn side-by-side as two panels of
        ONE figure. If False, each plot is shown in its own separate figure
        (the original behavior).
    --- figsize : tuple
        Size of the combined figure. Only used when combined=True.
    Output: None
    """

    if not combined:
        plot_label_frequency_distribution(
            label_stats,
            y_tick_interval=y_tick_interval,
            title=f"UAT Label Frequency Distribution - {dataset_name}"
        )

        print(" ")
        plot_label_frequency_bins(
            label_stats,
            title=f"UAT Label Frequency Ranges - {dataset_name}"
        )
        return None

    # Combined: one figure, two panels sharing the same label_stats.
    fig, (ax_dist, ax_bins) = plt.subplots(1, 2, figsize=figsize)

    plot_label_frequency_distribution(
        label_stats,
        y_tick_interval=y_tick_interval,
        title="Frequency Distribution",
        ax=ax_dist
    )

    plot_label_frequency_bins(
        label_stats,
        title="Frequency Ranges",
        ax=ax_bins
    )

    fig.suptitle(f"UAT Label Analysis - {dataset_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return None


########################################################################################################################
# Section:  EDA - Overall:
# Plots the distributions of given dataframes.
########################################################################################################################

def EDA_Analysis():
    """
    The full implementation of the EDA step.
    It can be called once and later several values of k use the results.

    Input: None
    Outputs:
    --- dict_of_dfs : dictionary of pandas dataframes (representing the original splits)
    --- dict_of_label_stats : dictionary of pandas dataframes (representing the label statistics for each dataset)
    """

    # Load the datasets:
    dict_of_dfs = data_loading_combining()
    dict_of_label_stats = {}  # Stores label statistics for each dataset

    # Analyze each dataset:
    print("\nAnalyzing original datasets by focusing on the labels:")
    for name, dataframe in dict_of_dfs.items():
        print(f"===== Original {name} =====")

        # Calculate label statistics:
        label_stats = get_label_stats(name, dataframe, True)
        # Save statistics for later use:
        dict_of_label_stats[name] = label_stats
        # Plot label analysis:
        plot_label_analysis(label_stats, dataset_name=name)
        print(" ")

    print("\nDone plotting the original splits datasets.\n")
    return dict_of_dfs, dict_of_label_stats


########################################################################################################################
# Section: K Analysis and Data Splitting
# This section analyses the data according to our threshold k, in order to find the k* we shall work with.
########################################################################################################################


def analyze_and_split_by_k(
        dataframe,
        label_stats,
        k,
        chosen_seed,
        labels_col="verified_uat_ids",
        split_ratios=(0.7, 0.1, 0.2)
    ):

    """
    Analyze the effect of a minimum label-frequency threshold K.

    The function:
    1. Keeps labels appearing at least K times.
    2. Removes invalid labels from each document.
    3. Removes documents left with no valid labels.
    4. Plots the filtered label distribution.
    5. Performs a multi-label stratified Train / Validation / Test split.
    6. Reports document and label counts in each split.
    7. Creates a label coverage table across all three splits.

    Inputs:
    --- dataframe : pandas.DataFrame
    --- label_stats : pandas.DataFrame
    --- k : positive int
    --- chosen_seed: positive int
    --- labels_col : str
    --- split_ratios : tuple
        (train_ratio, validation_ratio, test_ratio)

    Outputs:
    --- results : dict
        Contains:
        - filtered_dataframe
        - filtered_label_stats
        - splits
        - split_label_stats
        - coverage_table
        - problematic_labels
    """

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    if k < 1:
        raise ValueError("k must be at least 1.")

    if len(split_ratios) != 3:
        raise ValueError(
            "split_ratios must contain exactly three values: "
            "(train, validation, test)."
        )

    train_ratio, validation_ratio, test_ratio = split_ratios

    if not abs(
            train_ratio + validation_ratio + test_ratio - 1.0
    ) < 1e-9:
        raise ValueError("split_ratios must sum to 1.")

    if dataframe.empty:
        raise ValueError("dataframe cannot be empty.")

    if labels_col not in dataframe.columns:
        raise ValueError(
            f"'{labels_col}' does not exist in dataframe."
        )

    # --------------------------------------------------------
    # K filtering
    # --------------------------------------------------------

    print(f"========  K = {k} ANALYSIS  ========")

    documents_before = len(dataframe)
    labels_before = len(label_stats)

    # Labels that survive K:
    valid_labels = set(
        label_stats.loc[
            label_stats["count"] >= k,
            "label"
        ]
    )

    # Copy dataframe:
    filtered_dataframe = dataframe.copy()

    # Remove IDs below K together with their corresponding label names:
    def filter_uat_pairs(row):
        uat_ids = row["verified_uat_ids"]
        uat_labels = row["verified_uat_labels"]

        if len(uat_ids) != len(uat_labels):
            raise ValueError(
                "verified_uat_ids and verified_uat_labels "
                "must have the same length."
            )

        retained_pairs = [
            (uat_id, uat_label)
            for uat_id, uat_label in zip(uat_ids, uat_labels)
            if uat_id in valid_labels
        ]

        row["verified_uat_ids"] = [
            uat_id
            for uat_id, _ in retained_pairs
        ]

        row["verified_uat_labels"] = [
            uat_label
            for _, uat_label in retained_pairs
        ]

        return row

    filtered_dataframe = filtered_dataframe.apply(
        filter_uat_pairs,
        axis=1
    )

    # Remove documents that have no labels left:
    filtered_dataframe = (
        filtered_dataframe[
            filtered_dataframe[labels_col].apply(len) > 0
            ]
        .reset_index(drop=True)
    )

    documents_after = len(filtered_dataframe)

    # Recalculate label statistics:
    filtered_label_stats = get_label_stats(
        f"K={k}",
        filtered_dataframe,
        labels_col=labels_col
    )

    labels_after = len(filtered_label_stats)

    # --------------------------------------------------------
    # General K summary
    # --------------------------------------------------------

    print("\n----- FILTERING SUMMARY -----")

    print(f"Documents before: {documents_before:,}")
    print(f"Documents after:  {documents_after:,}")
    print(
        f"Documents removed: "
        f"{documents_before - documents_after:,}"
    )

    print()

    print(f"Labels before: {labels_before:,}")
    print(f"Labels after:  {labels_after:,}")
    print(
        f"Labels removed: "
        f"{labels_before - labels_after:,}"
    )

    # --------------------------------------------------------
    # Plot filtered label distribution
    # --------------------------------------------------------

    plot_label_analysis(
        filtered_label_stats,
        dataset_name=f"K = {k}"
    )

    # --------------------------------------------------------
    # Convert labels to multi-hot representation
    # --------------------------------------------------------

    mlb = MultiLabelBinarizer()

    y = mlb.fit_transform(
        filtered_dataframe[labels_col]
    )

    # --------------------------------------------------------
    # First split:
    # FULL -> TRAIN + TEMP
    # --------------------------------------------------------

    temp_ratio = validation_ratio + test_ratio

    first_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=temp_ratio,
        random_state=chosen_seed
    )

    train_idx, temp_idx = next(
        first_splitter.split(
            filtered_dataframe,
            y
        )
    )

    train_df = (
        filtered_dataframe
        .iloc[train_idx]
        .copy()
        .reset_index(drop=True)
    )

    temp_df = (
        filtered_dataframe
        .iloc[temp_idx]
        .copy()
        .reset_index(drop=True)
    )

    y_temp = y[temp_idx]

    # --------------------------------------------------------
    # Second split:
    # TEMP -> VALIDATION + TEST
    # --------------------------------------------------------

    test_ratio_inside_temp = (
            test_ratio / temp_ratio
    )

    second_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=test_ratio_inside_temp,
        random_state=chosen_seed
    )

    validation_idx, test_idx = next(
        second_splitter.split(
            temp_df,
            y_temp
        )
    )

    validation_df = (
        temp_df
        .iloc[validation_idx]
        .copy()
        .reset_index(drop=True)
    )

    test_df = (
        temp_df
        .iloc[test_idx]
        .copy()
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Save splits
    # --------------------------------------------------------

    splits = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df
    }

    # --------------------------------------------------------
    # Calculate label statistics for each split
    # --------------------------------------------------------

    split_label_stats = {}

    print("\n----- ACTUAL STRATIFIED SPLIT -----")

    for name, split_df in splits.items():
        stats = get_label_stats(
            name,
            split_df,
            labels_col=labels_col
        )

        split_label_stats[name] = stats

        print(
            f"{name.upper()}: "
            f"{len(split_df):,} documents, "
            f"{len(stats):,} unique labels"
        )

    # --------------------------------------------------------
    # Build label coverage table
    # --------------------------------------------------------

    coverage_table = (
        filtered_label_stats[
            ["label", "count"]
        ]
        .rename(
            columns={
                "count": "total_count"
            }
        )
        .copy()
    )

    for split_name in [
        "train",
        "validation",
        "test"
    ]:
        split_counts = (
            split_label_stats[split_name]
            .set_index("label")["count"]
        )

        coverage_table[
            f"{split_name}_count"
        ] = (
            coverage_table["label"]
            .map(split_counts)
            .fillna(0)
            .astype(int)
        )

    # --------------------------------------------------------
    # Presence indicators
    # --------------------------------------------------------

    coverage_table["in_train"] = (
            coverage_table["train_count"] > 0
    )

    coverage_table["in_validation"] = (
            coverage_table["validation_count"] > 0
    )

    coverage_table["in_test"] = (
            coverage_table["test_count"] > 0
    )

    coverage_table["in_all_splits"] = (
            coverage_table["in_train"]
            & coverage_table["in_validation"]
            & coverage_table["in_test"]
    )

    # --------------------------------------------------------
    # Coverage summary
    # --------------------------------------------------------

    total_retained_labels = len(coverage_table)

    labels_in_all = (
        coverage_table["in_all_splits"]
        .sum()
    )

    missing_train = (
        ~coverage_table["in_train"]
    ).sum()

    missing_validation = (
        ~coverage_table["in_validation"]
    ).sum()

    missing_test = (
        ~coverage_table["in_test"]
    ).sum()

    only_train = (
            coverage_table["in_train"]
            & ~coverage_table["in_validation"]
            & ~coverage_table["in_test"]
    ).sum()

    only_validation = (
            ~coverage_table["in_train"]
            & coverage_table["in_validation"]
            & ~coverage_table["in_test"]
    ).sum()

    only_test = (
            ~coverage_table["in_train"]
            & ~coverage_table["in_validation"]
            & coverage_table["in_test"]
    ).sum()

    print("\n----- LABEL COVERAGE -----")

    print(
        f"Labels present in all 3 splits: "
        f"{labels_in_all:,} / "
        f"{total_retained_labels:,} "
        f"({labels_in_all / total_retained_labels:.2%})"
    )

    print(
        f"Labels missing from TRAIN:      "
        f"{missing_train:,}"
    )

    print(
        f"Labels missing from VALIDATION: "
        f"{missing_validation:,}"
    )

    print(
        f"Labels missing from TEST:       "
        f"{missing_test:,}"
    )

    print()

    print(
        f"Labels appearing only in TRAIN:      "
        f"{only_train:,}"
    )

    print(
        f"Labels appearing only in VALIDATION: "
        f"{only_validation:,}"
    )

    print(
        f"Labels appearing only in TEST:       "
        f"{only_test:,}"
    )

    # --------------------------------------------------------
    # Problematic labels
    # --------------------------------------------------------

    problematic_labels = (
        coverage_table[
            ~coverage_table["in_all_splits"]
        ]
        .sort_values(
            "total_count",
            ascending=True
        )
        .reset_index(drop=True)
    )

    print(
        f"\nLabels missing from at least one split: "
        f"{len(problematic_labels):,}"
    )

    if len(problematic_labels) > 0:
        print(
            "\nFirst 20 problematic labels:"
        )

        print(
            problematic_labels[
                [
                    "label",
                    "total_count",
                    "train_count",
                    "validation_count",
                    "test_count"
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # Return everything for further analysis
    # --------------------------------------------------------

    results = {
        "filtered_dataframe": filtered_dataframe,
        "filtered_label_stats": filtered_label_stats,
        "splits": splits,
        "split_label_stats": split_label_stats,
        "coverage_table": coverage_table,
        "problematic_labels": problematic_labels,
    }

    return results


########################################################################################################################
# Section: OVERALL
########################################################################################################################

def main_EDA_DataSplit(k_list, chosen_seed):
    """
    The main function for the whole EDA and K-Analysis section.
    This function should be called from the main project file.

    If k_list contains multiple values, all values are analyzed,
    but only the splits corresponding to the first K are returned.
    If k_list contains a single value, its splits are returned.

    Assumptions:
    --- Data already exists in hf_cache. (user already ran "download_data.py")
    Inputs:
    --- k_list : list of positive integers
    --- chosen_seed: positive int
    Outputs:
    --- results : dict,  contains the splits to train, validation and test
    """

    # Validates input:
    if not (isinstance(k_list, list)):
        raise ValueError("EDA: Invalid type for input k.")

    if len(k_list) == 0 or (False in [isinstance(k, int) and k > 0 for k in k_list]):
        raise ValueError("EDA: k_list input must only contain positive integers.")

    print("\n======== Starts EDA and Data Splitting process ========")
    # Loads data and EDA process:
    dict_of_dfs, dict_of_label_stats = EDA_Analysis()

    # Extracts the full dataset:
    full_df = dict_of_dfs["full"]

    # analyze_and_split_by_k filters/splits on verified_uat_ids (see its
    # labels_col default), so it needs label_stats computed on that same
    # column -- not the label-string stats used for the EDA plots above.
    full_label_stats_ids = get_label_stats("full", full_df, labels_col="verified_uat_ids")

    # Analysis:
    print("======== Starts analyzing and splitting according to chosen k ========")
    results_by_k = {}
    for checked_k in k_list:
        results_by_k[checked_k] = analyze_and_split_by_k(
            dataframe=full_df,
            label_stats=full_label_stats_ids,
            k=checked_k,
            chosen_seed=chosen_seed
        )

    print("\n======== Done running EDA and Data Splitting ! ========")

    # Returns only the split
    # A reminder: this function should be called from the main project file
    chosen_k = k_list[0]
    chosen_results = results_by_k[chosen_k]
    return chosen_results["splits"]



if __name__ == "__main__":
    k_values = [10]  ## Change me everytime, TODO ##
    seed = 42           ## Change me everytime, TODO ##
    main_EDA_DataSplit(k_values, seed)
