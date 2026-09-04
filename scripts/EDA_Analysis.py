# This python file implements the Exploratory Data Analysis (EDA) part.


# General imports:
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.ticker import MultipleLocator



def get_label_stats(name, dataframe, labels_col="verified_uat_labels"):
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

    label_counts = Counter(label for labels in dataframe[labels_col] for label in labels)
    label_stats = pd.DataFrame(label_counts.items(), columns=["label", "count"])

    label_stats = (
        label_stats
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    print(f"--- {name} dataset has: {len(dataframe):,} papers; {len(label_stats):,} unique labels.")
    return label_stats



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



def EDA_Analysis(datasets_dict):
    """
    The full implementation of the EDA step.
    This method plots the EDA graphs for each original (train, validation) dataset.

    Input:
    --- datasets_dict: dictionary of pandas dataframes (representing the original splits)
    Outputs: None
    """

    print("\nAnalyzing datasets by focusing on the labels:")

    # Analyze each dataset:
    for name, dataframe in datasets_dict.items():

        # Calculate label statistics:
        label_stats = get_label_stats(name, dataframe)

        # Plot label analysis:
        plot_label_analysis(label_stats, dataset_name=name)

    print("Done plotting the original splits datasets.")
    return None

