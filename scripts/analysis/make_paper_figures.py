#!/usr/bin/env python3
"""Build the paper's figures as vector PDF.

The course guidelines (line 143) require figures as PDF and not PNG, so these are
written straight out of matplotlib as vector rather than converted from an image.

Two figures:

  paper/figures/fig1_per_band_micro_f1.pdf
      Per-band test micro-F1 for all six systems, with the coverage@50 recall
      ceiling marked over the two Gemma arms. Every value is read from a
      committed artefact and matches Table 1 in paper/results.tex digit for digit.

  paper/figures/fig2_coverage_at50_per_epoch.pdf
      Validation coverage@50 per band across the eight saved SciBERT epochs,
      recomputed from the per-epoch parquet tables. Reproduces the epoch-8 row in
      results/scibert_full/PROVENANCE.md (0.829 overall, 0.980 / 0.908 / 0.528).

Colours come from a categorical palette validated for colour-vision deficiency
separation. Three slots sit below 3:1 contrast against the chart surface, which
obliges either visible labels or a table view. Table 1 carries every value in
Figure 1, and Figure 2 direct-labels its three lines, so that obligation is met
without printing a number on all eighteen bars.

Run from the project root:

    python scripts/analysis/make_paper_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT / "paper" / "figures"

# Validated categorical palette, assigned in fixed order and never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdbd6"

BANDS = ("head", "torso", "tail")

# Test micro-F1, from results/*/test_results_*.json and
# results/scibert_lora/run2/lora/test_results_scibert_lora.json.
# Order is Table 1's order, weakest to strongest.
SYSTEMS = [
    ("Majority prior", [0.0914, 0.0385, 0.0055]),
    ("TF-IDF + LR", [0.0719, 0.0159, 0.0014]),
    ("SciBERT-LoRA", [0.4764, 0.2636, 0.1190]),
    ("SciBERT-alone", [0.5518, 0.3566, 0.2150]),
    ("Gemma, untuned", [0.5530, 0.3703, 0.2943]),
    ("Gemma, fine-tuned", [0.5690, 0.3783, 0.2897]),
]

# Test coverage@50 per band, from results/scibert_full/PROVENANCE.md.
# This bounds RECALL for the two Gemma arms, because only they are restricted to
# SciBERT's top 50. It is not a bound on the baselines or on SciBERT-alone, which
# score over all 1,864 concepts.
CEILING = {"head": 0.982, "torso": 0.900, "tail": 0.648}


def house_style():
    """Matplotlib settings for an ACL-format PDF."""

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,  # TrueType, not Type 3, which some venues reject.
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 7,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
        }
    )


def recessive_axes(ax):
    """Drop the chart junk so the marks carry the reading."""

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def figure_one(path):
    """Per-band micro-F1 for every system, with the Gemma recall ceiling."""

    fig, ax = plt.subplots(figsize=(6.3, 2.5))
    recessive_axes(ax)

    n = len(SYSTEMS)
    group_width = 0.78
    bar_width = group_width / n
    centres = np.arange(len(BANDS))

    for slot, (name, values) in enumerate(SYSTEMS):
        # A 2px surface gap between adjacent bars, expressed in data units.
        offset = -group_width / 2 + bar_width * (slot + 0.5)
        ax.bar(
            centres + offset,
            values,
            width=bar_width * 0.88,
            color=SERIES[slot],
            label=name,
            linewidth=0,
            zorder=3,
        )

    # The ceiling spans only the two Gemma bars, because it binds only them. The
    # Gemma arms are slots 4 and 5, so the span runs from the left edge of slot 4
    # to the right edge of slot 5.
    gemma_left = -group_width / 2 + bar_width * 4
    gemma_right = -group_width / 2 + bar_width * 6
    for i, band in enumerate(BANDS):
        y = CEILING[band]
        ax.plot(
            [centres[i] + gemma_left, centres[i] + gemma_right],
            [y, y],
            color=INK_MUTED,
            linewidth=1.0,
            linestyle=(0, (3, 2)),
            zorder=4,
        )
        ax.annotate(
            f"{y:.3f}",
            xy=(centres[i] + (gemma_left + gemma_right) / 2, y),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=6.5,
            color=INK_MUTED,
        )

    # Named once, on the first group, rather than seated in the legend where it
    # would read as a seventh system.
    # Sits in the empty band under the head ceiling, clear of the y tick labels
    # on the left and of the torso ceiling mark on the right.
    ax.annotate(
        "coverage@50, the recall ceiling\nfor the two Gemma arms",
        xy=(centres[0] + gemma_left, CEILING["head"]),
        xytext=(2, -11),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=6.5,
        color=INK_MUTED,
        linespacing=1.35,
    )

    ax.set_xticks(centres)
    ax.set_xticklabels([b.capitalize() for b in BANDS])
    ax.set_ylabel("Test micro-F1")
    ax.set_ylim(0, 1.06)
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    # Matplotlib fills a multi-column legend column-major, so three columns put
    # the six systems into their three families: non-neural, encoder, Gemma.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.5,
        columnspacing=2.2,
        labelcolor=INK,
    )

    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def figure_two(path, per_epoch):
    """Validation coverage@50 per band across the saved epochs."""

    fig, ax = plt.subplots(figsize=(3.15, 2.3))
    recessive_axes(ax)

    epochs = sorted(int(e) for e in per_epoch)
    # Bands reuse the first three categorical slots, in head/torso/tail order.
    for slot, band in enumerate(BANDS):
        ys = [per_epoch[str(e)][band] for e in epochs]
        ax.plot(
            epochs,
            ys,
            color=SERIES[slot],
            linewidth=1.6,
            marker="o",
            markersize=3.2,
            markeredgecolor=SURFACE,
            markeredgewidth=0.6,
            zorder=3,
        )
        # Direct label at the line end, so identity is never colour alone.
        ax.annotate(
            band.capitalize(),
            xy=(epochs[-1], ys[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
            color=INK,
        )

    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Validation coverage@50")
    ax.set_xticks(epochs)
    ax.set_xlim(0.7, epochs[-1] + 1.5)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    coverage_path = PROJECT / "results/scibert_full/coverage_at50_per_epoch.json"

    if not coverage_path.exists():
        print(
            f"missing {coverage_path}\n"
            "Run scripts/analysis/coverage_at50_per_epoch.py first.",
            file=sys.stderr,
        )
        return 1

    per_epoch = json.loads(coverage_path.read_text(encoding="utf-8"))["epochs"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    house_style()

    one = OUT_DIR / "fig1_per_band_micro_f1.pdf"
    two = OUT_DIR / "fig2_coverage_at50_per_epoch.pdf"

    figure_one(one)
    figure_two(two, per_epoch)

    for path in (one, two):
        print(f"wrote {path.relative_to(PROJECT)}  ({path.stat().st_size / 1024:.1f} KB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
