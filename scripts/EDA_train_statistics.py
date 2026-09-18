from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from baselines import build_paper_texts

def analyze_train_vocabulary(train_df):
    """
    Analyses unigram and bigram distributions using training texts only.

    The analysis helps choose:
    --- ngram_range
    --- min_df
    --- max_features

    Input:
    --- train_df: training DataFrame containing title and abstract.
    Output:
    --- dictionary containing summary tables and raw counters.
    """

    print("Building training texts ...")
    train_texts = build_paper_texts(train_df)

    # Uses the same tokenisation planned for the real TF-IDF model.
    unigram_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 1),
    )
    unigram_analyzer = unigram_vectorizer.build_analyzer()

    unigram_term_frequency = Counter()
    unigram_document_frequency = Counter()
    word_lengths = []

    print("Analysing unigrams ...")

    for text in train_texts:
        tokens = unigram_analyzer(text)

        # Counts every occurrence in the corpus.
        unigram_term_frequency.update(tokens)

        # Counts each token at most once per paper.
        unigram_document_frequency.update(set(tokens))

        # Measures token length in characters.
        word_lengths.extend(len(token) for token in tokens)

    # Uses the same tokenisation, but produces adjacent word pairs.
    bigram_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(2, 2),
    )
    bigram_analyzer = bigram_vectorizer.build_analyzer()

    bigram_term_frequency = Counter()
    bigram_document_frequency = Counter()

    print("Analysing bigrams ...")

    for text in train_texts:
        bigrams = bigram_analyzer(text)

        bigram_term_frequency.update(bigrams)
        bigram_document_frequency.update(set(bigrams))

    # Summarises word lengths in characters.
    word_lengths_array = np.asarray(word_lengths)

    word_length_summary = {
        "number_of_token_occurrences": int(len(word_lengths_array)),
        "mean": float(word_lengths_array.mean()),
        "median": float(np.median(word_lengths_array)),
        "percentile_75": float(np.percentile(word_lengths_array, 75)),
        "percentile_90": float(np.percentile(word_lengths_array, 90)),
        "percentile_95": float(np.percentile(word_lengths_array, 95)),
        "percentile_99": float(np.percentile(word_lengths_array, 99)),
        "maximum": int(word_lengths_array.max()),
    }

    # Counts how many features survive possible min_df choices.
    min_df_rows = []

    for min_df in (1, 2, 3, 5, 10, 20, 50):
        surviving_unigrams = sum(
            document_count >= min_df
            for document_count in unigram_document_frequency.values()
        )
        surviving_bigrams = sum(
            document_count >= min_df
            for document_count in bigram_document_frequency.values()
        )

        min_df_rows.append({
            "min_df": min_df,
            "unigrams": surviving_unigrams,
            "bigrams": surviving_bigrams,
            "total_features": surviving_unigrams + surviving_bigrams,
        })

    min_df_table = pd.DataFrame(min_df_rows)

    # Examines how much corpus frequency is covered by top-N features.
    combined_term_frequency = (
        unigram_term_frequency + bigram_term_frequency
    )

    sorted_frequencies = np.asarray(
        sorted(
            combined_term_frequency.values(),
            reverse=True,
        ),
        dtype=np.int64,
    )

    cumulative_frequencies = np.cumsum(sorted_frequencies)
    total_frequency = cumulative_frequencies[-1]

    coverage_rows = []

    for max_features in (
        10_000,
        25_000,
        50_000,
        75_000,
        100_000,
        150_000,
        200_000,
    ):
        number_kept = min(max_features, len(sorted_frequencies))

        covered_frequency = cumulative_frequencies[number_kept - 1]
        coverage = covered_frequency / total_frequency

        coverage_rows.append({
            "max_features": max_features,
            "features_actually_available": number_kept,
            "term_occurrence_coverage": float(coverage),
        })

    coverage_table = pd.DataFrame(coverage_rows)

    print("\nWord-length distribution:")
    for name, value in word_length_summary.items():
        print(f"  {name}: {value}")

    print("\nUnique vocabulary:")
    print(f"  Unigrams: {len(unigram_term_frequency):,}")
    print(f"  Bigrams:  {len(bigram_term_frequency):,}")
    print(
        "  Combined: "
        f"{len(unigram_term_frequency) + len(bigram_term_frequency):,}"
    )

    print("\nFeatures remaining after min_df:")
    print(min_df_table.to_string(index=False))

    print("\nTop-N feature coverage:")
    print(coverage_table.to_string(index=False))

    return {
        "word_length_summary": word_length_summary,
        "min_df_table": min_df_table,
        "coverage_table": coverage_table,
        "unigram_term_frequency": unigram_term_frequency,
        "unigram_document_frequency": unigram_document_frequency,
        "bigram_term_frequency": bigram_term_frequency,
        "bigram_document_frequency": bigram_document_frequency,
    }


def main_eda_train_statistics():

    from main import load_dataset_from_split_json_file

    datasets_dict = load_dataset_from_split_json_file()
    train_df = datasets_dict["train"]
    analyze_train_vocabulary(train_df)

if __name__ == "__main__":
    main_eda_train_statistics()
