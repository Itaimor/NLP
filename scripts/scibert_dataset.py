#!/usr/bin/env python3
"""
SciX UAT Dataset & Data-Loading Front End
=========================================
Implements text cleaning, dataset handling, and tokenization for SciBERT
classification over the Unified Astronomy Thesaurus (UAT) label space.

Inputs:
- Paper title and abstract from adsabs/SciX_UAT_keywords
- Verified UAT integer/string IDs

Outputs:
- PyTorch Dataset and DataLoaders yielding:
  (input_ids, attention_mask, labels)
  where labels is a multi-hot torch.float32 tensor of length num_labels.
"""

import re
import html
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

SCIBERT_MODEL_NAME = "allenai/scibert_scivocab_uncased"
DEFAULT_MAX_LENGTH = 512

# Common XML/HTML formatting tags in NASA ADS astrophysics abstracts
HTML_TAG_RE = re.compile(r"<\/?(sub|sup|i|b|em|strong|p|br|span|div|a)[^>]*>", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def clean_astronomy_text(text: Optional[str]) -> str:
    """
    Cleans astronomy abstract/title text by stripping inline formatting HTML tags
    (e.g., <SUB>, <SUP>, <P>), unescaping HTML entities, and normalizing whitespace.

    Parameters:
    --- text: input string or None

    Returns:
    --- cleaned string
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Unescape HTML entities (e.g., &amp; -> &, &lt; -> <)
    cleaned = html.unescape(text)

    # 2. Strip XML/HTML tags
    cleaned = HTML_TAG_RE.sub(" ", cleaned)

    # 3. Normalize whitespace (newlines, tabs, multiple spaces)
    cleaned = WHITESPACE_RE.sub(" ", cleaned).strip()

    return cleaned


def build_topic_mapping(
    topics: Union[List[Any], pd.Series, set]
) -> Tuple[Dict[Any, int], Dict[int, Any]]:
    """
    Constructs bidirectional mappings between topic identifiers (e.g. UAT IDs)
    and consecutive integer indices in [0, num_labels - 1].

    Parameters:
    --- topics: iterable of unique topic identifiers

    Returns:
    --- topic_to_idx: dict mapping identifier -> int index
    --- idx_to_topic: dict mapping int index -> identifier
    """
    # Sort for deterministic, reproducible indexing
    sorted_topics = sorted(list(set(topics)), key=lambda x: str(x))
    topic_to_idx = {topic: idx for idx, topic in enumerate(sorted_topics)}
    idx_to_topic = {idx: topic for idx, topic in enumerate(sorted_topics)}
    return topic_to_idx, idx_to_topic


class SciXDataset(Dataset):
    """
    PyTorch Dataset for SciBERT multi-label topic classification.

    Combines paper title and abstract, tokenizes with SciBERT vocabulary,
    and converts verified UAT IDs to multi-hot float32 tensors.

    Yields tuples of (input_ids, attention_mask, labels) to match the
    exact assumptions of train_one_epoch() and evaluate().
    """

    def __init__(
        self,
        dataframe_or_records: Union[pd.DataFrame, List[Dict[str, Any]]],
        topic_to_idx: Dict[Any, int],
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        title_col: str = "title",
        abstract_col: str = "abstract",
        labels_col: str = "verified_uat_ids",
        bibcode_col: str = "bibcode",
    ):
        """
        Parameters:
        --- dataframe_or_records: pandas DataFrame or list of dicts with papers
        --- topic_to_idx: dict mapping UAT label identifier to index in [0, num_labels - 1]
        --- tokenizer: HuggingFace tokenizer (instantiated if None)
        --- max_length: maximum token sequence length (512 for SciBERT)
        --- title_col: column name for paper title
        --- abstract_col: column name for paper abstract
        --- labels_col: column name for verified UAT IDs
        --- bibcode_col: column name for NASA ADS bibcode identifier
        """
        if isinstance(dataframe_or_records, pd.DataFrame):
            self.df = dataframe_or_records.reset_index(drop=True)
        else:
            self.df = pd.DataFrame(dataframe_or_records).reset_index(drop=True)

        self.topic_to_idx = topic_to_idx
        self.num_labels = len(topic_to_idx)
        self.max_length = max_length

        self.title_col = title_col
        self.abstract_col = abstract_col
        self.labels_col = labels_col
        self.bibcode_col = bibcode_col

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(SCIBERT_MODEL_NAME)
        else:
            self.tokenizer = tokenizer

        # Pre-extract bibcodes for quick reference
        if self.bibcode_col in self.df.columns:
            self.bibcodes = self.df[self.bibcode_col].tolist()
        else:
            self.bibcodes = [f"doc_{i}" for i in range(len(self.df))]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        # 1. Format and clean text: Title + "\n" + Abstract
        raw_title = row.get(self.title_col, "")
        raw_abstract = row.get(self.abstract_col, "")

        clean_title = clean_astronomy_text(raw_title)
        clean_abs = clean_astronomy_text(raw_abstract)

        if clean_title and clean_abs:
            full_text = f"{clean_title}\n{clean_abs}"
        elif clean_title:
            full_text = clean_title
        else:
            full_text = clean_abs

        # 2. Tokenize with SciBERT tokenizer
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # 3. Create multi-hot label vector of shape [num_labels]
        label_tensor = torch.zeros(self.num_labels, dtype=torch.float32)
        raw_labels = row.get(self.labels_col, [])

        # FIX: The HuggingFace dataset stores verified_uat_ids as numpy.ndarray in the DataFrame.
        # Python's isinstance(np.ndarray, (list, tuple, set)) returned False, which previously
        # caused all labels to be skipped and generated all-zero target vectors.
        # We now check for any iterable sequence (excluding str/bytes) so that numpy.ndarray,
        # lists, tuples, and sets are all correctly recognized and parsed.
        if raw_labels is not None and hasattr(raw_labels, "__iter__") and not isinstance(raw_labels, (str, bytes)):
            for label in raw_labels:
                # Labels may be numpy integers (int64/int32), Python ints, or string IDs.
                # Cast to standard int first to match topic_to_idx keys loaded from label_order.json.
                try:
                    int_label = int(label)
                    if int_label in self.topic_to_idx:
                        label_tensor[self.topic_to_idx[int_label]] = 1.0
                        continue
                except (ValueError, TypeError):
                    pass

                # Fallback for string or raw representations:
                if label in self.topic_to_idx:
                    label_tensor[self.topic_to_idx[label]] = 1.0
                elif str(label) in self.topic_to_idx:
                    label_tensor[self.topic_to_idx[str(label)]] = 1.0

        return input_ids, attention_mask, label_tensor

    def get_bibcode(self, idx: int) -> str:
        """Returns the bibcode corresponding to sample index idx."""
        return self.bibcodes[idx]


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    topic_to_idx: Optional[Dict[Any, int]] = None,
    batch_size: int = 8,
    eval_batch_size: int = 16,
    max_length: int = DEFAULT_MAX_LENGTH,
    tokenizer: Optional[AutoTokenizer] = None,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Convenience factory to instantiate DataLoaders for all splits.

    Parameters:
    --- train_df: training DataFrame
    --- val_df: validation DataFrame
    --- test_df: optional test DataFrame
    --- topic_to_idx: topic-to-index mapping dictionary (inferred if None)
    --- batch_size: batch size for training DataLoader (default: 8)
    --- eval_batch_size: batch size for eval/test DataLoaders (default: 16)
    --- max_length: maximum token sequence length (default: 512)
    --- tokenizer: instantiated tokenizer (loaded if None)
    --- num_workers: DataLoader worker count

    Returns:
    --- dict with keys 'train', 'validation', and optionally 'test'
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(SCIBERT_MODEL_NAME)

    if topic_to_idx is None:
        # Extract unique topics across splits, casting numeric elements to standard int
        all_topics = set()
        for df_split in (train_df, val_df, test_df):
            if df_split is not None and "verified_uat_ids" in df_split:
                for labels in df_split["verified_uat_ids"]:
                    if labels is not None and hasattr(labels, "__iter__") and not isinstance(labels, (str, bytes)):
                        for l in labels:
                            try:
                                all_topics.add(int(l))
                            except (ValueError, TypeError):
                                all_topics.add(l)
        topic_to_idx, _ = build_topic_mapping(all_topics)

    train_dataset = SciXDataset(
        train_df, topic_to_idx=topic_to_idx, tokenizer=tokenizer, max_length=max_length
    )
    val_dataset = SciXDataset(
        val_df, topic_to_idx=topic_to_idx, tokenizer=tokenizer, max_length=max_length
    )

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
        ),
        "validation": DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
    }

    if test_df is not None:
        test_dataset = SciXDataset(
            test_df, topic_to_idx=topic_to_idx, tokenizer=tokenizer, max_length=max_length
        )
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

    return loaders
