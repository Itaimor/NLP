#!/usr/bin/env python3
"""Download adsabs/SciX_UAT_keywords into the project Hugging Face cache."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))

from datasets import load_dataset  # noqa: E402

DATASET_ID = "adsabs/SciX_UAT_keywords"


def main() -> int:
    cache_dir = Path(os.environ["HF_HOME"])
    print(f"Downloading {DATASET_ID}")
    print(f"Cache directory: {cache_dir}")

    ds = load_dataset(DATASET_ID)

    print("\nDownload complete.")
    print(ds)
    for split in ds:
        print(f"  {split}: {len(ds[split]):,} rows")

    example = ds["train"][0]
    print("\nSample row:")
    print(f"  bibcode: {example['bibcode']}")
    print(f"  title: {example['title'][:80]}...")
    print(f"  verified_uat_ids: {example['verified_uat_ids']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
