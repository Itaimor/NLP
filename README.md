# NLP

Does AI Neglect Rare Science? Long-Tail Topic Classification of Astronomy Papers with Small Encoders vs. Quantized Generative LLMs

## Setup

Requires Python 3.10+.

### 1. Clone and enter the repo

```bash
git clone <repo-url>
cd NLP
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

### 4. Download the dataset

```bash
python scripts/download_data.py
```

This downloads [adsabs/SciX_UAT_keywords](https://huggingface.co/datasets/adsabs/SciX_UAT_keywords) into `.hf_cache/` inside the project (~19 MB). The folder is gitignored, so each teammate downloads it locally once.

Expected output:

- **train:** 18,677 rows
- **val:** 3,025 rows
- columns: `bibcode`, `title`, `abstract`, `verified_uat_ids`, `verified_uat_labels`

Re-running the script is safe. If the data is already cached, it completes in a few seconds.

### 5. Verify in Python (optional)

```bash
python
```

```python
from datasets import load_dataset
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(".").resolve() / ".hf_cache"))
ds = load_dataset("adsabs/SciX_UAT_keywords")
print(ds)
```

## Project layout

```
NLP/
├── scripts/
│   └── download_data.py   # fetch dataset into .hf_cache/
├── requirements.txt       # Python dependencies
├── .venv/                 # local virtualenv (gitignored)
└── .hf_cache/             # Hugging Face dataset cache (gitignored)
```

## Dataset

- **Source:** https://huggingface.co/datasets/adsabs/SciX_UAT_keywords
- **License:** MIT
- **Task:** multi-label UAT keyword classification from paper title + abstract

Use `verified_uat_ids` (not label strings) for training and evaluation.
