# Does AI Neglect Rare Science?

### Long-Tail Topic Classification of Astronomy Papers with Small Encoders vs. Quantized Generative LLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20%7C%20CUDA-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete experimental pipeline, models, evaluation protocols, and deliverables for benchmarking long-tail multi-label concept classification on astrophysics papers from the [Unified Astronomy Thesaurus (UAT)](https://astrothesaurus.org/).

We evaluate six systems across 1,864 UAT concepts stratified into frequency bands (**Head**, **Torso**, and **Tail**):
1. **Majority Frequency Prior** (frequency-based empirical baseline)
2. **TF-IDF + One-vs-Rest Logistic Regression** (sparse n-gram baseline)
3. **SciBERT-alone** (full transformer backbone fine-tuning)
4. **SciBERT-LoRA** (parameter-efficient adapter fine-tuning)
5. **Gemma 3 4B Untuned** (top-50 shortlist generative extraction)
6. **Gemma 3 4B Fine-Tuned / QLoRA** (shortlist-conditioned generative re-ranker)

---

## Quickstart

### 1. Setup Virtual Environment

Requires Python 3.10+.

```bash
# Clone the repository
git clone <repo-url>
cd NLP

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows
```

### 2. Install Dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

### 3. Download Dataset (One-Time)

Fetch the raw NASA ADS dataset ([adsabs/SciX_UAT_keywords](https://huggingface.co/datasets/adsabs/SciX_UAT_keywords)) into `.hf_cache/` (~19 MB):

```bash
python scripts/download_data.py
```

The dataset is partitioned into the canonical stratified split (`data/split.json`):
* **Train**: 15,822 papers
* **Validation**: 2,855 papers
* **Test**: 3,025 papers
* **Label Space**: 1,864 concepts stratified by training frequency:
  * **Head**: $>500$ occurrences (12 concepts, 1,353 test mentions)
  * **Torso**: $50 - 500$ occurrences (383 concepts, 9,269 test mentions)
  * **Tail**: $<50$ occurrences (1,469 concepts, 2,617 test mentions)

---

## Reproducing Results

### Reproduce Table 1 (All 6 Systems in ~15 Seconds)

All validation and test probability tables (§6 Parquet format) and Gemma generation picks are tracked in `results/`. Anyone can reproduce all results in Table 1 on CPU in seconds without retraining or needing a GPU:

```bash
python scripts/main.py --arm rescore_all
```

#### Canonical Table 1 Results Summary:

| System | Head (Micro-F1) | Torso (Micro-F1) | Tail (Micro-F1) | Overall (Micro-F1) | Predictions / Paper |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Majority prior** | 0.0914 | 0.0385 | 0.0055 | 0.0283 | 37.00 |
| **TF-IDF + LR** | 0.0719 | 0.0159 | 0.0014 | 0.0135 | 531.85 |
| **SciBERT-alone (tuned)** | 0.5518 | 0.3566 | 0.2150 | 0.3435 | 5.48 |
| **SciBERT-LoRA (tuned)** | 0.4764 | 0.2636 | 0.1190 | 0.2501 | 7.20 |
| **Gemma, untuned** | 0.5530 | 0.3703 | 0.2943 | 0.3778 | 8.79 |
| &nbsp;&nbsp;↳ shuffled candidate order | 0.4631 | 0.2906 | 0.2272 | 0.2908 | 8.94 |
| **Gemma, fine-tuned (QLoRA)** | 0.5690 | 0.3783 | 0.2897 | 0.3781 | 5.68 |
| &nbsp;&nbsp;↳ shuffled candidate order | 0.5661 | 0.3702 | 0.2794 | 0.3698 | 5.65 |

The two *shuffled* rows re-score the same two arms on shuffled candidate lists. Only the
fine-tuned arm keeps its gains, which is the paper's second headline result.

---

## Running Individual Arms & CLI Options

The main orchestrator (`scripts/main.py`) supports running or re-scoring specific arms:

```bash
# Re-score or run individual systems
python scripts/main.py --arm majority           # Train or rescore Majority baseline
python scripts/main.py --arm tfidf              # Train or rescore TF-IDF + LR
python scripts/main.py --arm scibert_full       # Rescore SciBERT full from disk
python scripts/main.py --arm scibert_lora       # Rescore SciBERT-LoRA from disk
python scripts/main.py --arm gemma              # Rescore untuned and fine-tuned Gemma

# Force re-scoring from §6 Parquet tables
python scripts/main.py --arm majority --rescore
python scripts/main.py --arm tfidf --rescore

# By default, rescoring runs in read-only observation mode without modifying files.
# To explicitly overwrite/update on-disk JSON results and predictions:
python scripts/main.py --arm rescore_all --save-results

# Run Exploratory Data Analysis (saves figures to results/eda_plots/)
python scripts/main.py --eda

# Set custom random seed (default: 42)
python scripts/main.py --seed 42
```

---

## Diagnostics & Sanity Checks

### 1. Real Generator Candidate Recall Ceiling
Evaluates candidate pool quality (Coverage@K) on the actual `scibert_full` and `scibert_lora` generators across depths $K \in \{10, 20, 25, 50, 100\}$ on validation and test sets:

```bash
python scripts/analysis/real_recall_ceiling.py
```
*Outputs: `scripts/analysis/real_recall_ceiling.out` and `scripts/analysis/real_recall_ceiling.json`.*

### 2. LoRA Overfit-a-Small-Subset Sanity Check
Verifies PEFT gradient backpropagation, optimizer dynamics, and representational capacity by overfitting 32 samples across 1,864 classes on MPS/CPU:

```bash
python scripts/scibert_lora_overfit_check.py
```
*Outputs: `results/scibert_lora/scibert_lora_overfit_sanity_check.png` and `results/scibert_lora/OVERFIT_SANITY_CHECK.md`.*

### 3. Standalone Exploratory Data Analysis
```bash
python scripts/EDA_Analysis.py --save-dir results/eda_plots
```

---

## Project Structure

```
NLP/
├── data/
│   ├── build_split.py              # Stratified train/val/test data loader
│   ├── split.json                  # Canonical split indices (15,822 / 2,855 / 3,025)
│   ├── label_order.json            # 1,864 UAT concepts in fixed column order
│   └── band_map_train.json         # UAT concept-to-frequency-band mapping (head/torso/tail)
├── scripts/
│   ├── main.py                     # Primary pipeline runner and Table 1 generator
│   ├── baselines.py                # Majority and TF-IDF+LR baselines & rescoring
│   ├── SciBERT_experiment.py       # SciBERT training and inference engine
│   ├── scibert_dataset.py          # PyTorch dataset & astronomy text cleaning
│   ├── evaluate.py                 # Threshold sweep (choose_tau) and test evaluator
│   ├── score_gemma.py              # Gemma candidate selection scorer
│   ├── scibert_lora_overfit_check.py # LoRA overfit sanity check
│   ├── export_scibert_lora_results.py # LoRA §6 Parquet exporter
│   └── analysis/
│       ├── real_recall_ceiling.py  # Empirical generator recall ceiling analysis
│       ├── make_paper_figures.py   # Publication vector PDF figure generator
│       └── ...
├── results/
│   ├── majority_baseline/          # cost.json, tau_majority.json, test_results_majority.json
│   ├── tfidf_logistic_regression/  # cost.json, tau_tfidf_lr.json, test_results_tfidf_lr.json
│   ├── scibert_full/               # §6 Parquets, val_epochs/, cost.json, overfit check
│   ├── scibert_lora/               # §6 Parquets, val_epochs/, cost.json, overfit check, run/
│   └── gemma_select/               # Model picks JSONL and scored test results
└── requirements.txt                # Python dependencies
```

---

## License
MIT License. Unified Astronomy Thesaurus concepts and metadata are subject to the [UAT terms of use](https://astrothesaurus.org/).
