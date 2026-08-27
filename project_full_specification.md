# Project Specification — Long-Tail Topic Classification of Astronomy Papers

*A complete, self-contained description of the project — its research question, data, models, methodology, decisions, and constraints.*

---

## 1. One-paragraph summary

We study **multi-label topic classification of astronomy papers**: given a paper's title + abstract, predict which topics (from a large controlled vocabulary) it belongs to. The vocabulary is heavily **long-tailed** — a few topics are very common, most are rare. Our central question is a **model comparison**: does a large, quantized *generative* LLM (Gemma, fine-tuned with QLoRA) do better than a small, specialized *encoder* (SciBERT) at this task — especially on the rare, long-tail topics — and is the large model's extra compute worth it? We answer by measuring each model's performance **broken down by label frequency** (common / medium / rare), not just as an overall average. This is a course project: the deliverable is a white-paper / proof-of-concept, not a publication.

---

## 2. Research question (precise)

**Primary:** Can parameter-efficient fine-tuning (LoRA / QLoRA) let a large generative model (Gemma) match or beat a small specialized encoder (SciBERT) at multi-label topic classification of astronomy papers?

**Sub-questions:**
- Specifically on the **rare, long-tail topics** (where models tend to struggle most), which approach does better?
- Is the large model's **extra compute** (time, memory, parameters) justified by any accuracy gain?

**Framing:** the question is informative either way. If a long-tail performance gap exists, we characterize it and test whether each method narrows it. If the gap is unexpectedly small, that itself is a reportable finding. Both positive and negative results are acceptable (the course explicitly values negative results when the methodology is sound).

---

## 3. Motivation — why this is worth doing, and why astronomy specifically

Long-tail multi-label classification and parameter-efficient fine-tuning are well studied in general NLP. So the project must justify why astronomy is a distinct, worthwhile setting rather than a solved, generic problem. Two reasons:

1. **Real deployment stakes on an expert-built vocabulary.** The labels come from the Unified Astronomy Thesaurus (UAT), a formal thesaurus maintained by astronomers and used in practice by NASA's SciX/ADS to index the astronomy literature. If a model systematically under-tags rare topics, genuinely niche research becomes harder to discover — a concrete downstream harm that generic sentiment or news benchmarks do not carry.

2. **The domain gap is unusually large.** Astronomy abstracts are dense with jargon, object designations, and formula-like notation. This makes the "does a general generative LLM match a domain-pretrained encoder?" question *more* informative here, because domain specialization is exactly what stresses the models.

---

## 4. What is already known, and the exact gap we fill

**Established in prior work:**
- Domain-pretrained encoders (SciBERT) are strong on scientific text.
- LoRA / QLoRA make fine-tuning cheap and feasible on small hardware.
- Long-tail degradation is common in extreme multi-label classification — rare labels are where strong models collapse (Chang et al., X-Transformer, KDD 2020).
- Balancing methods help the tail but do not fully solve it (Huang et al., EMNLP 2021).
- Per-frequency-band (head/medium/tail) evaluation is itself an established technique (Huang et al., EMNLP 2021), including on a specialized scientific vocabulary (they use PubMed with ~18k MeSH medical labels). **So the frequency-band breakdown is NOT our contribution.**

**The gap we fill (our actual contribution):** a controlled comparison of a small domain-pretrained encoder (SciBERT) against a quantized generative LLM (Gemma + QLoRA), resolved across the label-frequency spectrum, on the astronomy UAT. A literature search (including a dedicated deep-search tool) found no single paper doing all four of: multi-label text classification + small-encoder-vs-large-generative-LLM + LoRA/QLoRA quantized fine-tuning + results split by label frequency. Near-miss papers each miss at least one of these four elements. Note: our generative model is **decoder-only** (Gemma), which distinguishes us from prior encoder-vs-generative work that used encoder-decoder models like T5.

---

## 5. Dataset (full detail)

- **Name / source:** `adsabs/SciX_UAT_keywords` on HuggingFace, maintained by the official NASA ADS / Science Explorer (SciX) team.
- **License:** MIT (fully open, no access gating).
- **Size:** ~21,702 papers total; ~19 MB.
- **Provided splits:** ~18,677 train, ~3,025 validation. **There is no dedicated test split.**
- **Fields per example:**
  - `title` — paper title (text)
  - `abstract` — paper abstract (text; may contain HTML-like tags such as `<SUB>` that need cleaning)
  - `verified_uat_labels` — list of topic keyword strings, e.g. `["solar wind", "exosphere", "the moon"]`
  - `verified_uat_ids` — the same labels as integer IDs (the dataset authors recommend working with IDs, since text labels can have synonyms)
- **Label space:** the Unified Astronomy Thesaurus (UAT), ~2,300 possible concepts.
- **Labels per paper:** variable — roughly 1 to 12 in observed examples (this is what makes it multi-label).

---

## 6. Key design decisions (locked)

These must be fixed once and shared across all models so the comparison is valid.

### 6.1 Train / validation / test split
- The dataset provides train + validation only; there is no dedicated test split.
- The provided validation split contains 805 of the 1,864 labels; the remaining 1,059 labels do not appear in it.
- The provided validation split is not a random sample: exactly one label has 3 instances in it, no label has 1, 2 or 4, and 187 labels have exactly 5.
- **We merge the provided train + validation splits and re-split 70/10/20 using iterative stratification** (`MultilabelStratifiedShuffleSplit`, `random_state=42`).

### 6.2 Minimum-frequency label filter (k)

- We keep only labels that appear at least **k** times in the **training split**, and drop labels below that. This is standard "minimum-support" practice in extreme multi-label classification.
- **k is not yet committed to a number.** It should be chosen from the data. Candidate methods:
  1. **Fixed absolute cutoff** (e.g. k = 5, 10, or 20) — simplest, easy to justify. **This includes k = 1, i.e. no filter at all.**
     *Check to run:* for each candidate k, report how many labels are retained and how many training documents are left with no labels and therefore dropped.
  2. **Evaluation-reliability cutoff** — set k so every kept label has enough *validation-set* instances (e.g. ≥ 2–3) to compute a meaningful F1. **The check runs on the carved validation split, never on the test split**; test-set label support may be *reported* as a dataset statistic, but never *selected* on.
     *Check to run:* for each candidate k, report the validation-split support of every retained label (minimum, and how many fall below the required threshold).
  3. **Frequency-curve "elbow"** — plot sorted label frequencies, cut where the curve flattens into the sparse tail.
     *Check to run:* plot the sorted label-frequency curve and inspect whether a clear elbow exists at all.
  4. **Coverage-based** — keep labels covering ~95% of all label occurrences.
     *Check to run:* for each candidate k, report the share of training label-occurrences retained.
  5. **Sensitivity analysis** — train at several k values (e.g. 5, 10, 20) using only the cheap TF-IDF + Logistic Regression baseline (§7.1), so the sweep costs almost nothing, then compare the results and decide which k to adopt for all models.
     *Check to run:* confirm whether the per-band conclusions are stable across candidate k values, or whether they depend on the threshold.
  6. **Follow prior work** — see how the papers we cite handled the same threshold problem: whether they applied a minimum-support filter, at what value, and on what justification. Adopting or adapting an established rule is easier to defend in the report than inventing one.
     *Check to run (not done yet):* read each of the following and record whether it filters its data, on what criterion, and how it justifies it —
     - Huang et al., EMNLP 2021 (`Articles/`)
     - Chang et al., KDD 2020 (`Articles/`)
     - SciBERT — Beltagy, Lo & Cohan, EMNLP 2019 (`Articles/`)
     - LoRA — Hu et al., ICLR 2022 (`Articles/`)
     - QLoRA — Dettmers et al., NeurIPS 2023 (`Articles/`)
     - the `adsabs/SciX_UAT_keywords` dataset card / SciX documentation
     - the near-miss papers referred to in §11 (not yet named there — identify them first)

- **Suggested approach (to be confirmed, not decided):** combine (2) and (3) — use the frequency plot to see the shape, set k by evaluation-reliability *on the validation split*, and add a sensitivity check.
- Whatever k is chosen, record for the limitations section how many label *types* it removes, so the report is explicit that the study measures the tail it retains, not the full UAT tail.
- IMPORTANT: the minimum-frequency filter (removing ultra-rare labels) is a SEPARATE decision from the frequency-band grouping below. Do not conflate them.

### 6.3 Frequency bands for evaluation (head / medium / tail)
- After filtering, we group the retained labels into three bands by frequency for per-band evaluation.
- **Method: tertiles** — sort labels by frequency and split into three equal-sized groups (this follows Huang et al., EMNLP 2021, who split labels into equal-sized head/medium/tail groups; the exact instance-count boundaries then fall out of the data per-dataset).
- Rationale for three bands (not 2 or 4+): three is the smallest number that shows a *trend with a middle* (head → medium → tail) while keeping enough labels/test-data per band for stable metrics. More bands = finer detail but noisier per-band estimates; fewer = stabler but coarser.
- Optional enhancement: also plot performance as a continuous curve (F1 vs. label frequency) alongside the three-band table, if the dataset is large enough per band.

---

## 7. Models to compare

All models are trained and evaluated under an **identical protocol** (same data, same splits, same k, same bands, same evaluation harness) so differences are attributable to the model, not to setup.

### 7.1 Baselines (to make results meaningful — the "compared to what?")
- **Majority / frequency baseline:** ignore the text; always predict the most common labels. Sets the floor.
- **TF-IDF + Logistic Regression:** classical features (word-frequency vectors) + a simple one-vs-rest linear classifier. No neural network, no GPU. A real reference point the neural models must beat.

### 7.2 Model arm A — SciBERT, full fine-tuning
- Base model: `allenai/scibert_scivocab_uncased` (~110M parameters; BERT-base pretrained on scientific text).
- Add a **multi-label classification head**: one output score per retained label, using **sigmoid activation + binary cross-entropy (BCE) loss** (multi-label, so labels are independent — NOT softmax).
- Fine-tune **all** weights.

### 7.3 Model arm B — SciBERT, LoRA
- Same SciBERT base and same multi-label head.
- **Freeze the base model; train only LoRA adapters** (via HuggingFace `peft`). LoRA injects small low-rank matrices (typically into the attention query/value projections) and trains only those — well under 1% of parameters.
- Keep everything else identical to arm A so the full-vs-LoRA comparison is clean.

### 7.4 Model arm C — Gemma, QLoRA (the ambitious arm)
- Base model: a small **Gemma** (start with 1B, scale to 4B if resources allow). Gemma is a **decoder-only generative** model.
- **Load in 4-bit quantization** (via `bitsandbytes`) and fine-tune with **LoRA adapters** (via `peft`) — together this is **QLoRA**.
- This model is **generative**: it produces the labels as **text output**, which must then be **parsed** back into a clean multi-label prediction (e.g. matching generated strings/IDs to the label set). This parsing step is unique to this arm and needs care.
- Note on Gemma version: Gemma 3 (4B or 1B) is the recommended workhorse for stability/tooling maturity; Gemma 4 exists (released 2026) but is newer with thinner tooling.

---

## 8. Evaluation methodology (do not omit any part)

- **Metrics:** precision, recall, and F1, computed **per frequency band** (head / medium / tail) as well as overall (report macro-averaged F1 across labels, since micro would be dominated by common labels).
- **The core output:** a table (and/or plot) showing, for every model (2 baselines + SciBERT-full + SciBERT-LoRA + Gemma-QLoRA), the per-band precision/recall/F1. This is where the long-tail behavior and the model comparison become visible.
- **Compute cost reporting:** for each model, record trainable parameters and GPU-hours (and peak memory). This makes the small-vs-large / cheap-vs-expensive trade-off explicit — central to the research question.
- **Sanity check (fallback):** if a model can't get meaningful results, confirm it can **overfit a small subset** of the data. This proves the pipeline is bug-free rather than mis-designed. (This mirrors the course's own methodology requirement.)
- **Control for randomness:** use fixed seeds; where feasible run multiple seeds and report variation. Account for any decoding randomness in the generative (Gemma) arm.
- **Avoid data leakage:** strict train/val/test separation; the label filter and band definitions are computed from training data only, then applied to val/test.
- **Reproducibility:** log exact settings (model versions, hyperparameters, seeds); keep code in the shared repo.

---

## 9. Technical stack and resources

- **Libraries:** HuggingFace `transformers` and `datasets`; `peft` (LoRA); `bitsandbytes` (4-bit quantization); `scikit-learn` (TF-IDF + Logistic Regression baseline); standard `torch`.
- **Compute:** all models are small (SciBERT ~110M; Gemma 4B in 4-bit ~3 GB). Free Google Colab (T4 GPU, ~16 GB) suffices; the TAU Slurm `studentkillable` partition is available for the Gemma arm and repeated runs.
- **Storage:** minimal; avoid frequent checkpointing (do not save a new checkpoint every few hundred steps — it fills shared storage). Back up code + critical results on GitHub/Drive (Slurm storage is not backed up).
- **No external API credits or special hardware required.**

---

## 10. Scope, constraints, and Plan B

- **Scope:** this is a course project — a white-paper proof-of-concept, completable in a few days of focused work, NOT a conference paper. Avoid heavy engineering, large-scale human annotation, or huge compute.
- **Plan B:** the SciBERT-vs-Gemma comparison is the core contribution, so if the full Gemma + QLoRA arm proves too costly in time, **scale it down rather than drop it** — a smaller Gemma (1B instead of 4B) or fewer runs — keeping the comparison intact but cheaper.

---

## 11. Reference papers

**Anchor papers (directly used; from approved NLP/ML venues, post-2018):**
1. **SciBERT** — Beltagy, Lo & Cohan, EMNLP 2019 (the base encoder).
2. **LoRA** — Hu et al., ICLR 2022 (parameter-efficient fine-tuning).
3. **QLoRA** — Dettmers et al., NeurIPS 2023 (4-bit quantized fine-tuning; used for the Gemma arm). *(Verify venue/year before final citation.)*

**Related-work citations (context / prior art):**
- **Huang et al., EMNLP 2021** — "Balancing Methods for Multi-label Text Classification with Long-Tailed Class Distribution." Balancing-loss methods; head/medium/tail evaluation (via tertiles); tested on Reuters (news) and PubMed (biomedical/MeSH). Verified from the actual paper.
- **Chang et al., KDD 2020** — "Taming Pretrained Transformers for Extreme Multi-label Text Classification" (X-Transformer). Foundational transformer XMC; documents that the vast majority of labels are long-tail. Verified from the actual paper.
- Additional near-miss papers exist (found via literature search) that each cover *some* of our four elements but not all four; these are leads to verify and cite in the final literature review, not confirmed here.

---

## 12. Notes / open items

- **k value: NOT YET CHOSEN** (see §6.2). To be determined from the data by the checks listed there. Any figures quoted for a specific k in earlier drafts were derived by counting on the merged corpus and must be recomputed on the training portion of the §6.1 re-split.
- **Gemma version/size:** decide between Gemma 3 1B vs 4B based on resource/time budget (start small).
- **Citations to verify** before the final report: QLoRA (Dettmers et al., NeurIPS 2023) venue/year, and any near-miss papers added to related work.
- **AI Disclosure & Reflection section** is required in the FINAL report (not the proposal) — plan to write it.
- **Final report format:** ACL template (LaTeX/Overleaf), up to 8 pages excluding references/appendix.
