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

**Concurrent work on this exact corpus.** Alkan et al. (arXiv:2604.02156, April 2026 — an unrefereed preprint by the dataset's own authors at NASA ADS) introduce this corpus as *AstroConcepts*. It is a **corpus / resource paper**: its five stated contributions are the corpus itself, a systematic baseline comparison, a frequency-stratified evaluation framework, the finding that domain adaptation helps rare terminology most, and baseline establishment. It benchmarks four method families — rule-based lexical matching, k-NN over embeddings, fine-tuned encoders (BERT / SciBERT / astroBERT), and a vocabulary-constrained LLM — with results split by frequency band. That covers three of the four elements we originally claimed as novel, on our data, with our encoder. Full text in `Articles/AstroConcepts - Alkan et al., arXiv 2026.md`. **We cite it as concurrent work** — a preprint posted in April, not a published paper.

**Their LLM arm is a two-stage reranker, not free-form generation.** This matters for how we compare against it. Quoting §4.2.3: *"(1) use our best supervised model (astroBERT ...) to generate top-50 candidate labels for each abstract, (2) prompt DeepSeek-V3-reasoner via its API to select relevant concepts from these candidates."* They chose top-50 because astroBERT's top-50 covers ~82% of ground-truth labels, and they report that prompting over the full 2,367-concept vocabulary failed outright: *"Preliminary experiments on a small subset yielded very poor results when using the complete label list, as the model hallucinated concepts or became overwhelmed by the extensive vocabulary."* So their DeepSeek row is a **system** score — a fine-tuned astroBERT plus an LLM selector — and is conditional on the quality of that candidate generator.

**The gap that remains (our actual contribution), in two parts:**

1. **A generative model with updated weights — which no arm of theirs has.** Every generative component in their work is used *without* weight updates: DeepSeek-V3-reasoner is called through an API as a selector. No open-weight generative model is fine-tuned anywhere in the paper, and parameter-efficient or quantized fine-tuning is not attempted. That is the cell we fill. Our question, stated precisely: **an API-scale LLM selecting from astroBERT's candidates reaches 0.198 on the tail against astroBERT's own 0.081 — does that advantage survive when the selector is a 1-4B open-weight model, tuned with QLoRA on a single GPU?** Reference points: SciBERT 0.023, astroBERT 0.081, DeepSeek 0.198 (tail F1, Table 6).

   **A caution on evidence.** The sentence *"Due to computational resource constraints, we were unable to include Qwen models in the fine-tuning experiments"* appears in §4.2.2 on **supervised encoders**, and "Qwen" there refers to Qwen3-Embedding-8B, an *embedding* model used in their k-NN arm. **It is not a statement about generative PEFT and must not be quoted as one.** The gap is real — they simply never fine-tune a generative model — but the argument rests on what the paper does, not on that sentence.

   **The ablation they explicitly name as missing.** Their limitations section states that evaluating DeepSeek *"over candidate sets generated by BERT and SciBERT"* would *"isolate the contribution of the candidate generator's quality from the LLM's own re-ranking ability."* Our arm can answer exactly that, and framing it this way turns a confound into the contribution (see §7.4).

2. **An evaluation finding that costs no compute.** Two facts about this benchmark, both verified against the data, and both unreported by its authors:

   **(i) Three quarters of the tail band cannot be scored.** Of the 1,418 concepts in the tail, only **359** have any test instance; the other **1,059 (74.7%)** are held entirely in training by the corpus's own split rule. Any per-label average that includes them is divided by a denominator 1,418/359 = **3.95x** larger than the number of concepts that can actually contribute.

   **Stated carefully — this is a reporting problem, not proof that their numbers are wrong.** The paper says it uses *"Macro-F1, which averages per-label F1 scores"* but never states the per-band denominator, and their Table 5 (overall) cannot be reconciled with their Table 6 (per-band) under either label-weighting (gives 0.135 for astroBERT vs the 0.324 reported) or instance-weighting (0.254), so the two tables demonstrably use different conventions. **The honest claim is therefore: the published tail numbers cannot be interpreted without a denominator the paper does not give — and depending on which convention was used, they are either deflated ~4x or not deflated at all.** We must not assert the 4x as fact. Our own reporting fixes this by giving both denominators explicitly (§8).

   **(ii) The decision threshold moves tail F1 by up to 24.6x** (measured on our own baseline, §8), and no reference number in this literature reports the threshold used. Their encoder rows are thresholded; their LLM row emits a free set of 1-10 labels with no threshold at all, so their headline head-to-tail gap is partly a decision-rule artifact.

   Together: **band-wise long-tail comparisons in this literature are not currently interpretable**, and demonstrating that is a contribution in its own right — one the course guidelines explicitly invite: *"ways to improve the evaluation of existing methods, identifying problems that LLMs struggle with."*

---

## 5. Dataset (full detail)

- **Name / source:** `adsabs/SciX_UAT_keywords` on HuggingFace, maintained by the official NASA ADS / Science Explorer (SciX) team.
- **License:** MIT per the HuggingFace dataset card (not stated in the paper itself); fully open, no access gating.
- **Size:** ~21,702 papers total; ~19 MB.
- **Provided splits:** 18,677 train, 3,025 labelled `validation` on HuggingFace. **That second split is in fact the published benchmark TEST split of Alkan et al. (2026)** — the numbers match exactly, and their construction rule is quoted in §6.1. We therefore treat it as the test set and never train or tune on it.
- **Fields per example:**
  - `title` — paper title (text)
  - `abstract` — paper abstract (text; may contain HTML-like tags such as `<SUB>` that need cleaning)
  - `verified_uat_labels` — list of topic keyword strings, e.g. `["solar wind", "exosphere", "the moon"]`
  - `verified_uat_ids` — the same labels as integer IDs (the dataset authors recommend working with IDs, since text labels can have synonyms)
- **Label space:** the Unified Astronomy Thesaurus (UAT) defines ~2,367 concepts, of which **1,864 actually occur** in this corpus. Both numbers are correct and should be stated together to avoid an apparent contradiction; the label space we model is the 1,864 that occur.
- **Total label assignments:** 93,547 (paper, label) pairs across 21,702 papers — a mean of 4.31 labels per paper. Note that a count of "occurrences" is a count of pairs, not of papers.
- **Divergence from the paper, to state in the report:** Alkan et al. predict over *"the complete label space of 2,367 labels"*; we model the **1,864 that actually occur**. Since the 503 unused concepts have no instances anywhere, they cannot affect recall, but they do enlarge the denominator of any macro-average computed over the full vocabulary. Our numbers and theirs are therefore not automatically comparable at the macro level, and every macro-average must state its denominator (§8).
- **Labels per paper:** variable — roughly 1 to 12 in observed examples (this is what makes it multi-label).

---

## 6. Key design decisions (locked)

These must be fixed once and shared across all models so the comparison is valid.

### 6.1 Train / validation / test split

**Decision: adopt the shipped split verbatim — 18,677 train / 3,025 test. Do not merge and re-split.**

The shipped `validation` split is the published benchmark test split of Alkan et al. (2026), built with this rule, quoted from their **§4.1 "Data Partitioning"** (which describes it as *"label-aware stratification"*):

> *"Labels appearing >=15 times are stratified to achieve approximately 85/15 distribution per label, while labels with <15 occurrences are placed entirely in the training set to maximize training signal."*

Verified against the data: 935 concepts occur fewer than 15 times, and **exactly zero** of them appear in the shipped `validation` split. This also explains the apparent non-randomness we had noted (minimum support 3, a spike of 187 labels at exactly 5) — it is the <15 rule, not an anomaly.

**Why we do not re-split:**
- Re-splitting destroys the only route to comparing our numbers against published ones on identical data.
- It would take ~1,447 (paper, label) pairs away from the rarest concepts — the ones least able to spare them — and buy almost nothing: ~392 of the 935 rare concepts would still have no test instance, and ~433 more would have only one or two, where per-label F1 is a step function rather than a measurement.
- It removes an entire class of leakage bug: with the split given, there is no ordering to get wrong.

**Validation set.** The shipped split has only two parts, so we **carve ~15% of the training papers as a validation set** (~2,802 validation / ~15,875 train), used for tuning the decision threshold and any hyperparameters. Measured: this gives at least one validation instance for 97.2% of the 359 test-scoreable tail concepts, and ~1,407 tail positives to tune on. **We never tune on the 3,025 test papers.**

**Terminology warning.** HuggingFace labels the shipped 3,025-paper split `validation`, but we treat it as *test*. Everywhere else in this document, "validation" means the set carved from train.

### 6.2 Minimum-frequency label filter (k) — NOT USED

**Decision: no minimum-frequency filter. k is deleted, not chosen.** The label space is the 1,864 concepts that occur in the corpus.

Rationale: the shipped split already performs the minimum-support function upstream, by holding every concept with fewer than 15 occurrences entirely in training. Adding our own filter on top would remove concepts from a study whose subject is rare concepts, while duplicating a decision the corpus authors already made. Inheriting the label universe by citation is also easier to defend than inventing a threshold.

This supersedes the earlier plan to select k from the data; the six candidate selection methods previously listed here are no longer applicable.

### 6.3 Frequency bands for evaluation (head / medium / tail)
**Decision: absolute frequency cutoffs, matching Alkan et al. (2026) — NOT tertiles.**

| Band | Cutoff (total corpus occurrences) | Concepts | Assignments |
|---|---|---|---|
| Head | > 500 | 17 (0.9%) | 12,288 (13.1%) |
| Torso | 50-500 | 429 (23.0%) | 62,808 (67.1%) |
| Tail | < 50 | 1,418 (76.1%) | 18,451 (19.7%) |

**Why tertiles must not be used here.** Sorting concepts by frequency and cutting into three equal groups (Huang et al., EMNLP 2021) puts the bottom third at train-frequencies **1 to 6**. But the lowest train-frequency of any concept appearing in the test split is **13**. Every concept in a tertile-tail would therefore be absent from the test set by construction, and the band would contain **zero test instances** — producing a tail column of 0.000 for every model that would look like a dramatic finding but is an empty set. Huang et al. could use tertiles because their concepts averaged ~150 examples; ours average 50.2 and our bottom third averages 2.85. The method does not transfer.

Using the published cutoffs also keeps our per-band numbers directly comparable to theirs.

**Band assignment basis — the paper is ambiguous here, so we must be explicit.** Their Table 3 reports these counts over the *whole corpus*, while their Table 6 caption describes the thresholds as *"training examples."* The two bases differ because of the 85/15 split. **We assign bands once from training-split frequencies and never recompute**, and we report the corpus-frequency assignment alongside, so a reader can match either convention. The counts above (17/429/1,418) are corpus-wide, matching their Table 3. Note also that with bands cut on corpus frequency but models trained on train frequency, the two overlap slightly at the edges (torso concepts reach down to 39 training examples, tail concepts up to 44) — one sentence in the limitations.

**Their vocabulary is inconsistent between sections:** the abstract and Table 3 say 76% of concepts have fewer than 50 examples; §5.2 says 78%. Cite the Table 3 figure (76.1%) and do not repeat the other without checking.

**Scoreability.** Of the 1,418 tail concepts, only **359** have at least one test instance; **1,059 (74.7%)** cannot be scored at all. This is a property of the corpus, not of our models, and it must be reported (see §8).

---

## 7. Models to compare

All models are trained and evaluated under an **identical protocol** (same data, same splits, same bands, same decision-threshold procedure, same evaluation harness) so differences are attributable to the model, not to setup.

**Honest scope of "identical."** The four scoring arms emit a probability per label; the generative arm does not. Data, splits, bands and metric computation are identical across all arms, but the decision rule necessarily differs for the generative arm unless it is given comparable scores (see §7.4). That difference is reported, not hidden.

### 7.1 Baselines (to make results meaningful — the "compared to what?")
- **Majority / frequency baseline:** ignore the text; always predict the most common labels. Sets the floor.
- **TF-IDF + Logistic Regression:** classical features (word-frequency vectors) + a simple one-vs-rest linear classifier. No neural network, no GPU. A real reference point the neural models must beat.

### 7.2 Model arm A — SciBERT, full fine-tuning
- Base model: `allenai/scibert_scivocab_uncased` (~110M parameters; BERT-base pretrained on scientific text).
- Add a **multi-label classification head**: one output score per label in the 1,864-concept label space, using **sigmoid activation + binary cross-entropy (BCE) loss** (multi-label, so labels are independent — NOT softmax).
- Fine-tune **all** weights.

### 7.3 Model arm B — SciBERT, LoRA
- Same SciBERT base and same multi-label head.
- **Freeze the base model; train only LoRA adapters** (via HuggingFace `peft`). LoRA injects small low-rank matrices (typically into the attention query/value projections) and trains only those — well under 1% of parameters.
- Keep everything else identical to arm A so the full-vs-LoRA comparison is clean.

### 7.4 Model arm C — Gemma, QLoRA (the ambitious arm)
- Base model: a small **Gemma** (start with 1B, scale to 4B if resources allow). Gemma is a **decoder-only generative** model.
- **Load in 4-bit quantization** (via `bitsandbytes`) and fine-tune with **LoRA adapters** (via `peft`) — together this is **QLoRA**.
- This model is **generative**: it produces labels as **text output**, which must be mapped back to the label set. This is the hardest engineering in the project and is specified below rather than left as "needs care."

**Scoring, so the comparison stays fair.** Free-form generation yields a *set* of labels with no per-label score, so the tuned decision threshold used by every other arm cannot be applied to it — the operating point would be an artefact of decoding settings. To avoid that, we score candidates: for each test paper, take a **shortlist of candidate concepts** and have Gemma score each by **length-normalised teacher-forced log-probability**, reusing one cached prefill per paper. This yields a continuous, thresholdable score and reduces the work from 1,864 x 3,025 = 5.6M scorings to a feasible number. Scoring every label without a shared prefix cache is not feasible on our hardware.

**Use OUR OWN fine-tuned SciBERT as the candidate generator, at top-50 — not TF-IDF at top-100.** This is the single most important design choice in this arm, and it follows directly from how Alkan et al. built theirs. Their DeepSeek row is astroBERT-top-50 plus an LLM selector, with a measured **82% recall ceiling** at top-50. If we shortlist with a *different* model at a *different* depth, then a Gemma-vs-DeepSeek comparison confounds two things at once — the quality of the candidate generator and the quality of the selector — and any deficit we observe is unattributable. Matching the *architecture* of their pipeline (an encoder we fine-tuned, top-50) makes our arm an interpretable variant rather than an incomparable one.

**Measure the recall ceiling before committing.** Compute recall@50 and recall@100, per band, for both TF-IDF and our fine-tuned SciBERT, and report the chosen ceiling next to every result. If our generator's ceiling is materially below their 82%, Gemma's recall is capped before inference begins and no comparison to their 0.198 is meaningful. **This check is CPU-only and takes minutes — run it before any GPU work.**

**This design answers a question the paper itself poses.** Their limitations section states that evaluating the LLM stage *"over candidate sets generated by BERT and SciBERT"* would *"isolate the contribution of the candidate generator's quality from the LLM's own re-ranking ability."* By holding the candidate set fixed and varying only the selector — their API model versus our QLoRA-tuned open-weight model — we answer exactly that, and the confound becomes the contribution.

**Format failures are a reported result, not a confound.** Separately, run unconstrained generation on ~200 papers and report: out-of-vocabulary rate, normalisation-recoverable near-misses vs true hallucinations, duplicate rate, and mean predicted cardinality. This is the behaviour no prior work on this corpus has measured, because no prior work fine-tuned a generative model on it. We do **not** constrain decoding to the vocabulary: doing so would delete this finding while leaving the threshold asymmetry untouched.

**Sequence length.** Our measured token lengths for title + abstract plus target: p50 ~400, p95 ~500, p99 ~550, max ~780. Alkan et al. fine-tune their encoders at **max_length=512** and report that *"94% fit within 512 tokens"* — consistent with our measurement once the target tokens are excluded. **For the SciBERT arms, use 512 to match their setup**, so our SciBERT row is comparable to their published one. For the Gemma arm, where the target labels also occupy the sequence, use **640**. State both in the report.

**Model size is decided by measurement, not in advance.** Run a timed benchmark first (§9). If a projected epoch is <=4h and loss stays finite, train Gemma-3-4B. Otherwise train **Gemma-3-1B** and use the 4B model prompted-only — which is the direct analogue of the preprint's prompted arm, and makes the contrast *tuned-small vs prompted-large*, a defensible question in its own right.
- Note on Gemma version: Gemma 3 (1B or 4B) is the workhorse. A "Gemma 4" is referenced in 2026 secondary sources but we have not confirmed it against official documentation — check `ai.google.dev` before considering it.

**Their encoder settings, for matching our SciBERT arms** (§4.2.2): classification heads on the `[CLS]` representation, fine-tuned end-to-end, `max_length=512`, `batch_size=8`, AdamW; grid search over lr {2e-5, 3e-5, 5e-5} x epochs {1, 2, 3, 5, 8, 10}, with Table 5 reporting the best (lr=2e-5, epochs=8). Matching these makes our SciBERT row a genuine replication check rather than an unrelated number.

**A caution worth carrying into the discussion.** Their own conclusion attributes the LLM arm's advantage to *constraining the label space* rather than to model parameters — "domain expertise can be effectively incorporated through vocabulary constraints rather than solely through model parameters." Our project tests whether *tuning weights* adds anything on top of that constraint. That is a legitimate and unanswered question, but we should not assume the answer is yes; a null result here would be consistent with their finding and is still reportable.

---

## 8. Evaluation methodology (do not omit any part)

- **Decision threshold (tau) — the largest single factor, and previously missing from this spec.** Every scoring model outputs a probability per label; the cutoff for calling it a positive is a free parameter. Measured on our own baseline, moving it off the default 0.5 changes tail macro-F1 by **24.6x** (0.0139 -> 0.3420) — larger than any modelling choice in this project. **Tune one global tau per model arm on the carved validation split, and apply it identically to every arm.** A single shared tau costs only 1.06-1.23x against per-band tuning, and per-band tuning risks fitting noise given thin tail validation support. Tune the tail-band tau only over concepts that are test-scoreable, so the tuning population matches the evaluation population. Without this, any SciBERT-vs-Gemma difference may only reflect which model happened to suit the default.
- **Metrics:** precision, recall and F1 **per frequency band**. Report per-band **micro-F1** as the headline (it pools counts across the band and is not distorted by concepts with one or two test instances), with macro-F1 alongside. **Every macro-average must state its denominator** and be reported under both: over all concepts in the band, and over only the test-scoreable ones. The two differ by 3.95x in the tail, and the published reference numbers do not say which they use.
- **Store per-label TP/FP/FN** for every arm, so any averaging convention can be recomputed later without retraining. This is what lets us report against either of the paper's possible denominators without rerunning anything.
- **Report their metrics too, so our table is comparable to theirs.**
  - **Frequency robustness, Delta = Head F1 - Tail F1** — their contribution #3. Published values: BERT 0.159, SciBERT 0.170, astroBERT 0.135, DeepSeek 0.045. Cheap to compute and it is their headline robustness number. Caveat to state: Delta is threshold- and denominator-dependent, and it goes to zero for a model that predicts nothing, so it must be read alongside absolute F1.
  - **P@k and R@k for k in {1, 3, 5}** — they chose these to match the mean of 4.31 concepts per paper. **Ranking metrics sidestep the threshold asymmetry entirely**, which makes them the fairest available comparison between our thresholded encoder arms and any set-emitting generative arm. Worth reporting for that reason alone. Their per-band P@3/R@3 are in Table 6; overall, DeepSeek R@5 = 0.5050 vs SciBERT 0.2563.
  - **Their overall (not per-band) F1, Table 5**, which we should cite for context: rule-based 0.1950 / 0.2140 (without / with UAT synonym variants), k-NN 0.2017, BERT 0.1880, SciBERT 0.2127, astroBERT 0.3243, DeepSeek 0.3770. Note that **astroBERT (0.3243 overall, 0.081 tail) is the real bar**, not SciBERT — beating only SciBERT would be a weak target, and astroBERT is domain-pretrained on astrophysics text where SciBERT is general-scientific.
- **The core output:** a table (and/or plot) showing, for every model (2 baselines + SciBERT-full + SciBERT-LoRA + Gemma-QLoRA), the per-band precision/recall/F1. This is where the long-tail behavior and the model comparison become visible.
- **Compute cost reporting:** for each model record **trainable parameters, peak GPU memory (`torch.cuda.max_memory_allocated`), and seconds per training step measured on the same device at the same batch size and sequence length**. Do *not* headline wall-clock GPU-hours: on a preemptible queue that figure absorbs waiting time, and an hour on one card is not an hour on another. Log these from the first run — a cost figure reconstructed after seeing the accuracy is a narrative, not a measurement. Note that trainable-parameter count alone flatters the QLoRA arm (it trains ~10-30M parameters against SciBERT's ~110M while still back-propagating through 4B frozen weights), so it must never be reported alone.
- **What counts as a difference.** Measured on this test set, the bootstrap 95% interval on tail micro-F1 has a half-width of ~1.66pp, so the smallest difference we can resolve is roughly **2-3pp unpaired, ~1.5pp paired**. Differences below that are not results. We do not pre-register a success threshold; we report the difference with its interval and describe it. Whether the extra compute is "justified" is a judgement for the reader (and the marker), not something this data can settle — it should be asked, not assumed.
- **Sanity check (fallback):** if a model can't get meaningful results, confirm it can **overfit a small subset** of the data. This proves the pipeline is bug-free rather than mis-designed. (This mirrors the course's own methodology requirement.)
- **Control for randomness — three separate sources, not one.**
  - *Split variance:* removed by fixing the split, but every claim is conditional on this split and the paper says so once.
  - *Test-set sampling:* reported via **paired bootstrap over the 3,025 test papers** (Dror et al., ACL 2018), comparing arms on the same papers. Cheap, no retraining.
  - *Training seed:* **not captured by the bootstrap**, which conditions on one trained model. Train SciBERT-LoRA at **3 seeds** and report the spread as a descriptive sentence. Do not wire it into any threshold: a range over 2-3 runs is itself high-variance. Not affordable for the Gemma arm at 4B. Note that the TF-IDF baseline is deterministic (`liblinear`, convex objective), so running it at several seeds would show exactly zero variation and prove nothing.
  - *Decoding:* zeroed by greedy decoding (`do_sample=False, num_beams=1`) — state it.
  - No published source endorses bootstrap intervals as a substitute for multiple training seeds; we present it as a compute-constrained choice, not a citation-backed equivalence.
- **Avoid data leakage:** strict train/validation/test separation. Band definitions and the decision threshold are computed on training and carved-validation data only, never on the test split. Persist the split, the label set, the fitted binarizer and the band map once, with hashes, and have every arm read those files rather than recomputing them.
- **Reproducibility:** log exact settings (model versions, hyperparameters, seeds); keep code in the shared repo.

---

## 9. Technical stack and resources

- **Libraries:** HuggingFace `transformers` and `datasets`; `peft` (LoRA); `bitsandbytes` (4-bit quantization); `scikit-learn` (TF-IDF + Logistic Regression baseline); standard `torch`.
- **`requirements.txt` is currently missing `torch`, `transformers`, `peft` and `bitsandbytes`.** Pin them before any model work.
- **Apple Silicon laptops CAN train these models — verified by measurement, not assumption.** An earlier version of this section claimed `bitsandbytes` was NVIDIA-only and would not install on Apple Silicon. **That was wrong.** bitsandbytes has shipped macOS arm64 wheels since 0.49.0 (Dec 2025), and 0.50.0 (Jul 2026) states that "all 4-bit and LLM.int8() configurations now work on MPS"; the official support matrix lists QLoRA 4-bit as supported on Metal. Measured on an M1 Air, 16 GB, macOS 26.5.1, torch 2.13.0 + bitsandbytes 0.50.2:

  | Test | Result |
  |---|---|
  | NF4 `quantize_4bit` / `dequantize_4bit` on MPS | works; mean reconstruction error 0.073 |
  | SciBERT 110M, 1,864-label head, batch 8, seq 512 | **2.29 s/step -> ~76 min/epoch** over 15,875 docs |
  | QLoRA on a 0.5B decoder, batch 2, seq 640 | 3.69 s/step; loss finite and falling 16.4 -> 6.9 over 5 steps |
  | QLoRA on a **1.5B** decoder, bf16 compute, batch 1, seq 640, grad-ckpt on | 5.28 s/step; loss 7.6 -> 3.3; **peak memory 10.2 GB of 16 GB**; ~23 h/epoch |

  Scaling is near-linear per document (1.85 s/doc at 0.5B, 5.28 s/doc at 1.5B — 2.9x for 3x the
  parameters), so **Gemma-3-1B interpolates to ~15 h/epoch** on this hardware. bf16 compute is
  confirmed working on MPS, which is what Gemma 3 needs.

  **So both SciBERT arms are comfortably laptop work**, and the QLoRA path genuinely runs. What the laptop is not good for is *large* models: measured scaling puts **Gemma-3-1B at ~15 h/epoch and 4B at roughly 60-80 h/epoch**, because bitsandbytes' MPS `gemm_4bit` only uses its optimised kernel at batch 1 (inference) and falls back to dequantize-then-matmul for batched training. **Gemma-3-4B on a laptop is not viable; Gemma-3-1B is a slow but real fallback** — an overnight run per epoch.

  **Two optimisations that should cut the 1B figure substantially, both untested:** (a) the timings above are language-model loss over a 150k+ vocabulary; our arms use a **1,864-label classification head**, which removes a large logits tensor and its gradient — plausibly 30-50% of step cost and most of the memory pressure; (b) seq 640 heavily over-pads abstracts averaging ~250 tokens, so **length-grouped dynamic padding at 320-384** could roughly halve it again. Measure both before concluding the laptop is too slow.

- **Apple Silicon gotchas, all load-bearing:** pass `device_map={"": "mps"}` explicitly — with `device_map=None` the 4-bit quantizer silently places the model on **CPU**; 8-bit optimizers are unsupported on MPS, so use `adamw_torch`; set `TORCHDYNAMO_DISABLE=1` to avoid observed recompile thrash inside the dequantization path.
- **We own a dedicated NVIDIA GPU: an RTX 3060 Ti (8 GB, Ampere sm_86).** Verified from NVIDIA's specs: ~16.2 TFLOPS FP32, 448 GB/s, and — crucially — **bfloat16 and FlashAttention-2 support**, both of which the Colab T4 (Turing sm_75) lacks. Since Gemma 3 overflows fp16 (activations reach ~800,000 against a 65,504 ceiling), bf16 support means this card sidesteps the NaN problem natively. It is roughly **2x the T4's FP32 throughput** with no queue, no preemption and no session cap. Computed VRAM budget: Gemma-3-4B QLoRA fits in ~4.5 GB with a 1,864-label classification head, but ~6.5-7.5 GB with the 262k-vocab generative head — the latter is at or past the ~7.3 GB usable limit. **This has not yet been measured on the card; do that before committing.**

- **Colab is now redundant** — strictly worse than the 3060 Ti on every axis except raw VRAM.

- **The Slurm partition remains necessary, for reasons that are not about raw compute.** The `studentkillable` partition is preemptible but gives **each student their own account and storage** at `/home/morg/NLP2526b/<username>`. Three reasons it stays in the plan: (i) **the approved proposal commits to it** — "We will also use the TAU Slurm `studentkillable` partition for the Gemma arm and repeated runs"; (ii) **the 3060 Ti belongs to one student**, so a single-machine plan leaves the other two with nothing to run and creates a bus factor of one on our only CUDA device; (iii) **multi-seed runs are serial on one card and parallel on a cluster**. Access is **unproven** until the request form is submitted (a stated "before you start" prerequisite), the gated Gemma licence is accepted, and a job actually allocates a GPU.

- **Team hardware.** Ilana: MacBook Air M1 16 GB (measured — SciBERT 76 min/epoch) and a desktop RTX 3060 Ti 8 GB. Itai: MacBook Air M5 24 GB, 10-core GPU (unmeasured). Shai: not yet asked. The 3060 Ti is the only machine that is not someone's daily driver and is therefore the natural home for long Gemma runs. The M5 has confirmed matrix-multiplication hardware (Apple "Neural Accelerators") and 24 GB, but Apple exposes it via Metal/MLX with **no stated PyTorch support**, so no speedup should be assumed until measured.

- **Comparability constraint on the compute-cost claim — resolved without moving the training.** Our research question asks whether the large model's extra compute is justified, and the proposal promises GPU-hours. Timings from different machines are not the same unit, but the arms do not all have to train on one machine. **Decouple where we train from where we time:**
  1. **Report training FLOPs as the primary compute figure** (approximately 6ND — 6 x parameters x tokens). It is hardware-independent and reviewer-proof.
  2. **Add calibrated GPU-hours from a single declared reference device.** Run a short timing slice (~200 steps) of *every* arm on the 3060 Ti, multiply by the total step count, and report that. Roughly 2-3 hours of work in total.
  3. **Footnote the hardware each arm actually trained on.** Optionally add an M1 column as a secondary "accessible hardware" result — our measured finding that bitsandbytes' MPS `gemm_4bit` fast path applies only at batch 1 is a genuine contribution in its own right.

  This satisfies the proposal's promise while letting each arm train wherever it runs best.
- **Memory: the 4-bit weight size is not the binding constraint.** For Gemma-3 at batch 1, seq 640, the logits + cross-entropy tensor over the 262,144-token vocabulary is ~**1.68 GB** — and being vocabulary-driven it is *identical* for the 1B and 4B models, so shrinking the model does not shrink it. Mitigate with batch 1 plus gradient accumulation, chunked cross-entropy, gradient checkpointing, and loss on completion tokens only. Assert `attn_implementation="sdpa"`.
- **The Gemma 3 fp16 hazard is architectural, not NVIDIA-specific.** Gemma 3 activations exceed fp16's +-65504 range and produce NaN losses (HuggingFace transformers issue #36822). This is a property of the model, so it applies on any device running fp16. **The fix is to compute in bf16 where available, or fp32 where it is not.**
  - **On Apple Silicon:** MPS supports bf16 on macOS 14+, so `bnb_4bit_compute_dtype=torch.bfloat16` is the correct and natural setting.
  - **On a Colab T4 (sm_75):** there is no bf16. The correct path is `bnb_4bit_compute_dtype=torch.float32` with autocast off — slower than fp16 but numerically safe. **The T4 is not disqualified by this**; at roughly 8 TFLOPS fp32 against the M1's ~2.6 it remains several times faster than the laptop even in fp32. The T4's real problems are the 12-hour session cap and the lack of persistence, not the dtype.
  - Either way, **assert loss finiteness over >=500 steps** — 20 steps settles peak memory but will not catch this.
- **Checkpointing is required, not optional.** A Gemma epoch is estimated at 3-14 hours against a 12-hour cap and a preemptible queue, and no checkpoint or resume code currently exists. Save every few hundred steps and on SIGTERM, and test resume by deliberately killing a job. (The earlier advice here to avoid frequent checkpointing was about not filling shared storage — keep few checkpoints, but keep them.)
- **Storage:** back up code and critical results to GitHub/Drive; Slurm storage is not backed up and disappears at the submission date.
- **No external API credits required.** Better GPUs are worth requesting but must not be assumed: plan for the T4 and treat an upgrade as a bonus.

---

## 10. Scope, constraints, and Plan B

- **Scope:** this is a course project — a white-paper proof-of-concept, NOT a conference paper. Avoid heavy engineering, large-scale annotation, or large compute. (The earlier phrasing "completable in a few days of focused work" is dropped: it is the course's description of an *idea's* scope, and reading it as an estimate of our remaining workload would be misleading.)
- **Plan B — a ladder, in order:** Gemma-3-4B QLoRA -> 4B with gradient checkpointing -> **Gemma-3-1B trained plus 4B prompted-only** -> 1B only. Note that scaling 4B down to 1B is not the main lever: cost is dominated by decode length and evaluation-set size more than parameter count.
- **The floor.** The evaluation findings in §4 (tail scoreability, the 3.95x denominator, the 24.6x threshold effect) require **no GPU** and are already measured. Structure the report so that a failed Gemma arm still leaves a complete paper, with the arm reported honestly as attempted, including its wall-clock and the error that stopped it. The course explicitly values negative results with sound methodology.

---

## 11. Reference papers

**Anchor papers (directly used; from approved NLP/ML venues, post-2018):**
1. **SciBERT** — Beltagy, Lo & Cohan, EMNLP 2019 (the base encoder).
2. **LoRA** — Hu et al., ICLR 2022 (parameter-efficient fine-tuning).
3. **QLoRA** — Dettmers et al., NeurIPS 2023 (4-bit quantized fine-tuning; used for the Gemma arm). **Venue and year verified** — the earlier "verify before citing" note is closed.

*(All three anchor venues verified against the ACL Anthology / official proceedings: SciBERT = EMNLP-IJCNLP 2019, D19-1371; LoRA = ICLR 2022; QLoRA = NeurIPS 2023. The `Articles/` notes are all the correct papers.)*

**Related-work citations (context / prior art):**
- **Huang et al., EMNLP 2021** — "Balancing Methods for Multi-label Text Classification with Long-Tailed Class Distribution." Balancing-loss methods; head/medium/tail evaluation (via tertiles); tested on Reuters (news) and PubMed (biomedical/MeSH). Verified from the actual paper.
- **Chang et al., KDD 2020** — "Taming Pretrained Transformers for Extreme Multi-label Text Classification" (X-Transformer). Foundational transformer XMC; documents that the vast majority of labels are long-tail. Verified from the actual paper.
- **Alkan, Grezes, Blanco-Cuaresma, Bartlett, Chivvis, Kelbert, Lockhart & Accomazzi, arXiv:2604.02156 (2026)** — *"AstroConcepts: A Large-Scale Multi-Label Classification Corpus for Astrophysics."* **Cite as a preprint, not a published paper** — arXiv metadata shows no journal reference, no DOI beyond the automatic arXiv one, and a single version posted 2 April 2026. This is the corpus we use and the source of our split, band cutoffs and reference baselines. Full text at `Articles/AstroConcepts - Alkan et al., arXiv 2026.md`. **Not citing it would read as a failed literature review**, since it is by the dataset's own authors and is findable by anyone searching the dataset name.
- **Jain, Prabhu & Varma, KDD 2016** — propensity-scored precision (PSP@k), the established metric for tail-label evaluation in extreme multi-label classification. Relevant to our metric discussion in §8.
- **Yang & Liu, SIGIR 1999** (after Joachims, 1998) — the ModApte split of Reuters-21578, which retained only categories with at least one training *and* one test document. This is the standard precedent for a minimum-support filter with a test-support guarantee, and the historical reason benchmark label sets arrive pre-curated.
- **Dror, Baumer, Shlomov & Reichart, ACL 2018** — *"The Hitchhiker's Guide to Testing Statistical Significance in NLP."* Cited for the paired significance procedure in §8.
- **Card, Henderson, Khandelwal, Jia, Mahowald & Jurafsky, EMNLP 2020** — *"With Little Power Comes Great Responsibility."* Cited for the power/minimum-detectable-effect framing in §8.
- Two further partial near-misses to consider for related work: arXiv:2512.12677 (quantized LoRA on causal LLMs vs BERT) and arXiv:2406.08660 (fine-tuned small LLMs vs zero-shot generative models). Both cover some elements but neither combines PEFT with long-tail band evaluation.

---

## 12. Decision log and open items

**Decisions now closed — do not reopen without new evidence.**

| Decision | Value | Basis |
|---|---|---|
| Split | Shipped 18,677 / 3,025, adopted verbatim | §6.1 — it is the published benchmark test split |
| Validation | ~15% carved from train | §6.1 — the shipped split has only two parts |
| Minimum-frequency filter `k` | **None** | §6.2 — the split filters upstream |
| Frequency bands | Absolute: head >500, torso 50-500, tail <50 | §6.3 — tertiles give an empty tail band here |
| Decision threshold | One global tau per arm, tuned on the carved validation set | §8 — worth up to 24.6x on the tail |
| Headline metric | Per-band micro-F1, with macro-F1 under both denominators | §8 |
| Gemma output | Free-form generation, plus shortlist log-probability scoring | §7.4 — constrained decoding is not used |
| QLoRA anchor citation | Dettmers et al., NeurIPS 2023 | §11 — venue verified |

**Open items, with owners.**

| Item | Owner | Notes |
|---|---|---|
| Submit the Slurm request form | __________ | Stated "before you start" prerequisite; days of lead time |
| Accept the Gemma licence on HuggingFace, generate a token | __________ | Gated model; code fails at load without it |
| Timed 200-step Gemma benchmark on the real GPU | __________ | Decides 4B vs 1B (§7.4). Must assert loss finiteness over >=500 steps |
| Email the lecturer | __________ | Disclose the concurrent preprint; get the metric change approved (macro-F1 was never in the approved proposal); request better GPUs |
| Checkpoint / resume for training | __________ | Gates the Gemma arm against preemption; test by killing a job |
| Re-run the threshold sweep with tau tuned on the carved validation set | __________ | Current 0.342 / 0.087 figures are tuned-on-eval and optimistic |
| Re-check whether the AstroConcepts preprint is accepted anywhere | __________ | Before submission, so the citation is accurate |
| AI Disclosure & Reflection section | __________ | Required in the final report; log as we go rather than reconstructing in late September |

**Format:** ACL template (LaTeX/Overleaf), up to 8 pages excluding references and appendix. Budget roughly four tables and one figure; every table beyond that trades presentation marks for results marks.
