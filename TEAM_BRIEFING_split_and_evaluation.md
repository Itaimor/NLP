## TL;DR — the five things that changed

1. **Someone has already posted a preprint on our exact dataset**, and the `validation` split we
   have been throwing away is actually *their benchmark test split*.
2. **We should stop re-splitting the data 70/10/20** and use the split that ships with the dataset.
3. **The `k` filter question can be deleted entirely** — it becomes unnecessary once we use their split.
4. **Our tertile frequency bands are broken** and must be replaced with absolute cutoffs.
5. **The biggest single factor in our results is the decision threshold**, which our spec never
   mentions. Measured: it changes tail F1 by **24.6x**. That is larger than every other decision
   we have argued about combined.

None of this means the project is in trouble. It means the project just got a lot more focused,
and we now have external numbers to compare ourselves against — which we never had before.

---

## Part 1 — We found the paper that made our data split

### What the paper is

**Title:** *AstroConcepts: A Large-Scale Multi-Label Classification Corpus for Astrophysics*
**Authors:** Alkan, Grezes, Blanco-Cuaresma, Bartlett, Chivvis, Kelbert, Lockhart, Accomazzi
**Who they are:** the NASA ADS / Harvard-Smithsonian team — i.e. *the people who built our dataset*
**Where:** arXiv:2604.02156, submitted 2 April 2026

**Important status note:** this is an **unrefereed preprint**. It has NOT been peer-reviewed or
accepted anywhere. We checked the arXiv metadata directly: the "Journal reference" field is empty,
the only DOI is the automatic arXiv one, and there is a single version (v1). An earlier draft of
this briefing said it was published at a conference — that was wrong and has been corrected.

This matters in our favour, and we come back to it in Part 6.

### What they did

They used **our exact corpus** (21,702 astrophysics abstracts, UAT concepts). It is a **corpus
paper** — its main contribution is releasing the dataset — and it benchmarks four families of method
on it: rule-based keyword matching, k-nearest-neighbours over embeddings, fine-tuned encoders (BERT,
SciBERT, astroBERT), and an LLM. Results are broken down by how frequent each label is — head, torso
and tail bands.

That is three of the four things our spec section 4 claims nobody has combined.

**Their LLM is not doing what it might sound like.** It is a *two-stage* system: first their
fine-tuned astroBERT proposes the top 50 candidate concepts for a paper, then DeepSeek-V3 picks from
those 50. They chose 50 because astroBERT's top 50 already contains about 82% of the correct answers.
They also tried the obvious approach first — handing the LLM all 2,367 concepts — and it failed:
*"the model hallucinated concepts or became overwhelmed by the extensive vocabulary."*

Two things follow. Their headline 0.198 on the tail is a score for the **whole pipeline**, astroBERT
included, not for an LLM working alone. And their failure with the full vocabulary is a direct warning
for our own Gemma arm (see Part 9, Question 2).

### Their reported results

| Band | BERT | SciBERT | astroBERT | DeepSeek-V3 |
|---|---|---|---|---|
| Head (>500 occurrences) | 0.180 | 0.193 | 0.216 | 0.243 |
| Torso (50-500) | 0.167 | 0.197 | 0.312 | 0.376 |
| Tail (<50) | 0.021 | 0.023 | 0.081 | 0.198 |

*(These are F1 scores. Higher is better. Note how everything collapses in the tail — that is the
long-tail problem our project is about.)*

### The critical discovery: their split is our "validation" split

Their paper explains exactly how they built their train/test split:

> "Labels appearing >=15 times are stratified to achieve approximately 85/15 distribution per
> label, while labels with <15 occurrences are placed entirely in the training set to maximize
> training signal."

Their split is **18,677 train / 3,025 test**.

Our HuggingFace download is **18,677 train / 3,025 "validation"**.

**These are the same thing.** What we have been calling a validation split is their published
benchmark *test* split. Our spec section 6.1 merges it back into training and re-splits — which
destroys a benchmark and, with it, any chance of comparing our numbers to anyone else's.

### A note on units: "papers" vs "occurrences"

Two different things get counted in this document, and it matters which one you are reading:

- **Papers** — actual rows in the dataset. There are 21,702 (18,677 train + 3,025 test).
- **Occurrences**, also called **(paper, label) pairs** — one paper tagged with one concept. A paper
  tagged with 4 concepts contributes 4 occurrences. There are **93,547** in total, i.e. an average of
  **4.31 labels per paper** (93,547 / 21,702).

So an occurrence count is always larger than the paper count behind it, and the two must never be read
as the same quantity. Where a number below counts pairs rather than papers, it says so explicitly.

**One special case worth knowing:** for a *single* concept, its occurrence count and its paper count
are the same number — a concept appearing 25 times appears in 25 different papers, since a paper is
never tagged with the same concept twice. So per-concept frequencies (the "<15", ">500", "20-49"
thresholds below) can be read either way. It is only when you *sum across concepts* that the two
diverge.

### How we verified this is not a coincidence

We recomputed their statistics from our own downloaded data in `.hf_cache`. The method: count how
many times each UAT concept ID appears across all 21,702 papers, then check where those concepts live.

| What we computed | How | Result |
|---|---|---|
| Concepts appearing <15 times total | Counted every concept ID across train+val, kept those with count < 15 | **935 concepts** |
| How many of those appear in val | Checked each of the 935 against the val split | **0 — exactly zero** |
| Band sizes | Grouped concepts by total count: >500, 50-500, <50 | **17 / 429 / 1,418** |
| Total tail occurrences | Summed the corpus-wide counts of all 1,418 tail concepts (an "occurrence" is one paper-concept pair) | **18,451** |

Their paper reports 17 (0.9%), 429 (23.0%), 1,418 (76.1%), and 18,451 occurrences (= (paper, label)
pairs, not papers).
**Every number matches exactly.**

Zero out of 935 rare concepts leaking into the val split is not chance. That is their rule,
operating on the files sitting on our laptops.

### This also solves a mystery our own spec wrote down

Our spec section 6.1 notes that the provided validation split "is not a random sample": exactly one
label has 3 instances, no label has 1, 2 or 4, and 187 labels have exactly 5. We observed that and
could not explain it.

The explanation is the <15 rule. Anything rarer than 15 occurrences was held out of the test set
entirely, which is why the minimum support we see is 3 and why counts pile up at 5.

---

## Part 2 — Why we should keep the given split instead of re-splitting 70/10/20

### First: our original decision was reasonable

Our spec section 5 says "There is no dedicated test split." Read the HuggingFace dataset card and
that is exactly what it looks like — two splits named `train` and `val`. When a dataset ships no
test set, standard practice is precisely what we did: merge, re-split three ways, use iterative
stratification because it is multi-label, fix the seed at 42. That is textbook and nobody would
mark us down for it.

What changed is not that the reasoning was bad. New information arrived: that `val` split is not a
validation set someone tossed in, it is a deliberately constructed benchmark. We had no way to know
that until the preprint surfaced.

### Reason 1 — We get an external reference frame (the big one)

Right now, if we train SciBERT and get tail F1 = 0.15, we have no idea whether that is good, bad, or
a bug. Our only comparisons are our own other models. If all five of our arms share the same bug,
every number is wrong together and looks perfectly consistent.

Use their split, and our numbers sit directly beside theirs on identical data. That buys three things:

- **A correctness check.** Train SciBERT; if we get roughly 0.19 head F1, our pipeline works. If we
  get 0.60 or 0.01, we have a bug — and we find out in week one instead of while writing the paper.
- **Meaning for our result.** "Our Gemma-QLoRA reaches 0.12 tail F1, above astroBERT's 0.081 and
  below prompted DeepSeek-V3's 0.198" is a sentence with content. "Our Gemma-QLoRA reaches 0.12" is
  a number floating in space.
- **Rubric points.** Results and discussion is 20 points, and the course guidelines say results must
  be framed against relevant baselines. External baselines on identical data is the strongest
  version of that available to us.

**This is the part that cannot be recovered later.** If we re-split, our test set contains a
different mix of papers with different difficulty. Comparing our 0.15 to their 0.023 would be
meaningless — different exam, different score. There is no post-hoc fix. We either share their test
set or we never compare to anything.

### Reason 2 — Re-splitting actively damages the rare labels our paper is about

This is the counter-intuitive one, and the most persuasive.

Their rule holds every concept with fewer than 15 occurrences entirely in training — "to maximize
training signal." That looks like it is hiding the rare labels. It is actually protecting them.

Here is the concrete trade, computed from our data:

**935 concepts have fewer than 15 occurrences, totalling 4,824 occurrences** — that is 4,824
(paper, label) pairs, spread across only 3,817 distinct papers, since one paper can carry several rare
concepts.

- Under their split: all 4,824 pairs go to training.
- Under our 70/10/20: about 30% get pulled out for val and test — roughly **1,447 (paper, label)
  pairs removed** from training, taken from the concepts that can least afford to lose any. These are
  label-annotations, not papers: the papers themselves stay in the dataset, but the rare-concept tags
  attached to them move out of the training portion. A concept with 6 training examples drops to 4.
  We would be degrading the model's ability to learn exactly the labels the paper is about.

And what do we buy with those 1,447 lost pairs? (1,447 = 4,824 (paper, label) pairs x 30% moved out
of the training portion.) We computed the expected test support:

| Outcome | Concepts |
|---|---|
| Still 0 test instances (unscoreable anyway) | **~392** |
| **1 test instance** | ~287 |
| **2 test instances** | ~146 |
| 3+ test instances | ~110 |

*(How this is computed: each concept with `c` total occurrences has, under a random 20% test split,
a Binomial(c, 0.2) chance of landing `j` times in test. Summing those probabilities over all 935 rare
concepts gives the expected number of concepts at each level. Cross-checked with a hypergeometric
model, which gives the same shape.)*

**One or two test instances is not a measurement.** With 1 test positive, per-label F1 can only
ever be 0, 0.5, 0.667 or 1.0. With 2, only 0, 0.4, 0.5, 0.667, 0.8 or 1.0. It is a step function —
a single lucky or unlucky prediction swings a label from 0 to 0.667. Average several hundred of
those and you get a number that moves with the random seed and tells you nothing.

So we sacrifice ~1,447 rare-concept training pairs and, in exchange, **392 concepts are still unscoreable** and
another 433 gain only the useless 1-2 instances. **We would be trading real training signal for fake
measurement.** Their rule declines that trade deliberately.

### Reason 3 — Three bugs in our pipeline simply stop existing

Our `scripts/EDA_DataSplit.py` currently counts label frequencies on the **merged** corpus and
filters *before* splitting. That means information from the future test set leaks into which labels
exist at all. It is a genuine data-leakage bug, and our own spec section 8 forbids it.

If we do not re-split, that entire class of bug disappears. There is no ordering to get wrong and no
counts computed on the wrong subset. The split is a given, not something we compute.

For the 20-point methodology score, "we use the standard split released with the corpus" is a
stronger and shorter sentence than three paragraphs defending a homemade stratification.

### Reason 4 — Their rule is documented and citable; ours would need defending

Every split rule needs justification. Theirs is written down, motivated, and citable. Ours would be
ours to defend — and defending it means explaining why we distributed 4-occurrence labels across
three splits, which is the harder argument to win.

*(Note: their preprint is unrefereed, but that is not what this reason rests on. The split itself
ships with the HuggingFace dataset — it is a property of the data we downloaded. The preprint only
explains the rule behind it.)*

### Reason 5 — The limitation becomes our contribution

The honest cost of their split: only 805 of 1,864 concepts appear in the test set. (1,864 = the
number of distinct UAT concepts that occur anywhere in the corpus; 805 = how many of those appear at
least once in the 3,025-paper test split. The other 1,059 are train-only.) **1,059 tail
concepts can never be scored.**

That is real. But as shown in Reason 2, under a re-split those concepts would appear with 1-2
instances each — statistically meaningless. So the real choice is:

- **(a)** 1,059 concepts openly unscoreable, stated as a limitation, or
- **(b)** 1,059 concepts *apparently* scoreable, producing noise we might mistake for signal.

(b) is worse, because it hides the problem inside a number that looks fine.

And under (a) we get to *report* it, which is a genuine finding. See Part 4.

---

## Part 3 — Why `k` can now be deleted entirely

Our spec section 6.2 spends a page on choosing a minimum-frequency filter `k`, lists six candidate
methods, and picks none.

**With their split adopted, the question dissolves.** Their split already performs the
minimum-support function upstream: concepts with <15 occurrences are held out of the test set by
construction. We do not need to invent our own filter on top of that.

So `k` is not "chosen" — it is **deleted**. The label universe becomes the 1,864 concepts that
actually occur in the corpus. We inherit it by citation instead of choosing it by argument, which is
what finally ends the debate.

### One warning: do not confuse the two "thresholds"

There are two numbers in this project that both get called a threshold, and they are opposites:

| Name | What it is | What we do |
|---|---|---|
| **k** | a *data* filter — which labels are allowed into the study | **DELETE it** |
| **tau** | a *decision* threshold — how confident the model must be to say "yes" | **KEEP and TUNE it** |

Deleting both would recreate the exact bug described in Part 5. We delete `k`. We very much keep `tau`.

---

## Part 4 — The benchmark cannot score its own tail (our free contribution)

On the given split we counted how many tail concepts have any test example at all:

- Tail band: **1,418** concepts
- With at least one test example: **359**
- **Unscoreable — locked entirely in train: 1,059 (74.7%)**

*(1,059 divided by 1,418 = 74.7%)*

**Why this matters.** "Macro-F1" means: compute F1 separately for each label, then average those
scores equally. If we average over all 1,418 tail labels, then 1,059 of those terms are structurally
forced to be exactly 0 — not because the model failed, but because there is nothing to score. If we
average over only the 359 scoreable ones, we divide the same total by 359 instead.

**Ratio: 1,418 / 359 = 3.95.** So a published tail number changes by roughly 4x depending on which
denominator was used — and the preprint never says which.

**State this carefully.** We know the 1,059 unscoreable concepts exist; that is verified from the data.
What we do *not* know is which denominator their table uses. We checked whether their overall results
(Table 5) reconcile with their per-band results (Table 6) and they do not — under label-weighting
astroBERT would be 0.135, under instance-weighting 0.254, against the 0.324 they report. So the two
tables use different conventions and we cannot infer the per-band one. **The claim we can defend is
that their tail numbers are uninterpretable without a denominator they do not state — not that they
are definitely deflated 4x.** Our own results avoid the problem by reporting both denominators.

| Model | Their number (if over 1,418) | Same result if over 359 |
|---|---|---|
| SciBERT | 0.023 | 0.091 |
| astroBERT | 0.081 | 0.320 |
| DeepSeek-V3 | 0.198 | 0.782 |

*(e.g. 0.023 x 3.95 = 0.091)*

This is a real methodological finding about the field's own astronomy benchmark, it costs **zero GPU
hours**, and it is ours.

### What "rare" means for us now

Because concepts with <15 occurrences are held out of the test set, our *evaluated* tail band is not
the ultra-rare concepts. It splits in two:

| Group | Concepts | Corpus frequency | Median |
|---|---|---|---|
| **Scoreable tail** (in test) | **359** | 20-49 | 31 |
| **Unscoreable tail** (train only) | **1,059** | 1-19 | 5 |

**Is 20-49 occurrences still rare? Yes.** (These are per-concept counts, so they read directly as
papers — a concept with 25 occurrences appears in 25 papers.) Median training frequency per band,
across the 18,677 training papers:

| Band (cut on corpus frequency) | Concepts | Median train occurrences | Share of training papers |
|---|---|---|---|
| Head (>500) | 17 | 543 | 2.91% (543 / 18,677) |
| Torso (50-500) | 429 | 98 | 0.52% (98 / 18,677) |
| Tail (<50) — all | 1,418 | 8 | 0.04% |
| **Tail — the 359 we evaluate** | **359** | **25** | **0.134%** (25 / 18,677) |

Two comparisons, answering two different questions:

- **Is the evaluated tail still rare in absolute terms?** Against the head: **21.7x rarer**
  (543 / 25). This is the dynamic range of the study — the evaluated tail is still two orders of
  magnitude away from the common concepts, so the long-tail effect has not been defined out of
  existence. The measured difficulty gap confirms it (see Part 5).
- **Is the tail band actually distinct from the band next to it?** Against the torso: **3.9x rarer**
  (98 / 25). This is the more demanding test, and the honest number to quote if someone asks whether
  the three bands are really separate populations rather than an arbitrary slicing.

**A subtlety to state rather than be caught by:** the bands are cut on *corpus* frequency (<50,
50-500, >500), but models train on *train* frequency, and those differ because of the 85/15 split and
the <15 rule. At the edges the two bands overlap: torso concepts go down to 39 training examples while
evaluated-tail concepts reach up to 44. So a few "torso" concepts have fewer training examples than
some "tail" concepts. The effect is small, but it belongs in the limitations paragraph.

### Why our current band method (tertiles) has to go

Spec section 6.3 sorts concepts by frequency and cuts them into three equal-sized groups — tertiles.
That follows Huang et al. and is reasonable in general. **On this split it breaks completely.**

The bottom third of our concepts spans train-frequency **1 to 6**. But the lowest train-frequency of
any concept that appears in the test set is **13**. So every concept in the tertile-tail is absent from
the test set by construction — **the band would contain zero test papers**.

We would report a tail column of 0.000 for every model, and it would look like a devastating finding
about long-tail collapse. It would be an empty set.

Huang et al. could use tertiles because their concepts averaged ~150 papers each. Ours average ~50
(93,547 (paper, label) pairs / 1,864 concepts = 50.2), and our bottom third averages about 3
(measured: 2.85). The method does not transfer. **Absolute cutoffs fix it**, and using the preprint's
boundaries keeps our numbers comparable to theirs.

**And the ultra-rare concepts are not excluded from the task, only from scoring.** All 1,059 of them
are in the training data (6,906 training occurrences — i.e. 6,906 (paper, label) pairs, the sum of how
often those 1,059 concepts are tagged across the 18,677 training papers; far fewer than 6,906 distinct
papers are involved), they are in the 1,864-way label space, they
compete for probability mass, and predicting one on a test paper counts as a false positive. The
task keeps its full difficulty; only the scoring is restricted to where scoring is meaningful.

One thing we can still measure on those 1,059 concepts, even without gold labels: **how often a model
predicts them at all.** Every such prediction is a false positive by construction, so the count is not
an accuracy number — but it does separate two failure modes that F1 alone conflates (a model that never
touches them, versus one that invents them freely). See the end of Part 5, where it turns out to be the
cost side of the threshold finding rather than a separate result.

---

## Part 5 — The decision cutoff, and why it changes everything

### What the cutoff is

The model does not output tags. For every paper it outputs a *confidence score* for each of the 1,864
possible topics. To turn scores into tags we need a rule: **how confident must the model be before we
accept a tag?** That cutoff is called **tau**. Every tool — scikit-learn, PyTorch, every tutorial —
defaults to **tau = 0.5**: accept a tag only if the model is more than 50% sure.

**That default destroys rare topics.** A worked example from our data:

> *"Solar radio flares"* appears in **22 of the 18,677 training papers** — 1 paper in 848, a base rate
> of **0.12%**. The model reads a paper and says *"3% likely."* That is a real signal: a **25-fold**
> lift over the base rate.
>
> But 3% is not above 50%, so the default rule discards it and records "predicted nothing." Repeat for
> every rare topic on every paper, and the model scores **zero** on rare topics.

The zero measures the cutoff, not the model.

### What we measured

Cheap baseline (TF-IDF + logistic regression, no GPU), each band scored at the default cutoff and at
the best cutoff for that band.

*(F1 combines how many applied tags were correct with how many of the correct tags were found, into one
number between 0 and 1. Higher is better.)*

| Band | Concepts | F1 at default (0.5) | Best F1 | Cutoff achieving it | Concepts ever tagged at 0.5 |
|---|---|---|---|---|---|
| Head | 17 | 0.4727 | 0.5917 | 0.18 | 17 of 17 |
| Torso | 429 | 0.1655 | 0.4150 | 0.08 | 276 of 429 |
| **Tail** | **359** | **0.0139** | **0.3420** | **0.02** | **22 of 359** |

**Reading the tail row:** the same model scores 0.0139 at the default and 0.3420 at a cutoff of 0.02 —
**24.6x higher**, no retraining. At the default it never applied 337 of the 359 rare topics to a single
paper out of 3,025, while its highest confidence on a rare topic was 0.828.

**The pattern down the rows:** the rarer the band, the lower the cutoff it needs (0.18 -> 0.08 -> 0.02)
and the more damage the default does (1.25x -> 2.51x -> 24.6x). A large part of what the field calls
"long-tail collapse" may be this cutoff left at its default.

### One cutoff per model, not one per band

The "cutoff achieving it" column above is a diagnostic — what each band would choose alone. **The
protocol is one cutoff per model arm**, tuned on a validation set and applied to every band, so all
arms are treated identically.

> **Where the validation set comes from — important.** The shipped split has only two parts, 18,677
> train and 3,025 test, so there is **no validation set to tune on**. We create one by carving ~15% out
> of the *training* papers (Part 7). We never tune on the 3,025 test papers; doing so would be exactly
> the leakage we are removing elsewhere.
>
> **Watch the word "validation" in this document.** HuggingFace labels the shipped 3,025-paper split
> `validation`, but Part 1 establishes it is really the benchmark *test* set, and we treat it as test
> throughout. Wherever Part 5 onwards says "validation," it means the new set carved from train — never
> the shipped one. Our final structure is three parts: **~15,875 train / ~2,800 validation (carved) /
> 3,025 test (shipped, untouched until the end)**.

A single cutoff maximising overall accuracy gives **tau = 0.04**:

| Band | F1 at shared cutoff (0.04) | F1 at its own cutoff | Difference |
|---|---|---|---|
| Head | 0.4815 | 0.5917 | +0.110 (1.23x) |
| Torso | 0.3912 | 0.4157 | +0.025 (1.06x) |
| Tail | 0.3001 | 0.3420 | +0.042 (1.14x) |

Per-band cutoffs add only 1.06-1.23x, against 24.6x for abandoning the default. The shared cutoff of
0.04 sits near the tail's own optimum of 0.02, so the tail keeps 88% of its achievable score; the head
loses most (19%), having only 17 concepts to vote with.

One cutoff per arm is simpler, matches Huang et al., and avoids fitting noise — validation support on
rare topics is thin. What makes the comparison fair is not the scheme but that **every arm goes through
the identical procedure**.

### The cost of a lower cutoff

A lower cutoff also makes the model apply tags that do not belong. Measured on the 1,059 rare concepts
with no test examples, where every prediction is wrong by definition:

| Cutoff | Wrong tags applied | Rare concepts used | Test papers affected |
|---|---|---|---|
| 0.5 (default) | 1 | 1 of 1,059 | 1 of 3,025 |
| 0.08 | 35 | 28 | 34 |
| **0.02** (tuned) | **314** | **138 of 1,059** | 263 of 3,025 |
| 0.01 | 1,037 | 251 | 770 |

Same dial, opposite side. At the default the model refuses three-quarters of the rare vocabulary — one
concept, once, in 3,025 papers, which is the "neglect" our title asks about, measured. At the tuned
cutoff it uses 138 of those concepts and makes 314 wrong tags. Both numbers go in the paper.

### What this means for us

1. **Spec section 8 never mentions decision cutoffs.** This setting moves the headline number 24.6x.
   It is the most consequential gap in the document.
2. **One cutoff per arm, tuned on validation, applied identically.** Otherwise a SciBERT-vs-Gemma
   difference may only mean one of them happened to suit the default.
3. **Worth chasing:** TF-IDF with a tuned cutoff scores 0.342 over the 359 scoreable rare concepts —
   0.087 on the preprint's apparent denominator (0.342 x 359 / 1,418), roughly astroBERT's published
   0.081 and 3.8x SciBERT's 0.023. A bag-of-words model matching a domain-specific neural encoder on
   rare topics is a reportable result.

**Caveat:** the cutoff here was chosen on the same data it was scored on, so 0.342 and 0.087 are
optimistic and must be recomputed with the cutoff tuned on the carved validation set (Part 7). The
24.6x effect size is robust to this; the absolute levels are not.

---

## Part 6 — Where this leaves our contribution

### Section 4 of our spec needs rewriting

Our current novelty claim is that no single paper combines: multi-label classification +
small-encoder-vs-generative-LLM + LoRA/QLoRA + frequency-band results. The preprint covers three of
those four, on our data, with our encoder. The claim cannot stand as written.

### But there is still a cell they left open

**No generative model in their paper ever has its weights updated.** DeepSeek is called through an API
as a selector; the only models they fine-tune are the three encoders. Parameter-efficient quantized
fine-tuning of an open-weight generative model is therefore untouched, and that is our contribution.

**One correction, so nobody quotes it wrongly.** An earlier version of this briefing cited their line
*"we were unable to include Qwen models in the fine-tuning experiments"* as proof they wanted to run
our experiment. That reading is wrong: the sentence sits in their section on *supervised encoders*,
and "Qwen" there means Qwen3-Embedding-8B, an embedding model used for k-nearest-neighbours. It says
nothing about generative fine-tuning. **The gap is real — they never fine-tune a generative model —
but do not cite that sentence as evidence for it.**

**What they *do* name as unfinished is more useful to us.** From their limitations: running the LLM
stage *"over candidate sets generated by BERT and SciBERT"* would *"isolate the contribution of the
candidate generator's quality from the LLM's own re-ranking ability."* If we hold the candidate list
fixed and swap only the selector — their API model versus our QLoRA-tuned Gemma — we answer the
question they posed, and what would otherwise be a confound becomes the contribution.

### The preprint status helps us

Because it is an unrefereed preprint from April rather than a published paper, the honest framing is
**concurrent work**, not a completed scoop. Concurrent work is acknowledged, not conceded. Something like:

> Concurrent unrefereed work (Alkan et al., 2026) introduces this corpus and reports encoder and
> prompted-LLM baselines across frequency bands. We extend that setting to parameter-efficient
> quantized fine-tuning, which the authors note they were unable to run, and we identify an
> evaluation issue affecting the tail band.

### Are we allowed to cite and compare to a preprint? Yes — checked against the course guidelines

- The venue restriction ("published in NLP/ML conferences... ideally after 2018") applies only to the
  **1-3 anchor papers**. Ours are SciBERT (EMNLP 2019), LoRA (ICLR 2022) and QLoRA (NeurIPS 2023) —
  all verified and compliant. AstroConcepts does not need to be an anchor paper.
- Citing it is **required**, not optional. Guidelines: "This criterion also considers proper citation
  of sources used in the project. When mentioning results or claims from prior work, you should give
  the correct attribution."
- The lecturer's own reference list in the guidelines contains **six arXiv preprints out of ten
  references**, including one from 2026. Citing preprints is normal practice in this course.
- Comparing is encouraged: "always compare with relevant baselines... the most common approach that
  is currently used in the literature."

---

## Part 7 — Proposed plan

**Immediate (this week)**

**1. Email the lecturer.** One message, three asks. Owner: __________

- **Disclose the preprint.** Say plainly that a preprint by the dataset's own authors covers part of
  what we proposed, and state our narrowed claim: they could not run PEFT fine-tuning for compute
  reasons, and that is the cell we fill. Better he hears it from us in August than notices it while
  grading in October. Conceding costs nothing; being caught costs literature-review marks.
- **Get the metric change approved.** Our approved proposal promised "precision/recall/F1 per
  frequency band." It never mentioned macro-averaging — we added that ourselves in spec section 8, and
  Part 5 shows it is the wrong choice here. Ask whether reporting per-band micro-F1, plus both
  denominators, is acceptable. This is a scope change and it is cheaper to clear now than to defend
  later.
- **Request GPUs.** The course guidelines explicitly invite this ("if you need GPUs beyond what is available to
  you via Slurm... please email me ASAP"), and ask for the hours and card type we want. They are not
  guaranteed, so we plan as if refused.

**2. Sort out where we will train Gemma.** Owner: __________

This is the item most likely to sink the project, so here is the problem in full.

**Our laptops can train more than we assumed — this was tested, not guessed.** An earlier version of
this briefing said `bitsandbytes` (the library that shrinks the model to 4-bit) was NVIDIA-only and
would not install on a Mac. **That was wrong.** It has shipped Apple Silicon builds since December
2025, and as of July 2026 all 4-bit configurations work on Apple's GPU. Measured on an M1 Air, 16 GB:

| What we tested | Result |
|---|---|
| 4-bit quantization on the Mac GPU | works |
| **SciBERT** (our arms A and B), batch 8, seq 512 | 2.29 s/step -> **~76 min per epoch** |
| **QLoRA** on a 0.5B model, batch 2, seq 640 | 3.69 s/step, loss falling 16.4 -> 6.9 |

**So the SciBERT arms and both baselines are laptop work — no GPU needed, no waiting on anyone.**

**Where the laptop does run out is size.** The 0.5B timing extrapolates to roughly **15-16 hours per
epoch for a 1B model, and about 80 hours for a 4B one.** The reason is specific: on Apple hardware the
4-bit maths only uses the fast path for single-item inference, and falls back to a slower route for
batched training. So **Gemma-3-1B on the laptop is a slow but genuine fallback; Gemma-3-4B is not.**

**One thing that surprised us, and it matters.** Gemma 3 produces corrupted (NaN) values when trained
in the fp16 number format — this is a property of the model, not of any particular GPU. The fix is to
use bf16 instead. Our Macs support bf16; **Colab's T4 does not**, so on Colab we would have to use
fp32, which is slower but numerically safe. The T4 is still faster than the laptop even so — its real
drawback is the 12-hour cutoff, not the number format.

**Between us we have enough hardware to do the whole project.** What is available, and what each
machine can technically do:

| Machine | Memory | Measured / known capability |
|---|---|---|
| MacBook Air **M1** (2021) | 16 GB unified | **measured**: SciBERT ~76 min/epoch; QLoRA on a 1.5B model ~23 h/epoch at 10.2 GB peak. Gemma-3-1B ~15 h/epoch; 4B infeasible. |
| Desktop, **RTX 3060 Ti** | 8 GB VRAM | not yet measured. Has bf16 + FlashAttention-2; ~2x a Colab T4; no queue or time limit. Gemma-3-4B ~4.5 GB with a classification head, ~6.5-7.5 GB generating text against ~7.3 GB usable. |
| MacBook Air **M5** | 24 GB unified | not yet measured. 153 GB/s (~2.2x the M1) and dedicated matrix hardware; 24 GB fits models the M1 cannot. |

**The 3060 Ti is the strongest machine for the Gemma arm.** It has bf16 and FlashAttention-2, which
Colab's T4 lacks, so it avoids Gemma 3's fp16 NaN problem natively, with no queue and no time limit.
Its 8 GB is the binding constraint, and the 4B generative configuration is close enough to the ceiling
that it needs measuring before anyone relies on it.

**The M5 is promising but unproven.** Apple confirmed dedicated matrix-multiplication hardware
("Neural Accelerators", one per GPU core, claimed up to 9.5x the M1 for AI work). Two caveats: Apple
exposes them through Metal/MLX and makes **no claim of PyTorch support**, and no PyTorch release note
mentions them — so our stack may capture little of that speedup. There is also an open PyTorch bug
where MPS reports itself unavailable on macOS 26.x. It should beat the M1 on SciBERT, and its 24 GB is
a real advantage, but treat any speed claim as unproven until someone runs the smoke test on it.

*(Who runs what is for the team to decide — this section records only what each machine can do.)*

**So why still bother with Slurm?** Not for raw speed — the 3060 Ti is enough for the models we need.
Three other reasons:

1. **Our approved proposal commits to it.** It says: *"We will also use the TAU Slurm studentkillable
   partition for the Gemma arm and repeated runs."* Dropping it silently is a deviation from a plan
   the lecturer approved.
2. **The desktop belongs to one of us.** Slurm gives each of the three of us an account and storage.
   A plan built only on one person's machines leaves the other two with nothing to run, and means a
   single broken computer stops the whole project.
3. **Multiple seeds run in parallel on a cluster and one-at-a-time on a single card.**

Filing the form takes about fifteen minutes and does not oblige us to use it. It is insurance, and it
is the only item on our list with a multi-day lead time.

**One measurement constraint worth knowing.** Our research question asks whether Gemma's extra compute
is justified, and we promised to report GPU-hours. **Hours on different machines are not comparable** —
an M1 hour is not a 3060 Ti hour. So every arm whose compute cost we report has to be timed on the
same machine, at the same batch size and sequence length. The 3060 Ti is the obvious candidate.

**And here is the actual danger.** One training run of Gemma takes somewhere between **3 and 14 hours**.
Both options above interrupt you before or during that. Our code currently saves nothing while it
runs. So the realistic failure is: training starts, runs for nine hours, gets cut off, and we have
**nothing** — we start again from zero, and it can happen again the next day. That is how a fortnight
disappears without a single result.

**The fix is small and dull: make training save its progress periodically**, so an interruption costs
us minutes instead of a day. Roughly half a day of work, and it should be done before the first real
training run rather than after we lose one.

**What to do this week — each item has days of lead time and cannot be rescued in late September:**

- **Submit the Slurm request form** at `cs.tau.ac.il/system/SlurmRequestForm`, entering "NLP class
  2025/2026" as the PI/lab. The guidelines list this under *"Before you start."*
- **Accept the Gemma licence on HuggingFace** and generate an access token. Gemma is gated: without
  this the code fails with a permissions error before it loads anything.
- **Pre-download the model weights** to `/home/morg/NLP2526b/<username>`. Cluster compute nodes
  usually have no internet, so a job that tries to download the model at runtime just fails.
- **Run one job that prints `nvidia-smi`**, then a second that imports torch and puts a tensor on the
  GPU. Until that works, every date in this plan is guesswork.
- **Find out which GPUs the cluster actually has.** Colab's T4 is an older design that cannot do
  `bfloat16` arithmetic, which is a known source of Gemma producing corrupted (NaN) numbers during
  training. If TAU has newer cards, that problem disappears. One `nvidia-smi` answers it.

**Then**
- `build_split.py` — load the given split, carve validation from train, compute absolute bands,
  persist everything with hashes. Runs once, ever.
- `evaluate.py` — takes a predictions file, returns the per-band table under both denominators,
  including tuned-threshold logic.
- **Save-and-resume for training** — write the model state to disk every few hundred steps, and make
  the training script able to pick up from the last saved point. Test it by deliberately killing a job
  and restarting it; untested resume code is not resume code. This must exist before the first long
  Gemma run, not after we lose one.
- Cheap arms first (majority, TF-IDF), then SciBERT full, then SciBERT-LoRA, then Gemma-QLoRA.

---

## Part 8 — Things we should double-check

In the interest of honesty, these are the parts that are least certain:

1. **Whether their tail macro-F1 averages over 1,418 or 359 concepts — still open, and it does not
   resolve cleanly.** We have now read the paper (`Articles/AstroConcepts - Alkan et al., arXiv
   2026.md`). Their section 4 says only *"we evaluate using Macro-F1, which averages per-label F1
   scores"* and never gives the per-band denominator. Three possibilities, and we can rule out only
   one: averaging over just the 359 scoreable concepts would put DeepSeek's tail at 0.782, above its
   own torso of 0.376 — implausible, so not that. But their per-band table may equally be pooled over
   *instances* rather than averaged over labels, in which case zero-support concepts drop out on their
   own and there is no deflation at all. Their Table 5 and Table 6 do not reconcile under either
   weighting, which confirms the two use different conventions but not which.
   **So: report the asymmetry as a reporting problem, not as a 4x correction to their numbers.**
   That claim is defensible whichever convention they used, and it is the version to put in the paper.
2. **The TF-IDF 0.342 number.** Tuned on the data it was scored on, so optimistic. Needs re-running
   with a proper carved validation split before it enters the paper.
3. **Whether the preprint gets accepted somewhere before 30 September.** Worth re-checking arXiv in
   mid-September so the citation is accurate.
4. **Whether the lecturer is comfortable** with us adopting an external split and changing the
   metric. Cheap to ask, expensive to guess wrong.

---

## Part 9 — Four questions we raised, and where we landed

### Question 1 — What counts as success? How much improvement would justify Gemma's extra training time and resources?

There are two questions tangled together here, and they have different kinds of answer.

**The first is: how big does a difference have to be before it is real?** We measured this. We ran the
cheap baseline, then re-scored it two thousand times on random resamples of the test set, to see how
much the result wobbles purely from which papers happen to land in the test set. The answer is about
**1.7 percentage points**. So if Gemma beats SciBERT on rare topics by 2 points, that sits inside the
noise and we cannot tell whether it is real. Above roughly **2 to 3 points**, we can start to believe
it. This is a useful thing to know in advance, because it tells us what our experiment is capable of
detecting before we run it.

**The second question — whether the extra compute is "justified" — is not something our data can
answer.** It is a value judgement. Justified compared to what, and for whom? A researcher with a spare
GPU and a company paying cloud bills would answer differently. The lecturer is the one grading it, so
we should ask him directly what he would count as justified. That is one sentence in the email we are
already sending (Part 7).

**One thing to avoid:** we should not commit in advance to "we will call it a success if Gemma wins by
X." We tried to construct that rule and it collapsed — the thresholds required were mutually
contradictory and would have pushed almost any real outcome into "inconclusive." Instead we report the
difference with its error bar and describe what we see. A clear "these two are indistinguishable" is a
perfectly good result, and the course guidelines say so explicitly.

**For the compute side**, we record three plain numbers per model: how many parameters we actually
trained, the peak memory used, and how long one training step took. Not "GPU-hours" — an hour on a
borrowed cluster includes queue time and depends on which card we happened to get, so it is not
comparable to anything.

### Question 2 — Is the comparison fair? SciBERT returns labels directly, while Gemma produces text that must be parsed. Should we limit Gemma to returning only valid, unique labels?

There is a real unfairness here, but the part we first noticed is not the part that matters most.

**On constraining the output:** it is worth remembering that SciBERT *cannot* produce an invalid label.
Its output layer is literally the list of 1,864 topics — it chooses among them and cannot invent one.
So forcing Gemma to emit only valid topics removes a genuine mismatch. The idea is sound.

**But there is a larger unfairness underneath, and constraining the output does not touch it.** SciBERT
gives a confidence score for every topic, so we can tune the cutoff — and Part 5 shows that tuning is
worth about **25x** on rare topics. Gemma simply hands us a list of topic names: no scores, no cutoff
to tune. We would be comparing a carefully tuned model against an untuned one and calling it fair
because the text parses cleanly. That is the comparison that would actually mislead us.

**There is also a reason not to hide the format errors.** How often a fine-tuned Gemma invents topics
that do not exist, or repeats itself, is genuinely interesting — it is exactly what the preprint
authors could not measure, because they never fine-tuned their generative model. Constraining the
output first would delete our own finding.

**So our approach:** do not build the constrained-output machinery. Instead, for each paper take a
shortlist of about 100 candidate topics from the cheap TF-IDF model and have Gemma score each one.
Gemma then has scores too, gets the same tunable cutoff, and the comparison is honest. Separately, let
Gemma generate freely on a couple of hundred papers and count what goes wrong — how often it invents a
topic, how often it repeats one, how many topics it tends to produce. That is a small table, and it is
a result rather than a problem swept away.

### Question 3 — Should we run several seeds or error bars, or is that unnecessary since we take the dataset split as given?

Fixing the split removes *one* kind of randomness. There are three, and they are separate.

**Which split we use.** Fixed now, so it is no longer a source of variation — but every claim we make
is "on this split," and the paper should say that once.

**Which papers happen to be in the test set.** This is what error bars measure, by resampling. Cheap —
seconds of computation, no retraining. Worth doing on every headline number.

**Where the model randomly starts from during training.** Neural networks initialise randomly and
shuffle their data differently on each run, so training the same model twice gives slightly different
results. **Error bars do not capture this at all** — they hold one trained model fixed. This is the
real gap. Published work finds roughly a point of F1 swing from this alone, which is about the size of
the effect we are hunting. So we train the cheap neural model (SciBERT-LoRA) three times with
different seeds and report the spread in one sentence. Not for Gemma — a single run takes hours and
three is unaffordable.

**Randomness in Gemma's text generation.** We turn it off (greedy decoding) and say so. Free.

**One trap:** do not test seed variation on the TF-IDF baseline. That model is deterministic — it has
no random starting point — so it would show exactly zero variation and we would wrongly conclude that
seeds do not matter.

### Question 4 — Should we check Gemma's memory requirements in advance, since the size after quantization may not reflect what training actually needs?

**Yes, and the instinct about quantization is exactly right.** The 4-bit model size does not tell us
what training needs. Training also stores gradients, optimiser state, and intermediate activations for
every layer.

**But there is a larger memory cost that our spec does not account for at all.** Gemma's vocabulary is
about 262,000 tokens, and for every position in every paper the model produces a score for each one.
That single table is around **1.7 GB** — and the awkward part is that it is driven by vocabulary size
rather than model size, so switching from the 4B model to the 1B model **does not shrink it**. It is
likely to be the first thing that overflows.

Also worth knowing: our papers need about 640 tokens to fit. The common default of 512 would silently
truncate roughly one paper in twenty.

**That said, memory is not the thing that will hurt us. Time is.** One training run takes somewhere
between 3 and 14 hours, and both free GPU options interrupt us inside that window. Memory problems can
be fixed by making the batch smaller. Losing nine hours of training to an interruption we did not
prepare for, repeatedly, is what actually costs us the project (see Part 7).

**The test to run this week** is short: load Gemma on the real GPU, train for a few hundred steps, and
record how much memory it peaked at, how long each step took (so we can work out the hours per full
run), and whether the numbers stay valid rather than turning into NaN. That last point matters because
older GPUs such as Colab's have a known problem with Gemma producing corrupted values — and it appears
after a while, not in the first twenty steps, so the test must run a few hundred.

Then we use what it tells us. If a full run looks like four hours or less, we train the 4B model. If it
is much longer, we train the 1B model instead and use the 4B model without fine-tuning. That fallback
is genuinely fine: it is the closest thing to what the preprint did, and comparing it against our tuned
smaller model is arguably a more interesting result anyway.
