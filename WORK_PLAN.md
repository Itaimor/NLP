# Work Plan — Who Does What, and In What Order

**Written 10 September 2026, last revised 13 September. Submission: 30 September 2026.**

## The numbers that keep coming up

Every figure in this document is one of these. If you hit a bare number below and cannot remember
what it is, it is almost certainly on this list.

**Papers**

| number | what it is |
|---|---|
| **21,702** | every paper in the dataset |
| **18,677** | the **training** papers — what the models learn from |
| **3,025** | the **test** papers — what we are scored on. HuggingFace confusingly calls this split `val` (`dataset["val"]` in the code), but it is the published benchmark *test* split, and we never tune anything on it |
| **2,855** | our **validation set** — papers we carve out of the 18,677 training papers and hold back (label-stratified, seed 42), so we can choose the cutoff without ever touching the test set. The models train on the **15,822** that remain; all 1,864 topics still occur in them. Careful: this is *not* the 3,025, even though HuggingFace uses the word "validation" for those |

**Topics (labels)**

| number | what it is |
|---|---|
| **2,367** | concepts defined in the UAT thesaurus. 503 of them never appear in any paper here, so we ignore those |
| **1,864** | topics that actually appear in our data. **This is our label space** — the list every model predicts over |
| **12 / 383 / 1,469** | **head / torso / tail — the band map the results table uses.** Cut on how often each topic appears in the **18,677 training papers** (head > 500, torso 50–500, tail < 50). Alkan et al. cut their per-band results the same way (Table 6: "training examples"; their "78% of concepts have < 50" is 1,469 ÷ 1,864), so this is the map that makes our tail number comparable to their 0.198. Saved by Stage 1 as the **primary** band map |
| **17 / 429 / 1,418** | the same bands cut on **corpus-wide** counts (train + test). This is Alkan et al.'s Table 3 ("76% of concepts"); we quote it when describing the dataset. Only 56 topics change band between the two maps; the tail score moves by less than a point |
| **410** | of the 1,469 rare topics, the ones appearing at least once in the test set — the only rare topics we can actually measure (359 under the corpus-wide map) |
| **1,059** | the rare topics that never appear in test, under either map. They sit entirely in training, so **they cannot be scored at all** — 72% of the tail, and the source of the denominator problem in §9.3 |
| **805** | every topic we can measure at all, across all three bands (12 + 383 + 410) |

**Other figures that recur**

| number | what it is |
|---|---|
| **4.31** | the average number of correct topics per paper |
| **50** | how many candidate topics we hand to Gemma to choose from |
| **81.6%** | how often the correct rare topic is among those 50 on the word-counting stand-in — a proxy for Gemma's ceiling on rare topics until SciBERT's own lists exist (Stage 4). Over all bands the stand-in reaches 90.4%; Alkan et al. report ~82% pooled and no tail figure |
| **τ (tau)** | the cutoff: how confident the model must be before we count it as a "yes". One per band per arm, chosen on the validation set (rule in §4) |
| **micro-F1** | the headline score: pool every decision in a band and compute one F1. **Macro-F1** (average the per-topic F1s) is reported alongside, under both denominators (§9.3) |

---

## 1. The three dates that matter

**30 September — submission.** Fixed by the course.

**20 September — the freeze.** This is a word we should agree on because it means two different
things in our documents. When the spec says "freeze the base model," that is a technical thing about
LoRA — some weights are held fixed while training. That is *not* what this date means. **The freeze
is a calendar deadline: after 20 September we stop running new experiments and only write.** Whatever
numbers we have on the 20th are the numbers in the paper. This exists because writing eight pages
properly takes about five working days, and a project that is still running experiments on the 28th
submits an unwritten paper full of results nobody had time to explain.

**17 September — the Gemma fine-tuning giving-up date.** Gemma is two arms (Stage 5): **5a**, the
untouched model picking from SciBERT's shortlist, and **5b**, the same model after fine-tuning. Only
5b needs training, training shortlists and GPU nights, so only 5b has a giving-up date. If 5b is not
actually training by the 17th, we stop 5b and write it up as an honest negative result; 5a still runs,
because it needs nothing but the test shortlists. The course guidelines explicitly say negative
results count when the method is sound. "We attempted this, here is how far we got, here is what
stopped us" is a real section of a real paper. What is not acceptable is discovering on the 27th that
it was never going to work.

---

## 2. How the whole thing fits together

Before the stage list, here is the shape of the thing, because the stages make more sense once you
see where they sit.

Every model we build does the same job: read a paper's title and abstract, and output **a confidence
score between 0 and 1 for each of the 1,864 possible topics** (1,864 = every topic that appears
anywhere in our data; see the table above). Not a yes/no — a score. That is true of the two baselines
and of both SciBERT arms. **The Gemma arms are the one exception:** they are shown a shortlist of 50
and hand back an ordered *list of picks*, not a score per topic — Stage 5 and §6 say exactly what that
file looks like and how it is scored.

Turning those scores into an actual answer needs one more decision: **where do you draw the line?** A
score of 0.73 probably means yes and 0.02 probably means no, but the cutoff in between is a free
choice. We call that cutoff **tau (τ)**. It matters enormously — a TF-IDF test showed the score on
rare topics moving from 0.0078 to 0.324 purely by moving that line. *How* the line is chosen matters
almost as much: the same predictions score 0.156 on rare topics with one cutoff for all bands chosen
for overall micro-F1, 0.258 with one chosen for overall macro-F1, and 0.324 with a cutoff per band.
So the rule is fixed in §4 (one cutoff per band, chosen on validation to maximise that band's
micro-F1) and nobody varies it. **We never choose tau by looking at test results.** That is the
single rule most easily broken and hardest to explain away afterwards.

Then we score the results **three times over**: once for the 12 very common topics ("head"), once for
the 383 medium ones ("torso"), once for the 1,469 rare ones ("tail") — 12 + 383 + 1,469 = the 1,864
topics we predict over. Our entire research question is about whether models fail on rare topics, so
a single overall average would hide exactly the thing we are studying.

So the pipeline is: **fixed data split → model → table of scores → choose tau on validation → apply
to test → per-band results table → paper.**

---

## 3. The stages

### Stage 1 — Fix the data split, once and forever
Decide the exact division of papers into training, validation and test — then **save it to a file**
and have every later script read that file rather than recomputing it. Also saved: the fixed list of
1,864 topics in a fixed order (the label space), and the head/torso/tail grouping — two maps: the
primary one cut on the 18,677 training papers (12 / 383 / 1,469) and the corpus-wide one (17 / 429 /
1,418). Never cut bands on the 15,822-paper carve: that gives a head band of seven topics.

### Stage 2 — Build the scoring script (`evaluate.py`)
This is the ruler. It takes a table of confidence scores plus the correct answers and produces the
numbers that go in the paper: precision, recall and **micro-F1 for each of the three bands — the
headline** — with macro-F1 alongside under both denominators (§9.3); the tau search (§4 says the
rule); P@1, P@3 and P@5 and R@1, R@3, R@5; the paired bootstrap interval; and **Δ (delta)**, which is
simply "head F1 minus tail F1" — one number for how much worse a model is on rare topics than common
ones.

**What P@1, P@3 and P@5 mean.** The "@" is read as "at", so P@3 is "precision at 3". The model ranks
all 1,864 topics by confidence; you then look only at its top few guesses and ask how many are right.

Say a paper's correct topics are *solar wind*, *exosphere* and *the moon*, and the model's ranking
comes out like this:

| rank | topic the model guessed | correct? |
|---|---|---|
| 1 | exosphere | yes |
| 2 | solar wind | yes |
| 3 | black hole | no |
| 4 | the moon | yes |
| 5 | galaxies | no |

Then **P@1** = 1 right out of 1 = 1.00. **P@3** = 2 right out of 3 = 0.67. **P@5** = 3 right out of
5 = 0.60. The number after the @ is simply how far down the list you look. (R@k, recall at k, is the
mirror image: of the correct topics, how many did you catch in the top k — here R@5 = 3 of 3 = 1.00.)

We use 1, 3 and 5 because an average paper carries 4.31 topics, so 3 and 5 sit either side of that.
It is also the set Alkan et al. report, so our numbers line up against theirs.

**Why it is worth having at all: P@k needs no cutoff.** You just take the top k, so it sidesteps the
whole tau problem. One rule for Gemma, which hands back a short list of picks rather than a ranking
of all 1,864 topics: P@k counts hits among its first min(m, k) picks and still divides by k (a paper
where it emitted three topics can score at most 3/5 on P@5), R@k divides by the number of correct
topics, and nothing is padded in below its last pick. That makes Gemma's P@k partly a measure of how
*many* topics it emits — which is why the cardinality-matched row in Stage 5 is the primary
comparison and P@k the secondary one.

**The important property: this script does not care where the scores came from.** Its input is a
table of numbers. You can generate that table with a random number generator before any model
exists, and check the script behaves: a perfect prediction should score 1.0, predicting nothing should
score 0, the tail group should contain the number of topics you expect (1,469 under the primary map,
410 of them scoreable).

That is why this gets built *now*, in parallel with everything else, and not after the first model
finishes. If we train SciBERT overnight and only then start writing the scoring code, we end up with a
folder of model outputs nobody can score — and when the numbers finally look wrong, we will not know
whether the model is bad or the ruler is bent. **Build the ruler before measuring anything.**

### Stage 3 — The simple baselines, then the encoders
First the two cheap ones: a **majority baseline** (ignore the text, always predict the most common
topics) and **TF-IDF + logistic regression** (classic word-counting, no neural network). Both run on a
laptop in minutes. The course guidelines require a naive baseline explicitly, and these give us our
first real numbers in the results table — which is worth a lot for morale as much as for marks.

Then the two encoder arms from our proposal: **SciBERT fully fine-tuned** and **SciBERT with LoRA**.

**We do not train astroBERT.** It is the strongest encoder in Alkan et al. — 0.32 overall against
SciBERT's 0.21 — and it is tempting for exactly that reason. We are not doing it, for three reasons.
It is not in our approved proposal. It needs its own tokenisation pass, because it is *cased* where
SciBERT is *uncased*, so no data can be shared between the arms. And it fixes nothing that is
actually broken: our headline compares Gemma against *itself* on the same shortlist, so the identity
of the candidate generator is held fixed either way — swapping it buys prettier absolute numbers
against an unrefereed preprint, not a more valid claim. We cite astroBERT's published numbers as the
bar we are measured against. We do not train it.

### Stage 4 — Measure Gemma's ceiling

**The point of this stage in one sentence: work out the highest score Gemma could possibly get,
before it runs.**

Gemma only ever sees a shortlist of 50 candidate topics. Any correct topic missing from that list is
one Gemma cannot possibly get right — it is not being tested, it is being denied the option. So the
fraction of correct topics that make it onto the shortlist is a hard cap on Gemma's score, fixed
before Gemma does anything. Producing that number is all this stage is.

**Most of it is already done.** Using a crude word-counting shortlist I measured **81.6%** for rare
topics at depth 50 (72.0% at 25, 89.2% at 100), and 90.4% over all bands — see §9. That was enough
to settle two arguments: the shortlist is not the weak link, so we do not need astroBERT (§9.1), and
50 is a sensible depth (§9.2).

**What is left is one re-measurement, after SciBERT trains.** Our real shortlist comes from SciBERT,
not from word counting, so the true ceiling is SciBERT's number and nobody has it yet. Once SciBERT is
producing top-50 lists, count the same thing again on those. It is minutes of work on output we will
already have.

It matters for two reasons. It is the ceiling we have to quote in the paper — every Gemma result has
to be read against it, or a reader cannot tell a weak model from a starved one. And it is a go/no-go
signal: if SciBERT's coverage of rare topics came back low, Gemma's rare-topic score is capped near
that level no matter how well it re-ranks, and we would want to know before spending nights on it.

**This is not a depth experiment.** We train at one length only, 50. The 25 and 100 figures describe
the shortlist and go in a small table that justifies our choice of 50; they are never fed to Gemma.

### Stage 5 — Gemma re-ranking SciBERT's shortlist: 5a untouched (committed), 5b fine-tuned

SciBERT scores all 1,864 topics and its 50 best guesses become a shortlist. Gemma reads the paper and
picks from those 50. We run this **twice with the same Gemma** — once untouched, once fine-tuned —
on the same papers and the same shortlists, so the only difference is whether we updated the weights.

**Careful: this is not a contest between Gemma and SciBERT**, even though the proposal is worded that
way. SciBERT is inside both arms. §9.4 explains what we are actually measuring and how to describe it
in the paper — read it before writing any results prose.

**The two arms have different needs, so they are two sub-stages.**

- **5a — Gemma untouched.** Needs SciBERT's top-50 for the 3,025 *test* papers and for the 2,855
  *validation* papers (handoff 4, Wed 16 Sept). No training, no GPU night — one inference pass. This
  is the arm we commit to. On its own it answers the *first* half of the question in §9.4 — does an
  untouched generative model select from SciBERT's shortlist better than SciBERT's own cutoff. It does
  not answer the question our proposal actually asks, which is about parameter-efficient
  *fine-tuning*; that half is 5b.
- **5b — Gemma fine-tuned.** Needs one more file: SciBERT's top-50 for the papers Gemma *trains* on,
  with the training target for each (handoff 4b, same inference pass, Wed 16 Sept), plus a training
  night on the card and a second inference pass. The 17 September giving-up date governs this arm,
  with Alternative B (below) as its safety net through the freeze. If neither lands, 5a still stands
  (re-run at 4B) and the paper reports 5b as attempted, with the wall-clock and the error that stopped
  it.

**How Gemma picks: prompt-and-select, not log-probability.** We put the paper and the 50 candidates in
the prompt and have the model output which ones apply — what DeepSeek did in Alkan et al. Far less
code, and its failures are visible: either the output parses or it does not. The cost is that a
generated set has no per-label score, so this arm cannot use a tuned tau. Two rules: **at validation
and test time both arms see the 50 candidates in SciBERT's order** (only 5b's *training* lists are
shuffled — see below), and Gemma lists its picks most-confident-first, so that a pick at position *r*
can be given the score 1 − r/51 for P@k (counted as Stage 2 says). The parser maps each output line
to a candidate by case-insensitive exact match; a line that matches nothing is counted, not silently
dropped — **off-list rate and empty-output rate are columns in the results table, per arm.**

**Everything about Gemma is developed on the validation papers, never on test.** The prompt wording,
the parser, `max_new_tokens`, the `NONE` rule, the 4B-or-1B choice and (for 5b) which checkpoint to
keep are all decided from outputs on the 2,855 validation papers' shortlists. The test lists are run
once per arm, with the prompt's commit hash logged before the run. A prompt is very easy to tweak
after seeing a result, which is why this is written down.

**What 5b trains on, and the trap in it.** Fine-tuning needs examples of the form *(paper, its 50
candidates) → the correct topics among those 50*. The candidates come from SciBERT. But SciBERT's
shortlists for its **own training papers** are much easier than its shortlists for unseen papers: on
papers it trained on, the correct topic is in the top-50 essentially always and sits near the top
(measured on the word-counting stand-in: 100% coverage, median rank 2–3); on test papers it is there
about 75–82% of the time and sits at rank 8–15. A Gemma trained only on the easy lists learns "the
answer is always here, near the top" — and at test time, where that is false, it over-picks and
misses. This does not leak the test answers to Gemma — nothing about the test set reaches it. It is
the opposite problem: it handicaps 5b, because the lists it learned on look nothing like the lists it
is scored on.

This is a known situation. Two-stage systems in retrieval, entity linking and extreme multi-label
classification routinely train the second stage on the first stage's lists for the training set, with
the correct answer guaranteed present: DPR's reader (Karpukhin et al., EMNLP 2020), FiD (Izacard &
Grave, EACL 2021), BLINK — which reports its own train/test coverage gap, 93% vs 82%, and trains that
way regardless (Wu et al., EMNLP 2020, Table 1) — X-Transformer (Chang et al., KDD 2020),
XR-Transformer (Zhang et al., NeurIPS 2021), AttentionXML (You et al., NeurIPS 2019), and the LLM
re-rankers RankLLaMA (Ma et al., SIGIR 2024), ListT5 (Yoon et al., ACL 2024) and FIRST (Reddy et
al., EMNLP 2024). What we do about the skew:

1. **Shuffle the candidate order in 5b's *training* prompts.** This is what the fine-tuned listwise
   re-rankers do — ListT5 "randomly shuffle[s] the positive and negative passages and assign[s]
   identifiers {1, ..., m} to them to form each training data" (Yoon et al., ACL 2024, §4.1), and
   training with shuffled lists is what makes a re-ranker insensitive to input order (Tang et al.,
   NAACL 2024). Trained on shuffled lists, the model stops learning "the answer is near the top".
   **At validation and test time both arms see SciBERT's order.** An untouched LLM's ranking is
   "highly sensitive to the initial passage order" — on TREC DL19, nDCG@10 falls from 65.8 with the
   retriever's order to 25.2 with a random order (Sun et al., EMNLP 2023, Table 5) — and ListT5's
   Table 5 shows shuffling the retriever's order at test costs an untouched model 8.8 points but a
   shuffle-trained one 1.7. Shuffling the *test* lists would therefore cripple 5a while barely
   touching 5b, and the comparison would flatter fine-tuning. The papers evaluate on the first-stage
   order "as is standard" (Tang et al., §3.2); so do we. Robustness check, one inference pass: score
   5b on shuffled test lists as well; the difference should be within a point or two.
2. **The training target is "the correct topics that are in the list"** (gold ∩ shortlist), written
   verbatim, one per line, **ordered by SciBERT's score, highest first**, or the literal word `NONE`
   if none of them made the list. The re-ranker papers train on an ordered target (Sun et al., EMNLP
   2023, §4.1; Yoon et al., ACL 2024); ours is ordered for the same reason — the "most-confident-first"
   instruction needs a training signal. Restricting the target to on-list names follows from our
   parser accepting only names that are on the list. At scoring time Gemma is scored against *all*
   correct topics, so a correct topic missing from the list counts as a miss — same rule as every
   other arm. (`NONE` happens in well under 1% of examples.)
3. **Measure what is left, after the fact — do not correct it in training.** Shuffling removes the
   position problem. What remains is that training lists always contain the correct topics and test
   lists do not (about 18% of rare-band correct topics are missing), so 5b may emit slightly too many
   labels. That is a small effect (around one F1 point, and it works *against* 5b), and it is
   directly visible in the test predictions, so we report it rather than pre-correct it: for each
   arm, how often it emits a rare-band label on papers whose rare-band correct topics are *all*
   missing from the list (the "substitution" rate), and how often it picks the correct topic as a
   function of where SciBERT ranked it. Plus, at the end of each training epoch, log the average
   number of labels emitted divided by the number of correct labels in the list, on 50 held-out
   training-style lists and on 50 validation lists — if that ratio is well above 1 on the validation
   lists, 5b is over-emitting and a weak result is not a real null.

   We do **not** use "gold-dropout" — deleting a correct topic from a share of training lists so
   they have holes like the test lists — in the main run: it spends rare-topic training examples on
   the very band we are studying (the 410 measurable rare topics have a median of 25 training papers
   each), the right rate is not known until SciBERT's lists exist, and applied to 5b alone with no
   comparison run it could not be separated from the fine-tuning itself. Its nearest precedent, RAFT
   (Zhang et al., 2024), is a preprint. If a GPU night is left after the main runs, it is a worthwhile
   ablation row, described as our own variation.

Report a small **skew table** in the appendix, the way BLINK reports its train/test recall gap: for
the training lists as used, the validation lists, and the test lists — how often the correct topic is
in the top-50, its median rank, the share of correct topics that made the list, the share of lists
with no rare-band correct topic in them, and the band mix of the *wrong* candidates. It is produced
the day the lists land, from the same files, before any GPU is spent. If SciBERT's training lists
turn out much less skewed than the word-counting stand-in (rare-band coverage below ~90%), this whole
section is a footnote.

**Alternative B — the fallback, not the first choice:** train 5b on the 2,855 validation papers
instead of the 15,822 training papers. SciBERT never saw them, so their shortlists look exactly like
test shortlists, which fixes every part of the skew at once — but it leaves about 4 training examples
per measurable rare topic instead of ~25, and it uses up the validation set. If it is used, hold 300
of the 2,855 papers out as Gemma's development slice (prompt, parser, checkpoint choice) and train on
the other 2,555. Use it only if a full pass does not fit the calendar; if the choice is 4B on the
2,855 or 1B on the full set, take 1B on the full set. **But try it before declaring 5b dead:** a 1B
fine-tune on 2,555 lists is one evening on the card, and its lists come from the same inference pass
as the test lists. If a night is free at the end, the most informative extra run is 1B on 2,555
randomly chosen training papers versus 1B on the 2,555 validation papers, same settings — that
measures the whole skew with data size held fixed.

**One rule for this paper: train Gemma on the same generator's lists it is scored on.** A re-ranker
trained on one retriever's candidates and scored on another's degrades (Gao, Dai & Callan, ECIR 2021,
Fig. 2). If SciBERT's lists are late and TF-IDF's are used, they are used for training *and* test,
and the paper says so. Switching generators after 5b has started training means 5b starts again.

**Why Gemma may lose — and why that is fine.** Three honest reasons: Gemma inherits every mistake in
the shortlist and can never recover a correct topic SciBERT left out (about 18% of rare-band correct
topics, before Gemma is prompted — "the upper bound of the ranking effect is contingent upon the
recall of the initial passage retrieval", Sun et al., EMNLP 2023); Gemma is a general language model
being asked to out-rank a model trained on exactly this vocabulary; and a measurable rare topic has a
median of ~25 training papers, which is little for fine-tuning to move. So the real question is not
"does Gemma win" but **"does fine-tuning help or hurt, compared with the same model untouched?"** —
and *hurt* is a perfectly reportable answer. Build 5b so that either answer is believable.

**Model size is decided by the probe, not chosen in advance.** The real prompt carries 50 candidate
names and runs to roughly 1,000 tokens, so the probe trains Gemma-3-4B in 4-bit on the 3060 Ti at
**1,024 tokens, batch 1, for at least 500 steps** (Gemma 3's overflow problems can appear late),
logging peak memory and checking the loss stays finite. The same run trains on 64 papers until it
overfits them — the sanity check the course guidelines ask to be shown; save the loss plot. If 4B
fits, use it. If not, 1B — and then both arms are 1B, so that the only difference between them is
still the weights. The probe is a *training* probe; 5a does not train, and 4B in 4-bit runs
inference comfortably on the card. So if 5b is cut, re-run 5a at 4B (one inference pass).

**Four things the results table must carry** (§9.4 says why the first three are needed):

1. A **cardinality-matched SciBERT row** — per paper, take as many of SciBERT's best guesses as Gemma
   emitted, and score those. Computed separately for 5a and 5b, each at its own count.
2. A **mean-|ŷ| column** on every row — the average number of topics that arm emits.
3. The **recall@50 ceiling**, stated in Methodology before any results (Stage 4).
4. **Compute cost, logged from the first run** — the proposal promises it ("trainable parameters,
   GPU-hours") and the research question's second half has no answer without it. For every trained
   arm: trainable parameters, peak memory, seconds per step on the machine it trained on, and a
   **200-step timing slice on the 3060 Ti** at the arm's own batch size and sequence length, so every
   arm has a number on one declared device (ten minutes each for the SciBERT arms).

Also report P@1/3/5 and R@1/3/5, counted for Gemma as Stage 2 says, as the secondary comparison.

**The code does not wait for SciBERT.** The prompt, the parser and the inference loop are written and
smoke-tested against a stand-in shortlist — TF-IDF's top-50 for the *validation* papers — then
pointed at SciBERT's real lists the moment they exist. The stand-in buys a working parser, a parse
rate and a seconds-per-paper figure; nothing tuned on stand-in lists is kept. Run 5a the day handoff 4
lands; run 5b the day handoff 4b lands.

**One optional extra, if time allows:** swap the shortlist generator — SciBERT's top 50 versus
TF-IDF's top 50 — which isolates how much the generator's quality mattered. Skip it if Gemma is late.
Varying the shortlist depth was considered and rejected — see §9.2.

**The papers this stage relies on, with venues** — cite the peer-reviewed ones as evidence; the
preprints only as what they are:
- Sun et al., "Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents",
  EMNLP 2023 — order sensitivity of an untouched LLM re-ranker; the first stage's recall ceiling.
- Yoon et al., "ListT5: Listwise Reranking with Fusion-in-Decoder Improves Zero-shot Retrieval",
  ACL 2024 — shuffled training lists; cost of shuffling the retriever's order at test.
- Tang et al., "Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large
  Language Models", NAACL 2024 — positional bias; evaluation on the first-stage order.
- Ma et al., "Fine-Tuning LLaMA for Multi-Stage Text Retrieval", SIGIR 2024; Reddy et al., "FIRST",
  EMNLP 2024 — LLM re-rankers trained on first-stage lists of the training queries.
- Karpukhin et al. (DPR), EMNLP 2020; Izacard & Grave (FiD), EACL 2021; Wu et al. (BLINK), EMNLP
  2020; Gao, Dai & Callan, ECIR 2021; Chang et al. (X-Transformer), KDD 2020; Zhang et al.
  (XR-Transformer), NeurIPS 2021; You et al. (AttentionXML), NeurIPS 2019.
- Zhu & Zamani, "ICXML", Findings of NAACL 2024 — an LLM selecting labels from a candidate set in
  extreme multi-label classification.
- Preprints, not peer-reviewed: Alkan et al. 2026 (the concurrent work whose design we follow);
  RankVicuna and RankZephyr (Pradeep et al., 2023); RAFT (Zhang et al., 2024).

### Stage 6 — What Gemma does when left unconstrained
Let Gemma — the fine-tuned one if 5b lands, otherwise the untouched one — generate topic names freely
on about 200 papers, and measure what comes out:
how often it invents topics that do not exist, how often it is nearly right (a spelling or plural
variant), how often it repeats itself. If it is the fine-tuned model, nobody has measured this on
this vocabulary, because nobody has fine-tuned a generative model on it; if only the untouched model
ran, Alkan et al. already report that free prompting "hallucinated concepts", and our numbers put a
rate on that. Either way it is cheap — inference only, no training — and it is the paragraph that
justifies the shortlist design (§9.4).

### Stage 7 — The paper
Eight pages, ACL template, in Overleaf. Plus the required AI Disclosure and Reflection section.

**All three of us write.** Nobody hands their results to somebody else to write up — whoever ran an
experiment writes that section, because they are the one who can explain it, and the oral tests
exactly that. The split:

| section | who | why |
|---|---|---|
| Introduction, abstract, related work, dataset | Ilana | has done most of the work so far and holds the reading notes |
| Method + results for SciBERT, full and LoRA | Itai | ran them |
| Evaluation methodology, the threshold and denominator findings, baselines | Shai | built the ruler and the baselines |
| Method + results for Gemma, and the unconstrained-generation study | Ilana | ran them |
| Discussion, limitations, conclusion, AI disclosure | all three, together at the end | — |

**Worth remembering: 60 of the 100 marks need no experiments at all** — research question (10),
ambition (10), literature review (20), and presentation (20). Forty of those marks are completely
unstarted right now, and we already have 40,000 words of reading notes in `Articles/`. That work can
be finished before a single model finishes training, and it should be.

---

## 4. Who does what

The split is built around one fact: **Stage 2 needs Stage 1's five files and nothing else — never a
model.** The scoring script can be written and fully tested against fake numbers while the models do
not exist. That is what lets three people work at once instead of one working and two waiting. The
best SciBERT epoch is chosen by Stage 2's tau search, not by validation loss — but the training loop
does not need to import it: the loop saves every epoch's validation probability table and weights,
and `evaluate.py` picks the best epoch afterwards (see the tau row below). So SciBERT can start
training the moment its front end exists; the ruler only has to be finished before the scoring.

**The paper is shared work.** Each of us writes up what we ran — the split is in Stage 7. The lanes
below are about the experiments, not about who writes.

### Ilana — the foundation, Gemma, and the introduction

**You own Stage 1, Stages 5 and 6 (all of Gemma), the graphics card, and the introduction.**

Start with `build_split.py` — about 45 minutes: `main.py::split_train_validation` already computes
the split with seed 42, so this is saving what exists. Save the split, the topic list in a fixed
order, the two band maps (training-basis 12 / 383 / 1,469, marked primary; corpus-basis 17 / 429 /
1,418) and the id↔name map to disk with checksums, commit, and then re-run the §9 measurement scripts
on the saved split and the primary map so the numbers in this document and the paper come from the
same files everyone else reads.

You can start Gemma now without waiting for SciBERT: build the Gemma code against a stand-in
shortlist — TF-IDF's top-50 for the **validation** papers, never the test papers — to prove the
machinery runs, then swap in SciBERT's real candidates the moment they exist. The first thing to
learn, before any card time: does untouched Gemma-1B even parse — twenty validation papers on the
M1, under an hour, giving the parse rate, off-list rate and mean number of picks. If 1B is
degenerate (emits nothing, everything, or off-list names), the whole Gemma plan is 4B-or-nothing.

**Gemma is yours end to end** — the prompt, the parser, the
inference loop, the frozen-vs-fine-tuned pair, the three required table entries in Stage 5, and the
unconstrained-generation study in Stage 6. Read §9.4 before you write any of the results prose — the
obvious way to describe this experiment is wrong.

**The introduction is yours, and so is the framing of the paper.** You have done most of the work so
far and you hold the reading notes, so the introduction, abstract, related work and dataset
description are yours to write — plus the Gemma sections, since you ran them. Get the Overleaf
skeleton up this week and start on the literature review from `Articles/`. None of that needs a single
result to exist. The rest of the split is in Stage 7.

**Two short jobs this week that unblock Gemma, both yours because the card is yours:**

- **Accept the Gemma licence on HuggingFace and generate an access token.** Gemma is a gated model —
  without this the code fails at the first line that tries to load it. Approval is near-instant, so
  this is five minutes, but it is five minutes that otherwise costs somebody an evening.
- **Run the 4B probe** (Stage 5 says how). **This is what decides whether 5b uses 4B or 1B.** Budget
  an evening: the card has never run torch + bitsandbytes, so the first hour is an environment build
  and an ~8 GB download.

### Shai — the measuring stick and the baselines

**You own Stage 2 (`evaluate.py`), the baselines in Stage 3, and Stage 4.**

Everything in this lane runs on your laptop in minutes, so nothing here waits on hardware.

Start with fake data. Write `evaluate.py` so it takes a table of scores and a table
of correct answers and produces the full results table: per-band precision, recall and **micro-F1
(headline)** with macro-F1 under both denominators; the tau search on validation (the rule is in the
tau row below — one cutoff per band, maximising that band's micro-F1); P@1/P@3/P@5 and R@1/R@3/R@5;
the paired bootstrap; and delta. Test all of it against a randomly generated score table before any
model exists. You are done when you can hand it a perfect prediction and get 1.0, hand it nothing
and get 0, and the band sizes come out as **12 / 383 / 1,469** from the primary band map (and 17 /
429 / 1,418 from the corpus map — the script must read whichever map it is given, never hard-code
either).

Then the two baselines — majority and TF-IDF — which run on your laptop in minutes and give us the
first real rows of the results table.

Then Stage 4, the recall ceiling count, which is pure arithmetic on Itai's test table (handoff 5) and
tells us the highest score Gemma could possibly reach.

And one job that is easy to skip because it is not a model: **prove the ruler on real bytes before the
first SciBERT run.** Run TF-IDF, write its validation and test tables in the §6 format, score them
through `evaluate.py`, regenerate the row from the files. If that works by Monday night, every later
arm is a file drop.

**One thing to get right that is easy to get wrong: choose tau on the validation set, never on test.**
Picking the threshold by trying values on the test set and keeping the best is the leakage mistake the
rubric names explicitly, and it inflates every number that follows.

### Itai — the encoders

**You own Stage 3's encoder arms — SciBERT full and SciBERT LoRA — code and runs both. This is the
critical path.** Nothing downstream can start until SciBERT produces scores, and Gemma cannot start at
all until its top-50 candidate lists exist.

**The first job in this lane is a 20-step smoke test on your M5.** Load SciBERT, run twenty
training steps on a handful of papers, and report the seconds per step. This is not a formality. Our
briefing records that Apple makes *no claim of PyTorch support* for the M5's matrix hardware, and there
is an open PyTorch bug where MPS reports itself unavailable on macOS 26.x. If that bites, your M5
silently falls back to CPU and is not the fast machine the plan is counting on. Ten minutes, and it
decides where SciBERT trains.

**The training engine already exists.** Shai wrote it — the training loop, `evaluate()` returning
probabilities so several thresholds can be swept without re-running, and working checkpoint and
resume. Take an hour with him this week and walk through it. What is missing is the front end: the
tokenizer, the dataset object that hands the model batches, and the code that turns a list of topic
names into the yes/no vector the model trains against. Then fill in `run_SciBERT_experiment`, which is
currently only comments.

Two things in the existing code to settle before the first long run:

- **The best epoch is currently chosen by validation loss.** Our headline number is F1 on rare topics
  at a tuned tau. With 1,864 topics to score and only about four correct per paper, the loss is
  dominated by the vast number of easy "no" answers, so the epoch with the lowest loss is not
  reliably the epoch with the best F1. The fix: **save every epoch's validation probability table
  (2,855 × 1,864 floats, ~21 MB) and that epoch's weights, and let Shai's `evaluate.py` choose the
  best epoch afterwards with `choose_tau`.** The training script then never needs a tau and can
  start before the ruler is finished. Delete the pseudo-code steps 5.5–5.7 in
  `SciBERT_experiment.py`, which choose tau inside the training script. There must be exactly one
  function in the project that picks a threshold, because "who chose the threshold, and where?" is a
  question that will be asked, and the answer has to be the same from all three of us.
- **`num_epochs` defaults to 20.** The paper we are comparing against found its best at 8. Twenty is a
  lot of hours to spend discovering we overfit at six.
- **`run_SciBERT_experiment` loops over configurations.** We cannot afford a grid: 8 epochs × ~1.3 h ×
  2 arms is already ~21 hours. One configuration per arm — the paper's (lr 2e-5, batch 8, ≤8 epochs).
- **Log the cost from the first run.** Seed, learning rate, batch size, epochs, trainable parameters,
  seconds per step and peak memory, written to a file next to the checkpoint — and, once the card is
  free, a 200-step timing slice of each arm on the 3060 Ti (Stage 5, item 4). The proposal promises
  this and it cannot be reconstructed afterwards. Also save the full validation *and* test
  probability tables in the §6 format — right now `best_model.pt` keeps validation probabilities as
  tensors and nothing writes the test table at all.

**Three shortlist files, not one.** Ilana's Gemma arm waits on SciBERT-full's top-50 lists, and they
are three files with two deadlines: the lists for the 3,025 **test** papers and for the 2,855
**validation** papers (handoff 4, **16 September** — test is what 5a is scored on, validation is
where the prompt and parser are developed and where 5b's checkpoint is chosen), and the lists plus
training targets for the papers 5b **trains** on (handoff 4b, **17 September** — the 15,822 training
papers, or the validation papers under Stage 5's Alternative B). All are one inference pass over
papers the model has already seen or held out; none needs any new training. Stage 5 explains why the
training file needs the gold ∩ shortlist targets, ordered by SciBERT's score, next to each list.

### What depends on what

Most of the work runs in parallel, but three things genuinely block other people. If one of these
slips, somebody else sits idle — so they are worth naming rather than discovering.

**The whole workflow, drawn:**

```
      ┌──────────────────────────────────────────┐   ┌──────────────────────────────┐
      │  ILANA · build_split.py · Sun 13 Sept    │   │  ILANA · Overleaf skeleton   │
      │  split + label order + 2 band maps + ids │   │  + stubbed results table     │
      └────────────────────┬─────────────────────┘   │  · Sun 13 Sept · grows daily │
                           │  five files; everything reads them           ▲
           ┌───────────────┼───────────────────┐                          │
           ▼               ▼                   ▼                          │
 ┌──────────────────┐ ┌──────────────────┐ ┌────────────────────┐         │
 │ SHAI             │ │ ITAI (or SHAI,   │ │ ILANA              │         │
 │ evaluate.py      │ │  Mon 20:00 rule) │ │ 1B parse test·M1   │         │
 │  choose_tau()    │ │ M5 smoke test    │ │ 4B probe · Mon 14  │         │
 │  on §6 examples  │ │ SciBERT front end│ │ Gemma prompt+parser│         │
 │ majority+TF-IDF  │ │                  │ │  on VAL stand-in   │─────────┤
 │  through the     │ │                  │ │ lecturer email     │         │
 │  ruler · Mon 14  │ │                  │ │ related work draft │         │
 └────────┬─────────┘ └────────┬─────────┘ └─────────┬──────────┘         │
          │  scores after      │                     │                    │
          └───────────────────►▼                     │                    │
                     ┌──────────────────┐            │                    │
                     │ SciBERT-full     │            │                    │
                     │ trains Tue 15    │            │                    │
                     │ night (LoRA aft) │            │                    │
                     └────────┬─────────┘            │                    │
              ┌───────────────┼─────────────────┐    │                    │
              ▼               ▼                 ▼    │                    │
   ┌───────────────┐ ┌────────────────┐ ┌──────────────────┐              │
   │ handoff 5     │ │ handoff 4      │ │ handoff 4b       │              │
   │ full val+test │ │ VAL+TEST top-50│ │ TRAIN top-50     │              │
   │ tables·Wed 16 │ │ · Wed 16       │ │ + targets·Wed 16 │              │
   └──────┬────────┘ └───────┬────────┘ └────────┬─────────┘              │
          │                  ▼                   ▼                        │
          │        ┌──────────────────┐ ┌──────────────────┐              │
          │        │ 5a Gemma frozen  │ │ 5b Gemma tuned   │              │
          │        │ · Wed 16 night   │ │ Thu 17–Sat 19 or │              │
          │        └───────┬──────────┘ │ Alt B · or cut   │              │
          │                │            └────────┬─────────┘              │
          │                ├◄────────────────────┤ (whichever ran)        │
          │                ▼                     ▼                        │
          │        ┌──────────────────┐                                   │
          │        │ Stage 6 · free   │                                   │
          │        │ generation       │                                   │
          │        └────────┬─────────┘                                   │
          ▼                 ▼                                             │
 ┌─────────────────────────────────────────────────────────────┐          │
 │ SHAI scores EACH ARM THE DAY IT LANDS · results table grows │──────────┘
 └─────────────────────────────────────────────────────────────┘   each row → its section
```

**The critical path is the middle column, and it is shorter than it was.** Ilana's split → SciBERT-full
→ the *test* top-50 file → 5a is the committed chain, and it is done on the night of Wednesday the
16th. The *training* top-50 file → 5b is a second chain with its own deadline; it is the one that
answers the proposal's fine-tuning question, so it is not "optional" — it is the one we can lose to
the calendar, and Alternative B is its safety net. A slip in the middle column now costs 5b, not all
of Gemma. The paper is not at the bottom: it is the box on the right, started on day one, that every
finished row is written into.

**The handoffs, in order:**

| # | From | To | What | Needed by |
|---|---|---|---|---|
| 1 | Ilana | everyone | `split.json` (15,822 / 2,855 / 3,025), `label_order.json`, `band_map_train.json` (cut on the 18,677 training papers — 12 / 383 / 1,469 — **primary**), `band_map_corpus.json` (17 / 429 / 1,418, for the dataset description and Alkan et al.'s Table 3), `id_to_name.json` — five files, committed with checksums | **Sun 13 Sept** — before anyone writes real code |
| 2 | all three | — | the §6 output format, as a schema plus a committed three-row example file | before the first long run |
| 3 | Shai | Itai | walkthrough of the training engine he wrote | before Itai writes the front end |
| 4 | Itai | Ilana | **`scibert_full_top50_test.jsonl`** and **`scibert_full_top50_val.jsonl`** — SciBERT-full's top-50 for the 3,025 test papers and the 2,855 validation papers, in SciBERT's order | **Wed 16 Sept → Stage 5a** (and Gemma development on the val file) |
| 4b | Itai | Ilana | **`scibert_full_top50_train.jsonl`** — SciBERT-full's top-50 for the 15,822 training papers, with gold ∩ shortlist beside each, ordered by SciBERT's score | **Wed 16 Sept, same inference pass → Stage 5b only** |
| 5 | Itai | Shai | full **validation and test** probability tables per encoder arm, §6 format | within 24 h of each arm finishing |
| 6 | Ilana | Shai | Gemma's ordered output lists per arm, §6 format — validation and test files | within 24 h of each arm finishing |

**Two measurements gate decisions:**

- **Itai's M5 smoke test** decides where SciBERT trains (fifteen minutes once a tokenizer exists).
- **Ilana's 4B probe** decides Gemma's model size (an evening).

**If an owner is not there, the work moves — by rule, not by meeting.** Ownership above is by
capability; this is the rule for presence. **Monday 14 September, 20:00:** if there is no M5
smoke-test log and no front-end commit from Itai, then on Tuesday Shai writes the SciBERT front end
(he wrote the engine; it is the shortest path), SciBERT-full trains on Ilana's M1 from Tuesday night
(1.3 h/epoch × 8 = one night and a morning, done Wednesday), and SciBERT-LoRA moves to 17–20
September or is cut. If that happens, whoever ran SciBERT writes it up — the Stage 7 table follows
the runs. Nobody has to decide this on the day; it is decided now. And if the M5 smoke test shows
more than an hour per epoch or MPS unavailable, do not wait for Wednesday: cap SciBERT at 4 epochs
and, if the lists still cannot land by the 16th, switch handoff 4/4b to TF-IDF's top-50 for train,
validation *and* test (Stage 5's same-generator rule) so that Gemma starts on time.

**Who decides which parameters:**

| parameter | who | how it is decided |
|---|---|---|
| **tau — the yes/no cutoff** | **Shai**, as one importable function `choose_tau(val_probs, val_labels, band_map) -> {band: tau}` in `evaluate.py`, for every scoring arm | **One cutoff per band, chosen on the validation set to maximise that band's micro-F1 over the topics that have at least one validation instance, from a fixed grid (0.005–0.10 in steps of 0.005, then 0.10–0.55 in steps of 0.02), then applied to the test set once.** The function writes `tau_<arm>.json` (the three values, the objective, the validation file's checksum, the commit) so a reader can see who chose the threshold and from what. Why the rule is this precise: on the same TF-IDF predictions, one cutoff for all bands chosen for overall micro-F1 gives a rare-band macro-F1 of 0.156; chosen for overall macro-F1, 0.258; one cutoff per band, 0.324 — a 2x swing, bigger than any difference we expect between arms. The pooled version lands the cutoff at 0.10, where the torso is happiest and the tail is starved — the operating-point artefact §9.3 accuses the reference paper of. A single global cutoff (macro objective) is reported once as a robustness row. |
| SciBERT's learning rate, batch size, number of epochs | Itai | he trains it. Cap epochs at 8, not the current default of 20. One configuration per arm, no grid |
| which epoch counts as "best" | **Shai's `evaluate.py`, after training**, from the per-epoch validation tables Itai's loop saves | best rare-band micro-F1 at the tau `choose_tau` returns for that epoch's validation probabilities — not lowest validation loss. The training script never picks a tau. See Itai's section |
| Gemma's model size, 4B or 1B | Ilana | decided by the probe, not chosen in advance; if 5b is cut, 5a is re-run at 4B |
| Gemma's prompt and decoding settings | Ilana | greedy decoding; developed on validation shortlists only; whatever is used gets stated in the paper with its commit hash |
| 5b's checkpoint | Ilana | the epoch with the best rare-band micro-F1 on the **validation** shortlists (one inference pass per epoch on the 2,855 validation lists, or on the 300-paper hold-out under Alternative B) |
| the shortlist depth, 50 | already fixed | §9.2. Not a parameter we tune |
| the frequency bands | already fixed | Stage 1, saved to disk and read by everything: 12 / 383 / 1,469 on the 18,677 training papers for the results table; 17 / 429 / 1,418 corpus-wide for the dataset description |

**The Gemma arms have no tau at all**, because prompt-and-select emits a list of labels rather than a
score per topic. That is the reason Stage 5 needs the cardinality-matched row — it is what lets a
thresholded arm and an unthresholded one be compared at all. (The 1 − r/51 rank score in Stage 5 is
for P@k only; it is never thresholded.)

**What does not depend on anything, and can start immediately:**

- Shai's `evaluate.py`, built and tested against randomly generated scores.
- Shai's majority and TF-IDF baselines — they need only the split.
- Ilana's Gemma code, smoke-tested against a stand-in shortlist.
- Ilana's introduction, related work and dataset sections. No result is needed for any of them.

**If you are blocked, this is what to do instead.** Itai waiting on the split: read the training
engine with Shai. Ilana waiting on top-50 lists: write the paper. Shai waiting on model outputs: he
never is — his whole lane runs on fake data first, which is the point of building the ruler before
anything is measured.

---

## 5. What we are cutting, and why

**The log-probability scoring machinery — not the re-ranking design.** The spec's original plan was
for Gemma to score each candidate by a length-normalised likelihood with a cached prefill. The idea is
sound, but its failures are silent: when it is wrong it does not crash, it returns numbers that look
completely reasonable and are simply incorrect. With twenty days left we cannot afford an experiment
that can lie to us quietly. **We keep the shortlist and score it by prompting instead** (Stage 5).

**Three random seeds → one.** Nice to have, not affordable. The paper says so in one sentence in
Limitations — the rubric asks that randomness be accounted for, and "one seed, paired bootstrap over
the test papers for the sampling variance, seed variance not estimated" is an honest account; silence
is not.

**The Slurm cluster → not used.** The proposal named it; the three machines we own cover every arm,
and the desktop is the one declared timing device. One sentence in the paper, and a line in the
lecturer email.

**Also not in the plan:** a prompt-sensitivity experiment for Gemma (Limitations states that the
prompt is fixed and gives it verbatim in the appendix); a shortlist-depth sweep (§9.2); training
astroBERT (§9.1).

---

## 6. The one thing we must agree on first

Before anyone writes code: **agree the format of a model's output — as files, not as a sentence.**

**Encoder arms (majority, TF-IDF, SciBERT-full, SciBERT-LoRA)** — one file per arm *per split*, so
two files each: `<arm>_val.parquet` and `<arm>_test.parquet`. One row per paper; a `paper_id` column;
then one column per topic, values between 0 and 1, columns in the order of `label_order.json` from
Stage 1. Both files are required: tau is chosen on the validation file and applied to the test file,
so an arm that only writes its test table cannot be scored honestly.

**Gemma arms (5a, 5b)** — two files per arm, like the encoders: `<arm>_val.jsonl` and
`<arm>_test.jsonl`. One line per paper: `paper_id`, the 50 candidates *as they were shown* — which
at validation and test time is **SciBERT's order, never shuffled** — and Gemma's output as an
**ordered** list of candidate names, most-confident first, plus any raw output lines that matched no
candidate (kept, so the off-list rate can be computed). The scorer turns position *r* into the score
1 − r/51 for P@k — counting hits only among the picks actually made, never padding below the last
one (Stage 2) — and treats the list as the predicted set for F1. It also needs each arm's per-paper
count to build the cardinality-matched SciBERT row, which is why the SciBERT *full* test table — all
1,864 columns, not the top-50 — must exist (handoff 5). The val file is what the prompt and parser
are developed on and what chooses 5b's checkpoint; the test file is written once per arm.

**Shortlist files** — `scibert_full_top50_test.jsonl` and `scibert_full_top50_val.jsonl` (handoff 4)
and `scibert_full_top50_train.jsonl` (handoff 4b): `paper_id`, the 50 topic names in SciBERT's rank
order, and for the training file the target `gold_in_list`, ordered by SciBERT's score, highest
first. Only 5b's *training* prompts are shuffled — Gemma's training code shuffles each list at
prompt time and records the shown order in the training log; the val and test lists are shown as
they are.

**Stage 1 also emits `id_to_name.json`** — SciBERT trains on `verified_uat_ids`, Gemma reads and
writes topic *names*, and the cardinality-matched row joins the two per paper. Without the map the
join is guesswork.

Commit a three-row example of each file next to the schema, so that Shai can write the scorer against
real bytes before any model exists.

---

## 7. The calendar

| Dates | What happens |
|---|---|
| **Sun 13 Sept** | Ilana: **commit and push this plan** with the split files (`main.py` already computes the split — a `to_json` and a commit, ~45 min), the `hf-xet` fix in `requirements.txt`, and the Overleaf skeleton with the results table stubbed; **message Itai** (see the Monday rule); accept the Gemma licence; run the twenty-paper 1B parse test on the M1 (validation papers, TF-IDF stand-in lists). Shai: read §6 and start `evaluate.py` against the three-row example files. |
| **Mon 14 Sept** | Ilana: **send the lecturer email**; **run the 4B probe** on the card (an evening: environment build, then ≥500 steps at 1,024 tokens on 64 papers — also the overfit check); Gemma prompt + parser against the validation stand-in lists; related-work draft begins from `Articles/`. Shai: `evaluate.py` passes the Stage 2 tests; `choose_tau` importable; majority + TF-IDF rows scored end-to-end through `evaluate.py` by tonight — the first real proof the ruler works on real bytes. Itai: M5 smoke test, handover from Shai, front end. **20:00 — the tripwire** (see §4). |
| **Tue 15 Sept** | Itai (or Shai, if the tripwire fired) finishes the front end; **SciBERT-full trains overnight** (≤8 epochs, one configuration, every epoch's validation table saved). Ilana: Gemma inference loop working end-to-end on validation stand-in lists; introduction drafted. |
| **Wed 16 Sept** | SciBERT-full finishes in the morning. One inference pass produces **handoff 4 (test + validation top-50), handoff 4b (training top-50 + targets) and handoff 5 (full val + test tables)** — all three the same day. Shai: Stage 4 ceiling and the skew table from those files; best epoch chosen through `choose_tau`; SciBERT-full scored. Ilana: Gemma prompt finalised on the validation lists, commit hash logged; **5a runs the same night** (one inference pass). |
| **Thu 17 Sept** | **5b decision day:** 5b training on the card by tonight (full 15,822 lists at 4B or 1B as the probe decided), or fall back to Alternative B (1B on 2,555 validation lists, 300 held out), or cut and written up as attempted with the wall-clock and the error that stopped it. SciBERT-LoRA trains on the Mac if the M5 held. Shai: 5a scored, with the cardinality-matched row and mean-|ŷ|. |
| **Fri 18 – Sat 19 Sept** | 5b (or Alternative B) trains and runs: checkpoint chosen on the validation lists, test pass once, the shuffled-test robustness pass; Stage 6 on whichever Gemma exists; the 200-step timing slices on the card for every arm; Shai scores each arm the day it lands. **18 Sept: related-work + dataset sections drafted.** |
| **Sun 20 Sept** | **FREEZE.** No new experiments after this point. **Methodology section drafted** — the ruler, the baselines, the threshold and denominator findings are all known by now. |
| **21–27 Sept** | Results and discussion written; every section revised. Each writes up what they ran — see Stage 7. |
| **28–29 Sept** | Revision, figures, AI disclosure section (assembled from the per-stage log, not reconstructed), proofreading. |
| **Wed 30 Sept** | Submit. |

---

## 8. How we know each stage is done

A stage is done when someone who did not build it could check it from the repo. Each line below is
that check.

- **Stage 1** — `split.json` (15,822 / 2,855 / 3,025), `label_order.json`, `band_map_train.json`
  (12 / 383 / 1,469, primary), `band_map_corpus.json` (17 / 429 / 1,418) and `id_to_name.json` are
  committed with checksums; two different people loading them get identical splits; **and the
  measurement scripts behind §9 have been re-run on the saved split and the primary map, with their
  `.out` files committed, and the numbers in §9 updated** — every figure in the paper comes from the
  saved files.
- **Stage 2** — `evaluate.py` reads the §6 files, not in-memory arrays; a perfect fake prediction
  scores 1.0, an empty one scores 0; band sizes match whichever band map it is handed (12 / 383 /
  1,469 for the primary); `choose_tau` implements the §4 rule, is importable, returns the same taus
  for the same inputs and writes `tau_<arm>.json`; per-band P, R and **micro-F1**, macro-F1 under
  **both** denominators, P@1/3/5 and R@1/3/5 (counted for set-emitters as Stage 2 says), Δ, and the
  paired bootstrap interval are all produced; per-topic TP/FP/FN are stored per arm; off-list and
  empty-output rates are computed for the Gemma files; and **the whole thing has run end-to-end on
  TF-IDF's real tables**, not only on fakes.
- **Stage 3** — for each of majority, TF-IDF, SciBERT-full and SciBERT-LoRA: the validation and test
  tables exist on disk in the §6 format; the run log records seed, learning rate, batch size, epochs,
  trainable parameters, seconds per step and peak memory on the training machine, and the 200-step
  timing slice on the 3060 Ti; the best epoch was chosen through `choose_tau` from the saved per-epoch
  validation tables, not inside the training script; and for the SciBERT arms the
  overfit-a-small-subset check has been run and its plot saved (the guidelines ask to be shown it).
  The results table has a row for each — regenerated from the files by one command, by someone who
  did not train the model.
- **Stage 4** — the fraction of correct rare topics in **SciBERT-full's** top 50, at 25/50/100,
  computed by Shai from the handoff-5 test table, so we can state Gemma's ceiling; and the skew table
  (train-as-used / validation / test lists) from the handoff-4/4b files. (The word-counting version
  of the ceiling already exists; this is the real one.)
- **Stage 5a** — untouched Gemma has been scored on SciBERT's test shortlists **with the
  cardinality-matched SciBERT row, the mean-|ŷ| column, and the off-list / empty-output rates beside
  it**; the prompt's commit hash was logged before the test pass, and the validation-file outputs it
  was developed on are on disk. Without those the comparison is not interpretable, so they are
  requirements, not extras.
- **Stage 5b** — fine-tuned Gemma has been scored on the *same* shortlists and papers as 5a, with its
  own cardinality-matched row and mean-|ŷ|; its checkpoint was chosen on the validation lists; the
  skew table is in the appendix; the training log records trainable parameters, peak memory, seconds
  per step, the per-epoch emission ratio and the shuffled-test robustness result; and the training
  recipe (training lists shuffled, SciBERT order at validation and test, gold ∩ shortlist target
  ordered by SciBERT score, which papers it trained on — full set or Alternative B) is written down —
  **or 5b has been formally cut and the attempt described with its wall-clock and the error that
  stopped it, and 5a has been re-run at 4B.** The shortlist-generator swap is a bonus.
- **Stage 6** — we can state the out-of-vocabulary rate over 200 papers, for whichever Gemma ran.
- **Stage 7** — related-work and dataset sections drafted by 18 Sept, methodology by 20 Sept; eight
  pages, all sections present; the AI-disclosure section assembled from a log that has an entry per
  stage, written when the stage happened.

---

## 9. Four questions we kept re-asking, answered for good

The full reasoning and the measurements behind each answer are in `SETTLED_QUESTIONS.md`. The
answers:

- **Why SciBERT and not astroBERT as the shortlist screener (§9.1).** The screener's job is already
  nearly maxed out: plain word counting gets the correct topic into its top 50 90.4% of the time
  (81.6% for rare topics), against the ~82% Alkan et al. report for astroBERT. And the headline
  compares Gemma against itself on the same shortlist, so the screener's identity cancels out.
  astroBERT is not in the approved proposal, and it would cost a ~10-hour training run and its own
  tokenisation pass. We cite its published numbers as the bar; we do not train it.
- **Why one shortlist length, 50 (§9.2).** Coverage of the correct rare topic is 72.0% at 25, 81.6%
  at 50, 89.2% at 100 (word-counting stand-in). Going to 100 buys 7.6 points for double the prompt
  and fifty more distractors; each length would need its own training run (~24 GPU hours we do not
  have); and a change in score could not be attributed to either cause. The three numbers go in the
  paper as a description of the ceiling, not as an experiment. 50 is also Alkan et al.'s depth.
- **What is actually different from Alkan et al. (§9.3).** Not the data, not the band split, not
  SciBERT. Two things: we change a generative model's weights (they never do — DeepSeek is used
  through an API), and we show that rare-topic scores in this field cannot currently be
  interpreted, because two unstated choices are worth more than any model: the denominator of the
  macro average (0.324 over the 359 measurable rare topics vs 0.082 over all 1,418 — same
  predictions) and the decision threshold (0.0078 at 0.5 vs 0.324 tuned — forty times). Their
  paper states neither, and their Table 5 cannot be reconciled with their Table 6 under any
  averaging convention. We report both denominators, choose the cutoff on validation, and store
  per-topic TP/FP/FN so any convention can be recomputed.
- **Are we really comparing Gemma against SciBERT (§9.4).** No — SciBERT is inside both arms, so
  Gemma competes against SciBERT's *cutoff rule*, not against SciBERT. The honest research question
  is:

  > Holding the candidate set fixed at our SciBERT's top-50, does Gemma — untouched, and, if 5b
  > lands, QLoRA-tuned — select from it better than SciBERT's own tuned cutoff, and does any gain
  > show up in the rare topics? And does fine-tuning help or hurt, compared with the same model
  > untouched?

  This is the ablation Alkan et al.'s own limitations section asks for ("evaluating [the LLM stage]
  over candidate sets generated by BERT and SciBERT ... would isolate the contribution of the
  candidate generator's quality from the LLM's own re-ranking ability"), and the paper should say
  so. It is why Stage 5 requires the cardinality-matched row, the mean-|ŷ| column and the recall@50
  ceiling — and why Stage 6's free-generation numbers are the justification for the shortlist
  design rather than a curiosity.
