# Work Plan — Who Does What, and In What Order

**Written 10 September 2026. Submission: 30 September 2026.**

## The numbers that keep coming up

Every figure in this document is one of these. If you hit a bare number below and cannot remember
what it is, it is almost certainly on this list.

**Papers**

| number | what it is |
|---|---|
| **21,702** | every paper in the dataset |
| **18,677** | the **training** papers — what the models learn from |
| **3,025** | the **test** papers — what we are scored on. HuggingFace confusingly calls this split `validation`, but it is the published benchmark *test* split, and we never tune anything on it |
| **~2,801** | our **validation set** — papers we carve out of the 18,677 training papers and hold back, so we can choose the cutoff without ever touching the test set. The models train on the ~15,876 that remain. Careful: this is *not* the 3,025, even though HuggingFace uses the word "validation" for those |

**Topics (labels)**

| number | what it is |
|---|---|
| **2,367** | concepts defined in the UAT thesaurus. 503 of them never appear in any paper here, so we ignore those |
| **1,864** | topics that actually appear in our data. **This is our label space** — the list every model predicts over |
| **17** | **head** topics — common ones, appearing more than 500 times |
| **429** | **torso** topics — medium, 50 to 500 times |
| **1,418** | **tail** topics — rare, fewer than 50 times. **This band is what the whole project is about** |
| **359** | of those 1,418 rare topics, the ones appearing at least once in the test set — so these are the only rare topics we can actually measure |
| **1,059** | the other rare topics (1,418 − 359). They sit entirely in training and never appear in test, so **they cannot be scored at all**. That is 74.7% of the rare band, and it is the source of the denominator problem explained in §9.3 |
| **805** | every topic we can measure at all, across all three bands (17 + 429 + 359) |

**Other figures that recur**

| number | what it is |
|---|---|
| **4.31** | the average number of correct topics per paper |
| **50** | how many candidate topics we hand to Gemma to choose from |
| **81.6%** | how often the correct rare topic is actually among those 50 — so this is Gemma's ceiling on rare topics |
| **τ (tau)** | the cutoff: how confident the model must be before we count it as a "yes" |

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

**17 September — the Gemma giving-up date.** If the Gemma arm is not actually training by the 17th,
we stop and write it up as an honest negative result. The course guidelines explicitly say negative
results count when the method is sound. "We attempted this, here is how far we got, here is what
stopped us" is a real section of a real paper. What is not acceptable is discovering on the 27th that
it was never going to work.

---

## 2. How the whole thing fits together

Before the stage list, here is the shape of the thing, because the stages make more sense once you
see where they sit.

Every model we build does the same job: read a paper's title and abstract, and output **a confidence
score between 0 and 1 for each of the 1,864 possible topics** (1,864 = every topic that appears
anywhere in our data; see the table above). Not a yes/no — a score. That is true of the simple
baselines and of Gemma alike.

Turning those scores into an actual answer needs one more decision: **where do you draw the line?** A
score of 0.73 probably means yes and 0.02 probably means no, but the cutoff in between is a free
choice. We call that cutoff **tau (τ)**. It matters enormously — a TF-IDF test I ran showed the score
on rare topics moving from 0.0093 to 0.3257 purely by moving that line. So we choose tau using the
validation set, and only then apply it to the test set. **We never choose tau by looking at test
results.** That is the single rule most easily broken and hardest to explain away afterwards.

Then we score the results **three times over**: once for the 17 very common topics ("head"), once for
the 429 medium ones ("torso"), once for the 1,418 rare ones ("tail") — 17 + 429 + 1,418 = the 1,864
topics we predict over. Our entire research question is
about whether models fail on rare topics, so a single overall average would hide exactly the thing we
are studying.

So the pipeline is: **fixed data split → model → table of scores → choose tau on validation → apply
to test → per-band results table → paper.**

---

## 3. The stages

### Stage 1 — Fix the data split, once and forever
Decide the exact division of papers into training, validation and test — then **save it to a file**
and have every later script read that file rather than recomputing it. Also saved: the fixed list of
1,864 topics in a fixed order (the label space), and the head/torso/tail grouping that sorts them
into the 17 common / 429 medium / 1,418 rare bands.

### Stage 2 — Build the scoring script (`evaluate.py`)
This is the ruler. It takes a table of confidence scores plus the correct answers and produces the
numbers that go in the paper: precision, recall and F1 for each of the three bands; the tau search;
P@1, P@3 and P@5; and **Δ (delta)**, which is simply "head F1 minus tail F1" — one number for how
much worse a model is on rare topics than common ones.

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
whole tau problem. That makes it the fairest way to compare our thresholded encoders against Gemma,
which hands back a set of labels rather than a score for every topic.

**The important property: this script does not care where the scores came from.** Its input is a
table of numbers. You can generate that table with a random number generator before any model
exists, and check the script behaves: a perfect prediction should score 1.0, predicting nothing should
score 0, the tail group should contain the number of topics you expect.

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
topics at depth 50 (72.0% at 25, 89.2% at 100) — see §9. That was enough to settle two arguments: the
shortlist is not the weak link, so we do not need astroBERT (§9.1), and 50 is a sensible depth (§9.2).

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

### Stage 5 — Gemma re-ranking SciBERT's shortlist

SciBERT scores all 1,864 topics and its 50 best guesses become a shortlist. Gemma reads the paper and
picks from those 50. We run this **twice with the same Gemma** — once untouched, once fine-tuned —
on the same papers and the same shortlists, so the only difference is whether we updated the weights.

**Careful: this is not a contest between Gemma and SciBERT**, even though the proposal is worded that
way. SciBERT is inside both arms. §9.4 explains what we are actually measuring and how to describe it
in the paper — read it before writing any results prose.

**Gemma waits for SciBERT.** Since Gemma re-ranks SciBERT's top-50, SciBERT must be trained and its
top-50 lists saved to disk before this can run. The Gemma code itself does not have to wait: the
prompt, the parser and the inference loop can be written and smoke-tested against a stand-in shortlist
— TF-IDF's, or even fifty random topics — then swapped over the moment SciBERT's real ones exist.
Build it in parallel; run it after.

**How Gemma picks: prompt-and-select, not log-probability.** We put the paper and the 50 candidates in
the prompt and have the model output which ones apply — literally what DeepSeek did. Far less code,
and its failures are visible: either the output parses or it does not. The cost is that a generated
set has no per-label score, so this arm cannot use a tuned tau.

**Model size is decided by the probe, not chosen in advance.** 4B is estimated at 6.5–7.5 GB against
roughly 7.3 GB usable on the 3060 Ti — a coin flip. If 4B fits, use it. If not, 1B, or 1B fine-tuned
with 4B prompted-only.

**Three things the results table must carry** (all zero GPU — §9.4 says why each is needed):

1. A **cardinality-matched SciBERT row** — per paper, take as many of SciBERT's best guesses as Gemma
   emitted, and score those.
2. A **mean-|ŷ| column** on every row — the average number of topics that arm emits.
3. The **recall@50 ceiling**, stated in Methodology before any results (Stage 4).

Also report P@1/P@3/P@5, which need no cutoff and so compare the arms on equal terms.

**One optional extra, if time allows:** swap the shortlist generator — SciBERT's top 50 versus
TF-IDF's top 50 — which isolates how much the generator's quality mattered. SciBERT's is the main arm;
TF-IDF's is the ablation. Skip it if Gemma is late. A third idea, varying the shortlist depth, was
considered and rejected — see §9.2.

### Stage 6 — What Gemma does when left unconstrained
Let the fine-tuned model generate topic names freely on about 200 papers, and measure what comes out:
how often it invents topics that do not exist, how often it is nearly right (a spelling or plural
variant), how often it repeats itself. Nobody has measured this on this vocabulary, because nobody has
fine-tuned a generative model on it. It is cheap — inference only, no training — and it is a genuine
finding rather than a footnote.

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

The split is built around one fact: **Stage 2 depends on nothing.** The scoring script can be written
and fully tested against fake numbers while the models do not exist. That is what lets three people
work at once instead of one working and two waiting.

**The paper is shared work.** Each of us writes up what we ran — the split is in Stage 7. The lanes
below are about the experiments, not about who writes.

### Ilana — the foundation, Gemma, and the introduction

**You own Stage 1, Stages 5 and 6 (all of Gemma), the graphics card, and the introduction.**

Start with `build_split.py` — a few hours, and everything else depends on it. Save the split, the
topic list, and the band grouping to disk so the other two can read them.

**Gemma is yours end to end** — the prompt, the parser, the
inference loop, the frozen-vs-fine-tuned pair, the three required table entries in Stage 5, and the
unconstrained-generation study in Stage 6. Read §9.4 before you write any of the results prose — the
obvious way to describe this experiment is wrong.

You can start now without waiting for SciBERT: build the Gemma code against a stand-in shortlist
(TF-IDF's, or fifty random topics) purely to prove the machinery runs, then swap in SciBERT's real
candidates the moment Shai produces them.

**The introduction is yours, and so is the framing of the paper.** You have done most of the work so
far and you hold the reading notes, so the introduction, abstract, related work and dataset
description are yours to write — plus the Gemma sections, since you ran them. Get the Overleaf
skeleton up this week and start on the literature review from `Articles/`. None of that needs a single
result to exist. The rest of the split is in Stage 7.

**Two short jobs this week that unblock Gemma, both yours because the card is yours:**

- **Accept the Gemma licence on HuggingFace and generate an access token.** Gemma is a gated model —
  without this the code fails at the first line that tries to load it. Approval is near-instant, so
  this is five minutes, but it is five minutes that otherwise costs somebody an evening.
- **Run the 4B probe.** Load Gemma-3-4B in 4-bit on the 3060 Ti, run about 200 steps at the real
  sequence length, log peak memory, and confirm the loss stays finite. **This is what decides whether
  we use 4B or 1B**, and right now that choice is a guess — the estimate is 6.5–7.5 GB against roughly
  7.3 GB usable, which is a coin flip. Fifteen to thirty minutes, and it has been on our open-items
  list since 30 August.

### Shai — the measuring stick and the baselines

**You own Stage 2 (`evaluate.py`), the baselines in Stage 3, and Stage 4.**

Everything in this lane runs on your laptop in minutes, so nothing here waits on hardware.

**One handover: sit down with Itai for an hour this week and walk him through the training engine you
wrote.** The training loop, `evaluate()` returning probabilities, and the checkpoint and resume logic
are done and they are yours. What is missing is the front end, which he writes.

Start with fake data. Write `evaluate.py` so it takes a table of scores and a table
of correct answers and produces the full results table: per-band precision, recall and F1; the tau
search on validation; P@1/P@3/P@5; and delta. Test all of it against a randomly generated score table
before any model exists. You are done when you can hand it a perfect prediction and get 1.0, hand it
nothing and get 0, and the band sizes come out as 17 / 429 / 1,418.

Then the two baselines — majority and TF-IDF — which run on your laptop in minutes and give us the
first real rows of the results table.

Then Stage 4, the recall ceiling count, which is pure arithmetic and tells us the highest score Gemma
could possibly reach.

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

- If the M5 works: SciBERT trains there. Expect well under the M1's measured 1.3 h/epoch.
- If it does not: fall back to **Ilana's M1**, measured at 1.3 h/epoch and free once Gemma is running
  on the desktop. Do **not** fall back to Shai's i7 — 3–6 h/epoch will not fit the calendar.

**The training engine already exists.** Shai wrote it — the training loop, `evaluate()` returning
probabilities so several thresholds can be swept without re-running, and working checkpoint and
resume. Take an hour with him this week and walk through it. What is missing is the front end: the
tokenizer, the dataset object that hands the model batches, and the code that turns a list of topic
names into the yes/no vector the model trains against. Then fill in `run_SciBERT_experiment`, which is
currently only comments.

Two things in the existing code to settle before the first long run:

- **The best epoch is currently chosen by validation loss.** Our headline number is F1 on rare topics
  at a tuned tau. With 1,864 topics to score and only about four correct per paper (4.31 on average),
  the loss is dominated by the vast number of easy "no" answers, so the epoch with the lowest loss is
  not reliably the epoch with the best F1. Since `evaluate()` already returns the probabilities each
  epoch, computing F1 inside the loop is a small change now and an expensive re-run later.
- **`num_epochs` defaults to 20.** The paper we are comparing against found its best at 8. Twenty is a
  lot of hours to spend discovering we overfit at six.

**Save SciBERT's top-50 candidate lists to disk as soon as the model is trained.** Ilana's whole Gemma
arm waits on that file, and the deadline for it is 16 September.

### What depends on what

Most of the work runs in parallel, but three things genuinely block other people. If one of these
slips, somebody else sits idle — so they are worth naming rather than discovering.

**The whole workflow, drawn:**

```
                          ┌──────────────────────────┐
                          │  ILANA · build_split.py  │
                          │  split + topics + bands  │
                          └────────────┬─────────────┘
                                       │   everything waits on this
                 ┌─────────────────────┼─────────────────────┐
                 ▼                     ▼                     ▼
      ┌────────────────────┐ ┌───────────────────┐ ┌────────────────────┐
      │ SHAI               │ │ ITAI              │ │ ILANA              │
      │ evaluate.py        │ │ M5 smoke test     │ │ 4B probe           │
      │   on fake scores   │ │ SciBERT front end │ │ Gemma code, on a   │
      │ majority + TF-IDF  │ │                   │ │   stand-in list    │
      │                    │ │                   │ │ intro + lit review │
      └─────────┬──────────┘ └─────────┬─────────┘ └─────────┬──────────┘
                │                      ▼                     │
                │            ┌───────────────────┐           │
                │            │ SciBERT trains    │           │
                │            │ full + LoRA       │           │
                │            └─────────┬─────────┘           │
                │                      ▼                     │
                │            ┌───────────────────┐           │
                │            │ top-50 lists      │  ─────────┤
                │            │ saved · 16 Sept   │           │
                │            └─────────┬─────────┘           ▼
                │                      │           ┌────────────────────┐
                │                      │           │ Gemma runs         │
                │                      │           │ frozen + tuned     │
                │                      │           └─────────┬──────────┘
                │                      │                     ▼
                │                      │           ┌────────────────────┐
                │                      │           │ free generation    │
                │                      │           │ Stage 6            │
                │                      │           └─────────┬──────────┘
                ▼                      ▼                     ▼
      ┌──────────────────────────────────────────────────────────────────┐
      │  SHAI scores every arm  →  the results table                     │
      └────────────────────────────┬─────────────────────────────────────┘
                                   ▼
      ┌──────────────────────────────────────────────────────────────────┐
      │  ALL THREE · the paper — each writes up what they ran            │
      └──────────────────────────────────────────────────────────────────┘
```

**The critical path is the middle column.** Ilana's split → Itai's SciBERT → the top-50 lists →
Ilana's Gemma is the only chain where three people queue behind each other, and everything to the
right of a slip moves with it.

**The handoffs, in order:**

| # | From | To | What | Needed by |
|---|---|---|---|---|
| 1 | Ilana | everyone | the saved split, topic list and band map | before anyone writes real code |
| 2 | all three | — | agreement on the output format (§6) | before the first long run |
| 3 | Shai | Itai | walkthrough of the training engine he wrote | before Itai writes the front end |
| 4 | Itai | Ilana | **SciBERT's top-50 candidate lists** | **16 Sept — Gemma cannot start without it** |
| 5 | Itai | Shai | SciBERT's score tables, to be scored | as each arm finishes |
| 6 | Ilana | Shai | Gemma's outputs, to be scored | as each arm finishes |
| 7 | Itai | Shai | trained SciBERT, for the real recall@50 ceiling (Stage 4) | after SciBERT trains |

**Two measurements gate decisions, and both are short:**

- **Itai's M5 smoke test** decides where SciBERT trains. Until it is run, we do not know whether our
  fastest machine is actually fast.
- **Ilana's 4B probe** decides Gemma's model size. Until it is run, 4B-versus-1B is a guess.

Both are due by 12 September, and neither takes more than half an hour.

**Who decides which parameters:**

| parameter | who | how it is decided |
|---|---|---|
| **tau — the yes/no cutoff** | **Shai**, inside `evaluate.py`, for every arm | searched on the validation set and applied to test. It lives in one script on purpose, so every arm gets the identical rule and nobody picks their own |
| SciBERT's learning rate, batch size, number of epochs | Itai | he trains it. Cap epochs at 8, not the current default of 20 |
| which epoch counts as "best" | Itai | should be best F1 on rare topics, not lowest validation loss — see his section |
| Gemma's model size, 4B or 1B | Ilana | decided by the probe, not chosen in advance |
| Gemma's prompt and decoding settings | Ilana | greedy decoding, and whatever is used gets stated in the paper |
| the shortlist depth, 50 | already fixed | §9.2. Not a parameter we tune |
| the frequency bands, 17 / 429 / 1,418 | already fixed | Stage 1, saved to disk and read by everything |

**The Gemma arm has no tau at all**, because prompt-and-select emits a set of labels rather than a
score per topic. That is the reason Stage 5 needs the cardinality-matched row — it is what lets a
thresholded arm and an unthresholded one be compared at all.

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

**Three random seeds → one.** Nice to have, not affordable.

---

## 6. The one thing we must agree on first

Before anyone writes code: **agree the format of a model's output.**

One row per paper, one column per topic, values between 0 and 1, and the column order fixed by the
topic list that `build_split.py` saves. Itai's encoders and Ilana's Gemma both produce it; Shai's
scoring script consumes it. (Gemma is the exception that proves the rule: it emits a *set* of labels
rather than a score per topic, which is why Stage 5 needs the cardinality-matched row — agree how that
output is written to disk at the same time.)

---

## 7. The calendar

| Dates | What happens |
|---|---|
| **10–12 Sept** | Ilana: split saved, Overleaf started, lecturer email sent, **Gemma licence accepted and the 4B probe run**, Gemma code started against a stand-in shortlist. Itai: **M5 smoke test first**, then the SciBERT handover from Shai, then the data loading front end. Shai: `evaluate.py` against fake data, then the two baselines. |
| **12 Sept** | **Model size decided** by the 4B probe: 4B, or 1B, or 1B tuned + 4B prompted. **And where SciBERT trains decided** by the M5 smoke test: M5, or fall back to Ilana's M1. No more guessing after this. |
| **13–16 Sept** | Itai runs both encoder arms — full and LoRA — on the M5. First real results table. Itai: recall ceiling measured at 25/50/100. **SciBERT must be trained and its top-50 candidates saved by the 16th — Gemma cannot start without them.** Ilana: Gemma code smoke-tested against a stand-in shortlist, introduction and related work drafted. |
| **17 Sept** | **Gemma decision day.** Training by now, or cut and write it up. |
| **18–20 Sept** | Gemma runs if it survived. Results tables finalised. |
| **20 Sept** | **FREEZE.** No new experiments after this point. |
| **21–27 Sept** | Writing. All three of us, each writing up what we ran — see the split in Stage 7. |
| **28–29 Sept** | Revision, figures, AI disclosure section, proofreading. |
| **30 Sept** | Submit. |

---

## 8. How we know each stage is done

- **Stage 1** — a saved file exists, and two different people loading it get identical splits.
- **Stage 2** — a perfect fake prediction scores 1.0, an empty one scores 0, band sizes are 17/429/1,418.
- **Stage 3** — the results table has real rows for majority, TF-IDF, SciBERT-full and SciBERT-LoRA.
- **Stage 4** — we know what fraction of correct rare topics appear in **SciBERT's** top 50, so we can
  state Gemma's ceiling. (The word-counting version of this number already exists; this is the real one.)
- **Stage 5** — frozen and fine-tuned Gemma have been scored on the same shortlists and the same test
  papers, **with the cardinality-matched SciBERT row and a mean-|ŷ| column beside them** — or the arm
  has been formally cut. Without those two the comparison is not interpretable, so they are
  requirements, not extras. The shortlist-generator swap is a bonus.
- **Stage 6** — we can state the out-of-vocabulary rate over 200 papers.
- **Stage 7** — eight pages, all sections present, AI disclosure written.

---

## 9. Four questions we kept re-asking, answered for good

### What I measured first, and why

Before answering the three questions, I ran three quick checks on my laptop. **No GPU, no model
training, about an hour in total.** They are what turned these questions from arguments into answers,
so they are worth knowing about first.

**1. How often the correct topic is on the shortlist at all.** Gemma never sees all 1,864 topics — it
is handed a shortlist of 50 candidates and picks from those. So: how often is the correct topic even
among those 50? If it is missing, Gemma gets marked wrong for a mistake it had no chance to avoid.

*How I checked it:* for each of the 1,864 topics I built a simple word profile from the training
papers tagged with it — essentially, which words tend to show up in papers about that topic. Then for
each of the 3,025 test papers I compared its words against all 1,864 profiles, ranked the topics from
best match to worst, and looked at the answer key to see whether the correct ones landed in the top
25, 50 or 100. *For rare topics the correct answer is on a list of 50 about 81.6% of the time.*

**2. How much the yes/no cutoff is worth.** Models do not answer yes or no — they give a confidence
per topic, and someone has to pick the line above which we count it as a yes. Nobody had checked how
much that choice was worth.

*How I checked it:* I put the 3,025 test papers aside untouched. Out of the 18,677 training papers I
held back about 2,800 as a **validation set**, and trained a simple classifier on the remaining
~15,876 for each of the 359 rare topics we can actually measure. I tried a range of cutoffs **on the
validation set only** — never on test — took whichever scored best there, and applied that one cutoff
to the test papers once at the end. *The cutoff is worth about 35x on rare topics: the score goes from
essentially zero (0.0093) at the obvious default of 0.5, up to 0.3257 at the cutoff the validation
set picked.* The reason is that on rare topics the model is never confident — its strongest hunch
might be 8% — so a 50% bar makes it say no to everything.

**3. Whether the 359 measurable rare topics really are measurable.** Only 359 of our 1,418 rare
topics appear in the test set at all. But if most of those appeared in just one test paper, a
per-topic score could only come out 0 or 1 — a coin flip dressed up as a measurement. This was worth
checking before building a headline on it.

*How I checked it:* pure counting — for each of those 359 topics, how many test papers actually carry
it. *Each appears in between 3 and 12 test papers, half of them in exactly 5. Coarse, but a real
measurement.*

**One caveat.** All three checks used TF-IDF — plain word counting — because it runs in minutes on a
laptop. So these are not our final numbers. The 81.6% matters most here: **that is TF-IDF's
shortlist, not SciBERT's.** Our real shortlist will come from SciBERT, and once it is trained and
producing top-50 lists, that number has to be measured again before it goes anywhere near the paper.
SciBERT should do better than word counting, so it will probably go up — but it has to be the true
number, not a stand-in.

### 9.1 Why SciBERT and not astroBERT

Short answer: **keep SciBERT.** Stage 3 already says we do not train astroBERT and gives three
reasons. What was missing until now is the measurement that settles it — check 1 above.

First, a distinction that caused a lot of confusion, because a model can do **two different jobs** in
this project:
this project:

- **Job A — be a contestant.** A row in the results table, being measured. "astroBERT scores X on
  rare topics."
- **Job B — be the screener.** The model that narrows 1,864 topics down to the 50 we hand to Gemma.
  It is not being measured; it is doing prep work.

Alkan et al. use astroBERT for **Job B** — astroBERT produces the top 50, then DeepSeek picks from
those 50. Our plan uses SciBERT for Job B.

**As the screener (Job B), the job is already nearly maxed out.** This is check 1 above. A
deliberately crude method — plain word counting, no neural network at all — already gets the correct
rare topic into its top 50 **81.6%** of the time. Alkan's astroBERT screener manages about 82%.
There is no room left to buy. A better screener would be paying for something that is already done.

**As a contestant (Job A), it buys nothing either.** Our headline compares *untrained Gemma against
trained Gemma on the same shortlist*. The screener is identical on both sides of that comparison, so
it cancels out completely. Swapping SciBERT for astroBERT gives us slightly prettier absolute numbers
to place next to an unrefereed preprint. It does not make our claim any more true.

**And it is not free, even though it is not technically hard.** astroBERT is the same size as SciBERT
— BERT-base, ~110M parameters, same 512-token limit, a normal ungated HuggingFace download. Nothing
about it is harder to train. But it is *one more* training run, roughly 10 hours, on a card that is
already promised to Gemma, at a moment when we have zero trained encoders. It would either queue in
front of Gemma and push it past its own 17 September give-up date, or need a machine nobody has
confirmed we have.

### 9.2 Do we need to train Gemma on more than one shortlist length?

**The question.** Gemma picks its answers from a shortlist of 50 candidate topics. It is fair to ask
whether 50 is the right number. Maybe a shorter list of 25 would be easier for it to handle, or a
longer list of 100 would give it more chances to find the rare topics. Should we train Gemma three
times, once at each length, and see which does best?

**What I checked before spending any GPU time.** I measured how much a different length could even
buy us. The screener ranks all 1,864 topics for every paper, and the shortlist of 50 is simply the
top 50 of that ranking — so we could just count how often the correct rare topic falls inside the top
25, the top 50 and the top 100 of a ranking that already existed:

| shortlist length | how often the correct rare topic is on the list |
|---|---|
| 25 | 72.0% |
| 50 | **81.6%** |
| 100 | 89.2% |

**The answer: no, it is not worth it.** Three reasons.

*The gains are small and come with a cost.* Going up to 100 buys 7.6 more points of coverage, but it
doubles the prompt length and hands the model fifty extra wrong options to be distracted by. Going
down to 25 throws away 9.6 points. 50 sits in a sensible place, and it is also what Alkan et al.
used — which is the only reason our number will be readable next to theirs.

*It would cost three trainings, not one.* This is the part that is easy to get wrong. A model
fine-tuned on lists of 50 has learned things specific to lists of 50 — how long the prompt is, roughly
how many topics to emit, where in the list good answers tend to sit. Show it a list of 100 at test
time and any change in score is tangled up with that mismatch. So you cannot train once and simply
evaluate three times; each length needs its own training run. That is roughly 24 hours of GPU
training we do not have.

*And the result would not be interpretable anyway.* A longer list helps and hurts at the same time —
more chances to contain the right answer, more wrong options to confuse the model. If the score
moved, we could not say which of the two caused it. One number moving for two opposite reasons is not
a finding.

**So: 50 candidates, one training run, one evaluation. Nothing varied.**

**What we still do with those three numbers.** They go in the paper, not as an experiment but as a
description of the shortlist itself — they cost nothing, since we already have them. They do two
jobs. They show that 50 was a reasoned choice rather than one plucked out of the air. And they state
Gemma's ceiling honestly: on rare topics Gemma can never score above **81.6%**, because that is the
share of correct answers it was ever shown. A reader needs that number to make sense of every Gemma
result in the paper.

Written out it is one small table and one short paragraph, something like:

> *We construct shortlists at depth k=50, matching Alkan et al. To characterise the ceiling this
> imposes, we measure the proportion of gold labels present in the shortlist at k ∈ {25, 50, 100}.
> On the tail band this is 72.0%, 81.6% and 89.2% respectively. We fix k=50: k=25 sacrifices 9.6
> points of recoverable tail labels, while k=100 doubles prompt length for a 7.6-point gain. All
> Gemma tail results are therefore bounded above by 81.6%.*

### 9.3 What is actually different between our project and Alkan et al.

Start with **what is not different**, because being straight about this is what makes the rest
credible:

- The dataset is not ours. It is theirs.
- Splitting results into common / medium / rare is not new. They did it, and Huang et al. did it
  before them.
- **SciBERT is not new — they already trained it and published the result.** This is where the 0.023
  rare-topic figure comes from. It is *their* SciBERT, not ours. We have not trained anything yet.

So three of the four things our original proposal called novel are now covered by their paper. That
is the situation, and pretending otherwise would read far worse than naming it.

Two real differences remain.

**Difference 1 — we change the model's weights. They never did, not once.**

Every generative model in their paper is DeepSeek, reached through an API. You send text, you get
text back, and you cannot change it. It arrives knowing what it knows and leaves exactly as it came.

We are doing something different: taking an open model we can actually modify (Gemma) and training it
on astronomy papers so it gets better at this specific job. Nobody in that paper fine-tuned a
generative model anywhere. **Be careful how we evidence this claim.** Their paper does contain the
sentence *"due to computational resource constraints, we were unable to include Qwen models in the
fine-tuning experiments"* — but that sits in their section on supervised *encoders*, and the "Qwen"
there is an embedding model used in their k-NN arm. It is not a statement about generative
fine-tuning and must never be quoted as one. The gap is real, but the evidence for it is what the
paper *does* across all its arms, not any single sentence.

And our comparison is cleaner than theirs: **the same Gemma, twice.** Once untouched, once trained.
Same papers, same 50 candidates, same everything. The only thing that differs is whether we updated
the weights, so whatever we observe was caused by the training and by nothing else. Note this is a
comparison they could not make and we can — not a head-to-head between an encoder and an LLM, which
nobody has run on this corpus (see Stage 5).

**Difference 2 — we show that rare-topic scores in this field cannot currently be interpreted.**

This is the one worth expanding, because it is cheap, it is already finished, and it survives even if
the Gemma arm never runs.

*The problem in one sentence:* reporting a score on rare topics requires two choices that nobody
writes down, and I measured that those two choices are worth more than any model in the comparison.

**Problem A — what do you divide by?**

"Macro-F1" means: score each topic separately, then average the scores. Fine. But average over which
topics?

Of our 1,418 rare topics, only **359 appear in the test set at all.** The other 1,059 are not merely
hard — they are unmeasurable, because there is nothing to check an answer against. This is not an
accident; it is how the dataset's own authors built the split. Their rule: *labels appearing ≥15
times are split 85/15, while labels with <15 occurrences are placed entirely in the training set.*
Every genuinely rare label was deliberately held back in training.

So when we average, do we divide by 1,418 (counting the 1,059 unmeasurable ones as zero) or by 359
(only what we can actually check)? A tiny worked example, with five rare topics:

| topic | in the test set? | its score |
|---|---|---|
| A | yes | 0.6 |
| B | yes | 0.4 |
| C | no | — |
| D | no | — |
| E | no | — |

Divide by 5: (0.6 + 0.4 + 0 + 0 + 0) ÷ 5 = **0.20**. Divide by 2: (0.6 + 0.4) ÷ 2 = **0.50**. Same
model, same predictions, 2.5x apart.

In our real data the gap is 1,418 ÷ 359 = **3.95x**. I measured it on a TF-IDF baseline: the
identical predictions score **0.3257** one way and **0.0825** the other.

**Problem B — where do you draw the yes/no line?**

The model does not answer yes or no. It gives a confidence per topic — "3% sure about solar wind, 71%
about exosphere." Somebody has to pick a cutoff. The obvious default is 0.5.

But on rare topics the model is never confident; its strongest hunch might be 8%. With the bar at
50%, it says no to every rare topic and scores essentially zero — not because it learned nothing, but
because we asked for certainty it was never going to have.

I measured this properly (cutoff chosen on a held-out slice of training data, never on test):
**0.0093 at a 0.5 cutoff, 0.3257 at a tuned cutoff. Thirty-five times.** Bigger than SciBERT
vs Gemma. Bigger than LoRA vs full fine-tuning. Bigger than anything else we will do.

**Neither choice is stated anywhere in their paper.** The full description of their metric is
*"Macro-F1, which averages per-label F1 scores."* That is the entire sentence. No denominator, no
threshold, anywhere in the document.

**And their two tables do not agree with each other.** Their Table 5 gives overall scores; Table 6
gives the same models split by band. Average the per-band numbers back up and you should recover the
overall number. You do not — under any convention I could reconstruct:

| model | they report | divide by all 1,864 topics | divide by the 805 measurable ones | divide by the full 2,367 vocabulary | weight each band by how many label-assignments it holds |
|---|---|---|---|---|---|
| SciBERT | 0.2127 | 0.0646 | 0.1193 | 0.0509 | 0.1622 |
| astroBERT | 0.3243 | 0.1354 | 0.2070 | 0.1066 | 0.2538 |
| DeepSeek | 0.3770 | 0.2394 | 0.2938 | 0.1885 | 0.3234 |

Every convention fails, for every model.

We can also rule one of those columns out completely, using nothing but arithmetic. Follow it slowly,
because it is the strongest single thing we can say about their paper.

An average is a sum divided by a count. So: (sum of all the per-topic scores) ÷ (however many topics
they divided by) = the number they printed.

Now, how big can that sum possibly get? Each topic's score is at most 1. And the 1,059 rare topics
with no test paper score exactly 0, because there is nothing to get right. So the sum can never
exceed **359** — the number of rare topics that are actually measurable.

DeepSeek's printed rare-topic score is 0.198. Rearranging, their denominator = sum ÷ 0.198, and since
the sum is at most 359, the denominator is at most 359 ÷ 0.198 = **1,813 topics**.

Why that matters: if they had averaged over the full UAT vocabulary of 2,367 concepts, their rare
band would contain 2,367 − 17 head − 429 torso = **1,921 rare topics**. But we just showed the
denominator cannot exceed 1,813, and 1,921 is larger than that. **So they did not average over the
full vocabulary.** That is not a guess; it is impossible.

That leaves 1,418 (all rare topics) or 359 (only the measurable ones). We can argue against 1,418
too, though this one is judgement rather than proof: if they divided by 1,418, then their sum was
0.198 × 1,418 = 281, spread across 359 measurable topics — an average of **0.782** per topic. A model
scoring 0.78 on the rarest, hardest topics while scoring only 0.377 overall is not coherent. So 359
is the likely reading, which is also the one we use.

Either way, the point for our paper is the same: **we had to reverse-engineer this, and we still
cannot be certain, because the paper never says.**

**What we do differently, and it costs nothing:**

1. Report the rare-topic score under **both** denominators, in the same table, clearly labelled, every
   single time.
2. Choose the cutoff on our carved validation split, never on test, and state exactly what we chose
   and how.
3. Store per-topic TP/FP/FN for every arm, so any averaging convention can be recomputed later
   without retraining anything.

**We are not accusing them of being wrong.** The honest claim — and the one to write — is that their
numbers cannot be interpreted without information the paper does not give, and that we are the first
to measure how much that missing information is worth. The course guidelines explicitly invite this:
*"ways to improve the evaluation of existing methods."*

### 9.4 Are we really comparing Gemma against SciBERT?

**No, and the paper must not say we are.** This is the easiest thing in the project to get wrong,
because the proposal is worded as a contest and the design is not one.

**What actually happens.** SciBERT scores all 1,864 topics, and its 50 best guesses become a
shortlist. Gemma then picks from those 50. So **SciBERT is inside both arms.** Gemma is not competing
against SciBERT — it is competing against SciBERT's *cutoff rule*, the line above which SciBERT says
yes. Both arms draw from the same pool of 50, share the same ceiling, and Gemma can never name a topic
SciBERT left out of the pool. About 18% of rare correct topics sit outside the top 50 before Gemma is
even prompted.

**The paper we benchmark against works the same way, which is easy to miss.** Their "DeepSeek" row is
not a standalone LLM. Their §4.2.3: *"(1) use our best supervised model (astroBERT) to generate top-50
candidate labels for each abstract, (2) prompt DeepSeek-V3-reasoner via its API to select relevant
concepts from these candidates."* So their headline "the LLM beats the encoder" is astroBERT against
astroBERT-plus-DeepSeek — a part against a whole containing that part. **Nobody has run a real
head-to-head on this corpus**, and they say why: prompting over the full vocabulary failed, with the
model inventing topics.

**So the honest research question is:**

> Holding the candidate set fixed at our SciBERT's top-50, does a QLoRA-tuned Gemma select from it
> better than SciBERT's own tuned cutoff — and does any gain show up in the rare topics?

That still serves the proposal's real interest — is the big generative model worth its compute. What
is *not* answerable is "match or beat" read literally, because Gemma is structurally incapable of
finding something SciBERT missed. If we leave the old wording in, one question at the oral — *"could
Gemma have found a topic SciBERT missed?"* — takes the claim apart.

**This is a gap the paper's own authors asked someone to fill.** Their Limitations section: *"more
extensive experiments with the LLM filtering stage are needed: evaluating DeepSeek over candidate sets
generated by BERT and SciBERT, in addition to astroBERT, would isolate the contribution of the
candidate generator's quality from the LLM's own re-ranking ability."* Their closing paragraph asks
for *"controlled ablations of the LLM re-ranking stage across candidate generators of varying
quality."* That is our experiment, named as future work by the people who built the dataset — a
stronger position than a head-to-head would have been, and the paper should say so.

**Why the three required table entries exist.** Gemma emits a free set of labels; SciBERT applies a
cutoff. So a raw F1 win cannot tell "Gemma ranks better" apart from "Gemma simply guessed more
topics" — which is exactly the flaw we identify in their paper in §9.3, where DeepSeek's precision
(0.2891) is *lower* than astroBERT's (0.3068) and the entire gain is recall. Hence: the
**cardinality-matched row** (give SciBERT the same number of guesses Gemma made), the **mean-|ŷ|
column** that makes the problem visible, and the **recall@50 ceiling** stated before any results so a
reader can tell a weak model from a starved one.

**One useful consequence.** Stage 6 — letting Gemma generate freely over all 1,864 topics on ~200
papers — stops being a curiosity and becomes the *justification* for this whole design. "We tried
letting it choose from everything; here is the rate at which it invented topics that do not exist;
that is why the shortlist exists." That is a Methodology paragraph which defends itself.
