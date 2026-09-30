# Settled questions — the reasoning behind the work plan

*Companion to `WORK_PLAN.md` §9. These four questions were argued at length before the plan was fixed; the
answers are in the plan, the reasoning is here. Nothing in this file is an open decision.*

> **STATUS (30 September 2026): a reasoning record from mid-September. The paper in `paper/` is
> authoritative wherever the two differ.** Two known differences:
>
> 1. **The threshold figures below (0.0078 / 0.156 / 0.258 / 0.324, and the 0.324 vs 0.082
>    denominator pair) are the macro-optimised threshold row** of
>    `scripts/analysis/honest_tau_seed42_protocol.out`. The paper reports the **micro-optimised**
>    row — 0.008, 0.16 global, **0.30** per band, and 0.30 vs 0.08 — because micro-F1 is the
>    objective its protocol states. Both are real measurements of the same predictions under
>    different tuning objectives.
> 2. **The depth-50 coverage figures (72.0 / 81.6 / 89.2%) are the TF-IDF stand-in's**, as the note
>    at the end of that section already says. The real SciBERT generator measures **33.9 / 52.8 /
>    72.4% on validation and 42.2 / 64.8 / 81.9% on test** (`real_recall_ceiling.out`), which is
>    what `appendix.tex` Table 4 reports. The draft sentence "bounded above by 81.6%" was never
>    used; the paper states the real ceiling and treats it as a limitation.
>
> Note for anyone sweeping this file: the `0.3243` at the astroBERT row of the Table 5 comparison
> is **Alkan et al.'s own reported score**, not ours. It is correct and coincidental.


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
held back the 2,855-paper **validation set** (the split `main.py` produces with seed 42), and trained
a simple classifier on the rest for every topic we can measure. I tried a range of cutoffs **on the
validation set only** — never on test — took whichever scored best there, and applied that one
cutoff to the test papers once at the end. *The cutoff is worth about 40x on rare topics: the score
goes from essentially zero (0.0078) at the obvious default of 0.5, up to 0.324 at the cutoff the
validation set picked.* The reason is that on rare topics the model is never confident — its
strongest hunch might be 8% — so a 50% bar makes it say no to everything. **And the rule for
choosing the cutoff matters almost as much as choosing one:** the same predictions score 0.156 on
rare topics with one cutoff for all bands chosen for overall micro-F1, 0.258 with one cutoff chosen
for overall macro-F1, and 0.324 with a cutoff per band. That is why §4 writes the rule down.

**3. Whether the 359 measurable rare topics really are measurable.** Only 359 of our 1,418 rare
topics appear in the test set at all. But if most of those appeared in just one test paper, a
per-topic score could only come out 0 or 1 — a coin flip dressed up as a measurement. This was worth
checking before building a headline on it.

*How I checked it:* pure counting — for each of those 359 topics, how many test papers actually carry
it. *Each appears in between 3 and 12 test papers, half of them in exactly 5. Coarse, but a real
measurement.*

**Two caveats.** All three checks used TF-IDF — plain word counting — because it runs in minutes on
a laptop, and on the corpus-basis bands (359 measurable rare topics; 410 under the primary map);
Stage 1 re-runs every script on the saved split and the primary map before any figure goes in the
paper. And the 81.6% matters most here: **that is TF-IDF's shortlist, not SciBERT's.** Once SciBERT
is producing top-50 lists, that number has to be measured again before it goes anywhere near the
paper.

### 9.1 Why SciBERT and not astroBERT

Short answer: **keep SciBERT.** Stage 3 already says we do not train astroBERT and gives three
reasons; check 1 above is the measurement that settles it.

First, a distinction, because a model can do **two different jobs** in this project:

- **Job A — be a contestant.** A row in the results table, being measured. "astroBERT scores X on
  rare topics."
- **Job B — be the screener.** The model that narrows 1,864 topics down to the 50 we hand to Gemma.
  It is not being measured; it is doing prep work.

Alkan et al. use astroBERT for **Job B** — astroBERT produces the top 50, then DeepSeek picks from
those 50. Our plan uses SciBERT for Job B.

**As the screener (Job B), the job is already nearly maxed out.** This is check 1 above. A
deliberately crude method — plain word counting, no neural network at all — already gets the correct
topic into its top 50 **90.4%** of the time over all bands, and **81.6%** of the time for rare
topics. Alkan's astroBERT screener manages about 82% — a figure they give pooled over all bands, with
no rare-band number published, so the like-for-like comparison is 90.4% against 82%, and the crude
screener is already ahead. There is no room left to buy. A better screener would be paying for
something that is already done.

**As a contestant (Job A), it buys nothing either.** Our headline compares *untrained Gemma against
trained Gemma on the same shortlist*. The screener is identical on both sides of that comparison, so
it cancels out completely. Swapping SciBERT for astroBERT gives us slightly prettier absolute numbers
to place next to an unrefereed preprint. It does not make our claim any more true.

**And it is not free, even though it is not technically hard.** astroBERT is the same size as SciBERT
— BERT-base, ~110M parameters, same 512-token limit, a normal ungated HuggingFace download. Nothing
about it is harder to train. But it is *one more* ~10-hour training run and one more tokenisation
pass, from the same person and on the same Mac that SciBERT-full and SciBERT-LoRA already fill
until the freeze, at a moment when we have zero trained encoders. It would queue behind the two
arms we promised, and its lists would arrive after Gemma's give-up date.

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

(Those are the TF-IDF stand-in's figures; the paragraph is written with SciBERT's own numbers from
Stage 4 once they exist, on the primary band map.)

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

Of our 1,418 rare topics (corpus-wide count; 1,469 on the results-table map, with 410 measurable —
the arithmetic below is the same either way), only **359 appear in the test set at all.** The other 1,059 are not merely
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

In our real data the gap is 1,418 ÷ 359 = **3.95x** (1,469 ÷ 410 = 3.58x on the results-table
map). I measured it on a TF-IDF baseline: the identical predictions score **0.324** one way and
**0.082** the other.

**Problem B — where do you draw the yes/no line?**

The model does not answer yes or no. It gives a confidence per topic — "3% sure about solar wind, 71%
about exosphere." Somebody has to pick a cutoff. The obvious default is 0.5.

But on rare topics the model is never confident; its strongest hunch might be 8%. With the bar at
50%, it says no to every rare topic and scores essentially zero — not because it learned nothing, but
because we asked for certainty it was never going to have.

I measured this properly (cutoff chosen on a held-out slice of training data, never on test):
**0.0078 at a 0.5 cutoff, 0.324 at a tuned cutoff. Forty times.** Bigger than SciBERT
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
full vocabulary.** That is not a guess; it is arithmetic — with one stated assumption, that a topic
with no test paper contributes 0 to the sum (the default in every standard implementation, e.g.
scikit-learn's `zero_division`; a convention that scored those topics 1.0 instead would break the
bound, but would also inflate their own tail number past anything they report).

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

> Holding the candidate set fixed at our SciBERT's top-50, does Gemma — untouched, and, if 5b lands,
> QLoRA-tuned — select from it better than SciBERT's own tuned cutoff, and does any gain show up in
> the rare topics? And does fine-tuning help or hurt, compared with the same model untouched?

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
