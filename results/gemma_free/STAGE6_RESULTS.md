# Stage 6 — unconstrained generation: results, caveats and what is left

**Run 25 September 2026 on the RTX 3060 Ti. Code committed before the reported runs, so the
provenance logged in each summary is clean.** Commits `6bbc8af` (script), `c37d558` (unique-line
correction), `bedb8fb` (results and the comparison).

This file is the handoff. If you are picking the project up on another machine, read this first.

---

## 1. Why this stage existed, and why it was nearly missed

`WORK_PLAN.md:392` defines Stage 6, `:457` assigns it, `:803` carries it as an examiner-verifiable
done-criterion, and `:848` plus `SETTLED_QUESTIONS.md:341` make it the **justification for the
shortlist design** rather than a curiosity. `project_full_specification.md:199` lists the four
deliverables and forbids constrained decoding.

It was never cut. The 16 September ruling (docket:229-235) rejected free generation as the *main
arm* while explicitly preserving it as the 200-paper behavioural study, "not either/or".

**The trap, recorded so nobody falls in it again.** From 22 September the project notes said "all
Gemma passes are done and scored". That was true of the 5a and 5b **selection** passes. Stage 6 is
not a selection pass, writes no `.npz`, and never appears in `results/gemma_select/scored/`, so
reading that directory alone makes the card look free when it is not. Walk the WORK_PLAN stage list,
not the results directory.

---

## 2. The headline

Unique-line out-of-vocabulary rate. Both conditions re-parsed from their stored `raw_output`
against the full 1,864-name vocabulary, so the measurement is identical in every cell.

| | free generation | with SciBERT's top 50 |
|---|---|---|
| untouched (5a) | **0.6496** | 0.0401 |
| fine-tuned (5b, epoch-2) | **0.6290** | 0.0259 |

Left unconstrained, Gemma puts about **65 % of the topics it names outside the UAT**. Given a
shortlist that falls to 4.0 % and 2.6 %, a 16x and 24x reduction. All four figures were recomputed
independently from the raw files and reproduce exactly.

**About 97 % of the out-of-vocabulary output is true hallucination**, not a spelling variant, so
this is not a formatting problem that a string normaliser would fix.

Full micro-averaged picture on the same 200 papers, gold mean 4.45 topics per paper:

| | free/untouched | free/fine-tuned | shortlist/untouched | shortlist/fine-tuned |
|---|---|---|---|---|
| unique topics per paper | 9.21 | 2.79 | 9.22 | 5.80 |
| emitted lines per paper | 9.21 | 9.59 | 9.23 | 10.63 |
| OOV rate (unique lines) | 0.6496 | 0.6290 | 0.0401 | 0.0259 |
| hallucination share of OOV | 0.969 | 0.937 | 0.973 | 0.933 |
| loop rate (papers) | 0.000 | 0.185 | 0.010 | 0.250 |
| micro precision | 0.2930 | 0.4976 | 0.2802 | 0.3248 |
| micro recall | 0.2126 | 0.1159 | 0.5579 | 0.4128 |
| micro F1 | 0.2464 | 0.1880 | 0.3731 | 0.3635 |

**The loss is recall, not precision.** Micro-F1 falls 0.373 to 0.246 untouched and 0.364 to 0.188
fine-tuned, while free-generation precision actually *rises* to 0.498 for the fine-tuned arm. That
arm emits only 2.79 unique topics against 5.80 with a shortlist. It is starvation, not accuracy.

Typical failure, from `free-5a_validation.jsonl`. The model names plausible astronomy that is not
UAT: `equation of state`, `machine learning`, `nuclear matter`, `accretion disks`, `spectral
analysis`. Checked against the thesaurus, UAT has `accretion`, `galaxy accretion disks` and
`stellar accretion disks`, but no bare `accretion disks`.

---

## 3. Two caveats that must travel with these numbers

**(a) The decoding loop is a fine-tuning artefact, not a free-generation one.** It fires on 18.5 %
of papers without a shortlist but **25.0 % with one**, against 1.0 % for the untouched model. The
21 September ruling recorded 13-21 % for epoch-end checkpoints, so the shortlist figure sits just
above that band. Worst case emitted `plasma` 124 times until it hit the token cap.

**(b) Because of the loop, line-weighted rates are wrong and unique-line rates are the reportable
ones.** The fine-tuned arm's OOV rate reads **0.521 per emitted line against 0.629 per unique
line**, a 10.8 point gap, because some loops land on in-vocabulary terms and dilute the numerator.
The first version of the script reported the line-weighted figure. `c37d558` fixed it, both arms
were re-run, and the two weightings agree exactly wherever the duplicate rate is zero, which is the
case for the untouched arm. **Quote the unique-line numbers.**

---

## 4. Method, and why the near-miss split is conservative

No candidate list in the prompt and no constrained decoding. The prompt is `gemma_common.py`'s
committed Stage 5 template with the candidate block and the "copied exactly as written above"
clause removed, kept as a local copy in the Stage 6 script so the frozen Stage 5 prompt is not
perturbed. The "between 1 and 10" cap is retained so cardinality stays comparable.

`max_new_tokens` is 256 where Stage 5 used 160. With no list to copy from, truncation would
understate both the duplicate rate and the cardinality. The truncation rate is reported and is
0.00 for the untouched arm and 0.10 for the fine-tuned one.

The in-vocabulary split reuses the committed `parse_picks()` rule unchanged, exact match after
case-folding and list-marker stripping, never fuzzy. The near-miss classification is a reporting
layer applied afterwards to the lines that rule rejected, never a second chance to score. Four
rungs, tried strongest first, each folding only naming differences over the same content words:

1. `near_miss_punct`, case, separators or punctuation only
2. `near_miss_plural`, singular and plural forms
3. `near_miss_stopword`, function words only, for example `the interstellar medium`
4. `near_miss_wordset`, same content words reordered, for example `clusters of galaxies`

No paraphrase and no synonym is ever folded. So **near-miss is a lower bound and hallucination is
an upper bound**, which is the direction that does not flatter the argument this stage supports.
Normalisation collisions inside the vocabulary are 0, 0, 5 and 6 of 1,864 and are logged in every
summary.

Papers are 200 sampled with seed 42 from the **tracked** handoff file
`results/scibert_full/scibert_full_top50_val.jsonl`, so every Stage 6 paper also has a
prompt-and-select counterpart and a fresh clone reproduces the sample. Validation only. The test
split is never touched by this stage.

---

## 5. Files

| path | what |
|---|---|
| `scripts/gemma_free_generation.py` | the pass, both arms |
| `results/gemma_free/free-5a_validation.{jsonl,summary.json}` | untouched arm, 200 papers |
| `results/gemma_free/free-5b_validation.{jsonl,summary.json}` | fine-tuned arm, epoch-2 adapter |
| `results/gemma_free/free-smoke_validation.*` | 8-paper smoke test |
| `scripts/analysis/stage6_free_vs_shortlist.{py,json,out}` | the free-vs-shortlist comparison, registered in that README |

Reproduce, from the repo root:

```
python scripts/gemma_free_generation.py --arm free-5a --n 200 --batch-sizes 8
python scripts/gemma_free_generation.py --arm free-5b --n 200 --batch-sizes 8 \
    --adapter results/gemma_train/5b/checkpoints_kept/epoch-2/adapter
python scripts/analysis/stage6_free_vs_shortlist.py
```

Cost was 0.51 s/paper untouched and 2.02 s/paper fine-tuned, peak 4.37 and 5.01 GB, about 100
seconds per arm. The comparison script is CPU only and needs no adapter, so **it re-runs on any
machine**. The two generation passes need the GPU box, because the epoch-2 adapter and the Gemma
weights are both gitignored and exist only on that disk.

---

## 6. What the paper must now say

1. **§4 Method gains the shortlist-justification paragraph.** It can now be argued from our own
   measurement rather than only from Alkan et al.'s report that full-vocabulary prompting
   "hallucinated concepts or became overwhelmed". The sentence is: unconstrained, 65 % of named
   topics are not UAT topics and 97 % of those are outright inventions; with a top-50 shortlist
   that is 4 % and 2.6 %.
2. **The recall framing, not the precision framing.** Free generation does not lose because it is
   inaccurate. It loses because it is starved, recall 0.213 and 0.116 against 0.558 and 0.413.
   Free-generation precision for the fine-tuned arm is the highest precision anywhere in the study
   at 0.498, and quoting that without the recall beside it would be misleading.
3. **The loop belongs in §8 Limitations,** with the honest attribution that it is a fine-tuning
   artefact present in both conditions and worse with a shortlist, not a free-generation pathology.
4. **AI Disclosure needs a Stage 6 entry**, written now while the stage is fresh, per the
   done-criterion at `WORK_PLAN.md:805` that the disclosure log has one entry per stage.
5. **Nothing already reported changes.** Stage 6 adds a condition. Every 5a, 5b, 5b-shuffled,
   5a-shuffled and step-8500 figure stands.

---

## 7. What is left, and where it can be done

Everything below is prose or CPU work and **none of it needs the GPU box**.

| item | where | note |
|---|---|---|
| §2, §3, §4, §5, §8, §9, §10 prose, about 3,676 words | any machine | paper is at 2,822 of 6,498 |
| ACL template into Overleaf | browser | still the one blocking external dependency |
| Figure 1 as PDF | any machine | it is the two-stage pipeline **diagram**, not a data plot, so TikZ or any vector tool. PNG is forbidden by guidelines:143 |
| `gemma3` and `frey2018uat` citations | any machine | `gemma3` has no on-disk source. Decide: find one offline or drop the citation. Do not web-search, the bibliographic facts are meant to come from `Articles/` and `WORK_PLAN.md:374-389` |
| remaining `% TODO verify` fields | any machine | QLoRA venue and year, three truncated author lists |
| owner for the AI-disclosure section | organisational | still unassigned |
| number-provenance sweep | any machine | every reported figure regenerated from one commit. All inputs are tracked, verified 25 Sept |

### Running on a different machine

Every path the `scripts/analysis/` scripts open is tracked, so a fresh clone can reproduce all of
them. Two setup steps:

1. `python scripts/download_data.py`. The analysis needs the HuggingFace dataset for gold labels
   and it is only **19 MB**. The 10 GB in `.hf_cache` is almost entirely Gemma weights, which
   nothing outside Stage 6 touches.
2. **Export `HF_HOME` once before running anything.** Nine of the older analysis scripts hardcode
   an absolute cache path from the machine they were first written on, while the newer ones use a
   repo-relative one. Because they all use `os.environ.setdefault`, exporting the variable makes
   every script agree:

   ```
   export HF_HOME=/path/to/your/clone/.hf_cache
   ```

   The older scripts are exactly the ones the provenance sweep re-runs, so set this first.

`requirements.txt` records that these pins were verified on Apple Silicon, M1, macOS 26, MPS
backend, as of 30 August 2026.

### Still to do on the GPU box itself

**Back up `results/gemma_train/5b/checkpoints_kept/epoch-2/` and `step-8500/`, 228 MB total.**
They are gitignored and exist on one disk. `epoch-2/adapter` *is* 5b. No reported number depends on
having the file, since everything is computed and tracked, but regenerating it costs roughly 15
hours of training.
