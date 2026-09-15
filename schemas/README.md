# File Schemas (WORK_PLAN.md §6)

Three-row example files, on real bibcodes and real UAT topic names, so the scorer and the
shortlist producers can be written against real bytes instead of a sentence. Full prose spec:
`WORK_PLAN.md` §6 ("The one thing we must agree on first").

## `example_shortlist_val.jsonl` / `example_shortlist_train.jsonl`
**Producer: Itai (SciBERT).** One line per paper.
- `paper_id`: the bibcode.
- `candidates`: the 50 topic names SciBERT ranks highest for this paper, **in rank order**
  (highest score first). Validation and test files stop here — this is what handoff 4 delivers
  for `scibert_full_top50_val.jsonl` / `_test.jsonl`.
- `gold_in_list` (train file only, handoff 4b / `scibert_full_top50_train.jsonl`): the paper's
  gold topics that also appear in `candidates`, **ordered by SciBERT's score, highest first**.
  This is the training target Gemma's fine-tuning reads (`gemma_train.py`).

Real bytes at this shape: `results/shortlists/standin_tfidf_top50_val.jsonl` and `_train.jsonl`
(built by `scripts/standin_shortlist.py` from TF-IDF, a stand-in until SciBERT's lists exist).

## `example_gemma_picks.jsonl`
**Producer: Ilana (Gemma, both 5a and 5b), consumer: Shai (`evaluate.py`).** One line per paper.
- `paper_id`: the bibcode.
- `candidates_shown`: the 50 names exactly as Gemma saw them — SciBERT's order for validation
  and test, a shuffled order for 5b's *training* prompts only (never shuffled at val/test).
- `picks`: Gemma's ordered, on-list picks, most-confident-first (parsed by
  `gemma_common.parse_picks`, case-insensitive exact match). This is the predicted set for F1
  and the ranked list for P@k (position *r* → score 1 − r/51, only among the picks actually
  made — never padded below the last one).
- `off_list`: raw output lines that matched no candidate, kept (never dropped) so the off-list
  rate can be computed.
- `raw_output`: the full decoded generation, for debugging the parser.

Real bytes at this shape: `results/gemma_select/smoke4b_validation.jsonl` (untouched
gemma-3-4b-it on the TF-IDF stand-in shortlist, `scripts/gemma_select.py`).

## What this unblocks
`evaluate.py` (Shai, not yet in the repo) reads exactly these two file shapes — the encoder
score tables (`<arm>_val.parquet` / `_test.parquet`, §6, not shown here) and this Gemma picks
shape — and produces the results table. Agreeing this file *is* starting `evaluate.py`
(WORK_PLAN.md: "Shai: read §6 and start `evaluate.py` against the three-row example files").
