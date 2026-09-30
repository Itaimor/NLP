# results/scibert_full — provenance and status (20 Sept 2026)

**Source:** commit `366f46c` (Itai, 19 Sept 23:01), run stamped 19/09/2026 22:39. Replaces the 17 Sept tables
from `9586d53` in place (those are recoverable from git; they came from a model that had learned only the label
prior — val coverage@50 0.291, tail 0/2,787 — and must not be used).

| file | status | note |
|---|---|---|
| `scibert_full_val.parquet` (blob `d7dc410d5af5`) | **sound** | 2,855 × 1,864; ids == `split.json`; cols == `label_order.json`; val coverage@50 0.829 (head 0.980 / torso 0.908 / tail 0.528); max prob 0.997; random-pair top-50 Jaccard 0.067 |
| `scibert_full_test.parquet` (blob `72d82fc513c6`) | **sound** | 3,025 × 1,864; coverage@50 0.859 (head 0.982 / torso 0.900 / tail 0.648) — head/torso added 27 Sept, recomputed 1328/1353 and 8346/9269 from this blob and cross-checked against `scibert_full_top50_test.jsonl` (identical ordered top-50, no rank-50 ties) |
| `test_predictions_scibert_full.npz` | **regenerated 22 Sept — consistent with the JSON** | binary predictions at taus 0.92 / 0.82 / 0.54; **5.48 per paper** (was 15.3 per paper at the truncated 0.54/0.54/0.54) |
| `tau_scibert_full.json`, `test_results_scibert_full.json` | **re-scored 22 Sept — reportable** | Regenerated under the corrected `TAU_CANDIDATES` grid (0.0005–0.98, a strict superset of the old 0.005–0.54). Chosen taus are now **0.92 / 0.82 / 0.54, all interior** — the search is no longer truncated. Test micro-F1 head **0.5518**, torso **0.3566**, tail **0.2150**; predictions/paper 15.3 → **5.48** (gold 4.38 on test). Reproduces the 20 Sept chair re-run exactly. |

**Recorded run provenance is incomplete:** `commit = dcb82e2` (Shai's HEAD) with `has_tracked_changes = true`;
`--force-restart` vs resume, and whether `best_model.pt` or `checkpoint.pt`
produced the tensors are not recorded anywhere in the repo. **Superseded in part:** epochs completed (8),
both learning rates (2e-5 backbone / 1e-4 head), batch size, seed and seconds per step ARE now recorded, in
`results/scibert_full/cost.json`. Trainer changes in `366f46c`: `BCEWithLogitsLoss(pos_weight=30)`,
checkpoint selection on validation coverage@50, head LR ≥ 5× backbone.

**Derived here:** `results/shortlists/scibert_full_top50_{val,test}.jsonl` via `scripts/derive_scibert_shortlists.py`
(top-50 argsort of the tables above; identical to the exporter's `torch.topk(50)` up to float ties).
**Handoff lists (commit `e037005`, Itai, 20 Sept 10:37):** `scibert_full_top50_{train,val,test}.jsonl` + `gemma_candidates_*.json`
in this directory. **Verified 20 Sept 10:45:** the exported val and test lists equal the argsort of the parquet above
row for row (2,855/2,855 and 3,025/3,025, exact order, score diff 0.0) — so the train list comes from the same checkpoint.
Train list (blob `4bdc8b795620`): ids == `split.json[train]`, 50 distinct names/row, `gold_in_list` == gold ∩ candidates in
score order (0 mismatches), 38 `["NONE"]` rows (99.8 % non-empty). **In-sample coverage@50** all 0.910 / head 0.997 /
torso 0.966 / **tail 0.710** (median gold rank 6, tail 18) vs validation 0.829 / tail 0.528 and test 0.859 / tail 0.648 —
the pre-registered skew table; tail skew is +0.06 over test, far below the TF-IDF proxy's (0.98 in-sample vs 0.90 test).
The Gemma scripts read these through `results/shortlists/` (`scripts/derive_scibert_shortlists.py` re-keys
`candidate_names` → `candidates`, `["NONE"]` → `[]`).

Council session: `.council/session-2026-09-20_itai-19sept-push-readiness/`.

## 22 September 2026 — tau grid corrected, all arms re-scored

`scripts/evaluate.py` `TAU_CANDIDATES` was bounded at **both** ends (0.005–0.54), so any arm whose
optimum lay outside that window had its threshold search truncated rather than resolved. The grid now
spans 0.0005–0.98 and is a strict superset of the old one, so any arm whose optimum was already
interior keeps its exact previous tau (verified: the majority baseline's head 0.04 and torso 0.02 are
unchanged). Every probability-emitting arm was re-scored: `scripts/` was not otherwise modified, and
the re-score is reproducible via `.council/session-2026-09-22_next-step-after-itai-push/rescore.py`.

| arm | taus before | taus after | grid position now | test micro-F1 h/t/t | preds/paper |
|---|---|---|---|---|---|
| SciBERT-full | 0.54 / 0.54 / 0.54 (ceiling) | **0.92 / 0.82 / 0.54** | all interior | **0.5518 / 0.3566 / 0.2150** | 15.3 → **5.48** |
| TF-IDF + LR | 0.005 ×3 (floor) | 0.0005 / 0.0005 / 0.005 | **head+torso still at floor** | 0.0719 / 0.0159 / 0.0014 | 167 → **532** |
| majority | 0.04 / 0.02 / 0.005 | 0.04 / 0.02 / **0.0025** | all interior | 0.0914 / 0.0385 / **0.0055** | 18.0 → **37.0** |

**TF-IDF+LR remains pinned at the floor and is not a grid artefact.** Its validation Micro-F1 still
improves as the threshold approaches zero, i.e. its best available operating point is "predict almost
everything" (532 predictions per paper against 4.38 gold on test). That is a property of the model, not of the
sweep, and it should be reported as such rather than re-tuned further.

**The majority baseline's tail is no longer exactly zero** (0.0055) now that its tail tau is interior,
but it remains near-zero for the structural reason that a frequency prior cannot rank a rare label
above a common one.

## 23 September 2026 — `cost.json` parameter count is wrong: it was computed against 2,079 labels

`cost.json` records `total_parameters` (and `trainable_parameters`) as **111,517,215**. That figure is
exactly `109,918,464` (SciBERT backbone: vocab 31,090, 12 layers, hidden 768, with pooler) `+ 768*2079 + 2079`
— i.e. it was computed with a **2,079-wide** classification head. The corpus has **1,864** labels, confirmed
independently by `label_order.json` / `id_to_name.json` (1,864 unique ids), both band maps (1,469+383+12 and
1,418+429+17, each summing to 1,864), `2,367 defined − 503 never occurring = 1,864` in `WORK_PLAN.md:23`,
the committed val/test parquets (1,865 columns = 1,864 label columns matching `label_order.json` exactly and
in order, + `paper_id`), and `test_predictions_scibert_full.npz` (`predictions (3025, 1864)`).

**Correct value: `109,918,464 + 768*1864 + 1864` = 111,351,880 ≈ 111.4M.** Owner: Itai (asked 23 Sept).
`paper/method.tex` has been set to **111.4M** with a note not to re-copy the figure from `cost.json` until
it is corrected. No other number is affected: the probability tables, the taus, and every metric in
`test_results_scibert_full.json` are independent of this count.

The same 2,079 error appeared historically in early draft notes before being corrected to 1,864. In the LoRA artifacts it is demonstrably prose-only:
the recorded trainable count, 1,728,328, equals `294,912 (LoRA r=8, q&v, 12 layers) + 768*1864 + 1864`
to the parameter, confirming the code ran at 1,864. (The initial printed *total* in PEFT, 113,080,208, double-counts the head via
PEFT's `modules_to_save=["classifier"]` wrapper; the true total is 111,646,792 and the trainable share
1.55%, not 1.53%.)

Council session: `.council/docket.md` — entry [2026-09-23].

## 23 September 2026 — gold mean per paper corrected to 4.38 on test

The paper carried three different gold means (4.3, 4.31 and "4.3 gold") for the same quantity.
Recomputed two independent ways, agreeing exactly:

- **Band maps:** `band_map_corpus` 93,547 instances / 21,702 papers minus `band_map_train`
  80,308 / 18,677 papers leaves **13,239 test instances over 3,025 papers = 4.3765**.
- **Confusion counts:** `test_results_scibert_full.json` `per_label_counts`, tp 5,122 + fn 8,117
  = **13,239**. Cross-check, (tp + fp)/3,025 = 5.48, reproducing the recorded predictions/paper.

| basis | papers | gold instances | mean |
|---|---|---|---|
| project train (`split.json["train"]`) | 15,822 | 68,038 | 4.3002 |
| project validation | 2,855 | 12,270 | 4.2977 |
| HF training pool (= train + validation) | 18,677 | 80,308 | 4.2998 |
| corpus-wide (Alkan et al. Table 2) | 21,702 | 93,547 | **4.31** |
| **project test** (= HF `val`) | **3,025** | **13,239** | **4.3765 -> 4.38** |

**Corrected 23 Sept:** an earlier version of this table called the 18,677-paper row the
"train split". It is the HF training **pool**; the project's train split is 15,822 papers.
The three train-side bases all round to 4.30, so no downstream number was affected.
**4.31 is Alkan et al.'s published figure** (their Table 2, "21,702 abstracts with 93,547
total label assignments, averaging 4.31") and our corpus count reproduces it exactly, which
validates the split pipeline. Their P@k uses k in {1,3,5} "chosen to align with ... (4.31)",
so `WORK_PLAN.md:133` and `project_full_specification.md:219` correctly keep 4.31.

**Every predictions-per-paper comparison in the paper is scored on test** (5a 8.79, 5b 5.68,
SciBERT-alone 5.48, TF-IDF 532), so the correct comparator is **4.38**, not 4.3 or 4.31.
Updated in `introduction.tex`, `method.tex`, `results.tex` (x2) and this file (x2).
`task_and_data.tex` keeps the corpus-wide 4.31 because it describes the corpus, and now names
the basis explicitly so the three figures are not mixed again.
