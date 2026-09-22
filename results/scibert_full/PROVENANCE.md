# results/scibert_full — provenance and status (20 Sept 2026)

**Source:** commit `366f46c` (Itai, 19 Sept 23:01), run stamped 19/09/2026 22:39. Replaces the 17 Sept tables
from `9586d53` in place (those are recoverable from git; they came from a model that had learned only the label
prior — val coverage@50 0.291, tail 0/2,787 — and must not be used).

| file | status | note |
|---|---|---|
| `scibert_full_val.parquet` (blob `d7dc410d5af5`) | **sound** | 2,855 × 1,864; ids == `split.json`; cols == `label_order.json`; val coverage@50 0.829 (head 0.980 / torso 0.908 / tail 0.528); max prob 0.997; random-pair top-50 Jaccard 0.067 |
| `scibert_full_test.parquet` (blob `72d82fc513c6`) | **sound** | 3,025 × 1,864; coverage@50 0.859 (tail 0.648) |
| `test_predictions_scibert_full.npz` | **regenerated 22 Sept — consistent with the JSON** | binary predictions at taus 0.92 / 0.82 / 0.54; **5.48 per paper** (was 15.3 per paper at the truncated 0.54/0.54/0.54) |
| `tau_scibert_full.json`, `test_results_scibert_full.json` | **re-scored 22 Sept — reportable** | Regenerated under the corrected `TAU_CANDIDATES` grid (0.0005–0.98, a strict superset of the old 0.005–0.54). Chosen taus are now **0.92 / 0.82 / 0.54, all interior** — the search is no longer truncated. Test micro-F1 head **0.5518**, torso **0.3566**, tail **0.2150**; predictions/paper 15.3 → **5.48** (gold 4.3). Reproduces the 20 Sept chair re-run exactly. |

**Recorded run provenance is incomplete:** `commit = dcb82e2` (Shai's HEAD) with `has_tracked_changes = true`;
epochs completed, learning rate, `--force-restart` vs resume, and whether `best_model.pt` or `checkpoint.pt`
produced the tensors are not recorded anywhere in the repo. Trainer changes in `366f46c`: `BCEWithLogitsLoss(pos_weight=30)`,
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
everything" (532 predictions per paper against 4.3 gold). That is a property of the model, not of the
sweep, and it should be reported as such rather than re-tuned further.

**The majority baseline's tail is no longer exactly zero** (0.0055) now that its tail tau is interior,
but it remains near-zero for the structural reason that a frequency prior cannot rank a rare label
above a common one.
