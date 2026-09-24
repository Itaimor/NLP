# scripts/analysis — the measurements behind the reported numbers

One-off analysis scripts, kept because figures they produced are reported in the
paper. They are not part of the training or scoring pipeline (`scripts/main.py`,
`baselines.py`, `evaluate.py`, `gemma_*.py`, `score_gemma.py`); each was written to
answer one question and is preserved so the number it produced can be traced.

Several of the older scripts carry an absolute `HF_HOME` path from the machine they
were first run on. Point it at this repository's `.hf_cache` before re-running.

| script | what it measured | where it is reported |
|---|---|---|
| `threshold_sweep.py` | rare-band F1 against the decision threshold — 0.0078 at the 0.5 default vs 0.324 tuned | Introduction, the ~40x threshold claim |
| `global_vs_band_tau.py` | one threshold for all bands vs one per band — 0.156 / 0.258 / 0.324 | Introduction, the further 2x |
| `honest_tau_macro.py` | rare-band macro-F1 under each denominator — 0.32 vs 0.08 | Introduction, the denominator claim |
| `scoreability_curve.py`, `unscoreable_probe.py` | how many rare concepts have no test instance — 1,059 of 1,469 | Introduction, the scoreability claim |
| `recall_ceiling.py` | coverage@50 of the shortlist — the ceiling any selector inherits | Method / Results |
| `carve_cost_tfidf.py` | cost of carving the validation split | Method |
| `insample_gap.py` | in-sample vs out-of-sample shortlist skew | Method, the shuffle motivation |
| `bootstrap_mde.py` | minimum detectable difference for the planned comparison | pre-registration |
| `majority_baseline.py` | the frequency-prior baseline, later superseded by `scripts/baselines.py` | Results (baseline row) |
| `rescore_corrected_tau_grid.py` | re-selected tau per band under the corrected grid and re-scored every probability-emitting arm; output in `rescore_corrected_tau_grid.json` | Results, all encoder and baseline rows |
| `bootstrap_gemma_vs_scibert.py` | paired bootstrap, 2000 resamples, seed 42, per-band micro-F1 differences vs SciBERT-alone; output in `bootstrap_gemma_vs_scibert.json` | Introduction and Results, every confidence interval |
| `tfidf_floor_limit.py` | whether TF-IDF+LR's floor-pinned tau is a grid artefact — the closed-form `tau -> 0` predict-all limit against the chosen tau, per band, per arm; output in `tfidf_floor_limit.json` | Results / Baselines, the sentence defusing the TF-IDF floor |
| `bootstrap_order_robustness.py` | paired bootstrap, 2000 resamples, seed 42, for the {5a, 5b} x {SciBERT order, shuffled} design — each arm's order dependence and each shuffled arm's residual gain over SciBERT-alone; output in `bootstrap_order_robustness.json` | Introduction and Analysis, the candidate-order paragraph |

`.out` files are the captured stdout of the run that produced the figure.
