# Gemma probe: `google/gemma-3-4b-it`

**Date**: 2026-09-15 19:02  
**Code**: commit `c33eddc8e092ee4808a5c9a5b5607177a48a4be8`  
**GPU**: NVIDIA GeForce RTX 3060 Ti (8.0 GB)  
**Stack**: torch 2.14.0+cu130 / transformers 5.17.0 / peft 0.20.0 / bitsandbytes 0.50.2

## Verdict: FITS -- use this size for 5b

Status: `completed` after 500 of 500 steps.

| metric | value |
|---|---|
| sequence length / batch | 1024 tokens / 1 |
| tokens per example (prompt+target) | min 504, median 773, max 936 (0 abstracts shortened); label tokens mean 17.3 |
| trainable params | 29,802,496 of 4,329,881,968 (0.688%) -- LoRA r=16, alpha=32 |
| weights on card after 4-bit load | 3.01 GB |
| peak memory allocated / reserved | 3.87 GB / 4.15 GB |
| seconds per step (mean / min / max), padded to 1024 | 1.63 / 1.61 / 1.93 |
| train-mode loss, first step -> last step | 2.9090 -> 0.1939 |
| train-mode mean loss, first pass -> last pass | 1.2600 -> 0.2404 |
| gradient norm, max / mean over last pass | 248.54 / 7.26 (clip 1.0) |
| all losses finite | True |
| **eval-mode loss per paper, median before -> after** | 4.394 -> 0.024 (max after 1.923) |
| papers with eval loss < 0.05 / < 0.10 / > 0.50 after training | 61% / 72% / 3% |
| model load time | 14 s |
| training wall-clock | 13.5 min |

Loss-path check on example 0: HF forward-with-labels 2.5715 vs completion-only 2.5715 (|diff| 0.00e+00).

## Greedy generation on 8 randomly sampled trained-on papers, before vs after training

| | recall | precision | mean picks (gold) | off-list lines | empty | exact target match |
|---|---|---|---|---|---|---|
| zero-shot (before) | 0.76 | 0.55 | 5.9 (4.2) | 0 | 0 | 0/8 |
| after training | 0.94 | 0.89 | 4.5 (4.2) | 1 | 0 | 5/8 |

Candidates here are gold + random fillers, so these numbers describe format, stopping and memorisation only -- never accuracy.

Adapter saved to `adapter/`. Loss curve and per-paper eval loss: `loss_curve.png`. Full per-step log, per-example losses and generations: `probe_log.json`.
