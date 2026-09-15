# Gemma probe: `google/gemma-3-1b-it`

**Date**: 2026-09-15 19:09  
**Code**: commit `e89c88c84c8a9a295f66d528bec54f2566ae21a9`  
**GPU**: NVIDIA GeForce RTX 3060 Ti (8.0 GB)  
**Stack**: torch 2.14.0+cu130 / transformers 5.17.0 / peft 0.20.0 / bitsandbytes 0.50.2

## Verdict: FITS -- use this size for 5b

Status: `completed` after 500 of 500 steps.

| metric | value |
|---|---|
| sequence length / batch | 1024 tokens / 1 |
| tokens per example (prompt+target) | min 504, median 773, max 936 (0 abstracts shortened); label tokens mean 17.3 |
| trainable params | 13,045,760 of 1,012,931,712 (1.288%) -- LoRA r=16, alpha=32 |
| weights on card after 4-bit load | 0.90 GB |
| peak memory allocated / reserved | 1.32 GB / 1.77 GB |
| seconds per step (mean / min / max), padded to 1024 | 0.58 / 0.56 / 0.80 |
| train-mode loss, first step -> last step | 10.2984 -> 0.3197 |
| train-mode mean loss, first pass -> last pass | 2.2920 -> 0.2771 |
| gradient norm, max / mean over last pass | 376.70 / 11.63 (clip 1.0) |
| all losses finite | True |
| **eval-mode loss per paper, median before -> after** | 6.084 -> 0.063 (max after 0.511) |
| papers with eval loss < 0.05 / < 0.10 / > 0.50 after training | 50% / 55% / 3% |
| model load time | 6 s |
| training wall-clock | 4.9 min |

Loss-path check on example 0: HF forward-with-labels 4.6237 vs completion-only 4.6237 (|diff| 0.00e+00).

## Greedy generation on 8 randomly sampled trained-on papers, before vs after training

| | recall | precision | mean picks (gold) | off-list lines | empty | exact target match |
|---|---|---|---|---|---|---|
| zero-shot (before) | 0.29 | 0.15 | 8.1 (4.2) | 40 | 0 | 0/8 |
| after training | 0.82 | 0.97 | 3.6 (4.2) | 1 | 0 | 5/8 |

Candidates here are gold + random fillers, so these numbers describe format, stopping and memorisation only -- never accuracy.

Adapter saved to `adapter/`. Loss curve and per-paper eval loss: `loss_curve.png`. Full per-step log, per-example losses and generations: `probe_log.json`.
