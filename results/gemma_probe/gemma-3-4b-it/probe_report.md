# Gemma probe: `google/gemma-3-4b-it`

**Date**: 2026-09-15 13:15  
**GPU**: NVIDIA GeForce RTX 3060 Ti (8.0 GB)  
**Stack**: torch 2.14.0+cu130 / transformers 5.17.0 / peft 0.20.0 / bitsandbytes 0.50.2

## Verdict: FITS -- use this size for 5b

Status: `completed` after 500 of 500 steps.

| metric | value |
|---|---|
| sequence length / batch | 1024 tokens / 1 |
| tokens per example (prompt+target) | min 505, median 774, max 937 (0 abstracts shortened) |
| trainable params | 29,802,496 of 4,300,079,472 (0.693%) -- LoRA r=16, alpha=32 |
| weights on card after 4-bit load | 3.01 GB |
| peak memory allocated / reserved | 3.87 GB / 4.15 GB |
| seconds per step (mean / min / max) | 1.72 / 1.60 / 2.39 |
| loss, first step -> last step | 4.5858 -> 0.0256 |
| mean loss, first pass -> last pass over the 64 papers | 1.2799 -> 0.2394 |
| all losses finite | True |
| model load time | 22 s |
| training wall-clock | 14.4 min |

Loss-path check on example 0: HF forward-with-labels 3.1361 vs completion-only 3.1361 (|diff| 0.00e+00).

## Overfit check: greedy generation on trained-on papers

- `2020ApJ...891...91D`: 6/6 gold recovered, 9 picks, 0 off-list
- `2020ApJ...891..106A`: 2/2 gold recovered, 3 picks, 0 off-list
- `2020ApJ...892..134W`: 3/3 gold recovered, 3 picks, 0 off-list
- `2020ApJ...892L..35B`: 8/8 gold recovered, 8 picks, 2 off-list

Loss curve: `loss_curve.png`. Full per-step log: `probe_log.json`.
