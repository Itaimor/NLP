# M5 Hardware Smoke Test Report (SciBERT Stage 3 Gate)

**Date**: 2026-09-13  
**Machine / Architecture**: arm64 (arm)  
**Operating System**: macOS-26.6.2-arm64-arm-64bit-Mach-O  
**Python**: 3.14.6 | **PyTorch**: 2.14.0  
**Backend**: `MPS` (MPS Built: True, MPS Available: True)

---

## 1. Benchmark Configuration
Matches `project_full_specification.md` §4.2.2 & §5:
- **Base Model**: `allenai/scibert_scivocab_uncased` (110M parameters)
- **Classification Head**: 1,864 labels (Multi-label classification via `BCEWithLogitsLoss`)
- **Batch Size**: 8
- **Sequence Length**: 512 tokens
- **Optimizer**: AdamW (`lr=2e-5`)
- **Steps**: 2 warmup + 20 measured training steps

---

## 2. Measured Results

| Metric | Measured Value | Comparison / Reference |
|---|---|---|
| **Average Step Time** | **1.096 s/step** | M1: 2.29 s/step \| CPU: 3.4 s/step |
| **Speedup vs M1** | **2.09x faster** | Target: > 1.0x |
| **Minimum Step Time** | 1.076 s/step | |
| **Maximum Step Time** | 1.128 s/step | |
| **Initial Loss** | 0.6849 | |
| **Final Loss (Step 20)** | 0.5883 | Finite, no NaNs |
| **Est. Time per Epoch** | **~36.3 minutes** (0.60 hours) | M1: ~76 min/epoch |
| **Est. Full Run (8 Epochs)** | **~4.84 hours** | Fits comfortably within daily schedule |

### Step-by-Step Latency
```text
Step 01: 1.101 s
Step 02: 1.099 s
Step 03: 1.094 s
Step 04: 1.094 s
Step 05: 1.090 s
Step 06: 1.096 s
Step 07: 1.086 s
Step 08: 1.080 s
Step 09: 1.082 s
Step 10: 1.078 s
Step 11: 1.079 s
Step 12: 1.076 s
Step 13: 1.079 s
Step 14: 1.127 s
Step 15: 1.128 s
Step 16: 1.121 s
Step 17: 1.108 s
Step 18: 1.104 s
Step 19: 1.104 s
Step 20: 1.101 s
```

---

## 3. Decision for Stage 3 (SciBERT Training)

> **Gate Decision: PASS — SciBERT trains on this M5 machine.**

- **MPS Acceleration**: Fully operational without silent CPU fallback.
- **Loss Behavior**: Strictly finite, gradients propagating properly.
- **Feasibility**: A full 8-epoch training run over ~15,876 papers will take approximately **4.84 hours**, confirming that the M5 is the primary compute platform for both SciBERT Full and SciBERT LoRA arms.
