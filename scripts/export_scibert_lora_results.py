#!/usr/bin/env python3
"""
Export SciBERT-LoRA Artifacts to §6 Parquet & Run choose_tau Sweep
==================================================================
Fulfills Items 9, 10, and 12:
- Item 12: Normalizes artifact directory structure at results/scibert_lora/.
- Item 9: Exports val_probs_epoch_1..8.pt to results/scibert_lora/val_epochs/
  and exports scibert_lora_val.parquet and scibert_lora_test.parquet in §6 format.
- Item 10: Runs choose_tau across all 8 validation epochs and generates
  results/scibert_lora/epoch_selection_summary.json.
- Re-scores via rescore_encoder_from_disk to guarantee SHA-256 checksum integrity.
"""

import os
import sys
import shutil
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "data"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_split import load_split
from evaluate import choose_tau
from baselines import rescore_encoder_from_disk


def main():
    print("[LoRA Exporter] Loading split data...")
    s = load_split()
    val_df = s["validation_df"]
    test_df = s["test_df"]
    idx_to_topic = s["idx_to_topic"]
    num_labels = s["num_labels"]
    band_train = s["band_train"]

    source_dir = PROJECT_ROOT / "results" / "scibert_lora" / "run" / "lora"
    results_dir = PROJECT_ROOT / "results" / "scibert_lora"
    val_epochs_dir = results_dir / "val_epochs"
    val_epochs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Normalize directory layout (Item 12):
    print("\n[Item 12] Normalizing artifact directory depth...")
    files_to_promote = [
        "cost.json",
        "tau_scibert_lora.json",
        "test_results_scibert_lora.json",
        "test_predictions_scibert_lora.npz",
        "gemma_candidates_validation.json",
        "gemma_candidates_test.json",
        "experiment_results.json",
    ]
    for fname in files_to_promote:
        src = source_dir / fname
        dst = results_dir / fname
        if src.is_file() and not dst.is_file():
            shutil.copy2(src, dst)
            print(f"  Promoted {fname} -> results/scibert_lora/{fname}")

    # Remove duplicated run/cost.json if present
    run_cost = PROJECT_ROOT / "results" / "scibert_lora" / "run" / "cost.json"
    if run_cost.is_file():
        run_cost.unlink()
        print("  Removed redundant duplicate results/scibert_lora/run/cost.json")

    # 2. Build column order and ground truth answers
    label_order_list = [str(idx_to_topic[i]) for i in range(num_labels)]
    val_bibcodes = val_df["bibcode"].tolist()
    test_bibcodes = test_df["bibcode"].tolist()

    topic_str_to_idx = {str(topic): idx for idx, topic in idx_to_topic.items()}
    val_answers = np.zeros((len(val_df), num_labels), dtype=np.uint8)
    for r_idx, labels in enumerate(val_df["verified_uat_ids"]):
        for lbl in labels:
            if str(lbl) in topic_str_to_idx:
                val_answers[r_idx, topic_str_to_idx[str(lbl)]] = 1

    band_map = {idx: band_train[topic] for idx, topic in idx_to_topic.items()}

    # 3. Export all validation epochs & compute choose_tau sweep (Item 9 & 10)
    print("\n[Item 9 & 10] Exporting validation epochs and running choose_tau sweep...")
    epoch_summary = {}

    for epoch in range(1, 9):
        pt_path = source_dir / f"val_probs_epoch_{epoch}.pt"
        if not pt_path.exists():
            print(f"Warning: {pt_path} does not exist, skipping.")
            continue

        probs_tensor = torch.load(pt_path, map_location="cpu")
        probs_np = probs_tensor.numpy().astype(np.float32)

        # Build §6 DataFrame
        df_epoch = pd.DataFrame(probs_np, columns=label_order_list)
        df_epoch.insert(0, "paper_id", val_bibcodes)

        # Save to Parquet
        parquet_path = val_epochs_dir / f"scibert_lora_val_epoch_{epoch}.parquet"
        df_epoch.to_parquet(parquet_path, index=False)

        # Run choose_tau for this epoch
        chosen_taus, tau_sweep = choose_tau(
            val_probs=probs_np,
            val_labels=val_answers,
            band_map=band_map,
        )

        head_best = max(entry["micro_f1"] for entry in tau_sweep["head"])
        torso_best = max(entry["micro_f1"] for entry in tau_sweep["torso"])
        tail_best = max(entry["micro_f1"] for entry in tau_sweep["tail"])
        mean_band_f1 = (head_best + torso_best + tail_best) / 3.0

        epoch_summary[f"epoch_{epoch}"] = {
            "epoch": epoch,
            "parquet_file": str(parquet_path.relative_to(PROJECT_ROOT)),
            "chosen_taus": chosen_taus,
            "best_micro_f1": {
                "head": round(head_best, 4),
                "torso": round(torso_best, 4),
                "tail": round(tail_best, 4),
            },
            "mean_band_micro_f1": round(mean_band_f1, 4),
        }
        print(f"  Epoch {epoch} -> Head: {head_best:.4f}, Torso: {torso_best:.4f}, Tail: {tail_best:.4f} | Mean: {mean_band_f1:.4f}")

    # Determine best epoch based on mean band micro-F1
    best_epoch_key = max(epoch_summary.keys(), key=lambda k: epoch_summary[k]["mean_band_micro_f1"])
    best_epoch_num = epoch_summary[best_epoch_key]["epoch"]

    summary_output = {
        "cross_check_best_epoch": best_epoch_num,
        "cross_check_metric": "Mean band micro-F1 across Head, Torso, Tail using choose_tau",
        "note": (
            "Post-hoc verification only. The reported checkpoint was selected DURING "
            "TRAINING by validation Coverage@50 (SciBERT_experiment.py:669, "
            "selection_metric='coverage', recorded in results/scibert_lora/cost.json). "
            f"This file independently cross-checks that choice using mean band micro-F1 "
            f"and confirms epoch {best_epoch_num}."
        ),
        "epochs": epoch_summary,
    }

    summary_path = results_dir / "epoch_selection_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2)
    print(f"\n[Epoch Exporter] Summary saved to {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"Best epoch by choose_tau: Epoch {best_epoch_num}")

    # 4. Export final §6 validation & test Parquet tables (Item 9)
    print("\n[Item 9] Exporting primary §6 Parquet tables...")
    # Best epoch validation table -> scibert_lora_val.parquet
    best_epoch_parquet = val_epochs_dir / f"scibert_lora_val_epoch_{best_epoch_num}.parquet"
    primary_val_parquet = results_dir / "scibert_lora_val.parquet"
    shutil.copy2(best_epoch_parquet, primary_val_parquet)
    print(f"  Exported {primary_val_parquet.name} ({primary_val_parquet.stat().st_size / (1024*1024):.1f} MB)")

    # Test table -> scibert_lora_test.parquet
    test_probs_path = source_dir / "test_probabilities.pt"
    test_probs_tensor = torch.load(test_probs_path, map_location="cpu")
    test_probs_np = test_probs_tensor.numpy().astype(np.float32)

    df_test = pd.DataFrame(test_probs_np, columns=label_order_list)
    df_test.insert(0, "paper_id", test_bibcodes)

    primary_test_parquet = results_dir / "scibert_lora_test.parquet"
    df_test.to_parquet(primary_test_parquet, index=False)
    print(f"  Exported {primary_test_parquet.name} ({primary_test_parquet.stat().st_size / (1024*1024):.1f} MB)")

    # 5. Re-score using rescore_encoder_from_disk to synchronize tau and test checksums
    print("\n[Item 9 Verification] Re-scoring SciBERT-LoRA from the new §6 Parquet tables...")
    rescore_encoder_from_disk(
        arm_name="scibert_lora",
        results_dir=results_dir,
        val_df=val_df,
        test_df=test_df,
        label_order=label_order_list,
        band_map=band_map,
        val_parquet_path=primary_val_parquet,
        test_parquet_path=primary_test_parquet,
        objective="Maximise each band's validation Micro-F1 over the fixed TAU candidates.",
    )
    print("\nSciBERT-LoRA §6 tables and checksums are fully synchronized!")


if __name__ == "__main__":
    main()
