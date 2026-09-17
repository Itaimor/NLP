#!/usr/bin/env python3
"""
Export SciBERT-Full Handoffs & Official Evaluation
===================================================
Produces all required Stage 3/4 deliverables for SciBERT-Full:
1. Handoff 4b: scibert_full_top50_train.jsonl (15,822 training shortlists + gold_in_list)
2. Handoff 5: scibert_full_val.parquet & scibert_full_test.parquet (§6 schema)
3. Official evaluation via Shai's evaluate.py:
   - choose_tau() on validation split
   - tau_scibert_full.json
   - evaluate_test() on test split
   - test_results_scibert_full.json & test_predictions_scibert_full.npz
"""

import os
import sys
import json
import time
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "data"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_split import load_split
from scibert_dataset import SciXDataset
from SciBERT_experiment import build_scibert, load_model_state_dict
from evaluate import (
    choose_tau,
    save_tau_selection,
    evaluate_test,
    save_test_results,
    BANDS,
)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[SciBERT Handoffs] Using device: {device}")

    # 1. Load Stage 1 split data
    print("[SciBERT Handoffs] Loading split data...")
    s = load_split()
    train_df = s["train_df"]
    val_df = s["validation_df"]
    test_df = s["test_df"]
    topic_to_idx = s["topic_to_idx"]
    idx_to_topic = s["idx_to_topic"]
    id_to_name = s["id_to_name"]
    num_labels = s["num_labels"]
    band_train = s["band_train"]

    arm_dir = PROJECT_ROOT / "artifacts" / "scibert" / "full"
    results_dir = PROJECT_ROOT / "results" / "scibert_full"
    results_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # TASK 1: Generate Handoff 4b (scibert_full_top50_train.jsonl)
    # -------------------------------------------------------------
    train_shortlist_path = arm_dir / "scibert_full_top50_train.jsonl"
    if train_shortlist_path.exists():
        print(f"[SciBERT Handoffs] Training shortlist already exists at {train_shortlist_path}")
    else:
        print(f"\n[SciBERT Handoffs] Generating Handoff 4b: Training top-50 shortlists ({len(train_df)} papers)...")
        # Load model with Epoch 8 weights
        model = build_scibert(num_labels=num_labels, mode="full")
        best_model_path = arm_dir / "best_model.pt"
        checkpoint_path = arm_dir / "checkpoint.pt"

        if best_model_path.exists():
            print(f"Loading weights from {best_model_path}...")
            cp = torch.load(best_model_path, map_location="cpu", weights_only=False)
            sd = cp["model_state_dict"] if isinstance(cp, dict) and "model_state_dict" in cp else cp
            load_model_state_dict(model, sd)
        elif checkpoint_path.exists():
            print(f"Loading weights from {checkpoint_path}...")
            cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            sd = cp["model_state_dict"] if isinstance(cp, dict) and "model_state_dict" in cp else cp
            load_model_state_dict(model, sd)
        else:
            raise FileNotFoundError("Neither best_model.pt nor checkpoint.pt found!")

        model.to(device)
        model.eval()

        train_dataset = SciXDataset(
            dataframe_or_records=train_df,
            topic_to_idx=topic_to_idx,
            max_length=512,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
        )

        train_bibcodes = train_df["bibcode"].tolist()
        verified_labels = train_df["verified_uat_ids"].tolist()

        train_jsonl_lines = []
        train_json_dict = {}

        paper_cursor = 0
        start_time = time.time()
        with torch.no_grad():
            for input_ids, attention_mask, _ in tqdm(train_loader, desc="Train Set Inference"):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = torch.sigmoid(logits)

                scores_batch, top_idx_batch = torch.topk(probs, k=50, dim=1)
                scores_batch_np = scores_batch.cpu().numpy()
                top_idx_batch_np = top_idx_batch.cpu().numpy()

                batch_size_cur = input_ids.size(0)
                for b in range(batch_size_cur):
                    idx = paper_cursor + b
                    bibcode = train_bibcodes[idx]
                    gold_ids = verified_labels[idx]
                    if hasattr(gold_ids, "tolist"):
                        gold_ids = gold_ids.tolist()
                    gold_ids_set = {int(g) for g in gold_ids}
                    gold_names_set = {id_to_name.get(gid, str(gid)) for gid in gold_ids_set}

                    top_idx = top_idx_batch_np[b]
                    scores = scores_batch_np[b]

                    candidate_ids = [idx_to_topic[int(t)] for t in top_idx]
                    candidate_names = [id_to_name.get(cid, str(cid)) for cid in candidate_ids]
                    score_vals = [round(float(s), 5) for s in scores]

                    # Target: correct topics that made the list (gold ∩ shortlist), ordered by SciBERT score
                    gold_in_list = [cname for cname in candidate_names if cname in gold_names_set]
                    if not gold_in_list:
                        gold_in_list = ["NONE"]

                    entry = {
                        "paper_id": bibcode,
                        "candidate_ids": candidate_ids,
                        "candidate_names": candidate_names,
                        "scores": score_vals,
                        "gold_in_list": gold_in_list,
                    }
                    train_json_dict[bibcode] = entry
                    train_jsonl_lines.append(json.dumps(entry))

                paper_cursor += batch_size_cur

        elapsed = time.time() - start_time
        print(f"[SciBERT Handoffs] Train inference finished in {elapsed:.1f}s ({len(train_df)/elapsed:.1f} papers/sec)")

        # Save JSONL and JSON
        with open(train_shortlist_path, "w") as f:
            f.write("\n".join(train_jsonl_lines) + "\n")
        print(f"Saved {train_shortlist_path} ({train_shortlist_path.stat().st_size / (1024*1024):.1f} MB)")

        train_json_path = arm_dir / "gemma_candidates_train.json"
        with open(train_json_path, "w") as f:
            json.dump(train_json_dict, f, indent=2)
        print(f"Saved {train_json_path} ({train_json_path.stat().st_size / (1024*1024):.1f} MB)")

    # -------------------------------------------------------------
    # TASK 2: Export Parquet Tables in §6 Format (Handoff 5)
    # -------------------------------------------------------------
    print("\n[SciBERT Handoffs] Preparing Parquet tables for validation and test splits (§6 format)...")
    val_probs_path = arm_dir / "val_probs_epoch_8.pt"
    test_probs_path = arm_dir / "test_probabilities.pt"

    if not val_probs_path.exists():
        raise FileNotFoundError(f"Missing {val_probs_path}")
    if not test_probs_path.exists():
        raise FileNotFoundError(f"Missing {test_probs_path}")

    val_probs_tensor = torch.load(val_probs_path, map_location="cpu")
    test_probs_tensor = torch.load(test_probs_path, map_location="cpu")

    val_probs_np = val_probs_tensor.numpy()
    test_probs_np = test_probs_tensor.numpy()

    # Column ordering: paper_id, followed by UAT IDs as strings in label_order
    label_order_list = [str(idx_to_topic[i]) for i in range(num_labels)]

    # Validation DataFrame
    val_score_df = pd.DataFrame(val_probs_np, columns=label_order_list)
    val_score_df.insert(0, "paper_id", val_df["bibcode"].tolist())

    # Test DataFrame
    test_score_df = pd.DataFrame(test_probs_np, columns=label_order_list)
    test_score_df.insert(0, "paper_id", test_df["bibcode"].tolist())

    # Save to results/scibert_full/
    val_parquet_results = results_dir / "scibert_full_val.parquet"
    test_parquet_results = results_dir / "scibert_full_test.parquet"
    val_score_df.to_parquet(val_parquet_results, index=False)
    test_score_df.to_parquet(test_parquet_results, index=False)
    print(f"Saved {val_parquet_results} ({val_parquet_results.stat().st_size / (1024*1024):.1f} MB)")
    print(f"Saved {test_parquet_results} ({test_parquet_results.stat().st_size / (1024*1024):.1f} MB)")

    # Also keep copy in artifacts/scibert/full/
    val_parquet_artifacts = arm_dir / "scibert_full_val.parquet"
    test_parquet_artifacts = arm_dir / "scibert_full_test.parquet"
    val_score_df.to_parquet(val_parquet_artifacts, index=False)
    test_score_df.to_parquet(test_parquet_artifacts, index=False)

    # -------------------------------------------------------------
    # TASK 3: Official Scoring via Shai's evaluate.py
    # -------------------------------------------------------------
    print("\n[SciBERT Handoffs] Running official evaluation via evaluate.py...")
    # Build band_map with integer column index keys
    band_map = {idx: band_train[topic] for idx, topic in idx_to_topic.items()}

    # Build binary answer matrices
    topic_str_to_idx = {str(topic): idx for idx, topic in idx_to_topic.items()}

    val_answers = np.zeros((len(val_df), num_labels), dtype=np.uint8)
    for r_idx, labels in enumerate(val_df["verified_uat_ids"]):
        for lbl in labels:
            if str(lbl) in topic_str_to_idx:
                val_answers[r_idx, topic_str_to_idx[str(lbl)]] = 1

    test_answers = np.zeros((len(test_df), num_labels), dtype=np.uint8)
    for r_idx, labels in enumerate(test_df["verified_uat_ids"]):
        for lbl in labels:
            if str(lbl) in topic_str_to_idx:
                test_answers[r_idx, topic_str_to_idx[str(lbl)]] = 1

    # 1. choose_tau on validation probabilities
    print("\n1. Selecting optimal tau cutoffs per band on validation probabilities...")
    chosen_taus, tau_sweep = choose_tau(
        val_probs=val_probs_np,
        val_labels=val_answers,
        band_map=band_map,
    )
    print(f"   Chosen Taus: {chosen_taus}")

    objective_str = (
        "One cutoff per band, chosen on the validation set to maximise that "
        "band's micro-F1 over topics with >=1 validation instance."
    )
    save_tau_selection(
        chosen_taus=chosen_taus,
        tau_sweep=tau_sweep,
        arm_name="scibert_full",
        objective=objective_str,
        validation_filename="scibert_full_val.parquet",
        output_directory=results_dir,
    )
    print(f"   Saved tau selection to {results_dir / 'tau_scibert_full.json'}")

    # 2. evaluate_test with chosen taus
    print("\n2. Evaluating test set with selected band thresholds...")
    test_results = evaluate_test(
        test_probs=test_probs_np,
        test_labels=test_answers,
        band_map=band_map,
        chosen_taus=chosen_taus,
    )

    save_test_results(
        test_results=test_results,
        arm_name="scibert_full",
        output_directory=results_dir,
        paper_ids=test_df["bibcode"].tolist(),
        label_order=label_order_list,
    )
    print(f"   Saved test results to {results_dir / 'test_results_scibert_full.json'}")
    print(f"   Saved test predictions to {results_dir / 'test_predictions_scibert_full.npz'}")

    # Also copy to artifacts/scibert/full/
    with open(arm_dir / "test_results_scibert_full.json", "w") as f:
        json.dump(test_results, f, indent=2, default=str)
    with open(arm_dir / "tau_scibert_full.json", "w") as f:
        json.dump({"chosen_taus": chosen_taus, "objective": objective_str}, f, indent=2)

    # 3. Print Summary of Official Metrics
    print("\n" + "=" * 60)
    print("OFFICIAL SCIBERT-FULL RESULTS (evaluate.py)")
    print("=" * 60)
    print(f"Head  (tau={chosen_taus['head']:.3f}): Micro-F1 = {test_results['per_band']['head']['micro_f1']:.4f} | Precision = {test_results['per_band']['head']['precision']:.4f} | Recall = {test_results['per_band']['head']['recall']:.4f}")
    print(f"Torso (tau={chosen_taus['torso']:.3f}): Micro-F1 = {test_results['per_band']['torso']['micro_f1']:.4f} | Precision = {test_results['per_band']['torso']['precision']:.4f} | Recall = {test_results['per_band']['torso']['recall']:.4f}")
    print(f"Tail  (tau={chosen_taus['tail']:.3f}): Micro-F1 = {test_results['per_band']['tail']['micro_f1']:.4f} | Precision = {test_results['per_band']['tail']['precision']:.4f} | Recall = {test_results['per_band']['tail']['recall']:.4f}")
    print(f"Delta (Head - Tail): {test_results['delta_head_tail']:.4f}")
    print("\nRanking Metrics:")
    for k in (1, 3, 5):
        p_k = test_results['ranking_metrics'][f'precision_at_{k}']
        r_k = test_results['ranking_metrics'][f'recall_at_{k}']
        print(f"  P@{k}: {p_k*100:5.2f}% | R@{k}: {r_k*100:5.2f}%")
    print(f"\nMean predictions per paper: {test_results['mean_predictions_per_paper']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
