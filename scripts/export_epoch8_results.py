#!/usr/bin/env python3
"""
Exports SciBERT Epoch 8 Evaluation and Gemma Candidate Shortlists
================================================================
Loads the fully trained Epoch 8 model from checkpoint.pt, runs inference on the
test split, saves updated probability tables, and generates the Top-50 candidate
shortlists for both Validation and Test splits in JSON and JSONL formats.
"""

import os
import sys
import json
import time
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "data"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_split import load_split
from scibert_dataset import SciXDataset
from SciBERT_experiment import build_scibert, load_model_state_dict


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Epoch 8 Exporter] Using device: {device}")

    # 1. Load Stage 1 split data
    print("[Epoch 8 Exporter] Loading split data...")
    s = load_split()
    train_df = s["train_df"]
    val_df = s["validation_df"]
    test_df = s["test_df"]
    topic_to_idx = s["topic_to_idx"]
    idx_to_topic = s["idx_to_topic"]
    id_to_name = s["id_to_name"]
    num_labels = s["num_labels"]
    rare_indices = torch.tensor(s["rare_indices"])

    # 2. Load Epoch 8 Checkpoint
    arm_dir = PROJECT_ROOT / "artifacts" / "scibert" / "full"
    checkpoint_path = arm_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"[Epoch 8 Exporter] Loading Epoch 8 checkpoint from {checkpoint_path}...")
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    epoch_num = cp.get("completed_epoch", 7) + 1
    print(f"[Epoch 8 Exporter] Loaded weights from completed Epoch {epoch_num} (train loss: {cp.get('train_loss'):.6f})")

    # 3. Rebuild model and load weights
    model = build_scibert(num_labels=num_labels, mode="full")
    load_model_state_dict(model, cp["model_state_dict"])
    model.to(device)
    model.eval()

    # 4. Save best_model.pt with Epoch 8 weights
    best_model_path = arm_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": {k: v.cpu().clone() for k, v in cp["model_state_dict"].items()},
            "epoch": epoch_num,
            "train_loss": cp.get("train_loss"),
            "val_loss": cp.get("validation_loss"),
        },
        best_model_path,
    )
    print(f"[Epoch 8 Exporter] Saved Epoch 8 model to {best_model_path}")

    # 5. Run Test Inference
    print(f"[Epoch 8 Exporter] Running inference on {len(test_df)} test papers...")
    test_dataset = SciXDataset(test_df, topic_to_idx=topic_to_idx)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    test_probs_list = []
    test_labels_list = []

    start_time = time.time()
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(test_loader, desc="Test Inference"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.sigmoid(logits)

            test_probs_list.append(probs.cpu())
            test_labels_list.append(labels.cpu())

    test_probs = torch.cat(test_probs_list, dim=0)
    test_labels = torch.cat(test_labels_list, dim=0)
    elapsed = time.time() - start_time
    print(f"[Epoch 8 Exporter] Test inference finished in {elapsed:.1f}s ({len(test_df)/elapsed:.1f} papers/sec)")

    # Save test probabilities and labels
    torch.save(test_probs, arm_dir / "test_probabilities.pt")
    torch.save(test_labels, arm_dir / "test_labels.pt")
    print(f"[Epoch 8 Exporter] Saved test probabilities ({test_probs.shape}) to test_probabilities.pt")

    # 6. Load Validation Probabilities from Epoch 8
    val_probs_path = arm_dir / "val_probs_epoch_8.pt"
    val_probs = torch.load(val_probs_path, map_location="cpu")
    print(f"[Epoch 8 Exporter] Loaded validation probabilities ({val_probs.shape}) from {val_probs_path.name}")

    # 7. Evaluate Ranking Metrics on Test Set (P@k, R@k)
    print("\n=== SciBERT Epoch 8 Test Ranking Metrics (P@k, R@k) ===")
    total_gold = test_labels.sum().item()
    for k in [1, 3, 5, 10, 50]:
        topk = torch.topk(test_probs, k, dim=1).indices
        hits = torch.gather(test_labels, 1, topk).sum().item()
        p_at_k = hits / (len(test_probs) * k)
        r_at_k = hits / total_gold
        print(f"  P@{k:2d}: {p_at_k*100:5.2f}% | R@{k:2d}: {r_at_k*100:5.2f}% ({int(hits)} hits out of {int(total_gold)} gold topics)")

    # 8. Export Gemma Candidate Shortlists (Top 50)
    test_bibcodes = test_df["bibcode"].tolist()
    val_bibcodes = val_df["bibcode"].tolist()

    for split_name, probs_tensor, bibcodes in [
        ("test", test_probs, test_bibcodes),
        ("val", val_probs, val_bibcodes),
    ]:
        print(f"\n[Epoch 8 Exporter] Exporting Top-50 candidates for {split_name} ({len(bibcodes)} papers)...")
        json_dict = {}
        jsonl_lines = []

        for i in range(len(bibcodes)):
            scores, top_idx = torch.topk(probs_tensor[i], k=50)
            candidate_ids = [idx_to_topic[idx.item()] for idx in top_idx]
            candidate_names = [id_to_name.get(cid, str(cid)) for cid in candidate_ids]
            score_vals = [round(float(s.item()), 5) for s in scores]

            entry = {
                "paper_id": bibcodes[i],
                "candidate_ids": candidate_ids,
                "candidate_names": candidate_names,
                "scores": score_vals,
            }
            json_dict[bibcodes[i]] = entry
            jsonl_lines.append(json.dumps(entry))

        # Save JSON format
        json_out = arm_dir / f"gemma_candidates_{split_name}.json"
        with open(json_out, "w") as f:
            json.dump(json_dict, f, indent=2)

        # Save JSONL format (WORK_PLAN handoff 4 schema)
        jsonl_out = arm_dir / f"scibert_full_top50_{split_name}.jsonl"
        with open(jsonl_out, "w") as f:
            f.write("\n".join(jsonl_lines) + "\n")

        print(f"  Saved JSON:  {json_out.name} ({json_out.stat().st_size / 1e6:.1f} MB)")
        print(f"  Saved JSONL: {jsonl_out.name} ({jsonl_out.stat().st_size / 1e6:.1f} MB)")

    # 9. Update experiment_results.json
    results_path = arm_dir / "experiment_results.json"
    results = {
        "mode": "full",
        "device": str(device),
        "learning_rate": 2e-5,
        "num_epochs": 8,
        "best_epoch": 8,
        "epoch_8_train_loss": round(float(cp.get("train_loss", 0.0)), 6),
        "epoch_8_val_loss": round(float(cp.get("validation_loss", 0.0)), 6),
        "test_p_at_1": round(float((torch.gather(test_labels, 1, torch.topk(test_probs, 1, dim=1).indices).sum() / len(test_probs)).item()), 4),
        "test_p_at_3": round(float((torch.gather(test_labels, 1, torch.topk(test_probs, 3, dim=1).indices).sum() / (len(test_probs) * 3)).item()), 4),
        "test_p_at_5": round(float((torch.gather(test_labels, 1, torch.topk(test_probs, 5, dim=1).indices).sum() / (len(test_probs) * 5)).item()), 4),
        "test_r_at_50": round(float((torch.gather(test_labels, 1, torch.topk(test_probs, 50, dim=1).indices).sum() / total_gold).item()), 4),
        "test_candidates_json": str(arm_dir / "gemma_candidates_test.json"),
        "val_candidates_json": str(arm_dir / "gemma_candidates_validation.json"),
        "test_candidates_jsonl": str(arm_dir / "scibert_full_top50_test.jsonl"),
        "val_candidates_jsonl": str(arm_dir / "scibert_full_top50_val.jsonl"),
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Epoch 8 Exporter] Updated summary in {results_path.name}")
    print("[Epoch 8 Exporter] All exports completed successfully!")


if __name__ == "__main__":
    main()
