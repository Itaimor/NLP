#!/usr/bin/env python3
"""
Real Generator Recall Ceiling (Coverage@K) Analysis
===================================================
Calculates the empirical candidate recall ceiling (Coverage@K) across candidate
depths K in {10, 20, 25, 50, 100} for the actual neural candidate generators
(SciBERT Full and SciBERT LoRA) on both validation (2,855 papers) and test
(3,025 papers) splits.

Reports overall coverage and band-stratified coverage (HEAD, TORSO, TAIL).

Outputs:
  - scripts/analysis/real_recall_ceiling.out
  - scripts/analysis/real_recall_ceiling.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "data"))
sys.path.insert(0, str(PROJECT / "scripts"))

from data.build_split import load_split

DEPTHS = [10, 20, 25, 50, 100]
BANDS = ["head", "torso", "tail"]


def compute_coverage_for_split(
    parquet_path: Path,
    split_df: pd.DataFrame,
    label_order: List[str],
    col_band: np.ndarray,
    uat_to_col: Dict[int, int],
    depths: List[int],
) -> Dict[str, Any]:
    df = pd.read_parquet(parquet_path)
    order = {str(b): i for i, b in enumerate(split_df["bibcode"])}
    row_index = np.array([order[str(p)] for p in df["paper_id"]])
    assert len(set(row_index)) == len(row_index), "Duplicate paper IDs in parquet"

    # Construct gold binary matrix
    gold = np.zeros((len(split_df), len(label_order)), dtype=bool)
    for row, ids in enumerate(split_df["verified_uat_ids"]):
        for uat in ids:
            col = uat_to_col.get(int(uat))
            if col is not None:
                gold[row, col] = True

    gold_aligned = gold[row_index]
    probs = df[[str(u) for u in label_order]].to_numpy(dtype=np.float32)

    results = {}
    total_gold_all = int(gold_aligned.sum())
    total_gold_by_band = {
        band: int(gold_aligned[:, col_band == band].sum()) for band in BANDS
    }

    for K in depths:
        # Top-K candidate mask per document
        top = np.argpartition(-probs, K - 1, axis=1)[:, :K]
        in_top = np.zeros_like(gold_aligned)
        np.put_along_axis(in_top, top, True, axis=1)

        hit = gold_aligned & in_top
        k_results = {
            "overall": round(float(hit.sum() / total_gold_all), 4),
        }
        for band in BANDS:
            tot = total_gold_by_band[band]
            cols = col_band == band
            k_results[band] = round(float(hit[:, cols].sum() / tot), 4) if tot > 0 else 0.0

        results[str(K)] = k_results

    return {
        "num_papers": len(split_df),
        "total_gold_labels": total_gold_all,
        "gold_by_band": total_gold_by_band,
        "by_depth": results,
    }


def main():
    split = load_split()
    label_order = [str(u) for u in split["label_order"]]
    band_of_col = split["band_train"]
    val_df = split["validation_df"]
    test_df = split["test_df"]

    uat_to_col = {int(uat): i for i, uat in enumerate(label_order)}
    col_band = np.array([band_of_col[int(uat)] for uat in label_order])

    models = {
        "scibert_full": {
            "val": PROJECT / "results" / "scibert_full" / "scibert_full_val.parquet",
            "test": PROJECT / "results" / "scibert_full" / "scibert_full_test.parquet",
        },
        "scibert_lora": {
            "val": PROJECT / "results" / "scibert_lora" / "scibert_lora_val.parquet",
            "test": PROJECT / "results" / "scibert_lora" / "scibert_lora_test.parquet",
        },
    }

    all_results = {}
    lines = []

    lines.append("=" * 80)
    lines.append("REAL GENERATOR CANDIDATE RECALL CEILING (COVERAGE@K)")
    lines.append("=" * 80)

    for model_name, paths in models.items():
        all_results[model_name] = {}
        lines.append(f"\nMODEL: {model_name.upper()}")
        lines.append("-" * 80)

        for split_name, df in [("val", val_df), ("test", test_df)]:
            parquet_path = paths[split_name]
            if not parquet_path.exists():
                lines.append(f"  [{split_name.upper()}] Parquet not found at {parquet_path}")
                continue

            res = compute_coverage_for_split(
                parquet_path, df, label_order, col_band, uat_to_col, DEPTHS
            )
            all_results[model_name][split_name] = res

            lines.append(
                f"  [{split_name.upper()}] ({res['num_papers']} papers, {res['total_gold_labels']} gold labels: "
                f"head={res['gold_by_band']['head']}, torso={res['gold_by_band']['torso']}, tail={res['gold_by_band']['tail']})"
            )
            for K in DEPTHS:
                kd = res["by_depth"][str(K)]
                lines.append(
                    f"    K={K:3d} | Overall: {kd['overall'] * 100:5.2f}% | "
                    f"Head: {kd['head'] * 100:5.2f}% | "
                    f"Torso: {kd['torso'] * 100:5.2f}% | "
                    f"Tail: {kd['tail'] * 100:5.2f}%"
                )

    report_text = "\n".join(lines) + "\n"
    print(report_text)

    # Save outputs
    out_path = PROJECT / "scripts" / "analysis" / "real_recall_ceiling.out"
    json_path = PROJECT / "scripts" / "analysis" / "real_recall_ceiling.json"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Wrote text output to: {out_path}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote JSON output to: {json_path}")


if __name__ == "__main__":
    main()
