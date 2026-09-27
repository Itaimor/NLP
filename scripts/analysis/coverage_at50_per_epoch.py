"""Recompute validation coverage@50 per band for each saved epoch.

Verification first: epoch 8 must reproduce the committed figures in
results/scibert_full/PROVENANCE.md, namely 0.829 overall with
head 0.980 / torso 0.908 / tail 0.528. If it does not, the computation is wrong
and nothing downstream should be plotted.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT / "scripts"), str(PROJECT / "data"), str(PROJECT)]

from build_split import load_split  # noqa: E402

BANDS = ("head", "torso", "tail")
DEPTH = 50


def main():
    split = load_split()
    label_order = list(split["label_order"])
    band_of_col = split["band_train"]
    val_df = split["validation_df"]

    uat_to_col = {int(uat): i for i, uat in enumerate(label_order)}
    # band_train is keyed by UAT id, not by column index.
    col_band = np.array([band_of_col[uat] for uat in label_order])

    # Gold matrix, rows in validation_df order.
    gold = np.zeros((len(val_df), len(label_order)), dtype=bool)
    for row, ids in enumerate(val_df["verified_uat_ids"]):
        for uat in ids:
            col = uat_to_col.get(int(uat))
            if col is not None:
                gold[row, col] = True

    out = {}
    for epoch in range(1, 9):
        path = (
            PROJECT
            / "results/scibert_full/val_epochs"
            / f"scibert_full_val_epoch_{epoch}.parquet"
        )
        df = pd.read_parquet(path)

        # Align parquet rows to the gold matrix by paper id.
        order = {str(b): i for i, b in enumerate(val_df["bibcode"])}
        row_index = np.array([order[str(p)] for p in df["paper_id"]])
        assert len(set(row_index)) == len(row_index), "duplicate paper ids"

        probs = df[[str(u) for u in label_order]].to_numpy(dtype=np.float32)
        gold_aligned = gold[row_index]

        # Top-50 columns per paper.
        top = np.argpartition(-probs, DEPTH - 1, axis=1)[:, :DEPTH]
        in_top = np.zeros_like(gold_aligned)
        np.put_along_axis(in_top, top, True, axis=1)

        hit = gold_aligned & in_top
        record = {}
        for band in BANDS:
            cols = col_band == band
            total = int(gold_aligned[:, cols].sum())
            record[band] = round(float(hit[:, cols].sum() / total), 4) if total else None
        record["overall"] = round(float(hit.sum() / gold_aligned.sum()), 4)
        out[epoch] = record
        print(f"epoch {epoch}: {record}")

    expected = {"overall": 0.829, "head": 0.980, "torso": 0.908, "tail": 0.528}
    got = out[8]
    ok = all(abs(got[k] - v) <= 0.0015 for k, v in expected.items())
    print()
    print("epoch 8 committed :", expected)
    print("epoch 8 recomputed:", {k: got[k] for k in expected})
    print("REPRODUCES COMMITTED FIGURES:", ok)

    if ok:
        dest = PROJECT / "results/scibert_full/coverage_at50_per_epoch.json"
        dest.write_text(
            json.dumps(
                {
                    "what": "validation coverage@50 per band, per saved epoch",
                    "depth": DEPTH,
                    "basis": "training band map (primary), 2,855 validation papers",
                    "verified_against": "results/scibert_full/PROVENANCE.md epoch-8 row",
                    "epochs": out,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("wrote", dest)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
