"""Re-score every probability-emitting arm under the corrected tau grid.

Reads the committed validation/test probability tables, re-selects one tau per band on
validation with evaluate.choose_tau (which now sees the extended TAU_CANDIDATES), applies
those taus to test with evaluate.evaluate_test, and writes tau_<arm>.json /
test_results_<arm>.json / test_predictions_<arm>.npz through the project's own savers.

Alignment is asserted, never assumed: row order against split.json, column order against
label_order.json.
"""
import sys, os, json
from pathlib import Path

ROOT = Path(r"D:\Users\LOGIN\Desktop\Ilana\NLP\NLP")
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

import numpy as np
import pandas as pd

import main as M
from evaluate import (
    TAU_CANDIDATES, choose_tau, evaluate_test,
    save_tau_selection, save_test_results,
)

OBJECTIVE = (
    "Maximise each band's validation Micro-F1 over TAU_CANDIDATES "
    "(corrected grid, 0.0005-0.98, spanning both extremes)."
)

ARMS = [
    dict(arm="scibert_full", d=ROOT / "results/scibert_full",
         val="scibert_full_val.parquet", test="scibert_full_test.parquet"),
    dict(arm="tfidf_lr", d=ROOT / "results/tfidf_logistic_regression",
         val="tfidf_lr_val.parquet", test="tfidf_lr_test.parquet"),
    dict(arm="majority", d=ROOT / "results/majority_baseline",
         val="majority_val.parquet", test="majority_test.parquet"),
]


def gold_matrix(df, label_order):
    """Binary gold matrix, rows follow df, columns follow label_order."""
    idx = {str(l): i for i, l in enumerate(label_order)}
    Y = np.zeros((len(df), len(label_order)), dtype=np.uint8)
    for r, labs in enumerate(df["verified_uat_ids"]):
        for l in labs:
            j = idx.get(str(l))
            if j is not None:          # labels dropped by the k=10 filter
                Y[r, j] = 1
    return Y


def probs_matrix(path, label_order, expected_ids, name):
    df = pd.read_parquet(path)
    id_col = df.columns[0]
    ids = df[id_col].tolist()
    assert ids == list(expected_ids), (
        f"{name}: row order does not match split.json "
        f"({sum(a != b for a, b in zip(ids, expected_ids))} mismatches)"
    )
    cols = list(df.columns[1:])
    want = [str(l) for l in label_order]
    assert cols == want, (
        f"{name}: column order does not match label_order "
        f"(first mismatch at {next(i for i,(a,b) in enumerate(zip(cols,want)) if a!=b)})"
    )
    P = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    assert P.shape == (len(expected_ids), len(label_order)), f"{name}: bad shape {P.shape}"
    return P


def main():
    print(f"TAU_CANDIDATES: n={len(TAU_CANDIDATES)} "
          f"min={TAU_CANDIDATES[0]} max={TAU_CANDIDATES[-1]}\n")

    data = M.load_dataset_from_split_json_file()
    label_order = M.load_label_order()
    band_map = M.load_train_band_map(label_order)

    val_df, test_df = data["validation"], data["test"]
    Yval = gold_matrix(val_df, label_order)
    Ytest = gold_matrix(test_df, label_order)
    print(f"gold: val {Yval.shape} ({Yval.sum()} positives), "
          f"test {Ytest.shape} ({Ytest.sum()} positives)\n")

    summary = {}
    for a in ARMS:
        arm, d = a["arm"], a["d"]
        print("=" * 70)
        print(f"ARM: {arm}")
        vp, tp = d / a["val"], d / a["test"]
        if not vp.exists() or not tp.exists():
            print(f"  SKIP - missing {vp.name} or {tp.name}")
            continue

        Pval = probs_matrix(vp, label_order, val_df["bibcode"].tolist(), f"{arm}/val")
        Ptest = probs_matrix(tp, label_order, test_df["bibcode"].tolist(), f"{arm}/test")
        print(f"  aligned OK  val{Pval.shape} test{Ptest.shape}")

        before = {}
        tj = d / f"tau_{arm}.json"
        if tj.exists():
            before = json.load(open(tj)).get("taus", {})

        taus, sweep = choose_tau(val_probs=Pval, val_labels=Yval, band_map=band_map)

        edge = {b: ("FLOOR" if taus[b] == TAU_CANDIDATES[0]
                    else "CEILING" if taus[b] == TAU_CANDIDATES[-1] else "interior")
                for b in ("head", "torso", "tail")}
        print(f"  taus before: {before}")
        print(f"  taus after : {taus}")
        print(f"  grid position: {edge}")

        save_tau_selection(chosen_taus=taus, tau_sweep=sweep, arm_name=arm,
                           objective=OBJECTIVE, validation_filename=a["val"],
                           output_directory=d)

        res = evaluate_test(test_probs=Ptest, test_labels=Ytest,
                            band_map=band_map, chosen_taus=taus)
        save_test_results(test_results=res, arm_name=arm, output_directory=d,
                          paper_ids=test_df["bibcode"].tolist(), label_order=label_order)

        pb = res["per_band"]
        row = {b: round(float(pb[b]["micro_f1"]), 4) for b in ("head", "torso", "tail")}
        row["preds_per_paper"] = round(float(res.get("mean_predictions_per_paper", float("nan"))), 2)
        row["taus"] = taus
        row["edge"] = edge
        summary[arm] = row
        print(f"  TEST micro-F1 h/t/t: {row['head']} / {row['torso']} / {row['tail']}"
              f"  @ {row['preds_per_paper']} preds/paper")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(json.dumps(summary, indent=2))
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("rescore_summary.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
