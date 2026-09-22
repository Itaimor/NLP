"""Is TF-IDF+LR's floor-pinned tau a grid artefact, or the model's real optimum?

The corrected tau grid (0.0005-0.98) was adopted because an argmax on a grid boundary
means the search was truncated, not that a threshold was chosen. After the correction
SciBERT-full and the majority baseline are interior, but TF-IDF+LR's head and torso taus
sit exactly on the NEW floor, 0.0005. The same diagnostic fires again, and a referee who
has just read the ceiling justification will ask about the floor.

Narrating that away ("its micro-F1 just rises as tau falls") is an assertion. This script
proves it or refutes it, by evaluating the limit the grid cannot contain:

  tau -> 0 is not an unreachable asymptote. Any tau at or below the smallest predicted
  probability classifies every cell positive, so "predict everything" IS the limit, and it
  has a closed form: with G gold positives in a band over N papers x L labels,
  micro-F1 = 2G / (N*L + G). If the chosen floor value beats that limit, the optimum is
  interior-in-the-limit and 0.0005 is a legitimate approximation of a real peak. If it
  merely equals it, the arm's best operating point genuinely is predict-all and the row
  must be reported as a degenerate lower bound, not a tuned competitor.

Reported in: Results / Baselines, the sentence defusing the TF-IDF floor.
"""
import sys, os, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

import numpy as np
import pandas as pd
import main as M

BANDS = ("head", "torso", "tail")

# Below the corrected grid's floor (0.0005), spanning four more orders of magnitude,
# plus the floor itself and the two grid points above it for context.
SUB_FLOOR = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 2e-4, 3e-4, 4e-4,
             0.0005, 0.001, 0.005]

ARMS = [
    dict(arm="tfidf_lr", d=ROOT / "results/tfidf_logistic_regression",
         val="tfidf_lr_val.parquet", test="tfidf_lr_test.parquet"),
    dict(arm="scibert_full", d=ROOT / "results/scibert_full",
         val="scibert_full_val.parquet", test="scibert_full_test.parquet"),
    dict(arm="majority", d=ROOT / "results/majority_baseline",
         val="majority_val.parquet", test="majority_test.parquet"),
]


def gold_matrix(df, label_order):
    idx = {str(l): i for i, l in enumerate(label_order)}
    Y = np.zeros((len(df), len(label_order)), dtype=np.uint8)
    for r, labs in enumerate(df["verified_uat_ids"]):
        for l in labs:
            j = idx.get(str(l))
            if j is not None:
                Y[r, j] = 1
    return Y


def probs_matrix(path, label_order, expected_ids, name):
    df = pd.read_parquet(path)
    ids = df[df.columns[0]].tolist()
    assert ids == list(expected_ids), f"{name}: row order does not match split.json"
    assert list(df.columns[1:]) == [str(l) for l in label_order], \
        f"{name}: column order does not match label_order"
    return df.iloc[:, 1:].to_numpy(dtype=np.float64)


def micro_f1(P, Y, tau):
    """Pooled micro-F1 over a band's cells, matching evaluate_thresholds (>= tau)."""
    pred = P >= tau
    tp = float((pred & (Y == 1)).sum())
    s = float(pred.sum() + (Y == 1).sum())
    return 0.0 if s == 0 else 2.0 * tp / s


def predict_all_f1(P, Y):
    """Closed form for the tau -> 0 limit: every cell positive."""
    G = float((Y == 1).sum())
    N, L = P.shape
    return 2.0 * G / (N * L + G)


def main():
    data = M.load_dataset_from_split_json_file()
    label_order = M.load_label_order()
    band_map = M.load_train_band_map(label_order)
    val_df, test_df = data["validation"], data["test"]

    Yval = gold_matrix(val_df, label_order)
    Ytest = gold_matrix(test_df, label_order)
    column_bands = np.array([band_map[i] for i in range(len(label_order))])
    present_val = (Yval == 1).any(axis=0)

    out = {}
    for a in ARMS:
        arm, d = a["arm"], a["d"]
        vp, tp_path = d / a["val"], d / a["test"]
        if not vp.exists():
            print(f"SKIP {arm}: missing {vp.name}")
            continue

        Pval = probs_matrix(vp, label_order, val_df["bibcode"].tolist(), f"{arm}/val")
        Ptest = probs_matrix(tp_path, label_order, test_df["bibcode"].tolist(), f"{arm}/test")
        chosen = json.load(open(d / f"tau_{arm}.json", encoding="utf-8"))["taus"]

        print("=" * 78)
        print(f"ARM: {arm}    chosen taus: {chosen}")
        print(f"  global min predicted probability, validation: {Pval.min():.3e}")
        out[arm] = {"chosen_taus": chosen,
                    "val_min_probability": float(Pval.min()),
                    "bands": {}}

        for band in BANDS:
            # choose_tau's column selection: this band, labels seen in validation.
            sel_v = (column_bands == band) & present_val
            Pv, Yv = Pval[:, sel_v], Yval[:, sel_v]
            # evaluate_test scores the whole band, present or not.
            sel_t = (column_bands == band)
            Pt, Yt = Ptest[:, sel_t], Ytest[:, sel_t]

            tau = float(chosen[band])
            limit = predict_all_f1(Pv, Yv)
            at_tau = micro_f1(Pv, Yv, tau)
            pmin = float(Pv.min())

            # Any tau <= pmin is literally predict-all. If the chosen tau is above pmin,
            # the sub-floor region contains real distinctions the grid could not reach.
            reachable = tau > pmin

            sweep = []
            for t in SUB_FLOOR:
                f = micro_f1(Pv, Yv, t)
                ppp = float((Pv >= t).sum()) / Pv.shape[0]
                sweep.append({"tau": t, "micro_f1": round(f, 6),
                              "preds_per_paper_in_band": round(ppp, 2)})

            best = max(sweep, key=lambda r: r["micro_f1"])
            # Strict test: does anything BELOW the grid floor beat the chosen tau?
            below = [r for r in sweep if r["tau"] < tau]
            best_below = max(below, key=lambda r: r["micro_f1"]) if below else None
            beats_limit = at_tau > limit + 1e-12

            print(f"\n  -- {band.upper()}  (band min prob {pmin:.3e})")
            print(f"     chosen tau {tau}  ->  val micro-F1 {at_tau:.6f}")
            print(f"     tau->0 predict-all LIMIT (closed form) {limit:.6f}")
            print(f"     chosen beats the limit: {beats_limit}"
                  f"   (margin {at_tau - limit:+.6f})")
            print(f"     chosen tau is above the band's min probability: {reachable}")
            if best_below:
                print(f"     best value strictly BELOW the grid floor: tau={best_below['tau']}"
                      f"  micro-F1 {best_below['micro_f1']:.6f}"
                      f"  -> {'BEATS' if best_below['micro_f1'] > at_tau + 1e-12 else 'does NOT beat'}"
                      f" the chosen tau")
            print(f"     argmax over the sub-floor sweep: tau={best['tau']}"
                  f"  micro-F1 {best['micro_f1']:.6f}"
                  f"  @ {best['preds_per_paper_in_band']} preds/paper in band")

            out[arm]["bands"][band] = {
                "chosen_tau": tau,
                "band_min_probability": pmin,
                "val_micro_f1_at_chosen": round(at_tau, 6),
                "val_micro_f1_predict_all_limit": round(limit, 6),
                "chosen_beats_limit": bool(beats_limit),
                "margin_over_limit": round(at_tau - limit, 6),
                "chosen_tau_above_band_min_prob": bool(reachable),
                "best_strictly_below_grid_floor": best_below,
                "sub_floor_sweep": sweep,
                "test_micro_f1_at_chosen": round(micro_f1(Pt, Yt, tau), 6),
                "test_micro_f1_predict_all_limit": round(predict_all_f1(Pt, Yt), 6),
            }
        print()

    p = Path(__file__).with_suffix(".json")
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("=" * 78)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
