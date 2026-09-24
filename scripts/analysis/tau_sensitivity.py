"""How fragile is each chosen tau, and does the REPORTED test number move if we picked
the runner-up instead?

Two questions, both on the real committed tables:
  1. Validation: paired bootstrap over papers of micro-F1(tau_chosen) - micro-F1(tau_runnerup).
     If the CI straddles zero the argmax is noise-separated, not signal-separated.
  2. Test: recompute each band's test micro-F1 under the runner-up tau. This is the only
     thing that can actually change a number in the paper.
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
# chosen tau -> the runner-up on the validation sweep (from tau_scibert_full.json)
NEIGHBOURS = {"head": (0.92, 0.90), "torso": (0.82, 0.80), "tail": (0.54, 0.58)}
N_BOOT = 2000
SEED = 42


def gold_matrix(df, label_order):
    idx = {str(l): i for i, l in enumerate(label_order)}
    Y = np.zeros((len(df), len(label_order)), dtype=np.uint8)
    for r, labs in enumerate(df["verified_uat_ids"]):
        for l in labs:
            j = idx.get(str(l))
            if j is not None:
                Y[r, j] = 1
    return Y


def probs_matrix(path, label_order, expected_ids):
    df = pd.read_parquet(path)
    ids = df[df.columns[0]].tolist()
    assert ids == list(expected_ids), "row order mismatch vs split.json"
    assert list(df.columns[1:]) == [str(l) for l in label_order], "column order mismatch"
    return df.iloc[:, 1:].to_numpy(dtype=np.float64)


def cell_counts(P, Y, tau):
    """Per-paper TP / predicted / actual, so a bootstrap can resample papers."""
    pred = P >= tau
    tp = (pred & (Y == 1)).sum(axis=1).astype(np.float64)
    pp = pred.sum(axis=1).astype(np.float64)
    ap = (Y == 1).sum(axis=1).astype(np.float64)
    return tp, pp, ap


def micro_f1(tp, pp, ap):
    s = pp.sum() + ap.sum()
    return 0.0 if s == 0 else 2.0 * tp.sum() / s


def main():
    data = M.load_dataset_from_split_json_file()
    label_order = M.load_label_order()
    band_map = M.load_train_band_map(label_order)
    val_df, test_df = data["validation"], data["test"]

    Yval = gold_matrix(val_df, label_order)
    Ytest = gold_matrix(test_df, label_order)
    d = ROOT / "results/scibert_full"
    Pval = probs_matrix(d / "scibert_full_val.parquet", label_order, val_df["bibcode"].tolist())
    Ptest = probs_matrix(d / "scibert_full_test.parquet", label_order, test_df["bibcode"].tolist())

    column_bands = np.array([band_map[i] for i in range(Pval.shape[1])])
    present_val = (Yval == 1).any(axis=0)

    rng = np.random.default_rng(SEED)
    n = Pval.shape[0]
    boot_idx = rng.integers(0, n, size=(N_BOOT, n))

    out = {}
    for band in BANDS:
        chosen, runner = NEIGHBOURS[band]

        # --- validation, same column selection choose_tau uses
        sel_val = (column_bands == band) & present_val
        Pv, Yv = Pval[:, sel_val], Yval[:, sel_val]
        a = cell_counts(Pv, Yv, chosen)
        b = cell_counts(Pv, Yv, runner)
        obs = micro_f1(*a) - micro_f1(*b)

        deltas = np.empty(N_BOOT)
        for i in range(N_BOOT):
            k = boot_idx[i]
            deltas[i] = (micro_f1(a[0][k], a[1][k], a[2][k])
                         - micro_f1(b[0][k], b[1][k], b[2][k]))
        lo, hi = np.percentile(deltas, [2.5, 97.5])

        # --- test: the number that would appear in the paper
        sel_test = (column_bands == band)
        Pt, Yt = Ptest[:, sel_test], Ytest[:, sel_test]
        f_chosen = micro_f1(*cell_counts(Pt, Yt, chosen))
        f_runner = micro_f1(*cell_counts(Pt, Yt, runner))

        # how many validation positives survive the chosen threshold
        pos_above = int(((Pv >= chosen) & (Yv == 1)).sum())
        pos_total = int((Yv == 1).sum())

        out[band] = dict(
            chosen=chosen, runner_up=runner,
            val_delta=round(obs, 5),
            val_ci=[round(lo, 5), round(hi, 5)],
            straddles_zero=bool(lo <= 0 <= hi),
            test_micro_f1_at_chosen=round(f_chosen, 5),
            test_micro_f1_at_runner=round(f_runner, 5),
            test_delta=round(f_chosen - f_runner, 5),
            val_positives_above_chosen_tau=pos_above,
            val_positives_total=pos_total,
            val_recall_ceiling_at_chosen=round(pos_above / pos_total, 4) if pos_total else None,
        )

        print(f"== {band.upper()}  chosen {chosen} vs runner-up {runner}")
        print(f"   VAL  delta micro-F1 = {obs:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]"
              f"   {'STRADDLES ZERO -> noise-separated' if lo <= 0 <= hi else 'separated'}")
        print(f"   TEST micro-F1 {f_chosen:.5f} (chosen) vs {f_runner:.5f} (runner-up)"
              f"  -> reported number moves by {f_chosen - f_runner:+.5f}")
        print(f"   VAL  gold positives at/above chosen tau: {pos_above}/{pos_total}"
              f"  (recall ceiling {pos_above / pos_total:.3f})")
        print()

    p = Path(__file__).with_name("tau_sensitivity.json")
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
