"""Paired bootstrap CIs for the candidate-order robustness claims.

Completes the {5a, 5b} x {SciBERT order, shuffled} design. The 21 Sept run measured only
the 5b half; the 5a half was run 24 Sept. Four comparisons:

  1. 5a - 5a-shuffled   how much the untouched arm depends on SciBERT's ranking
  2. 5b - 5b-shuffled   the same for the fine-tuned arm (reported without a CI until now)
  3. 5a-shuffled - scibert   does the untouched arm still beat the encoder without the ranking
  4. 5b-shuffled - scibert   the residual gain, i.e. the claim-bearing quantity

Uses evaluate.paired_bootstrap unchanged, same 2000 resamples and seed 42 as
bootstrap_gemma_vs_scibert.py, so the intervals are directly comparable. All matrices are
aligned by paper_id and label_order before use; nothing is assumed to be in the same order.
"""
import sys, os, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

import numpy as np
import main as M
from evaluate import paired_bootstrap

SCORED = ROOT / "results/gemma_select/scored"
NPZ = {
    "scibert":     ROOT / "results/scibert_full/test_predictions_scibert_full.npz",
    "5a":          SCORED / "test_predictions_5a_test.npz",
    "5a-shuffled": SCORED / "test_predictions_5a-shuffled_test.npz",
    "5b":          SCORED / "test_predictions_5b_test.npz",
    "5b-shuffled": SCORED / "test_predictions_5b-shuffled_test.npz",
}


def gold_matrix(df, label_order):
    idx = {str(l): i for i, l in enumerate(label_order)}
    Y = np.zeros((len(df), len(label_order)), dtype=np.uint8)
    for r, labs in enumerate(df["verified_uat_ids"]):
        for l in labs:
            j = idx.get(str(l))
            if j is not None:
                Y[r, j] = 1
    return Y


def load_aligned(path, paper_ids, label_order, name):
    d = np.load(path, allow_pickle=True)
    P = d["predictions"]
    pid = [str(x) for x in d["paper_ids"]]
    lab = [str(x) for x in d["label_order"]]

    want_p, want_l = [str(x) for x in paper_ids], [str(x) for x in label_order]
    if pid != want_p:
        pos = {p: i for i, p in enumerate(pid)}
        missing = [p for p in want_p if p not in pos]
        assert not missing, f"{name}: {len(missing)} paper_ids absent"
        P = P[[pos[p] for p in want_p], :]
        print(f"    {name}: rows reordered to split order")
    if lab != want_l:
        pos = {l: i for i, l in enumerate(lab)}
        missing = [l for l in want_l if l not in pos]
        assert not missing, f"{name}: {len(missing)} labels absent"
        P = P[:, [pos[l] for l in want_l]]
        print(f"    {name}: columns reordered to label_order")

    P = P.astype(np.uint8)
    assert P.shape == (len(paper_ids), len(label_order)), f"{name}: bad shape {P.shape}"
    return P


def fmt(r, band):
    return (f"{r[band]['difference']:+.4f} "
            f"[{r[band]['ci_low']:+.4f}, {r[band]['ci_high']:+.4f}]")


def main():
    data = M.load_dataset_from_split_json_file()
    label_order = M.load_label_order()
    band_map = M.load_train_band_map(label_order)
    test_df = data["test"]
    pids = test_df["bibcode"].tolist()
    Y = gold_matrix(test_df, label_order)
    print(f"gold test: {Y.shape}, {Y.sum()} positives\n")

    P = {}
    for k, p in NPZ.items():
        if not p.exists():
            print(f"  SKIP {k}: {p.name} missing")
            continue
        P[k] = load_aligned(p, pids, label_order, k)
        print(f"  {k:12s} loaded {P[k].shape}  {P[k].sum()/len(pids):.2f} preds/paper")

    comparisons = [
        ("5a", "5a-shuffled", "5a - 5a-shuffled   [order dependence, untouched arm]"),
        ("5b", "5b-shuffled", "5b - 5b-shuffled   [order dependence, fine-tuned arm]"),
        ("5a-shuffled", "scibert", "5a-shuffled - SciBERT-alone   [residual gain, untouched]"),
        ("5b-shuffled", "scibert", "5b-shuffled - SciBERT-alone   [residual gain, fine-tuned]"),
    ]

    out = {}
    print("\n" + "=" * 78)
    print("PAIRED BOOTSTRAP - per-band micro-F1, A minus B, 2000 resamples, seed 42")
    print("=" * 78)
    for a, b, title in comparisons:
        if a not in P or b not in P:
            print(f"\nSKIP {title}")
            continue
        r = paired_bootstrap(P[a], P[b], Y, band_map, num_bootstraps=2000, seed=42)
        out[f"{a}_minus_{b}"] = r
        print(f"\n{title}")
        for band in ("head", "torso", "tail"):
            d = r[band]
            sig = "" if (d["ci_low"] <= 0 <= d["ci_high"]) else "  *"
            print(f"   {band:6s} {fmt(r, band)}{sig}")
        strad = [b_ for b_ in ("head", "torso", "tail")
                 if r[b_]["ci_low"] <= 0 <= r[b_]["ci_high"]]
        print(f"   -> CI straddles zero in: {strad if strad else 'no band'}")

    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "scripts/analysis/bootstrap_order_robustness.json"
    Path(dest).write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\n* = CI excludes zero\nwrote {dest}")


if __name__ == "__main__":
    main()
