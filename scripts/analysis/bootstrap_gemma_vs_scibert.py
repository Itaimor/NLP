"""Paired bootstrap CIs for the comparisons the paper's thesis sentence rests on.

Gemma arms vs the CORRECTED SciBERT-alone row (re-thresholded under the extended tau grid),
plus a 5b-5a replication as a pipeline sanity check against the recorded 21 Sept numbers.

Uses evaluate.paired_bootstrap unchanged. All matrices are aligned by paper_id and
label_order before use; nothing is assumed to be in the same order.
"""
import sys, os, json
from pathlib import Path

ROOT = Path(r"D:\Users\LOGIN\Desktop\Ilana\NLP\NLP")
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

import numpy as np
import main as M
from evaluate import paired_bootstrap

SCORED = ROOT / "results/gemma_select/scored"
NPZ = {
    "scibert": ROOT / "results/scibert_full/test_predictions_scibert_full.npz",
    "5a":      SCORED / "test_predictions_5a_test.npz",
    "5b":      SCORED / "test_predictions_5b_test.npz",
    "5b-step8500": SCORED / "test_predictions_5b-step-8500_test.npz",
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
        ("5a", "scibert", "5a (untouched Gemma) - corrected SciBERT-alone"),
        ("5b", "scibert", "5b (QLoRA epoch-2)  - corrected SciBERT-alone"),
        ("5b-step8500", "scibert", "5b step-8500        - corrected SciBERT-alone"),
        ("5b", "5a", "5b - 5a  [replication of the recorded 21 Sept run]"),
    ]

    out = {}
    print("\n" + "=" * 78)
    print("PAIRED BOOTSTRAP - per-band micro-F1, A minus B, 2000 resamples, seed 42")
    print("=" * 78)
    for a, b, title in comparisons:
        if a not in P or b not in P:
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

    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bootstrap_results.json")
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\n* = CI excludes zero\nwrote {dest}")


if __name__ == "__main__":
    main()
