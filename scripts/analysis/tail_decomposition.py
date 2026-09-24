"""Where the tail numbers come from, on the TEST split.

Two questions Section 7 asks and the scored JSONs do not answer on their own.

  1. Decomposition. How much tail gold never reaches the selector at all, because it is
     not in SciBERT's top 50, and of the tail gold that IS shown, what share each arm
     picks. This separates "the selector found more rare concepts" from "the shortlist
     contained more rare concepts".

  2. Tail macro-F1 with a paired bootstrap over papers. evaluate.paired_bootstrap returns
     per-band MICRO-F1 only, but the operating-point argument in Section 7.3 is about
     macro-F1 over the tail concepts that occur in the test split. Computed here at the
     repo's standard 2000 resamples and seed 42, so the intervals are comparable with
     bootstrap_gemma_vs_scibert.json and bootstrap_order_robustness.json.

An earlier version of these figures was measured on the VALIDATION split (docket 21 Sept)
and must not be quoted against test numbers. Test is the reported basis, so test is what
this script computes.
"""
import sys, os, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

import numpy as np
import main as M

N_BOOT = 2000
SEED = 42
# Read the TRACKED handoff file (key "candidate_names") rather than the derived copy in
# results/shortlists/, which is gitignored and would be absent on a fresh clone.
SHORTLIST = ROOT / "results/scibert_full/scibert_full_top50_test.jsonl"
SHORTLIST_DERIVED = ROOT / "results/shortlists/scibert_full_top50_test.jsonl"
ARMS = ("5a", "5b", "5b-step-8500")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def macro_present(P, G):
    """Macro-F1 over the columns with at least one positive in G (the 'present' denominator)."""
    tp = (P & G).sum(0)
    fp = (P & ~G).sum(0)
    fn = (~P & G).sum(0)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    present = G.sum(0) > 0
    return float(f1[present].mean())


def main():
    data = M.load_dataset_from_split_json_file()
    label_order = M.load_label_order()
    bands = M.load_train_band_map(label_order)
    id_to_name = json.load(open(ROOT / "data/id_to_name.json", encoding="utf-8"))

    test = data["test"]
    name_at = {i: id_to_name.get(str(l)) for i, l in enumerate(label_order)}
    tail_cols = [i for i in range(len(label_order)) if bands[i] == "tail" and name_at[i]]
    tail_name_to_col = {name_at[i]: j for j, i in enumerate(tail_cols)}

    pids = list(test["bibcode"])
    gold_names = {p: {id_to_name[str(u)] for u in labs if str(u) in id_to_name}
                  for p, labs in zip(pids, test["verified_uat_ids"])}

    src = SHORTLIST if SHORTLIST.exists() else SHORTLIST_DERIVED
    rows = read_jsonl(src)
    key = "candidate_names" if "candidate_names" in rows[0] else "candidates"
    shortlist = {r["paper_id"]: set(r[key]) for r in rows}
    print(f"shortlist: {src.relative_to(ROOT)} (key {key!r})")

    # ---- 1. decomposition -------------------------------------------------
    gold_tail = shown_tail = 0
    for p in pids:
        gt = {g for g in gold_names[p] if g in tail_name_to_col}
        gold_tail += len(gt)
        shown_tail += len(gt & shortlist[p])

    print(f"TEST tail gold                       : {gold_tail}")
    print(f"  present in the 50-candidate list   : {shown_tail}  ({shown_tail/gold_tail:.4f})")
    print(f"  never shown to the selector        : {gold_tail-shown_tail}  ({1-shown_tail/gold_tail:.4f})")

    decomposition = {
        "tail_gold_test": gold_tail,
        "tail_gold_in_shortlist": shown_tail,
        "coverage_at_50": shown_tail / gold_tail,
        "never_shown_fraction": 1 - shown_tail / gold_tail,
        "arms": {},
    }

    # ---- gold and prediction matrices over tail columns --------------------
    G = np.zeros((len(pids), len(tail_cols)), dtype=bool)
    for i, p in enumerate(pids):
        for g in gold_names[p]:
            j = tail_name_to_col.get(g)
            if j is not None:
                G[i, j] = True

    P = {}
    for arm in ARMS:
        rows = {r["paper_id"]: r for r in read_jsonl(ROOT / f"results/gemma_select/{arm}_test.jsonl")}
        Marm = np.zeros_like(G)
        picked_shown = 0
        for i, p in enumerate(pids):
            picks = set(rows[p]["picks"])
            for x in picks:
                j = tail_name_to_col.get(x)
                if j is not None:
                    Marm[i, j] = True
            gt_shown = {g for g in gold_names[p] if g in tail_name_to_col} & shortlist[p]
            picked_shown += len(gt_shown & picks)
        P[arm] = Marm
        rate = picked_shown / shown_tail
        decomposition["arms"][arm] = {"tail_gold_shown_and_picked": picked_shown,
                                      "selection_rate_of_shown_tail_gold": rate}
        print(f"  {arm:14s} picks {picked_shown:5d}/{shown_tail} of the tail gold it was shown = {rate:.4f}")

    # ---- 2. paired bootstrap on tail macro-F1 ------------------------------
    print(f"\nTail macro-F1 (present labels), paired bootstrap, {N_BOOT} resamples, seed {SEED}")
    point = {a: macro_present(P[a], G) for a in ARMS}
    for a in ARMS:
        print(f"  {a:14s} {point[a]:.4f}")

    rng = np.random.default_rng(SEED)
    n = len(pids)
    idx = [rng.integers(0, n, n) for _ in range(N_BOOT)]
    draws = {a: np.array([macro_present(P[a][i], G[i]) for i in idx]) for a in ARMS}

    out = {"decomposition": decomposition, "tail_macro_point": point, "paired": {}}
    print("\n  paired differences, A minus B")
    for a, b in (("5b", "5b-step-8500"), ("5b", "5a"), ("5b-step-8500", "5a")):
        d = draws[a] - draws[b]
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        diff = point[a] - point[b]
        sig = "" if lo <= 0 <= hi else "  *"
        out["paired"][f"{a}_minus_{b}"] = {"difference": diff, "ci_low": lo, "ci_high": hi}
        print(f"  {a:14s} - {b:14s} {diff:+.4f} [{lo:+.4f}, {hi:+.4f}]{sig}")

    dest = ROOT / "scripts/analysis/tail_decomposition.json"
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\n* = CI excludes zero\nwrote {dest}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
