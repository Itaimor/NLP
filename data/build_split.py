#!/usr/bin/env python3
"""
Stage 1 — fix the data split, once and forever.

Writes the five files every later script reads instead of recomputing anything:

  data/split.json            train / validation / test paper ids (bibcodes)
  data/label_order.json      the 1,864 UAT ids in fixed order — position = column index
  data/band_map_train.json   head / torso / tail per topic, cut on the 18,677 training papers (PRIMARY)
  data/band_map_corpus.json  the same, cut on corpus-wide counts (for the dataset description)
  data/id_to_name.json       UAT id -> topic name
  data/CHECKSUMS.sha256      sha256 of each file above

The shipped HuggingFace splits are kept as they are: "train" (18,677) is our training pool,
"val" (3,025) is the published benchmark TEST split and is never tuned on. The validation set is
carved out of the 18,677 with main.py::split_train_validation (label-stratified, seed 42), which is
the only place that carve is defined.

Usage:
    python data/build_split.py            # write the files and verify them
    python data/build_split.py --check    # only verify existing files against their checksums

From other scripts:
    sys.path.insert(0, "data"); from build_split import load_split
    s = load_split()
    s["train_df"], s["validation_df"], s["test_df"]   # pandas, rows in split.json order
    s["topic_to_idx"], s["idx_to_topic"]               # from label_order.json
    s["band_train"], s["band_corpus"]                   # {uat_id: "head" | "torso" | "tail"}
    s["rare_indices"]                                   # column indices of the primary-map tail
    s["id_to_name"]
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))  # for main.split_train_validation

DATA_DIR = Path(__file__).resolve().parent   # the files live next to this script
FILES = ["split.json", "label_order.json", "band_map_train.json", "band_map_corpus.json", "id_to_name.json"]

SEED = 42
VALIDATION_SIZE = 0.15
HEAD_MIN_EXCLUSIVE = 500   # head: more than 500 occurrences
TAIL_MAX_EXCLUSIVE = 50    # tail: fewer than 50 occurrences

# What the files must contain, or the build fails. Every number is stated in WORK_PLAN.md.
EXPECTED = {
    "n_train": 15822, "n_validation": 2855, "n_test": 3025, "n_labels": 1864,
    "bands_train": {"head": 12, "torso": 383, "tail": 1469},
    "bands_corpus": {"head": 17, "torso": 429, "tail": 1418},
    "scoreable_tail_train": 410, "scoreable_tail_corpus": 359,
}


def band_of(count):
    if count > HEAD_MIN_EXCLUSIVE:
        return "head"
    if count < TAIL_MAX_EXCLUSIVE:
        return "tail"
    return "torso"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


def build():
    from main import data_loading, split_train_validation  # the carve is defined there, nowhere else

    d = data_loading()
    train_df, validation_df = split_train_validation(train_df=d["train"], seed=SEED, validation_size=VALIDATION_SIZE)
    test_df = d["test"]

    # --- label space: every UAT id that occurs anywhere, in ascending id order
    corpus_counts = Counter()
    for df in (d["train"], test_df):
        for ids in df["verified_uat_ids"]:
            corpus_counts.update(ids)
    train_pool_counts = Counter()                 # the 18,677 shipped training papers
    for ids in d["train"]["verified_uat_ids"]:
        train_pool_counts.update(ids)
    test_counts = Counter()
    for ids in test_df["verified_uat_ids"]:
        test_counts.update(ids)

    label_order = sorted(int(i) for i in corpus_counts)
    id_to_name = {}
    for df in (d["train"], test_df):
        for ids, names in zip(df["verified_uat_ids"], df["verified_uat_labels"]):
            for i, n in zip(ids, names):
                id_to_name.setdefault(int(i), n)

    def band_map(counts, basis, n_papers):
        bands = {str(i): band_of(counts.get(i, 0)) for i in label_order}
        return {
            "basis": basis,
            "papers_counted": n_papers,
            "cutoffs": {"head": f"> {HEAD_MIN_EXCLUSIVE}", "torso": f"{TAIL_MAX_EXCLUSIVE}-{HEAD_MIN_EXCLUSIVE}", "tail": f"< {TAIL_MAX_EXCLUSIVE}"},
            "counts": dict(Counter(bands.values())),
            "scoreable_tail": sum(1 for i in label_order if bands[str(i)] == "tail" and test_counts.get(i, 0) > 0),
            "frequency": {str(i): counts.get(i, 0) for i in label_order},
            "bands": bands,
        }

    band_train = band_map(train_pool_counts, "shipped training split (18,677 papers) — PRIMARY, matches Alkan et al. Table 6", len(d["train"]))
    band_corpus = band_map(corpus_counts, "corpus-wide (train + test, 21,702 papers) — matches Alkan et al. Table 3", len(d["train"]) + len(test_df))

    split = {
        "source": "adsabs/SciX_UAT_keywords",
        "hf_split_mapping": {"train": "our training pool, carved into train + validation", "val": "TEST — the published benchmark split, never tuned on"},
        "validation_carve": {"function": "main.split_train_validation", "method": "iterstrat.MultilabelStratifiedShuffleSplit", "seed": SEED, "validation_size": VALIDATION_SIZE},
        "sizes": {"train": len(train_df), "validation": len(validation_df), "test": len(test_df)},
        "train": train_df["bibcode"].tolist(),
        "validation": validation_df["bibcode"].tolist(),
        "test": test_df["bibcode"].tolist(),
    }

    DATA_DIR.mkdir(exist_ok=True)
    write_json(DATA_DIR / "split.json", split)
    write_json(DATA_DIR / "label_order.json", label_order)
    write_json(DATA_DIR / "band_map_train.json", band_train)
    write_json(DATA_DIR / "band_map_corpus.json", band_corpus)
    write_json(DATA_DIR / "id_to_name.json", {str(i): id_to_name[i] for i in label_order})
    with open(DATA_DIR / "CHECKSUMS.sha256", "w") as f:
        for name in FILES:
            f.write(f"{sha256_of(DATA_DIR / name)}  {name}\n")

    verify(train_df, validation_df, test_df)


def verify(train_df=None, validation_df=None, test_df=None):
    """Checks the files on disk against their checksums and against the numbers in WORK_PLAN.md."""
    problems = []

    with open(DATA_DIR / "CHECKSUMS.sha256") as f:
        for line in f:
            digest, name = line.split()
            if sha256_of(DATA_DIR / name) != digest:
                problems.append(f"checksum mismatch: {name}")

    s = json.load(open(DATA_DIR / "split.json"))
    label_order = json.load(open(DATA_DIR / "label_order.json"))
    bt = json.load(open(DATA_DIR / "band_map_train.json"))
    bc = json.load(open(DATA_DIR / "band_map_corpus.json"))
    id_to_name = json.load(open(DATA_DIR / "id_to_name.json"))

    sets = {k: set(s[k]) for k in ("train", "validation", "test")}
    for k in sets:
        if len(sets[k]) != len(s[k]):
            problems.append(f"duplicate bibcodes in {k}")
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        problems.append("splits overlap")
    for k, n in (("train", EXPECTED["n_train"]), ("validation", EXPECTED["n_validation"]), ("test", EXPECTED["n_test"])):
        if len(s[k]) != n:
            problems.append(f"{k} has {len(s[k])} papers, expected {n}")

    if len(label_order) != EXPECTED["n_labels"] or label_order != sorted(label_order):
        problems.append(f"label_order: {len(label_order)} labels, sorted={label_order == sorted(label_order)}")
    if set(id_to_name) != {str(i) for i in label_order}:
        problems.append("id_to_name does not cover label_order exactly")
    for name, bm, key, sc in (("band_map_train", bt, "bands_train", "scoreable_tail_train"), ("band_map_corpus", bc, "bands_corpus", "scoreable_tail_corpus")):
        if bm["counts"] != EXPECTED[key]:
            problems.append(f"{name} counts {bm['counts']}, expected {EXPECTED[key]}")
        if bm["scoreable_tail"] != EXPECTED[sc]:
            problems.append(f"{name} scoreable tail {bm['scoreable_tail']}, expected {EXPECTED[sc]}")
        if set(bm["bands"]) != {str(i) for i in label_order}:
            problems.append(f"{name} does not cover label_order exactly")

    if train_df is not None:
        # every topic must still occur in the carved training set, or the model has an output it never sees
        present = set()
        for ids in train_df["verified_uat_ids"]:
            present.update(int(i) for i in ids)
        missing = [i for i in label_order if i not in present]
        if missing:
            problems.append(f"{len(missing)} topics have no paper in the carved training set")

    if problems:
        for p in problems:
            print("FAIL:", p)
        sys.exit(1)

    print(f"OK  split: train {len(s['train'])} / validation {len(s['validation'])} / test {len(s['test'])}"
          f"  (carve: {s['validation_carve']['method']}, seed {s['validation_carve']['seed']})")
    print(f"OK  labels: {len(label_order)}; id_to_name covers all")
    print(f"OK  bands, training basis (primary): {bt['counts']}  scoreable tail {bt['scoreable_tail']}")
    print(f"OK  bands, corpus basis:             {bc['counts']}  scoreable tail {bc['scoreable_tail']}")
    print(f"OK  checksums verified: {', '.join(FILES)}")


def load_split(data_dir=DATA_DIR, with_dataframes=True):
    """
    Reads the Stage 1 files. Returns a dict with the three DataFrames (rows in split.json order),
    topic_to_idx / idx_to_topic built from label_order.json, both band maps, rare_indices for the
    primary map, and id_to_name. This is the only supported way to obtain the split.
    """
    data_dir = Path(data_dir)
    s = json.load(open(data_dir / "split.json"))
    label_order = json.load(open(data_dir / "label_order.json"))
    band_train = {int(k): v for k, v in json.load(open(data_dir / "band_map_train.json"))["bands"].items()}
    band_corpus = {int(k): v for k, v in json.load(open(data_dir / "band_map_corpus.json"))["bands"].items()}
    id_to_name = {int(k): v for k, v in json.load(open(data_dir / "id_to_name.json")).items()}

    topic_to_idx = {uat_id: i for i, uat_id in enumerate(label_order)}
    idx_to_topic = {i: uat_id for uat_id, i in topic_to_idx.items()}
    out = {
        "split": s,
        "label_order": label_order,
        "topic_to_idx": topic_to_idx,
        "idx_to_topic": idx_to_topic,
        "band_train": band_train,
        "band_corpus": band_corpus,
        "rare_indices": [topic_to_idx[i] for i in label_order if band_train[i] == "tail"],
        "id_to_name": id_to_name,
        "num_labels": len(label_order),
    }
    if with_dataframes:
        from datasets import load_dataset
        ds = load_dataset("adsabs/SciX_UAT_keywords")
        pool = ds["train"].to_pandas().set_index("bibcode", drop=False)
        test = ds["val"].to_pandas().set_index("bibcode", drop=False)
        out["train_df"] = pool.loc[s["train"]].reset_index(drop=True)
        out["validation_df"] = pool.loc[s["validation"]].reset_index(drop=True)
        out["test_df"] = test.loc[s["test"]].reset_index(drop=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify the existing files instead of rebuilding them")
    args = ap.parse_args()
    if args.check:
        verify()
    else:
        build()
