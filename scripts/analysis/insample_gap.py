"""In-sample vs cross-fit vs test candidate coverage, for two generator types.

Question: when a candidate generator is trained on the same papers it later
shortlists, how much higher is gold coverage in its top-50 (and how much higher
does gold sit in the ranking) than on unseen papers?  And does 5-fold
cross-fitting remove the gap?

Generators:
  centroid  - TF-IDF label centroids (what recall_ceiling.py uses)
  sgd_ovr   - one-vs-rest logistic regression via SGD on the same TF-IDF
              features (a fitted, regularised classifier - closer in kind to
              a fine-tuned encoder than a centroid is)

Reports tail-band recall@50 and median gold rank on:
  IN-SAMPLE  : 2,000 training papers, generator fit on all 18,677 train papers
  CROSS-FIT  : the same 2,000 papers, each scored by a generator fit on the
               other 4/5 of the training papers (5-fold)
  TEST       : the 3,025 held-out papers, generator fit on all train papers
"""
import os, sys, time, numpy as np
os.environ.setdefault("HF_HOME", "/Users/ilanakolody/Desktop/School/ComputerScience/FourthYear/NLP/final_project/NLP/.hf_cache")
from datasets import load_dataset
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold
import scipy.sparse as sp

K = 50
N_PROBE = 2000
SEED = 42
GENS = sys.argv[1:] or ["centroid", "sgd_ovr"]

ds = load_dataset("adsabs/SciX_UAT_keywords"); tr, te = ds["train"], ds["val"]
txt = lambda r: (r["title"] or "") + " " + (r["abstract"] or "")
Xtr_raw = [txt(r) for r in tr]; Xte_raw = [txt(r) for r in te]
ytr = [r["verified_uat_ids"] for r in tr]; yte = [r["verified_uat_ids"] for r in te]
ctr = Counter(); cte = Counter()
for l in ytr: ctr.update(l)
for l in yte: cte.update(l)
tot = Counter(); tot.update(ctr); tot.update(cte)
labs = sorted(tot); li = {l: i for i, l in enumerate(labs)}
band = lambda l: "HEAD" if tot[l] > 500 else ("TORSO" if tot[l] >= 50 else "TAIL")

vec = TfidfVectorizer(max_features=50000, min_df=2, sublinear_tf=True, stop_words="english")
Xtr = normalize(vec.fit_transform(Xtr_raw)); Xte = normalize(vec.transform(Xte_raw))
rows, cols = [], []
for i, ls in enumerate(ytr):
    for l in ls: rows.append(i); cols.append(li[l])
Ytr = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ytr), len(labs)))

rng = np.random.default_rng(SEED)
probe = np.sort(rng.choice(len(ytr), N_PROBE, replace=False))

def fit_centroid(X, Y):
    return normalize(Y.T @ X)                      # label x vocab
def score_centroid(C, X):
    return np.asarray((X @ C.T).todense())
def fit_sgd(X, Y):
    clf = OneVsRestClassifier(SGDClassifier(loss="log_loss", alpha=2e-6, max_iter=20,
                                            tol=1e-3, random_state=SEED), n_jobs=-1)
    clf.fit(X, Y)
    return clf
def score_sgd(clf, X):
    S = clf.decision_function(X)
    # labels with no positive in the fit set get a constant predictor whose
    # decision_function is 0, which outranks real negative margins; push them down.
    absent = np.array([not hasattr(e, "coef_") for e in clf.estimators_])
    S[:, absent] = -1e9
    return S

FIT = {"centroid": (fit_centroid, score_centroid), "sgd_ovr": (fit_sgd, score_sgd)}

def report(name, S, gold):
    order = np.argsort(-S, axis=1)
    rank = np.empty_like(order);
    for i in range(order.shape[0]): rank[i, order[i]] = np.arange(order.shape[1])
    hit = Counter(); tot_g = Counter(); ranks = {"HEAD": [], "TORSO": [], "TAIL": []}
    for i, ls in enumerate(gold):
        for l in ls:
            b = band(l); tot_g[b] += 1; r = rank[i, li[l]]
            ranks[b].append(r + 1)
            if r < K: hit[b] += 1
    allh = sum(hit.values()); allt = sum(tot_g.values())
    tail = hit["TAIL"] / tot_g["TAIL"]
    print(f"  {name:<10} recall@{K}: pooled {allh/allt:.3f} | TAIL {tail:.3f} (n={tot_g['TAIL']}) "
          f"| median gold rank TAIL {int(np.median(ranks['TAIL']))}, pooled {int(np.median(sum(ranks.values(), [])))}",
          flush=True)

for g in GENS:
    fit, score = FIT[g]
    print(f"\n=== generator: {g} ===", flush=True)
    t0 = time.time()
    model = fit(Xtr, Ytr)
    print(f"  fit on all {Xtr.shape[0]} train papers: {time.time()-t0:.0f}s", flush=True)
    report("IN-SAMPLE", score(model, Xtr[probe]), [ytr[i] for i in probe])
    report("TEST", score(model, Xte), yte)
    # 5-fold cross-fit over the training set; score probe papers out-of-fold
    S_cf = np.zeros((N_PROBE, len(labs))); pos = {p: j for j, p in enumerate(probe)}
    t0 = time.time()
    for k, (fit_idx, hold_idx) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(np.arange(Xtr.shape[0]))):
        m = fit(Xtr[fit_idx], Ytr[fit_idx])
        held = [i for i in hold_idx if i in pos]
        if held:
            S_cf[[pos[i] for i in held]] = score(m, Xtr[held])
    print(f"  5-fold cross-fit: {time.time()-t0:.0f}s", flush=True)
    report("CROSS-FIT", S_cf, [ytr[i] for i in probe])
print("\nDONE", flush=True)
