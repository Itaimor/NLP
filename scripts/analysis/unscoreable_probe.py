import os, numpy as np
from pathlib import Path
os.environ.setdefault("HF_HOME","/Users/ilanakolody/Desktop/School/ComputerScience/FourthYear/NLP/final_project/NLP/.hf_cache")
from datasets import load_dataset
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
ds=load_dataset("adsabs/SciX_UAT_keywords"); tr,va=ds["train"],ds["val"]
def txt(r): return (r["title"] or "")+" "+(r["abstract"] or "")
Xtr_raw=[txt(r) for r in tr]; Xva_raw=[txt(r) for r in va]
ytr=[r["verified_uat_ids"] for r in tr]; yva=[r["verified_uat_ids"] for r in va]
ctr=Counter(); cva=Counter()
for l in ytr: ctr.update(l)
for l in yva: cva.update(l)
tot=Counter(); tot.update(ctr); tot.update(cva)
tail=[l for l,c in tot.items() if c<50]
unsc=[l for l in tail if cva.get(l,0)==0]   # the 1,059 with zero test instances
print(f"unscoreable concepts: {len(unsc)}", flush=True)
vec=TfidfVectorizer(max_features=30000,min_df=2,sublinear_tf=True,stop_words="english")
Xtr=vec.fit_transform(Xtr_raw); Xva=vec.transform(Xva_raw)
trs=[set(x) for x in ytr]
P=np.zeros((Xva.shape[0],len(unsc)),dtype=np.float32)
for j,l in enumerate(unsc):
    yt=np.fromiter((1 if l in s else 0 for s in trs),int,len(trs))
    c=LogisticRegression(max_iter=300,C=4.0,solver="liblinear"); c.fit(Xtr,yt)
    P[:,j]=c.predict_proba(Xva)[:,1]
print(f"max prob assigned to any unscoreable concept: {P.max():.4f}", flush=True)
for tau in (0.5,0.20,0.08,0.02,0.01):
    B=P>=tau
    npred=int(B.sum()); nconc=int((B.sum(axis=0)>0).sum()); ndocs=int((B.sum(axis=1)>0).sum())
    print(f"tau={tau:<5} -> {npred:6d} predictions | {nconc:4d}/{len(unsc)} concepts ever fired | {ndocs:4d}/3025 test docs affected", flush=True)
print("DONE", flush=True)
