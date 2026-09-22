import os, numpy as np
from pathlib import Path
os.environ.setdefault("HF_HOME", str(Path("/Users/ilanakolody/Desktop/School/ComputerScience/FourthYear/NLP/final_project/NLP").resolve()/".hf_cache"))
from datasets import load_dataset
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
ds=load_dataset("adsabs/SciX_UAT_keywords"); tr,va=ds["train"],ds["val"]
def txt(r): return (r["title"] or "")+" "+(r["abstract"] or "")
Xtr_raw=[txt(r) for r in tr]; Xva_raw=[txt(r) for r in va]
ytr=[r["verified_uat_ids"] for r in tr]; yva=[r["verified_uat_ids"] for r in va]
ctr=Counter(); cva=Counter()
for l in ytr: ctr.update(l)
for l in yva: cva.update(l)
total=Counter(); total.update(ctr); total.update(cva)
bands={"TAIL":[l for l,c in total.items() if c<50 and cva.get(l,0)>0],
       "TORSO":[l for l,c in total.items() if 50<=c<=500 and cva.get(l,0)>0],
       "HEAD":[l for l,c in total.items() if c>500 and cva.get(l,0)>0]}
print("scoreable:", {k:len(v) for k,v in bands.items()}, flush=True)
vec=TfidfVectorizer(max_features=30000,min_df=2,sublinear_tf=True,stop_words="english")
Xtr=vec.fit_transform(Xtr_raw); Xva=vec.transform(Xva_raw)
print("tfidf:",Xtr.shape, flush=True)
trs=[set(x) for x in ytr]; vas=[set(x) for x in yva]
for nm in ["TAIL","TORSO","HEAD"]:
    labs=bands[nm]
    P=np.zeros((Xva.shape[0],len(labs))); Y=np.zeros((Xva.shape[0],len(labs)),dtype=int)
    for j,l in enumerate(labs):
        yt=np.fromiter((1 if l in s else 0 for s in trs),int,len(trs))
        Y[:,j]=np.fromiter((1 if l in s else 0 for s in vas),int,len(vas))
        clf=LogisticRegression(max_iter=300,C=4.0,solver="liblinear"); clf.fit(Xtr,yt)
        P[:,j]=clf.predict_proba(Xva)[:,1]
    f05=f1_score(Y,(P>=0.5).astype(int),average="macro",zero_division=0)
    best=(0.0,0.0)
    for t in np.arange(0.02,0.52,0.02):
        f=f1_score(Y,(P>=t).astype(int),average="macro",zero_division=0)
        if f>best[0]: best=(f,t)
    fired=int(((P>=0.5).sum(axis=0)>0).sum())
    print(f"{nm:5s} n={len(labs):4d} | macroF1@0.5={f05:.4f} | best={best[0]:.4f}@tau={best[1]:.2f} | firing@0.5: {fired}/{len(labs)} | maxprob={P.max():.3f}", flush=True)
print("DONE", flush=True)
