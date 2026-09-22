import os, numpy as np
from pathlib import Path
os.environ.setdefault("HF_HOME","/Users/ilanakolody/Desktop/School/ComputerScience/FourthYear/NLP/final_project/NLP/.hf_cache")
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
tot=Counter(); tot.update(ctr); tot.update(cva)
bands={"HEAD":[l for l,c in tot.items() if c>500 and cva.get(l,0)>0],
       "TORSO":[l for l,c in tot.items() if 50<=c<=500 and cva.get(l,0)>0],
       "TAIL":[l for l,c in tot.items() if c<50 and cva.get(l,0)>0]}
allc=bands["HEAD"]+bands["TORSO"]+bands["TAIL"]
print("scoreable:", {k:len(v) for k,v in bands.items()}, "total", len(allc), flush=True)
vec=TfidfVectorizer(max_features=30000,min_df=2,sublinear_tf=True,stop_words="english")
Xtr=vec.fit_transform(Xtr_raw); Xva=vec.transform(Xva_raw)
trs=[set(x) for x in ytr]; vas=[set(x) for x in yva]
P=np.zeros((Xva.shape[0],len(allc)),dtype=np.float32); Y=np.zeros((Xva.shape[0],len(allc)),dtype=int)
for j,l in enumerate(allc):
    yt=np.fromiter((1 if l in s else 0 for s in trs),int,len(trs))
    Y[:,j]=np.fromiter((1 if l in s else 0 for s in vas),int,len(vas))
    c=LogisticRegression(max_iter=300,C=4.0,solver="liblinear"); c.fit(Xtr,yt)
    P[:,j]=c.predict_proba(Xva)[:,1]
idx={}; o=0
for k,v in bands.items(): idx[k]=list(range(o,o+len(v))); o+=len(v)
taus=np.arange(0.01,0.51,0.01)
def bandF1(k,tau): 
    c=idx[k]; return f1_score(Y[:,c],(P[:,c]>=tau).astype(int),average="macro",zero_division=0)
def overallF1(tau): return f1_score(Y,(P>=tau).astype(int),average="macro",zero_division=0)
# (a) ONE global tau, chosen to maximise overall macro-F1
gt=max(taus,key=overallF1)
print(f"\n(a) ONE GLOBAL tau maximising overall macro-F1  -> tau = {gt:.2f}", flush=True)
print(f"    overall macro-F1 = {overallF1(gt):.4f}")
for k in ("HEAD","TORSO","TAIL"): print(f"    {k:6s} F1 at that global tau = {bandF1(k,gt):.4f}")
# (b) PER-BAND tau
print("\n(b) PER-BAND tau, each maximising its own band", flush=True)
best={}
for k in ("HEAD","TORSO","TAIL"):
    bt=max(taus,key=lambda x:bandF1(k,x)); best[k]=(bt,bandF1(k,bt))
    print(f"    {k:6s} tau = {bt:.2f}  F1 = {best[k][1]:.4f}")
print("\n=== WHAT PER-BAND TUNING BUYS ===", flush=True)
for k in ("HEAD","TORSO","TAIL"):
    g=bandF1(k,gt); b=best[k][1]
    print(f"  {k:6s} global {g:.4f} -> per-band {b:.4f}   gain {b-g:+.4f} ({(b/g if g>0 else float('inf')):.2f}x)")
print("DONE", flush=True)
