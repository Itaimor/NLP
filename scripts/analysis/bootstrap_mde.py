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
tail=[l for l,c in tot.items() if c<50 and cva.get(l,0)>0]
print("tail scoreable:",len(tail),flush=True)
vec=TfidfVectorizer(max_features=30000,min_df=2,sublinear_tf=True,stop_words="english")
Xtr=vec.fit_transform(Xtr_raw); Xva=vec.transform(Xva_raw)
trs=[set(x) for x in ytr]; vas=[set(x) for x in yva]
P=np.zeros((Xva.shape[0],len(tail))); Y=np.zeros((Xva.shape[0],len(tail)),dtype=int)
for j,l in enumerate(tail):
    yt=np.fromiter((1 if l in s else 0 for s in trs),int,len(trs))
    Y[:,j]=np.fromiter((1 if l in s else 0 for s in vas),int,len(vas))
    c=LogisticRegression(max_iter=300,C=4.0,solver="liblinear"); c.fit(Xtr,yt)
    P[:,j]=c.predict_proba(Xva)[:,1]
np.savez("/private/tmp/claude-501/-Users-ilanakolody-Desktop-School-ComputerScience-FourthYear-NLP-final-project/60dc9cfc-dbb5-4f50-baf2-ad198bfb0314/scratchpad/council/tail_preds.npz",P=P,Y=Y)
def micro(Y,B):
    tp=(Y*B).sum(); fp=((1-Y)*B).sum(); fn=(Y*(1-B)).sum()
    return 0.0 if tp==0 else 2*tp/(2*tp+fp+fn)
tau=0.02; B=(P>=tau).astype(int)
pt=micro(Y,B)
print(f"tail MICRO-F1 @tau=0.02 = {pt:.4f}   (gold tail occurrences in test = {Y.sum()})",flush=True)
rng=np.random.default_rng(0); n=Y.shape[0]; boots=[]
for _ in range(2000):
    idx=rng.integers(0,n,n); boots.append(micro(Y[idx],B[idx]))
boots=np.array(boots); lo,hi=np.percentile(boots,[2.5,97.5])
print(f"bootstrap CI95 over DOCUMENTS: [{lo:.4f}, {hi:.4f}]  half-width = {(hi-lo)/2*100:.2f} pp",flush=True)
print(f"=> UNPAIRED minimum detectable difference ~ {(hi-lo)/2*100*1.41:.1f} pp",flush=True)
# paired: simulate a second model = same preds with r% of decisions flipped, measure paired CI
for eff in (0.02,0.05):
    B2=B.copy(); flip=rng.random(B.shape)<eff; B2=np.where(flip,1-B2,B2)
    d=[]
    for _ in range(2000):
        idx=rng.integers(0,n,n); d.append(micro(Y[idx],B[idx])-micro(Y[idx],B2[idx]))
    d=np.array(d); dlo,dhi=np.percentile(d,[2.5,97.5])
    print(f"PAIRED diff CI95 (perturb {eff:.0%}): [{dlo*100:+.2f}, {dhi*100:+.2f}] pp  half-width={(dhi-dlo)/2*100:.2f} pp",flush=True)
print("DONE",flush=True)
