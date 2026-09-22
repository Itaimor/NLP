import os, numpy as np
os.environ.setdefault("HF_HOME","/Users/ilanakolody/Desktop/School/ComputerScience/FourthYear/NLP/final_project/NLP/.hf_cache")
from datasets import load_dataset
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ds=load_dataset("adsabs/SciX_UAT_keywords")
tr_all,test=ds["train"],ds["val"]          # ds["val"] IS the designated TEST split
def txt(r): return (r["title"] or "")+" "+(r["abstract"] or "")
Xall_raw=[txt(r) for r in tr_all]; ya=[r["verified_uat_ids"] for r in tr_all]
Xte_raw=[txt(r) for r in test];    yte=[r["verified_uat_ids"] for r in test]

# ---- carve 15% VALIDATION out of train. tau is tuned HERE, never on test.
rng=np.random.default_rng(13)
n=len(Xall_raw); perm=rng.permutation(n); ncut=int(0.15*n)
vi,ti=set(perm[:ncut].tolist()),perm[ncut:]
Xtr_raw=[Xall_raw[i] for i in ti]; ytr=[ya[i] for i in ti]
Xva_raw=[Xall_raw[i] for i in sorted(vi)]; yva=[ya[i] for i in sorted(vi)]
print(f"train {len(Xtr_raw)} | carved-val {len(Xva_raw)} | test {len(Xte_raw)}",flush=True)

# ---- bands from FULL corpus frequency (spec 6.3), scoreable = has >=1 TEST instance
cnt=Counter()
for l in ya: cnt.update(l)
for l in yte: cnt.update(l)
cte=Counter()
for l in yte: cte.update(l)
tail=[l for l,c in cnt.items() if c<50 and cte.get(l,0)>0]
print("tail scoreable:",len(tail),flush=True)

vec=TfidfVectorizer(max_features=30000,min_df=2,sublinear_tf=True,stop_words="english")
Xtr=vec.fit_transform(Xtr_raw); Xva=vec.transform(Xva_raw); Xte=vec.transform(Xte_raw)
trs=[set(x) for x in ytr]; vas=[set(x) for x in yva]; tes=[set(x) for x in yte]

Pva=np.zeros((Xva.shape[0],len(tail))); Yva=np.zeros((Xva.shape[0],len(tail)),dtype=np.int8)
Pte=np.zeros((Xte.shape[0],len(tail))); Yte=np.zeros((Xte.shape[0],len(tail)),dtype=np.int8)
for j,l in enumerate(tail):
    yt=np.fromiter((1 if l in s else 0 for s in trs),int,len(trs))
    Yva[:,j]=np.fromiter((1 if l in s else 0 for s in vas),int,len(vas))
    Yte[:,j]=np.fromiter((1 if l in s else 0 for s in tes),int,len(tes))
    if yt.sum()==0: continue
    c=LogisticRegression(max_iter=300,C=4.0,solver="liblinear"); c.fit(Xtr,yt)
    Pva[:,j]=c.predict_proba(Xva)[:,1]; Pte[:,j]=c.predict_proba(Xte)[:,1]
    if j%60==0: print(" fit",j,flush=True)

def micro(Y,B):
    tp=(Y*B).sum(); fp=((1-Y)*B).sum(); fn=(Y*(1-B)).sum()
    return 0.0 if tp==0 else 2*tp/(2*tp+fp+fn)
def macro(Y,B):
    tp=(Y*B).sum(0); fp=((1-Y)*B).sum(0); fn=(Y*(1-B)).sum(0)
    den=2*tp+fp+fn
    return np.where(den==0,0.0,2*tp/np.maximum(den,1)).mean()

# ---- tune tau on the CARVED VALIDATION split
grid=np.concatenate([np.arange(0.005,0.10,0.005),np.arange(0.10,0.55,0.02)])
bmi=max(grid,key=lambda t:micro(Yva,(Pva>=t).astype(np.int8)))
bma=max(grid,key=lambda t:macro(Yva,(Pva>=t).astype(np.int8)))
print(f"\ntau tuned on CARVED VAL: micro-opt={bmi:.3f}  macro-opt={bma:.3f}",flush=True)

for nm,t in (("tau=0.5 (default)",0.5),(f"tau={bmi:.3f} (micro-tuned on val)",bmi),(f"tau={bma:.3f} (macro-tuned on val)",bma)):
    B=(Pte>=t).astype(np.int8)
    print(f"  TEST {nm:36s} micro={micro(Yte,B):.4f}  macro/359={macro(Yte,B):.4f}  macro/1418={macro(Yte,B)*len(tail)/1418:.4f}",flush=True)

# ---- honest threshold effect, macro, tuned on val
B05=(Pte>=0.5).astype(np.int8); Bt=(Pte>=bma).astype(np.int8)
m05,mt=macro(Yte,B05),macro(Yte,Bt)
print(f"\nHONEST threshold effect (macro/359, tau tuned on carved val): {m05:.4f} -> {mt:.4f} = {mt/max(m05,1e-9):.1f}x",flush=True)

# ---- bootstrap CI over DOCUMENTS for BOTH micro and macro at the honest tau
nte=Yte.shape[0]; rng2=np.random.default_rng(0); bmi_l=[]; bma_l=[]
for _ in range(1000):
    idx=rng2.integers(0,nte,nte)
    bmi_l.append(micro(Yte[idx],Bt[idx])); bma_l.append(macro(Yte[idx],Bt[idx]))
for nm,arr in (("MICRO",np.array(bmi_l)),("MACRO/359",np.array(bma_l))):
    lo,hi=np.percentile(arr,[2.5,97.5])
    print(f"bootstrap CI95 {nm:10s}: [{lo:.4f}, {hi:.4f}]  half-width={(hi-lo)/2*100:.2f} pp  => unpaired MDE ~{(hi-lo)/2*100*1.41:.1f} pp",flush=True)
print("DONE",flush=True)
