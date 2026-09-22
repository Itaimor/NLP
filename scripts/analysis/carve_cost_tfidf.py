# Does carving 2,855 validation papers out of the 18,677 cost tail F1? TF-IDF+LR on the 805 test-scoreable labels:
#  (1) train 15,822 -> per-band tau tuned on validation -> test    (the paper's protocol, option A)
#  (2) train 18,677 with the SAME tau                     -> test    (option B refit; tau transferred)
#  (3) train 15,822 -> tau tuned ON TEST                  -> test    (the inflation from test-selected tau, i.e. Alkan's practice)
# Bands = primary map (18,677 basis). Paired bootstrap over test papers for (2)-(1).
import os, sys, json, numpy as np, time
os.environ.setdefault("HF_HOME", os.path.abspath(".hf_cache")); sys.path.insert(0, "scripts")
from build_split import load_split
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
t0=time.time(); S=load_split()
tr,va,te=S["train_df"],S["validation_df"],S["test_df"]; band=S["band_train"]
s_=lambda v: v if isinstance(v,str) else ""
txt=lambda df:[s_(a)+" "+s_(b) for a,b in zip(df["title"],df["abstract"])]
cte=Counter(); [cte.update(int(i) for i in l) for l in te["verified_uat_ids"]]
labels=[l for l in S["label_order"] if cte.get(l,0)>0]; L=len(labels); print("scoreable labels", L)
def Y(df): 
    sets=[set(int(i) for i in l) for l in df["verified_uat_ids"]]
    return np.array([[1 if l in s else 0 for l in labels] for s in sets],dtype=np.int8)
Ytr,Yva,Yte=Y(tr),Y(va),Y(te)
def fit_predict(Xtr_raw,Ytr_,Xouts):
    vec=TfidfVectorizer(max_features=30000,min_df=2,sublinear_tf=True,stop_words="english"); Xtr=vec.fit_transform(Xtr_raw)
    Xo=[vec.transform(x) for x in Xouts]; P=[np.zeros((x.shape[0],L)) for x in Xo]
    for j in range(L):
        y=Ytr_[:,j]
        if y.sum()==0: continue
        c=LogisticRegression(max_iter=300,C=4.0,solver="liblinear").fit(Xtr,y)
        for k,x in enumerate(Xo): P[k][:,j]=c.predict_proba(x)[:,1]
    return P
def micro(Yt,B): tp=(Yt*B).sum(); fp=((1-Yt)*B).sum(); fn=(Yt*(1-B)).sum(); return 0.0 if tp==0 else 2*tp/(2*tp+fp+fn)
def macro(Yt,B):
    tp=(Yt*B).sum(0); fp=((1-Yt)*B).sum(0); fn=(Yt*(1-B)).sum(0); den=2*tp+fp+fn; return np.where(den==0,0.0,2*tp/np.maximum(den,1)).mean()
grid=np.concatenate([np.arange(0.005,0.10,0.005),np.arange(0.10,0.55,0.02)])
cols={b:np.array([j for j,l in enumerate(labels) if band[l]==b]) for b in ("head","torso","tail")}
def tune(P,Yt): return {b:max(grid,key=lambda t:micro(Yt[:,c],(P[:,c]>=t).astype(np.int8))) for b,c in cols.items()}
def score(P,Yt,tau): 
    B=np.zeros_like(P,dtype=np.int8)
    for b,c in cols.items(): B[:,c]=(P[:,c]>=tau[b])
    return {b:(micro(Yt[:,c],B[:,c]),macro(Yt[:,c],B[:,c])) for b,c in cols.items()}, B
# (1) protocol A
Pva,Pte=fit_predict(txt(tr),Ytr,[txt(va),txt(te)]); tauA=tune(Pva,Yva); A,BA=score(Pte,Yte,tauA)
print(f"\n(1) train 15,822, tau on VALIDATION {{{', '.join(f'{b}:{t:.3f}' for b,t in tauA.items())}}}  [{time.time()-t0:.0f}s]")
for b in cols: print(f"    {b:5s} micro={A[b][0]:.4f} macro={A[b][1]:.4f}")
# (2) refit on 18,677 with the same tau
import pandas as pd
full=pd.concat([tr,va],ignore_index=True); Yfull=np.vstack([Ytr,Yva])
(Pte2,)=fit_predict(txt(full),Yfull,[txt(te)]); B2,BB=score(Pte2,Yte,tauA)
print(f"\n(2) train 18,677, SAME tau -> test")
for b in cols: print(f"    {b:5s} micro={B2[b][0]:.4f} ({(B2[b][0]-A[b][0])*100:+.2f}pp) macro={B2[b][1]:.4f} ({(B2[b][1]-A[b][1])*100:+.2f}pp)")
# paired bootstrap on tail micro, (2)-(1)
rng=np.random.default_rng(0); n=Yte.shape[0]; c=cols["tail"]; d=[]
for _ in range(1000):
    idx=rng.integers(0,n,n); d.append(micro(Yte[idx][:,c],BB[idx][:,c])-micro(Yte[idx][:,c],BA[idx][:,c]))
lo,hi=np.percentile(d,[2.5,97.5]); print(f"    paired bootstrap, tail micro (2)-(1): [{lo*100:+.2f}, {hi*100:+.2f}] pp")
# (2b) refit with tau re-tuned on validation -- NOT honest (val is in training) but shows the calibration shift
tau2=tune(fit_predict(txt(full),Yfull,[txt(va)])[0],Yva)
print(f"    (calibration check: tau re-tuned on the now-in-sample validation set would be {{{', '.join(f'{b}:{t:.3f}' for b,t in tau2.items())}}} — not usable, shows the shift)")
# (3) tau tuned on TEST for model (1)
tauT=tune(Pte,Yte); T,_=score(Pte,Yte,tauT)
print(f"\n(3) train 15,822, tau tuned ON TEST {{{', '.join(f'{b}:{t:.3f}' for b,t in tauT.items())}}} (Alkan-style selection)")
for b in cols: print(f"    {b:5s} micro={T[b][0]:.4f} ({(T[b][0]-A[b][0])*100:+.2f}pp vs honest) macro={T[b][1]:.4f} ({(T[b][1]-A[b][1])*100:+.2f}pp)")
# tau stability: bootstrap the validation set, re-tune tail tau
taus=[]; nv=Yva.shape[0]
for _ in range(200):
    idx=rng.integers(0,nv,nv); taus.append(max(grid,key=lambda t:micro(Yva[idx][:,c],(Pva[idx][:,c]>=t).astype(np.int8))))
taus=np.array(taus); print(f"\ntail tau stability (200 validation bootstraps): median {np.median(taus):.3f}, 5-95% [{np.percentile(taus,5):.3f}, {np.percentile(taus,95):.3f}]; test tail micro at those taus: {np.percentile([micro(Yte[:,c],(Pte[:,c]>=t).astype(np.int8)) for t in np.unique(taus)],[0,100])[0]:.4f}-{np.percentile([micro(Yte[:,c],(Pte[:,c]>=t).astype(np.int8)) for t in np.unique(taus)],[0,100])[1]:.4f}")
print(f"DONE {time.time()-t0:.0f}s")
