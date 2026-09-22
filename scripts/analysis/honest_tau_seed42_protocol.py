# Re-run of the honest threshold measurement on the SEED-42 STRATIFIED carve that main.py::split_train_validation
# produces (the split build_split.py will persist), for ALL 805 test-scoreable labels, reporting the tail under
# (a) tau=0.5, (b) ONE GLOBAL tau tuned on validation (the protocol spec §8 locks), (c) a tail-band tau.
# Also reports the same predictions under the TRAIN(18,677)-basis band map, since WORK_PLAN calls that primary.
import os, numpy as np, time
os.environ.setdefault("HF_HOME", os.path.abspath(".hf_cache"))
from datasets import load_dataset
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
t0=time.time()
ds=load_dataset("adsabs/SciX_UAT_keywords"); tr_all,test=ds["train"],ds["val"]
def txt(r): return (r["title"] or "")+" "+(r["abstract"] or "")
Xall=[txt(r) for r in tr_all]; ya=[r["verified_uat_ids"] for r in tr_all]
Xte_raw=[txt(r) for r in test]; yte=[r["verified_uat_ids"] for r in test]
mlb=MultiLabelBinarizer(); Y=mlb.fit_transform(ya)
ti,vi=next(MultilabelStratifiedShuffleSplit(n_splits=1,test_size=0.15,random_state=42).split(np.zeros((len(ya),1)),Y))
Xtr_raw=[Xall[i] for i in ti]; ytr=[ya[i] for i in ti]; Xva_raw=[Xall[i] for i in vi]; yva=[ya[i] for i in vi]
print(f"carve (seed 42, iterstrat): train {len(ti)} val {len(vi)} test {len(yte)}",flush=True)
corpus=Counter(); [corpus.update(l) for l in ya]; [corpus.update(l) for l in yte]
train18=Counter(); [train18.update(l) for l in ya]
cte=Counter(); [cte.update(l) for l in yte]
def band(c): return "HEAD" if c>500 else "TORSO" if c>=50 else "TAIL"
labels=[l for l in corpus if cte.get(l,0)>0]  # 805 test-scoreable
bc={l:band(corpus[l]) for l in labels}; bt={l:band(train18[l]) for l in labels}
print("scoreable by CORPUS basis:",Counter(bc.values()),"| by TRAIN(18,677) basis:",Counter(bt.values()),flush=True)
vec=TfidfVectorizer(max_features=30000,min_df=2,sublinear_tf=True,stop_words="english")
Xtr=vec.fit_transform(Xtr_raw); Xva=vec.transform(Xva_raw); Xte=vec.transform(Xte_raw)
trs=[set(x) for x in ytr]; vas=[set(x) for x in yva]; tes=[set(x) for x in yte]
L=len(labels); Pva=np.zeros((Xva.shape[0],L)); Yva=np.zeros((Xva.shape[0],L),dtype=np.int8)
Pte=np.zeros((Xte.shape[0],L)); Yte=np.zeros((Xte.shape[0],L),dtype=np.int8)
for j,l in enumerate(labels):
    yt=np.fromiter((1 if l in s else 0 for s in trs),int,len(trs))
    Yva[:,j]=np.fromiter((1 if l in s else 0 for s in vas),int,len(vas)); Yte[:,j]=np.fromiter((1 if l in s else 0 for s in tes),int,len(tes))
    if yt.sum()==0: continue
    c=LogisticRegression(max_iter=300,C=4.0,solver="liblinear"); c.fit(Xtr,yt)
    Pva[:,j]=c.predict_proba(Xva)[:,1]; Pte[:,j]=c.predict_proba(Xte)[:,1]
    if j%100==0: print(f" fit {j}/{L}  {time.time()-t0:.0f}s",flush=True)
def micro(Y,B):
    tp=(Y*B).sum(); fp=((1-Y)*B).sum(); fn=(Y*(1-B)).sum(); return 0.0 if tp==0 else 2*tp/(2*tp+fp+fn)
def macro(Y,B):
    tp=(Y*B).sum(0); fp=((1-Y)*B).sum(0); fn=(Y*(1-B)).sum(0); den=2*tp+fp+fn
    return np.where(den==0,0.0,2*tp/np.maximum(den,1)).mean()
grid=np.concatenate([np.arange(0.005,0.10,0.005),np.arange(0.10,0.55,0.02)])
def cols(bmap,name): return np.array([i for i,l in enumerate(labels) if bmap[l]==name])
for basis,bmap,ntail in (("CORPUS",bc,1418),("TRAIN18677",bt,1469)):
    print(f"\n===== band basis: {basis} =====",flush=True)
    tail=cols(bmap,"TAIL")
    g_mi=max(grid,key=lambda t:micro(Yva,(Pva>=t).astype(np.int8)))
    g_ma=max(grid,key=lambda t:macro(Yva,(Pva>=t).astype(np.int8)))
    t_mi=max(grid,key=lambda t:micro(Yva[:,tail],(Pva[:,tail]>=t).astype(np.int8)))
    t_ma=max(grid,key=lambda t:macro(Yva[:,tail],(Pva[:,tail]>=t).astype(np.int8)))
    print(f"tau tuned on VAL: global-micro={g_mi:.3f} global-macro={g_ma:.3f} | tail-band micro={t_mi:.3f} macro={t_ma:.3f}",flush=True)
    for nm,t in (("tau=0.5 default",0.5),("ONE GLOBAL tau (micro-opt)",g_mi),("ONE GLOBAL tau (macro-opt)",g_ma),("TAIL-BAND tau (micro-opt)",t_mi),("TAIL-BAND tau (macro-opt)",t_ma)):
        B=(Pte>=t).astype(np.int8)
        for bn in ("HEAD","TORSO","TAIL"):
            c=cols(bmap,bn); m=macro(Yte[:,c],B[:,c])
            extra=f"  macro/{ntail}={m*len(c)/ntail:.4f}" if bn=="TAIL" else ""
            print(f"  TEST {nm:28s} {bn:5s} micro={micro(Yte[:,c],B[:,c]):.4f} macro/{len(c)}={m:.4f}{extra}",flush=True)
print(f"\nDONE {time.time()-t0:.0f}s",flush=True)
