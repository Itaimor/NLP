import os,numpy as np
from pathlib import Path
os.environ.setdefault("HF_HOME","/Users/ilanakolody/Desktop/School/ComputerScience/FourthYear/NLP/final_project/NLP/.hf_cache")
from datasets import load_dataset
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
ds=load_dataset("adsabs/SciX_UAT_keywords"); tr,va=ds["train"],ds["val"]
def txt(r): return (r["title"] or "")+" "+(r["abstract"] or "")
Xtr_raw=[txt(r) for r in tr]; Xva_raw=[txt(r) for r in va]
ytr=[r["verified_uat_ids"] for r in tr]; yva=[r["verified_uat_ids"] for r in va]
ctr=Counter(); cva=Counter()
for l in ytr: ctr.update(l)
for l in yva: cva.update(l)
tot=Counter(); tot.update(ctr); tot.update(cva)
labs=sorted(tot); li={l:i for i,l in enumerate(labs)}
def band(l):
    c=tot[l]
    return "HEAD" if c>500 else ("TORSO" if c>=50 else "TAIL")
print("n_labels",len(labs),"train",len(tr),"test",len(va),flush=True)
vec=TfidfVectorizer(max_features=50000,min_df=2,sublinear_tf=True,stop_words="english")
Xtr=vec.fit_transform(Xtr_raw); Xva=normalize(vec.transform(Xva_raw))
# label centroids from training docs
import scipy.sparse as sp
rows=[];cols=[]
for i,ls in enumerate(ytr):
    for l in ls: rows.append(li[l]); cols.append(i)
M=sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(labs),len(ytr)))
C=normalize(M@Xtr)   # label x vocab centroid
S=Xva@C.T            # test x label scores
S=np.asarray(S.todense()) if sp.issparse(S) else S
order=np.argsort(-S,axis=1)
for K in (25,50,100):
    topk=[set(order[i,:K]) for i in range(S.shape[0])]
    hit=Counter(); tot_gold=Counter()
    for i,ls in enumerate(yva):
        for l in ls:
            b=band(l); tot_gold[b]+=1
            if li[l] in topk[i]: hit[b]+=1
    allh=sum(hit.values()); allt=sum(tot_gold.values())
    print(f"recall@{K}: OVERALL {allh/allt:.3f} | "+" ".join(f"{b} {hit[b]/tot_gold[b]:.3f} (n={tot_gold[b]})" for b in ("HEAD","TORSO","TAIL")),flush=True)
# scoreable counts
sc={b:len({l for l in labs if band(l)==b and cva.get(l,0)>0}) for b in ("HEAD","TORSO","TAIL")}
al={b:len([l for l in labs if band(l)==b]) for b in ("HEAD","TORSO","TAIL")}
print("scoreable/all:",{b:(sc[b],al[b]) for b in sc},flush=True)
print("DONE",flush=True)
