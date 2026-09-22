import os, numpy as np
os.environ.setdefault("HF_HOME","/Users/ilanakolody/Desktop/School/ComputerScience/FourthYear/NLP/final_project/NLP/.hf_cache")
from datasets import load_dataset
from collections import Counter
ds=load_dataset("adsabs/SciX_UAT_keywords")
ya=[r["verified_uat_ids"] for r in ds["train"]]; yte=[r["verified_uat_ids"] for r in ds["val"]]
ctr=Counter()
for l in ya: ctr.update(l)
cnt=Counter(ctr); cte=Counter()
for l in yte: cnt.update(l); cte.update(l)
def band(l):
    c=cnt[l]; return "HEAD" if c>500 else ("TORSO" if c>=50 else "TAIL")
bands=("HEAD","TORSO","TAIL")
# majority = always predict the k most frequent TRAIN concepts (k = 4 ~ gold mean 4.31)
for k in (1,4,5):
    top=set(l for l,_ in ctr.most_common(k))
    print(f"\n--- MAJORITY baseline: always predict the {k} most frequent train concepts ---")
    for b in bands:
        labs=[l for l in cnt if band(l)==b]
        sco=[l for l in labs if cte.get(l,0)>0]
        tp=fp=fn=0; f1s=[]
        for l in labs:
            g=cte.get(l,0); p=len(yte) if l in top else 0
            t=g if l in top else 0
            tp+=t; fp+=p-t; fn+=g-t
            den=2*t+(p-t)+(g-t)
            if l in sco: f1s.append(0.0 if den==0 else 2*t/den)
        mic=0.0 if tp==0 else 2*tp/(2*tp+fp+fn)
        mac_s=float(np.mean(f1s)) if f1s else 0.0
        mac_a=mac_s*len(sco)/len(labs)
        print(f"  {b:6s} micro-F1={mic:.4f}  macro/scoreable({len(sco)})={mac_s:.4f}  macro/all({len(labs)})={mac_a:.4f}")
