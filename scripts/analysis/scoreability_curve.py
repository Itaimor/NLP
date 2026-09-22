import os, numpy as np
os.environ.setdefault("HF_HOME","/Users/ilanakolody/Desktop/School/ComputerScience/FourthYear/NLP/final_project/NLP/.hf_cache")
from datasets import load_dataset
from collections import Counter
ds=load_dataset("adsabs/SciX_UAT_keywords")
ya=[r["verified_uat_ids"] for r in ds["train"]]; yte=[r["verified_uat_ids"] for r in ds["val"]]
cnt=Counter(); cte=Counter()
for l in ya: cnt.update(l)
for l in yte: cnt.update(l); cte.update(l)
tail=[l for l,c in cnt.items() if c<50]
sc=[l for l in tail if cte.get(l,0)>0]
print(f"tail concepts: {len(tail)} | scoreable (>=1 test positive): {len(sc)}")
d=Counter(cte[l] for l in sc)
print("\ntest-positive count -> how many tail concepts have exactly that many:")
cum=0
for k in sorted(d):
    cum+=d[k]
    print(f"  {k:3d} test positives : {d[k]:4d} concepts   (cumulative {cum:4d} / {len(sc)} = {cum/len(sc):.1%})")
    if k>=12: break
for thr in (1,2,3,5,10):
    n=sum(1 for l in sc if cte[l]>=thr)
    print(f"\nconcepts with >= {thr:2d} test positives: {n:4d}  ({n/len(tail):.1%} of the 1,418 tail band)")
