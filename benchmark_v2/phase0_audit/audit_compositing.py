import sys, os, numpy as np
from PIL import Image
root=sys.argv[1]
def load(p, mode="RGB"): return np.asarray(Image.open(p).convert(mode)).astype(np.float32)
for task,sub in [("FI","CelebA_FI"),("FU","CelebA_FU")]:
    T=os.path.join(root,"data",sub); files=sorted(os.listdir(os.path.join(T,"gt")))
    G={f:load(os.path.join(T,"gt",f)) for f in files}
    Mraw={f:load(os.path.join(T,"mask",f),"L") for f in files}
    u=np.unique(np.concatenate([np.unique(m) for m in Mraw.values()]))
    print(task,"mask unique values:",len(u),u[:8],"...",u[-5:])
    M={f:Mraw[f]>127 for f in files}
    r=np.array([M[f].mean() for f in files]); print(" hole ratio: mean %.3f min %.3f max %.3f; >0.5: %d"%(r.mean(),r.min(),r.max(),(r>0.5).sum()))
    # masked input consistency
    mi=np.mean([np.abs(load(os.path.join(T,"masked",f))-G[f])[~M[f]].mean() for f in files[:100]])
    print(" masked-input visible diff vs gt (100 imgs): %.3f"%mi)
    for meth in sorted(os.listdir(os.path.join(T,"SOTA"))):
        d=os.path.join(T,"SOTA",meth); vis=[];hol=[];miss=0;exact=0
        for f in files:
            p=os.path.join(d,f)
            if not os.path.exists(p): miss+=1; continue
            o=np.asarray(Image.open(p).convert("RGB").resize((256,256),Image.LANCZOS)).astype(np.float32)
            e=np.abs(o-G[f]); v=e[~M[f]].mean(); vis.append(v); hol.append(e[M[f]].mean() if M[f].any() else 0); exact+= v<0.5
        print(f"  {meth:14s} n={len(vis)} miss={miss} vis={np.mean(vis):.3f} hole={np.mean(hol):.2f} composited(vis<0.5)={exact}")
