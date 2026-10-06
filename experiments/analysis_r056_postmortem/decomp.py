import torch,re,numpy as np,sys
NM,NI=4800,52970
w=torch.cat([torch.full((NM,),0.5/NM,dtype=torch.double),torch.full((NI,),0.5/NI,dtype=torch.double)])
def stream(seed=1000,n=16000):
    g=torch.Generator().manual_seed(seed*1_000_003+0)
    return list(torch.utils.data.WeightedRandomSampler(w,num_samples=n,replacement=True,generator=g))
def cat(i):
    if i>=NM: return 3            # IPEC replay
    f=i%48
    return 0 if f==0 else (1 if f<16 else 2)   # minted: frame0 / driven 1-15 / self-generated 16-47
def logs(path):
    out=[]
    for l in open(path,errors='ignore').read().replace('\r','\n').split('\n'):
        m=re.search(r'step:([0-9.K]+) .*?loss:([0-9.]+)',l)
        if m:
            s=m.group(1); s=int(round(float(s[:-1])*1000)) if s.endswith('K') else int(s); out.append((s,float(m.group(2))))
    return out
S=stream(); C=np.array([cat(i) for i in S])
names=['minted f0','minted driven 1-15','minted self 16-47','IPEC replay']
for tag,path in [('r16','/home/imerit/Documents/Code/VLA/runs/overnight/R056fix_train.log'),('r4','/home/imerit/Documents/Code/VLA/runs/overnight/R058_r4_train.log')]:
    L=logs(path)
    # keep only logs at multiples of 10 with a full preceding window; K-rounded steps (>=1000) are ambiguous -> use index order
    pass
    print(f'== {tag}: {len(L)} log lines, sample source shares',np.bincount(C,minlength=4)/len(C))
    # K formatting: steps >= 1000 print like 1.0K,... ambiguity: reconstruct by order (every 10 micro-steps)
    steps=[10*(k+1) for k in range(len(L))]; bad=[(s,x[0]) for s,x in zip(steps,L) if abs(s-x[0])>100]; print("order mismatches",len(bad),bad[:3])
    y=np.array([x[1] for x in L])
    X=np.stack([np.bincount(C[s-10:s],minlength=4)/10 for s in steps])
    for lo,hi in [(0,400),(400,1000),(1000,2000)]:
        idx=[k for k,s in enumerate(steps) if lo*8<s<=hi*8]
        Xs,ys=X[idx],y[idx]
        # merge f0 into driven for stability; categories: driven(0-15), self(16-47), IPEC
        Xm=np.stack([Xs[:,0]+Xs[:,1],Xs[:,2],Xs[:,3]],1)
        coef,res,_,_=np.linalg.lstsq(Xm,ys,rcond=None)
        pred=Xm@coef; r2=1-((ys-pred)**2).sum()/((ys-ys.mean())**2).sum()
        # shuffled control
        rng=np.random.default_rng(0); Xp=Xm[rng.permutation(len(Xm))]; cp=np.linalg.lstsq(Xp,ys,rcond=None)[0]; r2p=1-((ys-Xp@cp)**2).sum()/((ys-ys.mean())**2).sum()
        print(f'  opt {lo:4d}-{hi:4d} n={len(idx)}  loss by source: driven {coef[0]:.3f}  minted-self {coef[1]:.3f}  IPEC {coef[2]:.3f}   R2 {r2:.2f} (shuffled-order control R2 {r2p:.2f})')
