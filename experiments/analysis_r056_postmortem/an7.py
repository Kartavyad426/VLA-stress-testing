import pickle,numpy as np
rows=pickle.load(open('rows.pkl','rb'))
H={k:v for k,v in rows.items() if k[0]=='heldout'}
nomA={k[1]:v['base']['a0'] for k,v in rows.items() if k[0]=='nominal' and k[4]==0}
def disp(a): return a[:16,:3].sum(0)  # commanded translation over chunk 0 (normalised delta units)
print('chunk-0 commanded translation (sum dx,dy,dz over 16 steps), held-out, noise seed 0 only')
res={'r16':[],'r4':[]}
for k,v in sorted(H.items(),key=str):
    if k[4]!=0: continue
    nb=np.linalg.norm(disp(v['base']['a0'])-disp(nomA[k[1]]))
    for arm in res:
        na=np.linalg.norm(disp(v[arm]['a0'])-disp(nomA[k[1]])); dab=np.linalg.norm(disp(v[arm]['a0'])-disp(v['base']['a0']))
        res[arm].append((nb,na,dab))
for arm,L in res.items():
    A=np.array(L); print(arm,'median |base-nominal|',np.median(A[:,0]).round(3),' |arm-nominal|',np.median(A[:,1]).round(3),
        ' |arm-base|',np.median(A[:,2]).round(3),' arm closer to nominal than base in',int((A[:,1]<A[:,0]).sum()),'/',len(A))
# eef trajectory divergence from base at t=16,32,48 (same start & seed)
for arm in ['r16','r4']:
    for t in [16,32,48]:
        d=[np.linalg.norm(v[arm]['eef'][min(t,len(v[arm]['eef'])-1)]-v['base']['eef'][min(t,len(v['base']['eef'])-1)]) for v in H.values()]
        print(arm,'eef |arm-base| at t',t,'median cm',round(np.median(d)*100,2),'p90',round(np.percentile(d,90)*100,2))
