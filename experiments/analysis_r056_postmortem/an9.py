import pickle
from math import comb
rows=pickle.load(open('rows.pkl','rb'))
H={k:v for k,v in rows.items() if k[0]=='heldout'}
def mcn(b,c):
    n=b+c; return min(1,2*sum(comb(n,i) for i in range(min(b,c)+1))/2**n) if n else 1
for arm in ['r16','r4']:
    for name,f in [('1327 only',lambda k:k[1]==1327),('excl 1327',lambda k:k[1]!=1327)]:
        S={k:v for k,v in H.items() if f(k)}
        b=sum(1 for v in S.values() if v['base']['success'] and not v[arm]['success']); c=sum(1 for v in S.values() if not v['base']['success'] and v[arm]['success'])
        print(arm,name,len(S),'base',sum(v['base']['success'] for v in S.values()),arm,sum(v[arm]['success'] for v in S.values()),'lost',b,'gained',c,'p=%.4f'%mcn(b,c))
# failure mode of all base-success -> arm-failure flips
for arm in ['r16','r4']:
    fl=[v[arm] for v in H.values() if v['base']['success'] and not v[arm]['success']]
    nolift=sum(1 for x in fl if x['lift_t'] is None); print(arm,'lost starts',len(fl),'never lifted',nolift,'lifted-then-failed',len(fl)-nolift)
