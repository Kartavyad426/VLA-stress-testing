import pickle,numpy as np,collections
from math import comb
rows=pickle.load(open('rows.pkl','rb'))
H={k:v for k,v in rows.items() if k[0]=='heldout'}; N={k:v for k,v in rows.items() if k[0]=='nominal'}
def mcn(b,c):
    n=b+c; p=sum(comb(n,i) for i in range(0,min(b,c)+1))/2**n*2 if n else 1; return min(1,p)
for arm in ['r16','r4']:
    for name,sub in [('all',H),('<=8cm',{k:v for k,v in H.items() if v['base']['off']<=0.08}),('>8cm',{k:v for k,v in H.items() if v['base']['off']>0.08})]:
        b=sum(1 for v in sub.values() if v['base']['success'] and not v[arm]['success'])
        c=sum(1 for v in sub.values() if not v['base']['success'] and v[arm]['success'])
        print(f"{arm} {name:6s} n={len(sub):3d} base {sum(v['base']['success'] for v in sub.values()):3d} arm {sum(v[arm]['success'] for v in sub.values()):3d}  lost {b:2d} gained {c:2d}  exact McNemar p={mcn(b,c):.3f}")
# r16 vs r4 agreement
b=sum(1 for v in H.values() if v['r16']['success'] and not v['r4']['success']); c=sum(1 for v in H.values() if not v['r16']['success'] and v['r4']['success'])
print('r16 vs r4 discordant',b,c)
print()
print('<=8cm starts: per (scene,dir,radius) outcomes over noise seeds  base|r16|r4, offset, achieved joint radius')
g=collections.defaultdict(list)
for k,v in H.items():
    if v['base']['off']<=0.08: g[k[1:4]].append((k[4],v))
for s,L in sorted(g.items()):
    L.sort(key=lambda x:x[0])
    f=lambda a:''.join('1' if v[a]['success'] else '.' for _,v in L)
    print(s, f('base'),f('r16'),f('r4'), f"off {L[0][1]['base']['off']*100:.1f}cm jr {L[0][1]['base']['jr']:.3f}")
print()
print('nominal failures:',[(k[1],k[4],a) for k,v in N.items() for a in ['base','r16','r4'] if not v[a]['success']])
