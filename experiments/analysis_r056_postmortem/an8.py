import pickle,numpy as np,collections
rows=pickle.load(open('rows.pkl','rb'))
H={k:v for k,v in rows.items() if k[0]=='heldout'}
g=collections.defaultdict(list)
for k,v in H.items(): g[k[2]].append((v['base']['jr']/k[3], v['base']['off']))
for d,L in g.items(): A=np.array(L); print('dir',d,'achieved/requested median',A[:,0].mean().round(2),'range',A[:,0].min().round(2),A[:,0].max().round(2),' offset median cm',np.median(A[:,1]*100).round(1))
