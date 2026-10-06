import json,numpy as np,collections
R='/home/imerit/Documents/Code/VLA/'
def load(run):
    M=[json.loads(l) for l in open(R+f'runs/{run}/manifest.jsonl')]
    for m,l in zip(M,open(R+f'runs/{run}/rollouts.jsonl')):
        if m['set'] in('heldout','nominal'): yield m,json.loads(l)
def close_events(r):
    st=r['steps']; bowl=list(st[0]['obs_state']['_gt_object_pos'])[0]
    ev=[]; prev=-1
    for s in st:
        a=s.get('action'); 
        if a is None: continue
        g=a[6]
        if g>0 and prev<=0:
            e=np.array(s['obs_state']['eef_pos']); b=np.array(s['obs_state']['_gt_object_pos'][bowl])
            ev.append((s['t'], *(e-b)))
        prev=g
    return ev
out=collections.defaultdict(list)
for run in ['r056_r16','r058_r4']:
    for m,r in load(run):
        arm='base' if m['ckpt']=='base' else run[-3:].strip('_')
        ev=close_events(r)
        if not ev: continue
        t,dx,dy,dz=ev[0]
        out[(m['scene'],arm,r['success'])].append((t,dx,dy,dz,len(ev)))
for sc in [1327,1062,1201,984]:
    for arm in ['base','r16','r4']:
        for s in [True,False]:
            L=out.get((sc,arm,s))
            if not L: continue
            A=np.array(L)
            print(f"{sc} {arm:4s} {'S' if s else 'F'} n={len(L):2d} first close t={np.median(A[:,0]):5.1f}  eef-bowl dx {np.median(A[:,1])*100:5.1f} dy {np.median(A[:,2])*100:5.1f} dz {np.median(A[:,3])*100:5.1f} cm  #close events med {np.median(A[:,4]):.0f}")
