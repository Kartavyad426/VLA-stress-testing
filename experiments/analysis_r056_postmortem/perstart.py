"""Paired per-start table: base vs r16(opt1000) vs r4(opt500) on the R-056 held-out set + nominal."""
import json, numpy as np, collections, sys
R='/home/imerit/Documents/Code/VLA/'
nom=json.load(open(R+'runs/r047/nominal_starts.json'))
def load(run):
    M=[json.loads(l) for l in open(R+f'runs/{run}/manifest.jsonl')]
    out={}
    for i,(m,l) in enumerate(zip(M,open(R+f'runs/{run}/rollouts.jsonl'))):
        r=json.loads(l); assert r['rollout_id']==m['rollout_id'] and r['success']==m['success']
        out[i]=(m,r)
    return out
A=load('r056_r16'); B=load('r058_r4')
def feat(m,r):
    st=r['steps']; s0=st[0]['obs_state']; n=nom[str(m['scene'])]
    off=float(np.linalg.norm(np.array(s0['eef_pos'])-np.array(n['eef_pos'])))
    jr=float(np.linalg.norm(np.array(s0['joint_pos'])-np.array(n['joint_pos'])))
    objs=list(s0['_gt_object_pos'].keys()); bowl=objs[0]
    z0=s0['_gt_object_pos'][bowl][2]
    zs=[s['obs_state']['_gt_object_pos'][bowl][2] for s in st]
    lift_t=next((s['t'] for s,z in zip(st,zs) if z-z0>0.03),None)
    acts=np.array([s['action'] for s in st if s.get('action') is not None])
    # eef distance to bowl over time
    d=[s['obs_state']['_gt_eef_to_object'][bowl] for s in st]
    return dict(success=r['success'],steps=r['env_steps'],off=off,jr=jr,lift_t=lift_t,maxlift=max(zs)-z0,
                a0=acts[:16], acts=acts, d16=d[min(16,len(d)-1)], d32=d[min(32,len(d)-1)], d48=d[min(48,len(d)-1)],
                dmin=min(d), eef=np.array([s['obs_state']['eef_pos'] for s in st]))
rows=collections.defaultdict(dict)
for run,D,tag in [('r056',A,None),('r058',B,None)]:
    for rid,(m,r) in D.items():
        if m['set'] not in ('heldout','nominal'): continue
        k=(m['set'],m['scene'],m['dir_seed'],m['radius'],m['noise_seed'],m['env_seed'])
        lab='base' if m['ckpt']=='base' else ('r16' if run=='r056' else 'r4')
        rows[k][lab]=feat(m,r)
import pickle; pickle.dump(dict(rows),open(sys.argv[1],'wb'))
print(len(rows), collections.Counter(tuple(sorted(v)) for v in rows.values()))
