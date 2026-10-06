import json,numpy as np,collections
R='/home/imerit/Documents/Code/VLA/'
prov={json.loads(l)['rollout_id']:json.loads(l) for l in open(R+'data/r054_train48/meta/provenance.jsonl')}
n=0; res=collections.defaultdict(list)
for l in open(R+'runs/r054/rollouts.jsonl'):
    r=json.loads(l)
    if r['rollout_id'] not in prov or not r['success']: continue
    st=r['steps']
    if '_gt_object_pos' not in st[0]['obs_state']: print('no gt'); break
    bowl=list(st[0]['obs_state']['_gt_object_pos'])[0]; z0=st[0]['obs_state']['_gt_object_pos'][bowl][2]
    closes=[];opens=[];prev=-1
    for s in st:
        a=s.get('action')
        if a is None: continue
        if a[6]>0 and prev<=0: closes.append(s['t'])
        if a[6]<=0 and prev>0: opens.append(s['t'])
        prev=a[6]
    lift=next((s['t'] for s in st if s['obs_state']['_gt_object_pos'][bowl][2]-z0>0.03),None)
    sc=prov[r['rollout_id']]['provenance']['scene_task_id']
    c48=[c for c in closes if c<48]; o48=[o for o in opens if o<48]
    res[sc].append(dict(first_close=closes[0] if closes else None,n_close48=len(c48),reopen48=len(o48),lift=lift,
                        failed_first=(len(closes)>1 and (lift is None or lift>closes[1])), steps=len(st)))
    n+=1
print('matched',n)
print(f"{'scene':>5} {'n':>3} {'close<48':>8} {'lift<48':>7} {'reopen<48':>9} {'1st grasp failed':>16} {'med 1st close':>13} {'med lift':>8}")
tot=collections.Counter()
for sc,L in sorted(res.items()):
    c=sum(1 for x in L if x['n_close48']>0); lf=sum(1 for x in L if x['lift'] is not None and x['lift']<48)
    ro=sum(1 for x in L if x['reopen48']>0); ff=sum(1 for x in L if x['failed_first'])
    fc=[x['first_close'] for x in L if x['first_close'] is not None]; li=[x['lift'] for x in L if x['lift'] is not None]
    print(f"{sc:>5} {len(L):>3} {c:>8} {lf:>7} {ro:>9} {ff:>16} {np.median(fc):>13.0f} {np.median(li):>8.0f}")
    tot.update(n=len(L),c=c,lf=lf,ro=ro,ff=ff)
print(dict(tot))
