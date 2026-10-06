import json,numpy as np,collections
R='/home/imerit/Documents/Code/VLA/'
prov={json.loads(l)['rollout_id']:json.loads(l) for l in open(R+'data/r054_train48/meta/provenance.jsonl')}
rows=[]
for l in open(R+'runs/r054/rollouts.jsonl'):
    r=json.loads(l)
    if r['rollout_id'] not in prov or not r['success']: continue
    st=r['steps']; bowl=list(st[0]['obs_state']['_gt_object_pos'])[0]; z0=st[0]['obs_state']['_gt_object_pos'][bowl][2]
    prev=-1; closes=[]
    for s in st:
        a=s.get('action')
        if a is None: continue
        if a[6]>0 and prev<=0:
            e=np.array(s['obs_state']['eef_pos']); b=np.array(s['obs_state']['_gt_object_pos'][bowl]); closes.append((s['t'],e[2]-b[2]))
        prev=a[6]
    lift=next((s['t'] for s in st if s['obs_state']['_gt_object_pos'][bowl][2]-z0>0.03),None)
    ff=len(closes)>1 and (lift is None or lift>closes[1][0])
    p=prov[r['rollout_id']]['provenance']
    rows.append((p['scene_task_id'],p['band'],ff,closes[0][0],closes[0][1],len(closes),lift))
print('failed-first-grasp minted episodes: scene band first_close_t dz_cm n_closes lift_t')
for x in rows:
    if x[2]: print(x[0],x[1],x[3],f"{x[4]*100:.1f}",x[5],x[6])
ok=[x[4] for x in rows if not x[2]]; bad=[x[4] for x in rows if x[2]]
print('median dz at first close: good-first',np.median(ok)*100,'failed-first',np.median(bad)*100)
print('failed-first with first close inside 48:',sum(1 for x in rows if x[2] and x[3]<48),'of',sum(1 for x in rows if x[2]))
