"""prim's one-off probe (UNVERIFIED, scratchpad): do v1 event rules give a unique failure mode per episode,
and does the root depend on rule order? CPU, runs/r047/rollouts.jsonl only."""
import json, collections, sys
import numpy as np
LIFT_MIN, LIFT_OK, WIN, NT_LIFT, GOAL_XY = 0.015, 0.05, 40, 0.015, 0.06
PREC = ["stall","wrong_object","miss","slip","drop","wrong_location"]   # table order

def events(r):
    S=r['steps']; e2o=S[0]['obs_state']['_gt_eef_to_object']; names=list(e2o)
    tgt=names[0]; goal=names[1] if len(names)>1 else None
    P=[s['obs_state']['_gt_scene_object_pos'] for s in S]
    z0={k:v[2] for k,v in P[0].items()}
    cmd=[bool(s.get('action')) and s['action'][6]>0.5 for s in S]
    tz=np.array([p[tgt][2]-z0[tgt] for p in P])
    ev=[]
    onsets=[t for t in range(1,len(S)) if cmd[t] and not cmd[t-1]]
    if cmd[0]: onsets=[0]+onsets
    if not onsets: ev.append((len(S)-1,'stall'))
    for t in onsets:
        w=slice(t,min(t+WIN+1,len(S)))
        others=[k for k in P[0] if k not in (tgt,goal)]
        nt=max((max(p[k][2]-z0[k] for p in P[w]) for k in others),default=0)
        lift=float(tz[w].max())
        if nt>NT_LIFT and lift<LIFT_MIN: ev.append((t,'wrong_object')); continue
        if lift<LIFT_MIN: ev.append((t,'miss')); continue
        if lift<LIFT_OK: ev.append((t,'slip')); continue
        # held: find release (cmd closed->open) or drop (target back near rest while cmd closed)
        t_held=t+int(np.argmax(tz[w]>=LIFT_OK))
        rel=None
        for u in range(t_held,len(S)):
            if not cmd[u]: rel=u; break
            if tz[u]<LIFT_MIN and u>t_held:
                g=P[u]; near=goal and np.hypot(g[tgt][0]-g[goal][0],g[tgt][1]-g[goal][1])<GOAL_XY
                if not near: ev.append((u,'drop'))
                rel=-1; break
        if rel is None or rel==-1: continue
        p=P[min(rel+10,len(S)-1)]
        if goal:
            dxy=np.hypot(p[tgt][0]-p[goal][0],p[tgt][1]-p[goal][1])
            if not (dxy<GOAL_XY and p[tgt][2]>p[goal][2]): ev.append((rel,'wrong_location'))
    return sorted(ev)

rows=[json.loads(l) for l in open('runs/r047/rollouts.jsonl')]
stats=collections.Counter(); multi=collections.Counter(); diff=0; nf=0; first_c=collections.Counter(); prec_c=collections.Counter()
pairs=collections.Counter(); succ_fp=collections.Counter(); unexpl=0; examples=[]
for i,r in enumerate(rows):
    ev=events(r); modes=[m for _,m in ev]
    if r['success']:
        if modes: succ_fp[tuple(sorted(set(modes)))]+=1
        continue
    nf+=1
    if not ev: unexpl+=1; first_c['unexplained']+=1; prec_c['unexplained']+=1; continue
    ds=sorted(set(modes)); multi[len(ds)]+=1
    if len(ds)>1: pairs[tuple(ds)]+=1
    f=ev[0][1]; p=min(ds,key=PREC.index); first_c[f]+=1; prec_c[p]+=1
    if f!=p: diff+=1; len(examples)<5 and examples.append((i,ev[:6]))
print(f"failed episodes: {nf}")
print("distinct failure modes per failed episode:", dict(sorted(multi.items())), " unexplained:", unexpl)
print("root = first in time     :", dict(first_c.most_common()))
print("root = fixed precedence  :", dict(prec_c.most_common()))
print(f"episodes where the two disagree: {diff}/{nf-unexpl}")
print("most common multi-mode combos:", pairs.most_common(6))
print("successes firing failure events:", sum(succ_fp.values()), "of", len(rows)-nf, dict(succ_fp.most_common(4)))
print("examples (line, events):"); [print(" ",e) for e in examples]
