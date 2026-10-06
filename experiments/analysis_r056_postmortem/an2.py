import pickle,collections,json
rows=pickle.load(open('rows.pkl','rb'))
H={k:v for k,v in rows.items() if k[0]=='heldout'}
sc=collections.defaultdict(lambda: collections.Counter())
for k,v in H.items():
    for a in ['base','r16','r4']: sc[k[1]][a]+=v[a]['success']
    sc[k[1]]['n']+=1
# minted episodes per scene
prov=[json.loads(l) for l in open('/home/imerit/Documents/Code/VLA/data/r054_train48/meta/provenance.jsonl')]
print(json.dumps(prov[0])[:800])
pc=collections.Counter(p['provenance']['scene_task_id'] for p in prov)
print('noise seeds',collections.Counter(p['provenance']['noise_seed'] for p in prov),'env seeds',collections.Counter(p['provenance']['env_seed'] for p in prov))
print('bands',collections.Counter(p['provenance']['band'] for p in prov))
import statistics as st
print('full episode steps: median',st.median(p['outcome']['steps'] for p in prov),'min',min(p['outcome']['steps'] for p in prov))
print(f"{'scene':>6} {'n':>3} {'base':>4} {'r16':>4} {'r4':>4}  minted")
for s in sorted(sc):
    c=sc[s]; print(f"{s:>6} {c['n']:>3} {c['base']:>4} {c['r16']:>4} {c['r4']:>4}  {pc.get(s,0)}")
print('minted scenes not in heldout',{s:n for s,n in pc.items() if s not in sc})
