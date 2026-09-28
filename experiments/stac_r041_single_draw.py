"""CPU pass: single-draw chunk-overlap consistency (Sentinel's STAC, 1 sample) on stored R-041 chunks.
Chunk f covers env steps [16f, 16f+40); chunk f+1 covers [16(f+1), ...). Overlap = f[16:40] vs f+1[0:24].
Gripper (dim 6) dropped, as Sentinel does. Matched forward index to avoid the length confound."""
import json, glob, os, numpy as np
from scipy.stats import mannwhitneyu, spearmanr
K, H, D = 16, 40, 6
EARLY = 3   # overlaps f=0,1,2 -> needs >=4 forwards
def ov(a):  # a: (F,40,7) -> per-overlap distance (F-1,)
    a = a[..., :D].astype(np.float32)
    return np.linalg.norm((a[:-1, K:] - a[1:, :H-K]).reshape(len(a)-1, -1), axis=1)
def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    return mannwhitneyu(pos, neg).statistic / (len(pos)*len(neg))
rows = []
for ax in sorted(os.listdir('runs/r041')):
    man = f'runs/r041/{ax}/manifest.jsonl'
    if not os.path.exists(man): continue
    for l in open(man):
        r = json.loads(l); m = r['magnitude']
        tag = f"{r['task_id']}_{m:.6f}".replace('.', 'p')
        z = np.load(f'runs/r041/{ax}/{tag}.npz')
        sP, sN = ov(z['act_P']), ov(z['act_N'])
        if len(sP) < EARLY: continue
        rows.append(dict(ax=ax, task=r['task_id'], m=m, succ=r['success'], F=len(z['act_P']),
                         dpn0=float(z['d_pn'][0]), stacP=sP[:EARLY], stacN=sN[:EARLY]))
print(f'{len(rows)} rollouts with >= {EARLY+1} forwards; failures: {sum(not r["succ"] for r in rows)}')
# reference: magnitude-0 rollouts, per overlap index p95 of STAC_P (success-only calibration, Sentinel-style)
ref = np.array([r['stacP'] for r in rows if r['m'] == 0 and r['succ']])
thr = np.percentile(ref, 95, axis=0)
print('m=0 STAC_P per-overlap p95 (f0,f1,f2):', np.round(thr, 3), ' n_ref =', len(ref))
def feats(r):
    return {'magnitude': r['m'], '|P-N| f0': r['dpn0'],
            'STAC_P f0': r['stacP'][0], 'STAC_P mean f0-2': r['stacP'].mean(),
            'dSTAC f0': r['stacP'][0]-r['stacN'][0], 'dSTAC mean f0-2': (r['stacP']-r['stacN']).mean(),
            'STAC_P max/thr f0-2': (r['stacP']/thr).max()}
print('\nAUROC for failure (1 = higher score -> failure), perturbed rollouts only (m>0)')
names = list(feats(rows[0]))
axes = sorted({r['ax'] for r in rows})
print(f"{'signal':22s}" + ''.join(f'{a[:14]:>16s}' for a in axes) + f"{'pooled*':>10s}")
for n in names:
    line = f'{n:22s}'
    for a in axes:
        sub = [r for r in rows if r['ax'] == a and r['m'] > 0]
        nf = sum(not r['succ'] for r in sub)
        line += f"{auroc([feats(r)[n] for r in sub if not r['succ']], [feats(r)[n] for r in sub if r['succ']]):>10.2f} ({nf:2d}f)"
    sub = [r for r in rows if r['m'] > 0 and r['ax'] != 'light_intensity']
    # pooled excludes magnitude (different units per axis)
    v = auroc([feats(r)[n] for r in sub if not r['succ']], [feats(r)[n] for r in sub if r['succ']]) if n != 'magnitude' else float('nan')
    print(line + f'{v:>10.2f}')
print('\nSpearman with magnitude (m>0), per axis:')
for a in axes:
    sub = [r for r in rows if r['ax'] == a and r['m'] > 0]
    print(f'  {a:18s}', {n: round(spearmanr([r["m"] for r in sub], [feats(r)[n] for r in sub]).correlation, 2) for n in names[1:]})
print('\n2x2 (m>0): shifted = |P-N| f0 > tau 0.093; inconsistent = any STAC_P f0-2 above m=0 p95')
cells = {}
for r in rows:
    if r['m'] == 0: continue
    k = ('shifted' if r['dpn0'] > 0.093 else 'absorbed', 'inconsistent' if (r['stacP'] > thr).any() else 'consistent')
    c = cells.setdefault(k, [0, 0]); c[0 if r['succ'] else 1] += 1
for k, (s, f) in sorted(cells.items()): print(f'  {k[0]:9s} x {k[1]:12s}: {s:3d} succ  {f:2d} fail')
print('\nFailures in detail:')
for r in rows:
    if not r['succ']:
        print(f"  {r['ax']:17s} task {r['task']} m={r['m']:<7g} F={r['F']:2d} |P-N|f0={r['dpn0']:.2f} STAC_P={np.round(r['stacP'],2)} STAC_N={np.round(r['stacN'],2)}")
