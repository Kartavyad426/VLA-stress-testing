"""Render one rollout as a self-contained HTML page.

Purpose: make the mining layer LEGIBLE. Every number the classifier reasons
about -- distance to each object, gripper aperture, grasp attempts, phase
boundaries -- is plotted against the same time axis, beside the diagnosis it
produced. If the diagnosis is wrong, the picture should show why.

    python3 experiments/visualise_episode.py <run_dir> [rollout_id] [out.html]
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vla_harness.schema import rollout_from_dict
from vla_harness.mining.classify import classify
from vla_harness.mining.phases_libero import LiberoPhaseSegmenter
from vla_harness.mining import signals as sig

PHASE_COLOR = {"approach": "--seq-2", "pre_grasp": "--seq-3", "grasp": "--seq-5",
               "transport": "--good", "retry": "--crit", "idle": "--rule"}


def extract(r):
    seg = LiberoPhaseSegmenter()
    segs, info = seg(r)
    d = classify(r, segmenter=seg)
    S = sig.pick(r)
    objs = sorted((r.steps[0].obs_state.get("_gt_eef_to_object") or {}))
    series = {o: [] for o in objs}
    eef, grip, act, closed = [], [], [], []
    for s in r.steps:
        st = s.obs_state
        dd = st.get("_gt_eef_to_object") or {}
        for o in objs:
            series[o].append(dd.get(o))
        eef.append(st.get("eef_pos"))
        q = st.get("gripper_qpos") or [0, 0]
        grip.append(sum(abs(x) for x in q))
        closed.append(sum(abs(x) for x in q) < 0.030)
        act.append(s.action)
    return {
        "id": r.rollout_id, "task": r.task_id, "instruction": r.instruction,
        "success": r.success, "termination": r.termination,
        "perturbation": r.perturbation, "policy": r.policy_id, "env": r.env_id,
        "n": len(r.steps), "objects": objs, "dist": series, "eef": eef,
        "grip": grip, "closed": closed, "act": act,
        "phases": [{"p": s.phase, "a": s.t_start, "b": s.t_end} for s in segs],
        "diag": {k: d.get(k) for k in
                 ("family", "reason", "terminal", "final_error_m",
                  "grasp_attempts", "attempt_spread_m", "failure_cost",
                  "signals", "confident")},
        "scene": r.scene_descriptor,
        "target": objs[0] if objs else None,
    }


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Episode __ID__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{color-scheme:light;--bg:#f4f6f9;--sf:#fff;--sf2:#eef1f6;--ink:#12161d;--ink2:#555f6e;--ink3:#8a93a3;
--rule:#dfe4ec;--rule2:#c3ccda;--acc:#2a78d6;--crit:#e34948;--good:#1baf7a;--warn:#eda100;
--seq-2:#9ec5f4;--seq-3:#5598e7;--seq-5:#1c5cab;
--fd:"IBM Plex Sans Condensed",system-ui,sans-serif;--fb:"IBM Plex Sans",system-ui,sans-serif;--fm:"IBM Plex Mono",monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--bg:#0e1116;--sf:#161b23;--sf2:#1d232c;--ink:#eef1f6;--ink2:#a3adbd;--ink3:#6d7787;--rule:#262d38;--rule2:#39424f;
--acc:#3987e5;--crit:#e66767;--good:#199e70;--warn:#c98500;--seq-2:#184f95;--seq-3:#256abf;--seq-5:#6da7ec}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--fb);font-size:14px;line-height:1.55}
.w{max-width:1120px;margin:0 auto;padding:30px 22px 70px}
h1{font-family:var(--fd);font-size:27px;margin:0;letter-spacing:-.01em}
h2{font-family:var(--fd);font-size:16px;margin:0 0 3px}
.eyebrow{font-family:var(--fm);font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink3)}
.mono{font-family:var(--fm);font-variant-numeric:tabular-nums}
header{border-bottom:1px solid var(--rule);padding-bottom:16px;display:flex;flex-direction:column;gap:7px}
.instr{font-size:16px;color:var(--ink2);font-style:italic}
.meta{display:flex;flex-wrap:wrap;gap:6px 18px;font-family:var(--fm);font-size:11.5px;color:var(--ink3)}
section{margin-top:26px}
.panel{background:var(--sf);border:1px solid var(--rule);padding:16px}
.grid{display:grid;grid-template-columns:1fr 320px;gap:18px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.kv{display:flex;flex-direction:column;gap:7px;font-size:13px}
.kv div{display:flex;justify-content:space-between;gap:10px;border-bottom:1px dotted var(--rule);padding-bottom:6px}
.kv div:last-child{border:0}.kv .v{font-family:var(--fm);text-align:right}
.pill{display:inline-flex;gap:5px;font-family:var(--fm);font-size:10.5px;font-weight:600;letter-spacing:.06em;
text-transform:uppercase;padding:3px 9px;border-radius:2px}
.fail{color:var(--crit);background:color-mix(in srgb,var(--crit) 13%,transparent)}
.ok{color:var(--good);background:color-mix(in srgb,var(--good) 15%,transparent)}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin-bottom:10px;font-size:12.5px;color:var(--ink2)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:13px;height:3px;border-radius:2px;flex:none}
svg{display:block;max-width:100%}.scroll{overflow-x:auto}
figcaption{font-family:var(--fm);font-size:11.5px;color:var(--ink3);margin-top:8px;line-height:1.5}
footer{margin-top:40px;padding-top:16px;border-top:1px solid var(--rule);font-size:12.5px;color:var(--ink3)}
</style></head><body><div class="w">
<header>
  <span class="eyebrow">Episode trace &middot; __TASKID__</span>
  <h1>__FAMILY__</h1>
  <p class="instr">&ldquo;__INSTR__&rdquo;</p>
  <div class="meta"><span>__ID__</span><span>__N__ steps</span><span>__TERM__</span>
  <span>__PERT__</span><span>__POLICY__</span></div>
</header>
<section><div class="grid">
  <div class="panel">
    <h2>What the detectors saw</h2>
    <p class="eyebrow" style="margin-bottom:12px">every number the classifier reasons about, on one time axis</p>
    <div class="legend" id="lg"></div>
    <div class="scroll"><svg id="main" viewBox="0 0 900 430" width="900" height="430"></svg></div>
    <figcaption>Top: phase segmentation. Middle: end-effector distance to each BDDL object &mdash;
    the signal the family rule keys on. Bottom: gripper aperture; shaded spans are closures,
    and a closure that is not followed by the object rising is a grasp on empty air.</figcaption>
  </div>
  <div>
    <div class="panel"><h2>Diagnosis</h2><div class="kv" id="diag" style="margin-top:10px"></div></div>
    <div class="panel" style="margin-top:16px"><h2>Top-down</h2>
      <p class="eyebrow" style="margin-bottom:8px">where it went vs where things are</p>
      <svg id="xy" viewBox="0 0 300 280" width="300" height="280"></svg>
      <figcaption>End-effector path in the world XY plane.</figcaption>
    </div>
  </div>
</div></section>
<section class="panel"><h2>Actions emitted</h2>
  <p class="eyebrow" style="margin-bottom:10px">7-D: 6 end-effector deltas + gripper</p>
  <div class="scroll"><svg id="act" viewBox="0 0 900 200" width="900" height="200"></svg></div>
  <figcaption>The gripper channel is binary in the demonstrations, so intermediate values
  would themselves be out of distribution.</figcaption>
</section>
<footer>Generated from a stored rollout &middot; mining is re-runnable over these traces without a GPU.</footer>
</div>
<script>const D=__DATA__;</script>
<script>
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const NS="http://www.w3.org/2000/svg";
const el=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e};
const tx=(x,y,s,a)=>{const e=el("text",Object.assign({x,y},a));e.textContent=s;return e};
const PC={approach:"--seq-2",pre_grasp:"--seq-3",grasp:"--seq-5",transport:"--good",retry:"--crit",idle:"--rule"};
const SER=["--acc","--warn","--good","--crit"];
function main(){
  const s=document.getElementById("main");s.textContent="";
  const W=900,L=52,R=14,n=D.n,X=i=>L+(W-L-R)*i/Math.max(1,n-1);
  // phase ribbon
  const py=14,ph=22;
  D.phases.forEach(p=>{const x=X(p.a),w=Math.max(1.5,X(p.b)-X(p.a));
    s.appendChild(el("rect",{x,y:py,width:w,height:ph,fill:css(PC[p.p]||"--rule")}));
    if(w>54)s.appendChild(tx(x+w/2,py+15,p.p,{fill:"#fff","font-size":10,"font-family":css("--fm"),"text-anchor":"middle"}));});
  s.appendChild(tx(L-8,py+15,"phase",{fill:css("--ink3"),"font-size":11,"font-family":css("--fm"),"text-anchor":"end"}));
  // distance panel
  const dy=70,dh=200;
  let mx=0;D.objects.forEach(o=>D.dist[o].forEach(v=>{if(v!=null&&v>mx)mx=v}));
  mx=Math.ceil(mx*10)/10||1;const Y=v=>dy+dh-dh*v/mx;
  [0,mx/2,mx].forEach(g=>{s.appendChild(el("line",{x1:L,x2:W-R,y1:Y(g),y2:Y(g),stroke:css("--rule")}));
    s.appendChild(tx(L-8,Y(g)+4,g.toFixed(2),{fill:css("--ink3"),"font-size":10.5,"font-family":css("--fm"),"text-anchor":"end"}))});
  s.appendChild(tx(L-8,dy-6,"dist (m)",{fill:css("--ink3"),"font-size":11,"font-family":css("--fm"),"text-anchor":"end"}));
  D.objects.forEach((o,k)=>{let d="",started=false;
    D.dist[o].forEach((v,i)=>{if(v==null)return;d+=(started?" L":"M")+X(i)+" "+Y(v);started=true});
    s.appendChild(el("path",{d,fill:"none",stroke:css(SER[k%SER.length]),"stroke-width":k===0?2.2:1.5,
      "stroke-dasharray":k===0?"":"4 3","stroke-linejoin":"round"}))});
  // closure shading
  let run=null;D.closed.forEach((c,i)=>{if(c&&run===null)run=i;
    if((!c||i===D.n-1)&&run!==null){s.appendChild(el("rect",{x:X(run),y:dy,width:Math.max(1,X(i)-X(run)),
      height:dh,fill:css("--crit"),"fill-opacity":.09}));run=null}});
  // gripper panel
  const gy=300,gh=86;let gmx=Math.max(...D.grip,0.001);
  s.appendChild(tx(L-8,gy-4,"gripper",{fill:css("--ink3"),"font-size":11,"font-family":css("--fm"),"text-anchor":"end"}));
  let gd="";D.grip.forEach((v,i)=>{gd+=(i?" L":"M")+X(i)+" "+(gy+gh-gh*v/gmx)});
  s.appendChild(el("path",{d:gd,fill:"none",stroke:css("--ink2"),"stroke-width":1.6}));
  [0,n-1].forEach(i=>s.appendChild(tx(X(i),420,"t="+i,{fill:css("--ink3"),"font-size":10.5,"font-family":css("--fm"),"text-anchor":i?"end":"start"})));
  const lg=document.getElementById("lg");
  D.objects.forEach((o,k)=>{const sp=document.createElement("span");
    sp.innerHTML='<i class="sw" style="background:'+css(SER[k%SER.length])+'"></i>'+o.replace(/_main$/,"")+(k===0?" (target)":"");
    lg.appendChild(sp)});
  const sp=document.createElement("span");
  sp.innerHTML='<i class="sw" style="height:10px;background:color-mix(in srgb,'+css("--crit")+' 20%,transparent)"></i>gripper closed';
  lg.appendChild(sp);
}
function xy(){
  const s=document.getElementById("xy");s.textContent="";
  const pts=D.eef.filter(Boolean);if(!pts.length)return;
  const objs=(D.scene&&D.scene.objects)||[];
  const xs=pts.map(p=>p[0]).concat(objs.map(o=>o.pos_m[0]));
  const ys=pts.map(p=>p[1]).concat(objs.map(o=>o.pos_m[1]));
  const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  const pad=0.05,sx=v=>30+240*(v-x0+pad)/((x1-x0)+2*pad),sy=v=>250-220*(v-y0+pad)/((y1-y0)+2*pad);
  let d="";pts.forEach((p,i)=>{d+=(i?" L":"M")+sx(p[0])+" "+sy(p[1])});
  s.appendChild(el("path",{d,fill:"none",stroke:css("--ink2"),"stroke-width":1.4,"stroke-opacity":.85}));
  s.appendChild(el("circle",{cx:sx(pts[0][0]),cy:sy(pts[0][1]),r:4,fill:css("--ink3")}));
  s.appendChild(el("circle",{cx:sx(pts[pts.length-1][0]),cy:sy(pts[pts.length-1][1]),r:5,fill:css("--crit")}));
  objs.forEach((o,k)=>{s.appendChild(el("circle",{cx:sx(o.pos_m[0]),cy:sy(o.pos_m[1]),r:6,
      fill:"none",stroke:css(SER[k%SER.length]),"stroke-width":2}));
    s.appendChild(tx(sx(o.pos_m[0])+10,sy(o.pos_m[1])+4,o.name.replace(/_main$/,""),
      {fill:css("--ink2"),"font-size":9.5,"font-family":css("--fm")}))});
  s.appendChild(tx(30,270,"start",{fill:css("--ink3"),"font-size":10,"font-family":css("--fm")}));
  s.appendChild(tx(90,270,"ended here",{fill:css("--crit"),"font-size":10,"font-family":css("--fm")}));
}
function acts(){
  const s=document.getElementById("act");s.textContent="";
  const names=["dx","dy","dz","droll","dpitch","dyaw","grip"],W=900,L=52,R=14,n=D.n;
  const X=i=>L+(W-L-R)*i/Math.max(1,n-1);
  const rows=names.length,h=22;
  names.forEach((nm,k)=>{const y0=10+k*h+h/2;
    s.appendChild(tx(L-8,y0+4,nm,{fill:css("--ink3"),"font-size":10.5,"font-family":css("--fm"),"text-anchor":"end"}));
    s.appendChild(el("line",{x1:L,x2:W-R,y1:y0,y2:y0,stroke:css("--rule")}));
    let d="";D.act.forEach((a,i)=>{if(!a)return;const v=Math.max(-1,Math.min(1,a[k]||0));
      d+=(d?" L":"M")+X(i)+" "+(y0-v*(h/2-2))});
    s.appendChild(el("path",{d,fill:"none",stroke:css(k===6?"--crit":"--acc"),"stroke-width":1.2}))});
}
function diag(){
  const d=D.diag,c=document.getElementById("diag");
  const cost=d.failure_cost&&d.failure_cost.cost;
  const rows=[["family",d.family||"succeeded"],["confident",String(d.confident)],
    ["terminal",d.terminal],["final error","("+(d.final_error_m==null?"n/a":(d.final_error_m*100).toFixed(1)+" cm")+")"],
    ["grasp attempts",d.grasp_attempts],["attempt spread",d.attempt_spread_m==null?"n/a":(d.attempt_spread_m*100).toFixed(1)+" cm"],
    ["failure cost",cost||"n/a"],["signal set",d.signals]];
  rows.forEach(([k,v])=>{const r=document.createElement("div");
    r.innerHTML='<span>'+k+'</span><span class="v">'+v+'</span>';c.appendChild(r)});
  const r=document.createElement("div");r.style.borderBottom="0";
  r.innerHTML='<span style="color:var(--ink2);font-style:italic">'+(d.reason||"")+'</span>';
  c.appendChild(r);
}
main();xy();acts();diag();
</script></body></html>"""


if __name__ == "__main__":
    run_dir = sys.argv[1]
    want = sys.argv[2] if len(sys.argv) > 2 else None
    out = sys.argv[3] if len(sys.argv) > 3 else "episode.html"
    chosen = None
    for line in open(os.path.join(run_dir, "rollouts.jsonl")):
        r = rollout_from_dict(json.loads(line))
        if want is None or r.rollout_id == want:
            chosen = r
            break
    if chosen is None:
        sys.exit(f"rollout {want} not found in {run_dir}")
    D = extract(chosen)
    html = (HTML.replace("__DATA__", json.dumps(D))
                .replace("__ID__", D["id"]).replace("__TASKID__", D["task"])
                .replace("__INSTR__", D["instruction"])
                .replace("__FAMILY__", (D["diag"]["family"] or "success").replace("_", " "))
                .replace("__N__", str(D["n"]))
                .replace("__TERM__", D["termination"])
                .replace("__PERT__", json.dumps(D["perturbation"]) if D["perturbation"] else "nominal")
                .replace("__POLICY__", D["policy"].split("@")[0]))
    open(out, "w").write(html)
    print(f"wrote {out}  ({D['n']} steps, family={D['diag']['family']})")
