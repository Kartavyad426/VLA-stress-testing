"""Per-episode table for ab_wrist_photo.sh (prim's condition c: rc, peak GPU, peak RAM, wall, stages; d: counts, no
verdict). Run with the real2sim venv from the main checkout. Exploratory, UNVERIFIED."""
import glob, json, os, re, sys
import numpy as np
import imageio.v2 as imageio

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/imerit/Documents/Code/VLA-exp-taskgen/runs/taskgen/ab_wrist_photo"
OPEN_MAX = 0.02  # c11 gripper library open_max (photo weight target: clip((open_max + 0.05 - opening) / 0.05, 0, 1))
rows = []
for d in sorted(glob.glob(os.path.join(OUT, "photo_*_ep*")), key=lambda p: (p.split("_ep")[1], p)):
    if not os.path.isdir(d):
        continue
    r = dict(run=os.path.basename(d))
    log = open(os.path.join(d, "run.log")).read() if os.path.exists(os.path.join(d, "run.log")) else ""
    m = re.search(r"end rc (\d+); peak GPU (\d*) MiB; min MemAvailable (\d*) MiB", log)
    t = re.findall(r"(\d\d:\d\d:\d\d) (?:start|end)", log)
    r.update(rc=int(m.group(1)) if m else None, peak_gpu_mib=int(m.group(2)) if m and m.group(2) else None,
             min_ram_mib=int(m.group(3)) if m and m.group(3) else None)
    if len(t) >= 2:
        h0, h1 = [sum(int(x) * k for x, k in zip(s.split(":"), (3600, 60, 1))) for s in t[:2]]
        r["wall_s"] = h1 - h0
    rj = os.path.join(d, "rollout.json")
    if os.path.exists(rj):
        j = json.load(open(rj))
        st = j["stages"]
        r.update(photo=j.get("wrist_gripper_photo"), success=j["success"], reached=st["reached"], closed=st["closed_at_target"],
                 lifted=st["lifted"], placed=st["placed"], first=st["first_step"], min_dist_cm=round(100 * st["min_dist_to_target_m"], 1))
        z = np.load(os.path.join(d, "rollout.npz"))
        g, cmd = z["state"][:, 9], z["action"][:, 9] > 0.5
        r["grip_toggles"] = int((np.diff(cmd.astype(int)) != 0).sum())
        w = np.clip((OPEN_MAX + 0.05 - g) / 0.05, 0, 1) if j.get("wrist_gripper_photo") else np.zeros_like(g)
        r["look_swaps"] = int((np.abs(np.diff(w)) > 0.3).sum())
        fr = np.array([f[:, 320:] for f in imageio.get_reader(os.path.join(d, "rollout.mp4"))]).astype(np.float32)
        dw = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2, 3))
        r["wrist_diff_p95"] = round(float(np.percentile(dw, 95)), 1)
    rows.append(r)
for r in rows:
    print(json.dumps(r))
for c in ("on", "off"):
    rs = [r for r in rows if r["run"].startswith(f"photo_{c}") and "success" in r]
    if rs:
        print(f"photo {c}: {len(rs)} episodes | success {sum(r['success'] for r in rs)} | reached {sum(r['reached'] for r in rs)} | closed {sum(r['closed'] for r in rs)}"
              f" | lifted {sum(r['lifted'] for r in rs)} | placed {sum(r['placed'] for r in rs)} | toggles {[r['grip_toggles'] for r in rs]} | swaps {[r['look_swaps'] for r in rs]}")
json.dump(rows, open(os.path.join(OUT, "ab_report.json"), "w"), indent=1)
