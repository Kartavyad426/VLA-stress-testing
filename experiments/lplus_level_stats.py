"""What does each LIBERO-Plus difficulty level contain, per perturbation type?

Difficulty = how many of 4 reference models solved the variant (paper C.3), NOT
perturbation magnitude. This decodes each variant's parameters from its name
(parsing per libero/libero/envs/env_wrapper.py) so the doc can show it.
"""
import json, re, sys, statistics as st
from collections import defaultdict, Counter
suites = sys.argv[1:] or ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
cls = json.load(open("third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json"))
NOISE = [(10, "motion blur"), (20, "gaussian blur"), (30, "zoom blur"), (40, "fog"), (50, "glass blur")]

def decode(name):
    d = {}
    m = re.search(r"_view_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_initstate_(\d+)(?:_noise_(\d+))?", name)
    if m:
        h, v, s, r, e, init, noise = m.groups()
        d.update(azimuth=int(h), elevation=int(v), distance_pct=int(s), cam_rot=int(r),
                 cam_vert=int(e), initstate=int(init))
        if noise:
            n = int(noise); lo = 0
            for hi, kind in NOISE:
                if n <= hi:
                    d.update(noise_type=kind, noise_severity=n - lo); break
                lo = hi
    for key, pat in (("light", r"_light_(\d+)"), ("table", r"_(?:table|tb)_(\d+)"),
                     ("language", r"_language_(\d+)"), ("added_objects", r"_add_(\d+)"),
                     ("layout_level", r"_level(\d+)_sample")):
        m = re.search(pat, name)
        if m: d[key] = int(m.group(1))
    return d

out = {}
for suite in suites:
    rows = cls[suite]
    counts = defaultdict(Counter)
    params = defaultdict(lambda: defaultdict(list))
    for r in rows:
        c, lv = r["category"], r["difficulty_level"]
        counts[c][lv] += 1
        d = decode(r["name"])
        if c == "Camera Viewpoints":
            ang = max(min(d.get("azimuth", 0), 360 - d.get("azimuth", 0)), d.get("elevation", 0))
            params[c][lv].append(("view_off_axis_deg", ang)); params[c][lv].append(("distance_pct", d.get("distance_pct")))
        elif c == "Sensor Noise":
            params[c][lv].append(("noise", f"{d.get('noise_type')} s{d.get('noise_severity')}"))
        elif c == "Objects Layout":
            if "added_objects" in d: params[c][lv].append(("added_objects", d["added_objects"]))
            if "layout_level" in d: params[c][lv].append(("target_pose_shift_level", d["layout_level"]))
        elif c == "Robot Initial States":
            params[c][lv].append(("initstate", d.get("initstate")))
    out[suite] = {"counts": {c: dict(sorted(v.items())) for c, v in counts.items()},
                  "params": {c: {lv: vals for lv, vals in sorted(ls.items())} for c, ls in params.items()}}
    print(f"\n##### {suite}: {len(rows)} variants")
    for c in sorted(counts):
        print(f"  {c:22s} " + "  ".join(f"L{lv}:{counts[c][lv]:3d}" for lv in range(1, 6)))
    for c in ("Camera Viewpoints", "Sensor Noise", "Objects Layout"):
        print(f"  -- {c}")
        for lv in range(1, 6):
            vals = params[c].get(lv, [])
            if not vals: continue
            if c == "Camera Viewpoints":
                off = [v for k, v in vals if k == "view_off_axis_deg"]; dist = [v for k, v in vals if k == "distance_pct"]
                print(f"     L{lv}: off-axis angle median {st.median(off):.0f}° (range {min(off)}–{max(off)}), distance median {st.median(dist):.0f}% (range {min(dist)}–{max(dist)})")
            elif c == "Sensor Noise":
                kinds = Counter(v.split(' s')[0] for k, v in vals); sev = [int(v.split(' s')[1]) for k, v in vals]
                print(f"     L{lv}: types {dict(kinds.most_common())} severity median {st.median(sev):.0f} (range {min(sev)}–{max(sev)})")
            else:
                add = [v for k, v in vals if k == "added_objects"]; shift = [v for k, v in vals if k == "target_pose_shift_level"]
                print(f"     L{lv}: added-object variants {len(add)} (n added median {st.median(add) if add else '-'}), target-pose-shift variants {len(shift)} (shift levels {sorted(set(shift))})")
json.dump(out, open("experiments/repro/lplus_level_stats.json", "w"))
