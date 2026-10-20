"""Are LIBERO stable_scanned_objects copies of GSO models? Compare each LIBERO scanned visual mesh with every GSO
model.obj by vertex/face count, sorted bbox extents ratio, and texture hash / downscaled texture similarity."""
import sys, os, glob, json, hashlib
import numpy as np, trimesh
from PIL import Image
L, G, out = sys.argv[1:4]

def mesh_stats(p):
    m = trimesh.load(p, force="mesh", process=False)
    ext = np.sort(m.bounding_box.extents)
    return len(m.vertices), len(m.faces), ext

def tex_sig(p):
    if not os.path.exists(p): return None, None
    b = open(p, "rb").read()
    im = np.asarray(Image.open(p).convert("RGB").resize((32, 32)), dtype=float)
    return hashlib.md5(b).hexdigest(), im

gso = {}
for d in sorted(glob.glob(f"{G}/models/*")):
    try:
        v, f, ext = mesh_stats(f"{d}/model.obj")
    except Exception as e:
        continue
    gso[os.path.basename(d)] = dict(v=v, f=f, ext=ext, tex=f"{d}/texture.png")
res = {}
for d in sorted(glob.glob(f"{L}/*/")):
    name = os.path.basename(d.rstrip("/"))
    objs = [p for p in glob.glob(f"{d}*.obj") + glob.glob(f"{d}visual/*.obj") if "_col" not in p]
    if not objs: continue
    try:
        v, f, ext = mesh_stats(objs[0])
    except Exception as e:
        res[name] = dict(error=str(e)); continue
    texs = glob.glob(f"{d}*.png") + glob.glob(f"{d}visual/*.png")
    th, ti = tex_sig(texs[0]) if texs else (None, None)
    cands = []
    for g, s in gso.items():
        shape = float(np.abs(np.log((ext / ext[-1]) / (s["ext"] / s["ext"][-1]))).max())  # aspect-ratio mismatch
        cands.append((shape, abs(s["v"] - v) / max(v, 1), g, s))
    cands.sort(key=lambda c: (c[1] > 0.02, c[0]))
    top = []
    for shape, dv, g, s in cands[:3]:
        gh, gi = tex_sig(s["tex"])
        top.append(dict(gso=g, verts_lib=v, verts_gso=s["v"], faces_lib=f, faces_gso=s["f"], aspect_mismatch=round(shape, 4),
                        scale_lib_over_gso=round(float(ext[-1] / s["ext"][-1]), 4), tex_md5_equal=(th == gh and th is not None),
                        tex_rmse=None if ti is None or gi is None else round(float(np.sqrt(((ti - gi) ** 2).mean())), 2)))
    res[name] = dict(mesh=objs[0], tex=texs[0] if texs else None, top=top)
json.dump(res, open(out, "w"), indent=1)
for n, r in res.items():
    t = r.get("top", [{}])[0]
    print(f"{n:32s} -> {t.get('gso')}  v {t.get('verts_lib')}/{t.get('verts_gso')}  aspect {t.get('aspect_mismatch')}  "
          f"scale {t.get('scale_lib_over_gso')}  texmd5 {t.get('tex_md5_equal')}  texrmse {t.get('tex_rmse')}")
