"""Second pass: rank GSO models by texture similarity (32x32 RMSE) to each LIBERO scanned object, then report mesh
aspect/scale for the best ones."""
import sys, os, glob
import numpy as np, trimesh
from PIL import Image
L, G = sys.argv[1:3]
sig = lambda p: np.asarray(Image.open(p).convert("RGB").resize((32, 32)), dtype=float)
gso = {os.path.basename(os.path.dirname(p)): sig(p) for p in sorted(glob.glob(f"{G}/models/*/texture.png"))}
for name in sys.argv[3:]:
    d = f"{L}/{name}/"
    texs = glob.glob(f"{d}*.png") + glob.glob(f"{d}visual/*.png")
    objs = [p for p in glob.glob(f"{d}*.obj") + glob.glob(f"{d}visual/*.obj") if "_col" not in p]
    t = sig(texs[0])
    best = sorted(gso, key=lambda g: ((gso[g] - t) ** 2).mean())[:2]
    e1 = np.sort(trimesh.load(objs[0], force="mesh", process=False).bounding_box.extents)
    for g in best:
        e2 = np.sort(trimesh.load(f"{G}/models/{g}/model.obj", force="mesh", process=False).bounding_box.extents)
        print(f"{name:30s} -> {g:55s} texrmse {np.sqrt(((gso[g]-t)**2).mean()):6.2f}  extents lib {np.round(e1,4)} gso {np.round(e2,4)}")
