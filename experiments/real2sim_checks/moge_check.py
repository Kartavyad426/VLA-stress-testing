"""D6: image vs physics in relative form, with MoGe-2 (REAL2SIM_CHECKLIST_V1 D6; IMAGE_PHYSICS_CHECKS C5; implres31, 10-27).

MoGe's absolute error (a few % of distance, i.e. a few cm at 0.7-1 m) is the size of the errors we want to catch, so
nothing here is compared in metres directly (IMAGE_PHYSICS §10):
  (b) ours vs physics   MoGe depth of our composite vs the sim's own depth buffer of the same state, after one
                        robust affine fit per image (z_sim ~ a z_moge + b, on the table pixels). Residuals per object
                        (median over its sim mask) and per table patch (4 x 4 grid over the visible table), in cm.
                        Works on layout variants too (--variants), where no real image exists.
  (a) real vs ours      at the real layout only: MoGe depth of the real frame vs MoGe depth of our composite, same fit;
                        MoGe's own biases mostly cancel, differences point at seams (texture at the wrong height, an
                        object pasted at the wrong depth).
A residual that stands out from the table patches is the signal, not its absolute size (no bar yet, Q1).
--inject body:dz_m renders the sim depth with that body shifted up by dz (the composite keeps it where it is): the known
-bad case, the object's residual must move by about dz while the table patches stay.
Frames: frame 0 for both cameras, plus --wrist_frames more pre-touch wrist frames. MoGe-2 ViT-L on CPU (fp32).
Writes <out>/moge_check.json and <out>/moge_check/*.jpg. Exploratory, UNVERIFIED.

    MUJOCO_GL=osmesa /home/imerit/Documents/Code/VLA/.venvs/real2sim/bin/python experiments/real2sim_checks/moge_check.py \
        --scene /home/imerit/Documents/Code/VLA/runs/real2sim/ep_orange_block_stereo \
        --episode /home/imerit/Documents/Code/VLA/data/real2sim/droid/ep_orange_block
"""
import argparse, json, os, time
import common  # noqa: F401  (triton before mujoco; res31's path)
from common import SceneFrames, composites, to_uint8, MAIN
import numpy as np
import cv2
import mujoco

WORK = (360, 640)


class MoGe:
    def __init__(self):
        import torch
        from moge.model.v2 import MoGeModel
        self.torch = torch
        self.model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").eval()  # perceive.geometry's model

    def depth(self, rgb, fov_x_deg):
        x = self.torch.tensor(rgb / 255.0, dtype=self.torch.float32).permute(2, 0, 1)
        with self.torch.no_grad():
            o = self.model.infer(x, fov_x=fov_x_deg, use_fp16=False)
        z = o["depth"].numpy()
        ok = o["mask"].numpy().astype(bool) & np.isfinite(z) if "mask" in o else np.isfinite(z)
        return np.where(ok, z, np.nan)


def robust_affine(x, y, iters=5):
    """y ~ a x + b on finite pairs, iteratively dropping points beyond 2.5 MAD."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    keep = np.ones(len(x), bool)
    a, b = 1.0, 0.0
    for _ in range(iters):
        if keep.sum() < 50:
            break
        A = np.c_[x[keep], np.ones(keep.sum())]
        a, b = np.linalg.lstsq(A, y[keep], rcond=None)[0]
        r = y - (a * x + b)
        mad = np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-6
        keep = np.abs(r - np.median(r[keep])) < 2.5 * 1.4826 * mad
    return float(a), float(b), int(keep.sum())


def regions(sf, geom):
    m = sf.m
    tab = geom == m.geom("table").id
    out = {m.body(b).name: np.isin(geom, sf.obj_geoms[b]) for b in sf.free}
    ys, xs = np.nonzero(tab)
    patches = {}
    if len(ys):
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        for i in range(4):
            for j in range(4):
                p = np.zeros_like(tab)
                p[y0 + (y1 - y0) * i // 4:y0 + (y1 - y0) * (i + 1) // 4, x0 + (x1 - x0) * j // 4:x0 + (x1 - x0) * (j + 1) // 4] = True
                p &= tab
                if p.sum() >= 200:
                    patches[f"table_{i}{j}"] = p
    return tab, out, patches


def compare(zr, zt, tab, objs, patches):
    """Fit zt ~ a zr + b on the table, then median residual (cm) per object and per table patch."""
    a, b, n = robust_affine(zr[tab], zt[tab])
    res = (a * zr + b - zt) * 100.0
    obj = {k: (round(float(np.nanmedian(res[mk])), 2) if np.isfinite(res[mk]).sum() >= 20 else None) for k, mk in objs.items()}
    pat = {k: round(float(np.nanmedian(res[mk])), 2) for k, mk in patches.items() if np.isfinite(res[mk]).sum() >= 50}
    pv = np.array(list(pat.values())) if pat else np.array([np.nan])
    spread = float(np.nanpercentile(np.abs(pv - np.nanmedian(pv)), 90)) if pat else None
    stand = {k: (None if v is None or not pat else round(abs(v - float(np.nanmedian(pv))) / max(spread, 0.1), 2)) for k, v in obj.items()}
    return dict(fit_a=round(a, 4), fit_b_m=round(b, 4), fit_px=n, object_residual_cm=obj, table_patch_residual_cm=pat,
                table_patch_spread_cm_p90=None if spread is None else round(spread, 2),
                object_vs_table_spread=stand), res


def heat(res_cm, lim=5.0):
    r = np.nan_to_num(np.clip(res_cm / lim, -1, 1))
    img = np.zeros(res_cm.shape + (3,), np.uint8)
    img[..., 0] = (255 * np.clip(r, 0, 1)).astype(np.uint8)
    img[..., 2] = (255 * np.clip(-r, 0, 1)).astype(np.uint8)
    img[..., 1] = (255 * (1 - np.abs(r)) * 0.6).astype(np.uint8)
    img[~np.isfinite(res_cm)] = 0
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", default=None, help="default: <main checkout>/runs/real2sim_checks/<scene name>")
    ap.add_argument("--wrist_frames", type=int, default=2, help="pre-touch wrist frames besides frame 0")
    ap.add_argument("--variants", type=int, default=0, help="also check this many jittered layouts (make_tasks.randomise), (b) only")
    ap.add_argument("--inject", default=None, help="body:dz_m, e.g. orange_block:0.02 (known-bad test, sim depth only)")
    ap.add_argument("--tag", default="moge_check")
    ap.add_argument("--real2sim", default=None, help="res31's experiments/real2sim, imported read-only (read by common.py)")
    a = ap.parse_args()
    out = a.out or os.path.join(MAIN, "runs/real2sim_checks", os.path.basename(os.path.normpath(a.scene)))
    os.makedirs(os.path.join(out, a.tag), exist_ok=True)
    t0 = time.time()
    sf = SceneFrames(a.scene, a.episode)
    ec, wc, note = composites(sf, WORK)
    if ec is None:
        json.dump(dict(scene=a.scene, label="not run", reason=note), open(os.path.join(out, f"{a.tag}.json"), "w"), indent=1)
        print("not run:", note)
        return
    mg = MoGe()
    m = sf.m
    inj = None
    if a.inject:
        bn, dz = a.inject.split(":")
        inj = (m.body(bn).id, float(dz))
    pre = sorted(sf.info["frames"]["before_touch"])
    wrist_extra = [pre[int(round(i))] for i in np.linspace(len(pre) // 2, len(pre) - 1, a.wrist_frames)] if a.wrist_frames else []
    jobs = [("ext", 0, None), ("wrist", 0, None)] + [("wrist", f, None) for f in wrist_extra if f != 0]
    if a.variants:
        from make_tasks import randomise
        from scene_roles import body_kinds
        free_names = [m.body(b).name for b in sf.free]
        for v in range(1, a.variants + 1):
            jobs.append(("ext", 0, v))
    res = dict(scene=a.scene, episode=a.episode, composite=note, inject=a.inject, status="exploratory, UNVERIFIED", views=[])
    for cam, f, var in jobs:
        rf = sf.pose(f)
        if var is not None:
            rng = np.random.default_rng(1000 + var)
            if not randomise(m, sf.d, rng, free_names, kinds=body_kinds(sf.fit)):
                continue
            mujoco.mj_forward(m, sf.d)
        ours = to_uint8((ec if cam == "ext" else wc)(sf.d))
        if inj is not None:  # sim depth with the body moved; the composite above keeps it in place
            jadr = m.jnt_qposadr[m.body_jntadr[inj[0]]]
            sf.d.qpos[jadr + 2] += inj[1]
            mujoco.mj_forward(m, sf.d)
        geom, zsim = sf.render(cam, WORK)
        if inj is not None:
            sf.d.qpos[jadr + 2] -= inj[1]
            mujoco.mj_forward(m, sf.d)
        zsim = np.where(geom >= 0, zsim, np.nan)
        K = sf.K(cam, WORK)
        fov = float(np.degrees(2 * np.arctan(WORK[1] / (2 * K[0, 0]))))
        t1 = time.time()
        zo = mg.depth(ours, fov)
        tab, objs, patches = regions(sf, geom)
        row = dict(cam=cam, frame=f, real_frame=rf, variant=var, fov_x_deg=round(fov, 2))
        row["ours_vs_physics"], r_b = compare(zo, zsim, tab, objs, patches)
        tiles = [ours, heat(r_b)]
        if var is None:
            real = sf.real(cam, rf, WORK)
            zr = mg.depth(real, fov)
            row["real_vs_ours"], r_a = compare(zo, zr, tab, objs, patches)
            tiles += [real, heat(r_a)]
        row["moge_seconds"] = round(time.time() - t1, 1)
        res["views"].append(row)
        fn = f"{cam}_f{f}" + (f"_v{var}" if var else "") + ".jpg"
        cv2.imwrite(os.path.join(out, a.tag, fn), cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(cam, f, var, json.dumps(row["ours_vs_physics"]["object_residual_cm"]),
              "table spread", row["ours_vs_physics"]["table_patch_spread_cm_p90"], f"({row['moge_seconds']} s)", flush=True)
        json.dump(res, open(os.path.join(out, f"{a.tag}.json"), "w"), indent=1, default=float)
    res["seconds"] = round(time.time() - t0, 1)
    res["note"] = ("residual = a*z_moge + b - z_reference (cm), fit on table pixels; object_vs_table_spread = |object residual - "
                   "median table-patch residual| / p90 table-patch spread (>1: stands out from the table). Heat images: "
                   "composite | ours-vs-physics residual (red: MoGe farther, blue: nearer, +-5 cm) | real | real-vs-ours residual.")
    json.dump(res, open(os.path.join(out, f"{a.tag}.json"), "w"), indent=1, default=float)
    print(f"written {os.path.join(out, a.tag + '.json')} ({res['seconds']} s)")


if __name__ == "__main__":
    main()
