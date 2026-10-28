"""Image-physics and appearance checks of a built scene (REAL2SIM_CHECKLIST_V1 A2, C1-C5, D1, D3, D5;
implres31, 10-27). D6 (MoGe) is moge_check.py.

All on the frames before anything is touched (frame 0 + episode_info 'before_touch'; checklist rule 3), the sim posed at
the replayed recorded joints. Wrist view = the independent witness (pose from the joints); exterior view = the camera
the scene was built from, so its geometry numbers compare the build with its own input (rule 4: reported, marked).

  D3 nothing physical that's invisible  per free body and view: visible pixels in the sim render; in view but 0 px = error
  D1 objects drawn where physics has them  per object, view, frame: centroid offset real - sim in cm at the object's
     depth (K3), mask IoU; real masks by the counting protocol (common.Detector); K1 skips -> "unwitnessed"
  A2 table extent and edges  sim table edge (true edges only, not the image border or occluders) -> distance to the real
     table's edge, in cm at the edge's depth; table-mask IoU; each object's distance to the nearest table edge
  D5 contacts  physics: each free body after 0.5 s settle: what it rests on, gap / penetration; looks: contact-shadow
     ratio (table brightness 0.5-1.5 cm around the footprint / 3-5 cm around it), ours vs real
  C1 similarity per region (composite vs real at the policy's 320x180): table, background, robot/gripper, each object:
     colour error (raw and exposure-matched), edge F1 (Canny, 1 px tolerance), sharpness ratio (Laplacian variance)
  C2 background ghosts  (a) detector counts per noun on our composite vs sim objects visible vs real; (b) residue: share
     of background pixels (away from sim and real foreground) where |ours - real| > 40/255 after exposure matching
  C3 wrist table-texture coverage  share of wrist-view table pixels whose mosaic texel was observed (not inpainted);
     share of the reachable table (<= 0.855 m from the base) and of each object's surroundings (+-3 cm, the variant jitter)
  C4 light per view  each fit's recorded overlap, darkness, gap to the runner-up direction, angle between the two fits;
     measured: share of table pixels in shadow, ours vs real
  C5 objects as rendered  per object: median colour ours (sim mask) vs real (detector mask), clipped-pixel share
No bars yet (Q1; REAL2SIM_CHECKLIST_V1 rule 2): numbers, plus "unwitnessed" / "not independent" where they apply.
Writes <out>/image_checks.json and overlay JPGs under <out>/image_checks/. Exploratory, UNVERIFIED.

    MUJOCO_GL=osmesa /home/imerit/Documents/Code/VLA/.venvs/real2sim/bin/python experiments/real2sim_checks/image_checks.py \
        --scene /home/imerit/Documents/Code/VLA/runs/real2sim/ep_orange_block_stereo \
        --episode /home/imerit/Documents/Code/VLA/data/real2sim/droid/ep_orange_block
"""
import argparse, json, os, time
import common  # noqa: F401  (triton before mujoco; res31's path)
from common import SceneFrames, Detector, composites, assign, touches_border, head_noun, to_uint8, footprint, POLICY_HW, MAIN
import numpy as np
import cv2
import mujoco

WORK = (360, 640)  # render / detect size (block_size_check used 1280x720 with min_px 400: same share of the image = 100 px)
MIN_PX = 100
REACH_M = 0.855  # FR3 maximum reach (Franka datasheet value; ASSUMED, not checked against the Menagerie model)
RESIDUE_THR = 40.0


def gray(img):
    return cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2GRAY).astype(np.float32)


def cm_at_depth(px, z, f):
    return 100.0 * px * z / f


def centroid(mk):
    ys, xs = np.nonzero(mk)
    return np.array([xs.mean(), ys.mean()])


def edge_f1(a, b, region):
    """Canny edges of two images inside `region`; 1 px tolerance. (precision, recall, f1) of a against b."""
    ea, eb = cv2.Canny(to_uint8(a), 50, 150) > 0, cv2.Canny(to_uint8(b), 50, 150) > 0
    reg = cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ea, eb = ea & reg, eb & reg
    k = np.ones((3, 3), np.uint8)
    da, db = cv2.dilate(ea.astype(np.uint8), k) > 0, cv2.dilate(eb.astype(np.uint8), k) > 0
    if ea.sum() == 0 or eb.sum() == 0:
        return None
    p, r = float((ea & db).sum() / ea.sum()), float((eb & da).sum() / eb.sum())
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def lap_var(img, region):
    lap = cv2.Laplacian(gray(img), cv2.CV_32F)
    return float(lap[region].var()) if region.sum() > 20 else None


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def contours(img, masks_cols, thick=2):
    out = img.copy()
    for mk, col in masks_cols:
        if mk is None:
            continue
        cs, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, col, thick)
    return out


class Checks:
    def __init__(self, sf, out, use_detector=True, use_composite=True):
        self.sf, self.out = sf, out
        os.makedirs(os.path.join(out, "image_checks"), exist_ok=True)
        self.det = Detector() if use_detector else None
        self._dcache = {}
        self.comp_note = "not requested"
        self.comp_policy = self.comp_work = (None, None)
        if use_composite:
            e, w, note = composites(sf, POLICY_HW)
            self.comp_note = note
            if e is not None:
                self.comp_policy = (e, w)
                self.comp_work = composites(sf, WORK)[:2]
        m = sf.m
        self.nouns = {b: head_noun(sf.label.get(m.body(b).name, m.body(b).name)) for b in sf.free}

    def detect(self, key, img, noun, robot):
        k = (key, noun)
        if k not in self._dcache:
            self._dcache[k] = self.det.masks(img, noun, robot)
        return self._dcache[k]

    # ---------------------------------------------------------------- per-frame collection
    def collect(self):
        sf, m = self.sf, self.sf.m
        self.rows = []
        for f in sf.frames():
            rf = sf.pose(f)
            row = dict(frame=f, real_frame=rf, cams={})
            for cam in ("ext", "wrist"):
                geom, depth = sf.render(cam, WORK)
                robot = np.isin(geom, sf.robot)
                robot_d = cv2.dilate(robot.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                c = dict(geom=geom, depth=depth, robot=robot, robot_d=robot_d, real=sf.real(cam, rf, WORK),
                         R_t=sf.cam_pose(cam), K=sf.K(cam, WORK))
                ec, wc = self.comp_work
                if ec is not None:
                    c["ours"] = to_uint8((ec if cam == "ext" else wc)(sf.d))
                ec, wc = self.comp_policy
                if ec is not None:
                    c["ours_p"] = (ec if cam == "ext" else wc)(sf.d)
                    c["real_p"] = sf.real(cam, rf, POLICY_HW)
                    gp, _ = sf.render(cam, POLICY_HW)
                    c["geom_p"] = gp
                row["cams"][cam] = c
            self.rows.append(row)

    # ---------------------------------------------------------------- D3
    def d3(self):
        sf, m = self.sf, self.sf.m
        res = {}
        for b in sf.free:
            name = m.body(b).name
            r = dict(drawn=bool(any(m.geom_rgba[g, 3] > 0 for g in sf.obj_geoms[b])))
            for cam in ("ext", "wrist"):
                px, in_view = [], []
                for row in self.rows:
                    sf.pose(row["frame"])
                    uv, z = sf.project(cam, WORK, [sf.d.xpos[b]])
                    in_view.append(bool(z[0] > 0 and 0 <= uv[0, 0] < WORK[1] and 0 <= uv[0, 1] < WORK[0]))
                    px.append(int(np.isin(row["cams"][cam]["geom"], sf.obj_geoms[b]).sum()))
                vis = max(px)
                r[cam] = dict(max_px=vis, frames_in_view=int(sum(in_view)), frames=len(px),
                              label=("visible" if vis >= MIN_PX else "small (< %d px)" % MIN_PX if vis > 0 else
                                     "0 px although its centre projects into the image (hidden behind something, or not drawn)" if any(in_view) else "not in view before the first touch"))
            r["error"] = (not r["drawn"]) or any(r[c]["label"].startswith("0 px") for c in ("ext", "wrist"))
            res[name] = r
        return res

    # ---------------------------------------------------------------- D1 (+ masks reused by C5, A2)
    def d1(self):
        sf, m = self.sf, self.sf.m
        res, tiles = {}, {"ext": [], "wrist": []}
        self.real_masks = {}  # (frame, cam, body) -> real mask at WORK
        for b in sf.free:
            name = m.body(b).name
            res[name] = {}
            for cam in ("ext", "wrist"):
                rows, skips, far = [], {}, []
                for row in self.rows:
                    c = row["cams"][cam]
                    sim = np.isin(c["geom"], sf.obj_geoms[b]) & ~c["robot"]
                    if sim.sum() < MIN_PX:
                        skips["sim < %d px" % MIN_PX] = skips.get("sim < %d px" % MIN_PX, 0) + 1
                        continue
                    if touches_border(sim):
                        skips["sim touches the image edge"] = skips.get("sim touches the image edge", 0) + 1
                        continue
                    cands = self.detect((row["frame"], cam, "real"), c["real"], self.nouns[b], c["robot_d"])
                    real, iou = assign(cands, sim)
                    if real is None:
                        if not cands:
                            skips["no detection of this noun"] = skips.get("no detection of this noun", 0) + 1
                            continue
                        # 10-28: on c11 (built from the bad anchored calibration) every real block lay outside the gate,
                        # which read as "unwitnessed" and hid the misplacement. Report the nearest same-noun detection.
                        cs = centroid(sim)
                        near = min(cands, key=lambda mk: np.linalg.norm(centroid(mk) - cs))
                        off = centroid(near) - cs
                        z = float(np.median(c["depth"][sim]))
                        far.append(dict(frame=row["frame"], offset_px=off.round(1).tolist(), depth_m=round(z, 3),
                                        offset_cm=round(float(np.hypot(cm_at_depth(off[0], z, c["K"][0, 0]), cm_at_depth(off[1], z, c["K"][1, 1]))), 2)))
                        if len(tiles[cam]) < 8:
                            tiles[cam].append((row["frame"], name + " FAR", contours(c["real"], [(sim, (255, 255, 255)), (near, (255, 0, 0))])))
                        continue
                    if touches_border(real):
                        skips["real touches the image edge"] = skips.get("real touches the image edge", 0) + 1
                        continue
                    self.real_masks[(row["frame"], cam, b)] = real
                    off = centroid(real) - centroid(sim)
                    z = float(np.median(c["depth"][sim]))
                    dx, dy = cm_at_depth(off[0], z, c["K"][0, 0]), cm_at_depth(off[1], z, c["K"][1, 1])
                    rows.append(dict(frame=row["frame"], depth_m=round(z, 3), offset_px=off.round(1).tolist(), dx_cm=round(dx, 2),
                                     dy_cm=round(dy, 2), offset_cm=round(float(np.hypot(dx, dy)), 2), iou=round(iou, 3),
                                     sim_px=int(sim.sum()), real_px=int(real.sum())))
                    if len(tiles[cam]) < 8:
                        tiles[cam].append((row["frame"], name, contours(c["real"], [(sim, (255, 255, 255)), (real, (0, 0, 0))])))
                res[name][cam] = dict(n_frames=len(rows), skipped=skips,
                                      offset_cm_median=med([r["offset_cm"] for r in rows]),
                                      dx_cm_median=med([r["dx_cm"] for r in rows]), dy_cm_median=med([r["dy_cm"] for r in rows]),
                                      iou_median=med([r["iou"] for r in rows]),
                                      far_frames=len(far), nearest_detection_offset_cm_median=med([r["offset_cm"] for r in far]),
                                      label=("measured" if rows else
                                             "FAR: no real detection within 60 px of the sim object (nearest same-noun detection "
                                             f"{med([r['offset_cm'] for r in far]):.1f} cm away; may be another instance)" if far else "unwitnessed"),
                                      frames=rows, far=far)
                if cam == "ext":
                    res[name][cam]["independent"] = False
        self._tiles("d1", tiles)
        return res

    def _tiles(self, tag, tiles):
        for cam, ts in tiles.items():
            if not ts:
                continue
            ims = [cv2.putText(cv2.resize(t, (WORK[1] // 2, WORK[0] // 2)), f"{f} {n}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                               (255, 255, 0), 1) for f, n, t in ts]
            while len(ims) % 4:
                ims.append(np.zeros_like(ims[0]))
            grid = np.vstack([np.hstack(ims[i:i + 4]) for i in range(0, len(ims), 4)])
            cv2.imwrite(os.path.join(self.out, "image_checks", f"{tag}_{cam}.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])

    # ---------------------------------------------------------------- A2
    def a2(self):
        sf, m = self.sf, self.sf.m
        tab = sf.m.geom("table").id
        fit_t = sf.fit["table"]
        res = dict(table_label=sf.table_label(), views={}, objects_to_edge_cm={})
        tiles = {"ext": [], "wrist": []}
        for cam in ("ext", "wrist"):
            rows = []
            for row in self.rows:
                c = row["cams"][cam]
                sim_t = c["geom"] == tab
                if sim_t.sum() < MIN_PX:
                    continue
                if cam == "ext" and row["frame"] == 0:  # the build's own table mask (perception.npz), frame 0
                    per = np.load(os.path.join(sf.scene, "perception.npz"))
                    key = f"mask/{sf.table_label()}"
                    real_t = cv2.resize(per[key].astype(np.uint8), (WORK[1], WORK[0]), interpolation=cv2.INTER_NEAREST) > 0 if key in per else None
                else:
                    # a table fills much of the image: no max-area filter for it
                    old = self.det.max_frac
                    self.det.max_frac = 1.0
                    cands = self.detect((row["frame"], cam, "real_table"), c["real"], "table", c["robot_d"])
                    self.det.max_frac = old
                    real_t = max(cands, key=lambda mk: (mk & sim_t).sum()) if cands else None
                if real_t is None:
                    continue
                fg = np.isin(c["geom"], np.r_[sf.robot, [g for b in sf.free for g in sf.obj_geoms[b]]])
                fg_d = cv2.dilate(fg.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                edge = cv2.morphologyEx(sim_t.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
                edge &= sim_t
                b = np.zeros_like(edge)
                b[2:-2, 2:-2] = True
                edge &= b & ~fg_d  # true table edges only
                redge = cv2.morphologyEx(real_t.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
                redge &= b
                iou = float((sim_t & real_t).sum() / max((sim_t | real_t).sum(), 1))
                r = dict(frame=row["frame"], iou=round(iou, 3), sim_edge_px=int(edge.sum()))
                if edge.sum() >= 20 and redge.sum() > 0:
                    dist = cv2.distanceTransform((~redge).astype(np.uint8), cv2.DIST_L2, 3)
                    dpx = dist[edge]
                    dcm = cm_at_depth(dpx, c["depth"][edge], c["K"][0, 0])
                    r.update(edge_dist_cm_median=round(float(np.median(dcm)), 2), edge_dist_cm_p90=round(float(np.percentile(dcm, 90)), 2))
                rows.append(r)
                if len(tiles[cam]) < 4:
                    tiles[cam].append((row["frame"], "table", contours(c["real"], [(sim_t, (255, 255, 255)), (real_t, (0, 0, 0))])))
            with_edge = [r for r in rows if "edge_dist_cm_median" in r]
            res["views"][cam] = dict(n_frames=len(rows), iou_median=med([r["iou"] for r in rows]),
                                     edge_dist_cm_median=med([r["edge_dist_cm_median"] for r in with_edge]),
                                     label=("measured" if with_edge else "no table edge in view (edge unwitnessed)" if rows else "unwitnessed"),
                                     independent=cam == "wrist" or None, frames=rows)
            if cam == "ext":
                res["views"][cam]["independent"] = False  # the table's extent was fitted from this camera's mask
        self._tiles("a2", tiles)
        sf.pose(0)
        lo, hi = np.array(fit_t["xy_lo"]), np.array(fit_t["xy_hi"])
        for b in sf.free:
            p = sf.d.xpos[b][:2]
            res["objects_to_edge_cm"][m.body(b).name] = round(100 * float(min((p - lo).min(), (hi - p).min())), 1)
        return res

    # ---------------------------------------------------------------- D5
    def d5(self):
        sf, m = self.sf, self.sf.m
        d = mujoco.MjData(m)
        d.qpos[:] = sf.st["qpos"][0]
        arm_u = [m.actuator(f"fr3_joint{k}").id for k in range(1, 8)]
        arm_q = [m.joint(f"fr3_joint{k}").qposadr[0] for k in range(1, 8)]
        d.ctrl[arm_u] = d.qpos[arm_q]
        for _ in range(int(0.5 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        tz = sf.fit["table"]["z"]
        tab = m.geom("table").id
        res = {}
        for b in sf.free:
            og = set(sf.obj_geoms[b])
            with_table, others, pen = [], set(), []
            for i in range(d.ncon):
                ct = d.contact[i]
                g1, g2 = ct.geom1, ct.geom2
                if (g1 in og) == (g2 in og):
                    continue
                other = g2 if g1 in og else g1
                if other == tab:
                    with_table.append(float(ct.dist))
                elif m.body_rootid[m.geom_bodyid[other]] != b:
                    others.add(m.body(m.body_rootid[m.geom_bodyid[other]]).name)
                pen.append(float(ct.dist))
            _, _, zmin = footprint(m, d, sf.obj_geoms[b])
            on_floor = "world" in others  # the floor plane belongs to the world body (10-28: c1b's cups, 75 cm down)
            rests = ("table" if with_table else "FLOOR (off the table)" if on_floor else ("on " + ", ".join(sorted(others))) if others
                     else "FLOATING (no contact)")
            deepest = min(pen) if pen else None
            res[m.body(b).name] = dict(rests_on=rests, bottom_minus_table_cm=round(100 * (zmin - tz), 2),
                                       deepest_contact_cm=None if deepest is None else round(100 * deepest, 3),
                                       error=bool(not pen or on_floor or (deepest is not None and deepest < -0.002)),
                                       note="bottom from collision bounding boxes: loose for meshes (c3h apple reads -2 cm on the table)")
        # contact shadow, frame 0, both views (composite needed)
        if self.comp_policy[0] is not None:
            row = self.rows[0]
            sf.pose(row["frame"])
            for b in sf.free:
                name = m.body(b).name
                p, rad, _ = footprint(m, sf.d, sf.obj_geoms[b])
                ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
                ring = lambda r0: np.c_[p[0] + r0 * np.cos(ang), p[1] + r0 * np.sin(ang), np.full_like(ang, tz)]
                near = np.vstack([ring(rad + dr) for dr in (0.005, 0.01, 0.015)])
                far = np.vstack([ring(rad + dr) for dr in (0.03, 0.04, 0.05)])
                for cam in ("ext", "wrist"):
                    c = row["cams"][cam]
                    fg = np.isin(c["geom"], np.r_[sf.robot, [g for bb in sf.free for g in sf.obj_geoms[bb]]])
                    fg = cv2.dilate(fg.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                    vals = {}
                    for tag, img in (("ours", gray(c["ours"])), ("real", gray(c["real"]))):
                        out = []
                        for P in (near, far):
                            uv, z = sf.project(cam, WORK, P)
                            ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < WORK[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < WORK[0])
                            u, v = uv[ok, 0].astype(int), uv[ok, 1].astype(int)
                            keep = ~fg[v, u]
                            out.append(float(img[v[keep], u[keep]].mean()) if keep.sum() >= 10 else None)
                        vals[tag] = None if None in out else round(out[0] / max(out[1], 1.0), 3)
                    res[name][f"contact_shadow_ratio_{cam}"] = vals
        return res

    # ---------------------------------------------------------------- C1
    def c1(self):
        sf, m = self.sf, self.sf.m
        if self.comp_policy[0] is None:
            return dict(label="not run", reason=self.comp_note)
        tab = m.geom("table").id
        res = {}
        for cam in ("ext", "wrist"):
            regs = {}
            for row in self.rows:
                c = row["cams"][cam]
                ours, real, g = c["ours_p"], c["real_p"].astype(np.float32), c["geom_p"]
                regions = {"table": g == tab, "robot" if cam == "ext" else "gripper": np.isin(g, sf.robot)}
                objs = np.zeros_like(g, bool)
                for b in sf.free:
                    mk = np.isin(g, sf.obj_geoms[b])
                    objs |= mk
                    regions[m.body(b).name] = mk
                regions["background"] = ~(regions["table"] | np.isin(g, sf.robot) | objs)
                gain = np.median(gray(real)[regions["table"]]) / max(np.median(gray(ours)[regions["table"]]), 1.0) if regions["table"].sum() > 50 else 1.0
                for name, mk in regions.items():
                    if mk.sum() < 20:
                        continue
                    e = np.abs(ours[mk] - real[mk]).mean(0) / 255.0
                    eg = np.abs(np.clip(ours[mk] * gain, 0, 255) - real[mk]).mean(0) / 255.0
                    ef = edge_f1(ours, real, mk)
                    lo, lr = lap_var(ours, mk), lap_var(real, mk)
                    regs.setdefault(name, []).append(dict(frame=row["frame"], px=int(mk.sum()), colour_err=e.round(3).tolist(),
                                                          colour_err_exposure_matched=eg.round(3).tolist(), exposure_gain=round(float(gain), 3),
                                                          edge_f1=None if ef is None else round(ef[2], 3),
                                                          edge_precision=None if ef is None else round(ef[0], 3),
                                                          edge_recall=None if ef is None else round(ef[1], 3),
                                                          sharpness_ratio=None if not lo or not lr else round(lo / lr, 3)))
            res[cam] = {n: dict(n_frames=len(v), colour_err_mean=med([float(np.mean(x["colour_err"])) for x in v]),
                                colour_err_exposure_matched_mean=med([float(np.mean(x["colour_err_exposure_matched"])) for x in v]),
                                edge_f1=med([x["edge_f1"] for x in v]), sharpness_ratio=med([x["sharpness_ratio"] for x in v]), frames=v)
                        for n, v in regs.items()}
            # side-by-side tiles at the policy's size
            ims = [np.hstack([to_uint8(r["cams"][cam]["real_p"]), to_uint8(r["cams"][cam]["ours_p"])]) for r in self.rows[:4]]
            cv2.imwrite(os.path.join(self.out, "image_checks", f"c1_{cam}.jpg"), cv2.cvtColor(np.vstack(ims), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
        res["note"] = ("composite: " + self.comp_note + ". Region masks come from the sim; where the sim draws an object off the real "
                       "one (D1), its region mixes look and placement.")
        return res

    # ---------------------------------------------------------------- C2
    def c2(self, d3res):
        sf, m = self.sf, self.sf.m
        if self.comp_work[0] is None:
            return dict(label="not run", reason=self.comp_note)
        nouns = sorted(set(self.nouns[b] for b in sf.free) | {head_noun(lab) for lab in sf.fit.get("redacted", {})})
        res = dict(nouns=nouns, counts={"ext": [], "wrist": []}, residue={"ext": [], "wrist": []})
        for cam in ("ext", "wrist"):
            heat = []
            for row in self.rows:
                c = row["cams"][cam]
                cnt = {}
                for nn in nouns:
                    n_sim = sum(1 for b in sf.free if self.nouns[b] == nn and np.isin(c["geom"], sf.obj_geoms[b]).sum() >= MIN_PX)
                    n_ours = len(self.detect((row["frame"], cam, "ours"), c["ours"], nn, c["robot_d"]))
                    n_real = len(self.detect((row["frame"], cam, "real"), c["real"], nn, c["robot_d"]))
                    # ghost = on our composite but neither in the sim nor in the real frame; an extra instance the real
                    # frame shows too is a real thing that wasn't built, or a detector false positive in both (orange: tape)
                    cnt[nn] = dict(sim_visible=n_sim, ours=n_ours, real=n_real, ghost_suspect=n_ours > max(n_sim, n_real),
                                   unbuilt_or_false_positive=min(n_ours, n_real) > n_sim)
                res["counts"][cam].append(dict(frame=row["frame"], per_noun=cnt))
                # residue at the policy's size
                ours, real, g = c["ours_p"], c["real_p"].astype(np.float32), c["geom_p"]
                fg = np.isin(g, np.r_[sf.robot, [gg for b in sf.free for gg in sf.obj_geoms[b]]])
                realfg = np.zeros(POLICY_HW, bool)
                for nn in nouns:
                    for mk in self.detect((row["frame"], cam, "real"), c["real"], nn, c["robot_d"]):
                        realfg |= cv2.resize(mk.astype(np.uint8), (POLICY_HW[1], POLICY_HW[0]), interpolation=cv2.INTER_NEAREST) > 0
                excl = cv2.dilate((fg | realfg).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                bg = ~excl
                tabp = g == m.geom("table").id
                gain = np.median(gray(real)[tabp]) / max(np.median(gray(ours)[tabp]), 1.0) if tabp.sum() > 50 else 1.0
                diff = np.abs(np.clip(gray(ours) * gain, 0, 255) - gray(real))
                bad = (diff > RESIDUE_THR) & bg
                res["residue"][cam].append(dict(frame=row["frame"], bg_px=int(bg.sum()), residue_share=round(float(bad.sum() / max(bg.sum(), 1)), 4)))
                if len(heat) < 4:
                    h = to_uint8(real).copy()
                    h[bad] = (255, 0, 255)
                    heat.append(np.hstack([h, to_uint8(ours)]))
            if heat:
                cv2.imwrite(os.path.join(self.out, "image_checks", f"c2_residue_{cam}.jpg"), cv2.cvtColor(np.vstack(heat), cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
            res[f"residue_share_median_{cam}"] = med([r["residue_share"] for r in res["residue"][cam]])
            res[f"ghost_suspect_frames_{cam}"] = [r["frame"] for r in res["counts"][cam] if any(v["ghost_suspect"] for v in r["per_noun"].values())]
            res[f"unbuilt_or_false_positive_frames_{cam}"] = [r["frame"] for r in res["counts"][cam]
                                                              if any(v["unbuilt_or_false_positive"] for v in r["per_noun"].values())]
        res["note"] = ("(a) counts: one detector call per noun on our composite and on the real frame, robot masked; ghost_suspect = "
                       "more instances on our composite than both the sim objects visible and the real frame; "
                       "unbuilt_or_false_positive = an extra instance on both our composite and the real frame. (b) residue: magenta = background pixels (away from "
                       f"sim and real foreground) differing by > {RESIDUE_THR:.0f}/255 after exposure matching.")
        return res

    # ---------------------------------------------------------------- C3
    def c3(self):
        sf, m = self.sf, self.sf.m
        p = os.path.join(sf.scene, "table_tex_wrist.npz")
        if not os.path.exists(p):
            return dict(label="not run", reason="no table_tex_wrist.npz")
        z = np.load(p)
        if "observed" not in z.files:
            return dict(label="not run", reason="table_tex_wrist.npz has no 'observed' map (older cache)")
        obs, x0, y0, res_ = z["observed"].astype(bool), float(z["x0"]), float(z["y0"]), float(z["res"])
        ny, nx = obs.shape
        tab = m.geom("table").id
        frames = []
        for row in self.rows:
            c = row["cams"]["wrist"]
            tp = c["geom"] == tab
            if tp.sum() < MIN_PX:
                continue
            R, t = c["R_t"]
            K = c["K"]
            v, u = np.nonzero(tp)
            rays = np.c_[(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(u, float)]
            pw = (rays * c["depth"][v, u][:, None]) @ R.T + t
            ix = ((pw[:, 0] - x0) / res_).astype(int)
            iy = ((pw[:, 1] - y0) / res_).astype(int)
            ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            cov = float(obs[iy[ok], ix[ok]].mean()) if ok.any() else 0.0
            frames.append(dict(frame=row["frame"], table_px=int(tp.sum()), observed_share=round(cov, 3)))
        gx, gy = np.meshgrid(x0 + (np.arange(nx) + 0.5) * res_, y0 + (np.arange(ny) + 0.5) * res_)
        reach = np.hypot(gx, gy) <= REACH_M
        sf.pose(0)
        around = {}
        for b in sf.free:
            p0, r, _ = footprint(m, sf.d, sf.obj_geoms[b])
            dist = np.hypot(gx - p0[0], gy - p0[1]) - r
            share = lambda ring: round(float(obs[ring].mean()), 3) if ring.any() else None
            # 0-3 cm: what a +-3 cm variant uncovers; wrist_table_texture leaves out obj_margin (1.5 cm) + a 2 cm LaMa
            # zone around every object, so this ring is filled-in by construction. 3-6 cm: where a moved object lands.
            around[m.body(b).name] = {"0-3cm": share((dist > 0) & (dist <= 0.03)), "3-6cm": share((dist > 0.03) & (dist <= 0.06))}
        return dict(per_frame=frames, observed_share_wrist_view_median=med([f["observed_share"] for f in frames]),
                    observed_share_reachable_table=round(float(obs[reach].mean()), 3) if reach.any() else None,
                    observed_share_whole_table=round(float(obs.mean()), 3), observed_share_around_objects=around,
                    reach_m=REACH_M, note="observed = texel seen in a real wrist frame (not LaMa-filled); table_tex_wrist.npz 'observed'")

    # ---------------------------------------------------------------- C4
    def c4(self):
        """Light per view: what each fit recorded (res31's calibrate_light / light_fit_wrist), how distinct its optimum is,
        how far apart the two fits are, and, measured here, the share of table pixels in shadow ours vs real."""
        import ast
        sf, m = self.sf, self.sf.m
        res = {}
        dirs = {}
        for cam, fn, key in (("ext", "light.json", "shadow_iou"), ("wrist", "light_wrist.json", "iou")):
            p = os.path.join(sf.scene, fn)
            if not os.path.exists(p):
                res[cam] = dict(label="no fit", file=fn)
                continue
            j = json.load(open(p))
            top = j.get("top5")
            top = ast.literal_eval(top) if isinstance(top, str) else top
            scores = [t[0] if isinstance(t, (list, tuple)) else t.get("iou") for t in (top or [])]
            dirs[cam] = np.asarray(j["dir"], float)
            res[cam] = dict(file=fn, azimuth_deg=j.get("azimuth_deg"), elevation_deg=j.get("elevation_deg"), fit_iou=j.get(key),
                            real_shadow_ratio=j.get("real_shadow_ratio"), frames=j.get("frames"),
                            best_minus_second=None if len(scores) < 2 else round(scores[0] - scores[1], 4))
        if len(dirs) == 2:
            res["angle_between_fits_deg"] = round(float(np.degrees(np.arccos(np.clip(dirs["ext"] @ dirs["wrist"], -1, 1)))), 1)
        if self.comp_policy[0] is not None:
            tab = m.geom("table").id
            for cam in ("ext", "wrist"):
                rows = []
                for row in self.rows:
                    c = row["cams"][cam]
                    t = c["geom_p"] == tab
                    fg = cv2.dilate(np.isin(c["geom_p"], np.r_[sf.robot, [g for b in sf.free for g in sf.obj_geoms[b]]]).astype(np.uint8),
                                    np.ones((5, 5), np.uint8)) > 0
                    t &= ~fg
                    if t.sum() < 200:
                        continue
                    sh = {}
                    for tag, img in (("ours", c["ours_p"]), ("real", c["real_p"])):
                        g = gray(img)
                        local = cv2.medianBlur(to_uint8(g), 31).astype(np.float32)  # the table's own brightness nearby
                        sh[tag] = round(float(((g < 0.85 * np.maximum(local, 1)) & t).sum() / t.sum()), 4)
                        # broad shadows (wider than the 31 px window) are their own local median: also vs the whole table
                        sh[tag + "_broad"] = round(float(((g < 0.85 * np.median(g[t])) & t).sum() / t.sum()), 4)
                    rows.append(dict(frame=row["frame"], shadow_share_ours=sh["ours"], shadow_share_real=sh["real"],
                                     broad_shadow_share_ours=sh["ours_broad"], broad_shadow_share_real=sh["real_broad"]))
                res.setdefault(cam, {})["shadow_share"] = dict(ours_median=med([r["shadow_share_ours"] for r in rows]),
                                                              real_median=med([r["shadow_share_real"] for r in rows]),
                                                              broad_ours_median=med([r["broad_shadow_share_ours"] for r in rows]),
                                                              broad_real_median=med([r["broad_shadow_share_real"] for r in rows]), frames=rows)
        res["note"] = ("fit_iou: overlap of the real dark mask with the sim shadow at the fitted light (res31's fits); best_minus_second: "
                       "gap to the runner-up direction (small = flat optimum, the direction is a guess); shadow_share: share of visible "
                       "table pixels (robot/objects excluded) darker than 0.85x their 31-px neighbourhood median (sharp shadows); broad_ = darker than 0.85x the "
                       "visible table's median (wide shadows; also counts darker wood), at the policy's size.")
        return res

    # ---------------------------------------------------------------- C5
    def c5(self):
        sf, m = self.sf, self.sf.m
        if self.comp_work[0] is None:
            return dict(label="not run", reason=self.comp_note)
        res = {}
        k = np.ones((5, 5), np.uint8)
        for b in sf.free:
            name = m.body(b).name
            res[name] = {}
            for cam in ("ext", "wrist"):
                rows = []
                for row in self.rows:
                    real = self.real_masks.get((row["frame"], cam, b))
                    if real is None:
                        continue
                    c = row["cams"][cam]
                    sim = np.isin(c["geom"], sf.obj_geoms[b]) & ~c["robot"]
                    si, ri = cv2.erode(sim.astype(np.uint8), k) > 0, cv2.erode(real.astype(np.uint8), k) > 0
                    if si.sum() < 30 or ri.sum() < 30:
                        continue
                    o, r = c["ours"][si].astype(float), c["real"][ri].astype(float)
                    rows.append(dict(frame=row["frame"], ours_rgb=(np.median(o, 0) / 255).round(3).tolist(),
                                     real_rgb=(np.median(r, 0) / 255).round(3).tolist(),
                                     ours_clipped=round(float((o >= 250).any(1).mean()), 3), real_clipped=round(float((r >= 250).any(1).mean()), 3)))
                if rows:
                    O, Rr = np.array([x["ours_rgb"] for x in rows]), np.array([x["real_rgb"] for x in rows])
                    res[name][cam] = dict(n_frames=len(rows), ours_rgb=np.median(O, 0).round(3).tolist(), real_rgb=np.median(Rr, 0).round(3).tolist(),
                                          err_per_channel=np.abs(np.median(O, 0) - np.median(Rr, 0)).round(3).tolist(),
                                          ours_clipped=med([x["ours_clipped"] for x in rows]), real_clipped=med([x["real_clipped"] for x in rows]),
                                          frames=rows)
                else:
                    res[name][cam] = dict(n_frames=0, label="unwitnessed")
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", default=None, help="default: <main checkout>/runs/real2sim_checks/<scene name>")
    ap.add_argument("--checks", default="d3,d1,a2,d5,c1,c2,c3,c4,c5")
    ap.add_argument("--no_composite", action="store_true", help="geometry checks only (D1, D3, D5 physics, A2)")
    ap.add_argument("--hide", action="append", default=[], help="known-bad test: draw this body nowhere (alpha 0), keep its physics")
    ap.add_argument("--real2sim", default=None, help="res31's experiments/real2sim, imported read-only (read by common.py)")
    a = ap.parse_args()
    out = a.out or os.path.join(MAIN, "runs/real2sim_checks", os.path.basename(os.path.normpath(a.scene)))
    todo = a.checks.split(",")
    t0 = time.time()
    sf = SceneFrames(a.scene, a.episode)
    need_det = any(x in todo for x in ("d1", "a2", "c2", "c5"))
    ch = Checks(sf, out, use_detector=need_det, use_composite=not a.no_composite and any(x in todo for x in ("c1", "c2", "c4", "c5", "d5")))
    for name in a.hide:  # known-bad test: after the compositors set their colours, so it stays hidden in every render
        for g in sf.obj_geoms[sf.m.body(name).id]:
            sf.m.geom_rgba[g, 3] = 0.0
    ch.collect()
    res = dict(scene=a.scene, episode=a.episode, frames=sf.frames(), work_hw=WORK, policy_hw=POLICY_HW, composite=ch.comp_note,
               status="exploratory, UNVERIFIED", checks={})
    order = ["d3", "d1", "a2", "d5", "c1", "c2", "c3", "c4", "c5"]
    if "c5" in todo and "d1" not in todo:
        todo.append("d1")  # C5 uses D1's real masks
    for k in [x for x in order if x in todo]:
        t1 = time.time()
        res["checks"][k] = ch.c2(res["checks"].get("d3")) if k == "c2" else getattr(ch, k)()
        print(f"{k}: done ({time.time() - t1:.0f} s)", flush=True)
        json.dump(res, open(os.path.join(out, "image_checks.json"), "w"), indent=1, default=float)
    res["detector_calls"] = ch.det.calls if ch.det else 0
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(out, "image_checks.json"), "w"), indent=1, default=float)
    print(f"written {os.path.join(out, 'image_checks.json')} ({res['seconds']} s, {res['detector_calls']} detector calls)")


if __name__ == "__main__":
    main()
